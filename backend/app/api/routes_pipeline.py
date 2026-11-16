"""Reconstruction pipeline endpoints (stages 1–5)."""
from __future__ import annotations

import difflib
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import serializers as S
from app.api.deps import DB, deferrer, get_or_404
from app.database.models import (
    Analysis,
    Experiment,
    GenerationRun,
    Image,
    PromptVersion,
    Reconstruction,
    RefinementIteration,
    RefinementSession,
)
from app.evaluation.metrics import DEFAULT_EVALUATION_METRICS, DEFAULT_OPTIMIZATION_METRIC
from app.experiments import tracking
from app.generation.base import GenerationParams
from app.prompt_engine.builder import build_prompt, build_validated
from app.prompt_engine.schema import ImageAttributes
from app.refinement import controller as rc
from app.services import pipeline, storage

router = APIRouter(prefix="/api")


def _bad(e: Exception) -> HTTPException:
    return HTTPException(422, str(e))


# ------------------------------------------------------------------------------------------------ images
@router.get("/images/{image_id}/file")
def image_file(image_id: str, db: Session = DB) -> FileResponse:
    img = get_or_404(db, Image, image_id)
    return FileResponse(storage.image_path(img), media_type=img.mime)


# --------------------------------------------------------------------------------------- reconstructions
@router.post("/reconstructions", status_code=201)
async def create_reconstruction(file: UploadFile = File(...), name: str = Form(""), notes: str = Form(""),
                                experiment_id: str | None = Form(None), db: Session = DB) -> dict[str, Any]:
    data = await file.read()
    try:
        img = storage.save_image(db, data, file.filename or "upload", "original")
    except storage.InvalidImage as e:
        raise HTTPException(400, str(e)) from e
    if experiment_id:
        get_or_404(db, Experiment, experiment_id)
    r = Reconstruction(name=name.strip() or (file.filename or "Untitled"), notes=notes, image_id=img.id,
                       experiment_id=experiment_id or None)
    db.add(r)
    db.commit()
    return S.reconstruction_summary(r)


@router.get("/reconstructions")
def list_reconstructions(db: Session = DB) -> list[dict[str, Any]]:
    rs = db.scalars(select(Reconstruction).order_by(Reconstruction.created_at.desc())).all()
    return [S.reconstruction_summary(r) for r in rs]


@router.get("/reconstructions/{rid}")
def get_reconstruction(rid: str, db: Session = DB) -> dict[str, Any]:
    r = get_or_404(db, Reconstruction, rid)
    sessions = db.scalars(select(RefinementSession).where(RefinementSession.reconstruction_id == rid)
                          .order_by(RefinementSession.created_at)).all()
    return S.reconstruction_summary(r) | {
        "analyses": [S.analysis(a) for a in r.analyses],
        "prompt_versions": [S.prompt_version(p) for p in r.prompt_versions],
        "generations": [S.generation(g) for g in r.generations],
        "refinement_sessions": [S.refinement_session(s) for s in sessions],
    }


# ------------------------------------------------------------------------------------- stage 1: analysis
class AnalysisRequest(BaseModel):
    provider: str = "qwen3vl_local"
    model: str | None = None
    mode: Literal["structured", "caption"] = "structured"
    image_id: str | None = None  # default: the reference image


@router.post("/reconstructions/{rid}/analyses", status_code=202)
def request_analysis(rid: str, body: AnalysisRequest, background: BackgroundTasks, db: Session = DB) -> dict[str, Any]:
    r = get_or_404(db, Reconstruction, rid)
    img = get_or_404(db, Image, body.image_id) if body.image_id else None
    try:
        a = pipeline.start_analysis(db, r, body.provider, body.mode, body.model, deferrer(background), image=img)
    except (pipeline.PipelineError, ValueError) as e:
        raise _bad(e) from e
    return S.analysis(a)


class ManualAnalysisRequest(BaseModel):
    attributes: ImageAttributes | None = None
    caption: str | None = None
    annotator: str = Field(default="", max_length=64)


@router.post("/reconstructions/{rid}/analyses/manual", status_code=201)
def manual_analysis(rid: str, body: ManualAnalysisRequest, db: Session = DB) -> dict[str, Any]:
    r = get_or_404(db, Reconstruction, rid)
    try:
        return S.analysis(pipeline.manual_analysis(db, r, body.attributes, body.caption, body.annotator))
    except pipeline.PipelineError as e:
        raise _bad(e) from e


@router.get("/analyses/{aid}")
def get_analysis(aid: str, db: Session = DB) -> dict[str, Any]:
    return S.analysis(get_or_404(db, Analysis, aid))


# -------------------------------------------------------------------------------------- stage 2: prompts
class FromAnalysisRequest(BaseModel):
    analysis_id: str
    template_id: str = "structured_v1"


@router.post("/reconstructions/{rid}/prompts/from-analysis", status_code=201)
def prompts_from_analysis(rid: str, body: FromAnalysisRequest, db: Session = DB) -> list[dict[str, Any]]:
    r = get_or_404(db, Reconstruction, rid)
    a = get_or_404(db, Analysis, body.analysis_id)
    if a.reconstruction_id != rid or a.image_id != r.image_id:
        raise HTTPException(422, "Analysis must be of this reconstruction's reference image")
    try:
        return [S.prompt_version(p) for p in pipeline.prompts_from_analysis(db, r, a, body.template_id)]
    except (pipeline.PipelineError, ValueError) as e:
        raise _bad(e) from e


class EditPromptRequest(BaseModel):
    parent_id: str | None = None
    attributes: ImageAttributes
    template_id: str = "structured_v1"
    note: str = ""
    positive_override: str | None = None
    negative_override: str | None = None


@router.post("/reconstructions/{rid}/prompts", status_code=201)
def create_prompt_version(rid: str, body: EditPromptRequest, db: Session = DB) -> dict[str, Any]:
    r = get_or_404(db, Reconstruction, rid)
    parent = get_or_404(db, PromptVersion, body.parent_id) if body.parent_id else None
    if parent and parent.reconstruction_id != rid:
        raise HTTPException(422, "Parent version belongs to another reconstruction")
    try:
        pv = pipeline.edited_version(db, r, parent, body.attributes, body.template_id, body.note or "Manual edit",
                                     source="edited" if parent else "manual",
                                     positive_override=body.positive_override,
                                     negative_override=body.negative_override)
    except ValueError as e:
        raise _bad(e) from e
    return S.prompt_version(pv)


class PreviewRequest(BaseModel):
    attributes: ImageAttributes
    template_id: str = "structured_v1"
    caption: str | None = None


@router.post("/prompts/preview")
def preview_prompt(body: PreviewRequest) -> dict[str, Any]:
    try:
        if body.template_id == "caption_v1":
            bp = build_prompt(body.attributes, "caption_v1", caption=body.caption)
            return {"original": bp.model_dump(), "validated": bp.model_dump()}
        original, validated = build_validated(body.attributes, body.template_id)
    except ValueError as e:
        raise _bad(e) from e
    return {"original": original.model_dump(), "validated": validated.model_dump()}


@router.get("/prompts/diff")
def prompt_diff(a: str, b: str, db: Session = DB) -> dict[str, Any]:
    pa, pb = get_or_404(db, PromptVersion, a), get_or_404(db, PromptVersion, b)

    def ops(x: str, y: str) -> list[dict[str, str]]:
        xs, ys = x.split(), y.split()
        out = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=xs, b=ys, autojunk=False).get_opcodes():
            if tag == "equal":
                out.append({"op": "equal", "text": " ".join(xs[i1:i2])})
            else:
                if i2 > i1:
                    out.append({"op": "delete", "text": " ".join(xs[i1:i2])})
                if j2 > j1:
                    out.append({"op": "insert", "text": " ".join(ys[j1:j2])})
        return out

    attr_changes = []
    for k in sorted(set(pa.attributes) | set(pb.attributes)):
        if pa.attributes.get(k) != pb.attributes.get(k):
            attr_changes.append({"field": k, "before": pa.attributes.get(k), "after": pb.attributes.get(k)})
    return {"a": S.prompt_version(pa), "b": S.prompt_version(pb), "positive": ops(pa.positive, pb.positive),
            "negative": ops(pa.negative, pb.negative), "attributes": attr_changes}


# ----------------------------------------------------------------------------------- stage 3: generation
class GenerationRequest(BaseModel):
    prompt_version_id: str
    provider: str = "gpu_worker"
    params: GenerationParams = Field(default_factory=GenerationParams)
    condition: Literal["A_caption", "B_structured", "C_refined"] | None = None


@router.post("/reconstructions/{rid}/generations", status_code=202)
def request_generation(rid: str, body: GenerationRequest, background: BackgroundTasks, db: Session = DB
                       ) -> dict[str, Any]:
    r = get_or_404(db, Reconstruction, rid)
    pv = get_or_404(db, PromptVersion, body.prompt_version_id)
    if pv.reconstruction_id != rid:
        raise HTTPException(422, "Prompt version belongs to another reconstruction")
    try:
        g = pipeline.start_generation(db, r, pv, body.params, body.provider, deferrer(background), body.condition)
    except (pipeline.PipelineError, ValueError) as e:
        raise _bad(e) from e
    return S.generation(g)


@router.get("/generations/{gid}")
def get_generation(gid: str, db: Session = DB) -> dict[str, Any]:
    return S.generation(get_or_404(db, GenerationRun, gid))


# ----------------------------------------------------------------------------------- stage 4: evaluation
class EvaluateRequest(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_EVALUATION_METRICS))
    role: Literal["evaluation", "optimization"] = "evaluation"


@router.post("/generations/{gid}/evaluate", status_code=202)
def evaluate(gid: str, body: EvaluateRequest, background: BackgroundTasks, db: Session = DB) -> list[dict[str, Any]]:
    g = get_or_404(db, GenerationRun, gid)
    try:
        rows = pipeline.start_evaluation(db, g, body.metrics, body.role, deferrer(background))
    except pipeline.PipelineError as e:
        raise _bad(e) from e
    return [S.metric(m) for m in rows]


# ------------------------------------------------------------------------- comparison (demo A / B / C)
class ComparisonRequest(BaseModel):
    caption_prompt_version_id: str | None = None
    structured_prompt_version_id: str | None = None
    provider: str = "gpu_worker"
    params: GenerationParams = Field(default_factory=GenerationParams)


@router.post("/reconstructions/{rid}/comparison", status_code=202)
def start_comparison(rid: str, body: ComparisonRequest, background: BackgroundTasks, db: Session = DB
                     ) -> list[dict[str, Any]]:
    """Generate conditions A and B with *identical* model, seed and parameters (controlled comparison).
    Condition C comes from the best iteration of a refinement session started from B."""
    r = get_or_404(db, Reconstruction, rid)
    out = []
    for cond, pv_id in (("A_caption", body.caption_prompt_version_id),
                        ("B_structured", body.structured_prompt_version_id)):
        if not pv_id:
            continue
        pv = get_or_404(db, PromptVersion, pv_id)
        if pv.reconstruction_id != rid:
            raise HTTPException(422, "Prompt version belongs to another reconstruction")
        # A uses the caption's own negative prompt; parameters otherwise identical.
        params = body.params.model_copy(update={"negative_prompt": ""})
        try:
            out.append(S.generation(pipeline.start_generation(db, r, pv, params, body.provider, deferrer(background),
                                                              cond)))
        except (pipeline.PipelineError, ValueError) as e:
            raise _bad(e) from e
    if not out:
        raise HTTPException(422, "Provide at least one prompt version")
    return out


@router.get("/reconstructions/{rid}/comparison")
def get_comparison(rid: str, db: Session = DB) -> dict[str, Any]:
    r = get_or_404(db, Reconstruction, rid)
    comp = tracking.comparison(db, r)
    sess = comp.pop("_session")
    return {
        "reconstruction": S.reconstruction_summary(r),
        "conditions": {k: (S.generation(g) if g else None) for k, g in comp.items()},
        "refinement": ({"session_id": sess[0].id, "best_iteration": sess[1].iteration,
                        "optimization_metric": sess[0].optimization_metric,
                        "history": [{"iteration": i.iteration, "status": i.status, "score": i.optimization_score}
                                    for i in sess[0].iterations]} if sess else None),
    }


@router.post("/reconstructions/{rid}/demo-bundle", status_code=201)
def demo_bundle(rid: str, db: Session = DB) -> dict[str, str]:
    r = get_or_404(db, Reconstruction, rid)
    return {"bundle": tracking.export_demo_bundle(db, r)}


# ----------------------------------------------------------------------------------- stage 5: refinement
class RefinementRequest(BaseModel):
    base_prompt_version_id: str
    optimization_metric: str = DEFAULT_OPTIMIZATION_METRIC
    max_iterations: int = Field(default=5, ge=1, le=50)
    patience: int = Field(default=2, ge=0, le=50)
    min_delta: float = Field(default=0.005, ge=0)
    provider: str = "gpu_worker"
    params: GenerationParams = Field(default_factory=GenerationParams)
    analysis_provider: str | None = "qwen3vl_local"
    analysis_model: str | None = None


@router.post("/reconstructions/{rid}/refinement", status_code=201)
def start_refinement(rid: str, body: RefinementRequest, background: BackgroundTasks, db: Session = DB
                     ) -> dict[str, Any]:
    r = get_or_404(db, Reconstruction, rid)
    pv = get_or_404(db, PromptVersion, body.base_prompt_version_id)
    try:
        s = rc.create_session(db, r, pv, optimization_metric=body.optimization_metric,
                              max_iterations=body.max_iterations, patience=body.patience, min_delta=body.min_delta,
                              generation_provider=body.provider, params=body.params,
                              analysis_provider=body.analysis_provider, analysis_model=body.analysis_model,
                              defer=deferrer(background))
    except (rc.RefinementError, pipeline.PipelineError, ValueError) as e:
        raise _bad(e) from e
    return S.refinement_session(s)


@router.get("/refinement/{sid}")
def get_refinement(sid: str, db: Session = DB) -> dict[str, Any]:
    return S.refinement_session(get_or_404(db, RefinementSession, sid))


class ProposeRequest(BaseModel):
    max_edits: int = Field(default=3, ge=1, le=10)


@router.post("/refinement/{sid}/propose", status_code=201)
def propose(sid: str, body: ProposeRequest, db: Session = DB) -> dict[str, Any]:
    s = get_or_404(db, RefinementSession, sid)
    try:
        rc.create_proposal(db, s, body.max_edits)
    except rc.RefinementError as e:
        raise _bad(e) from e
    db.refresh(s)
    return S.refinement_session(s)


class ApproveRequest(BaseModel):
    edit_indices: list[int] | None = None


@router.post("/refinement/iterations/{iid}/approve")
def approve(iid: str, body: ApproveRequest, background: BackgroundTasks, db: Session = DB) -> dict[str, Any]:
    it = get_or_404(db, RefinementIteration, iid)
    try:
        rc.approve(db, it, body.edit_indices, deferrer(background))
    except (rc.RefinementError, pipeline.PipelineError) as e:
        raise _bad(e) from e
    return S.refinement_session(it.session)


class RejectRequest(BaseModel):
    reason: str = ""


@router.post("/refinement/iterations/{iid}/reject")
def reject(iid: str, body: RejectRequest, db: Session = DB) -> dict[str, Any]:
    it = get_or_404(db, RefinementIteration, iid)
    try:
        rc.reject(db, it, body.reason)
    except rc.RefinementError as e:
        raise _bad(e) from e
    return S.refinement_session(it.session)


@router.post("/refinement/{sid}/stop")
def stop(sid: str, db: Session = DB) -> dict[str, Any]:
    return S.refinement_session(rc.stop(db, get_or_404(db, RefinementSession, sid)))
