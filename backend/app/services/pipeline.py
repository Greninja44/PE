"""Orchestration of the five-stage pipeline.

Inline providers (remote HTTP endpoints, SSIM) run in background tasks inside the API process.
Worker providers (Qwen3-VL local, diffusers, torch metrics) become queued jobs for the GPU worker.
Either way the database row moves through ``queued/pending → running → completed | failed`` and never
holds a value it did not genuinely compute.
"""
from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.base import AnalysisMode, AnalysisParseError, parse_caption_output, parse_structured_output
from app.analysis.providers import get_vlm_provider
from app.database.models import Analysis, GenerationRun, Image, MetricResult, PromptVersion, Reconstruction
from app.database.session import get_sessionmaker
from app.evaluation.metrics import DEFAULT_EVALUATION_METRICS, METRICS, get_metric
from app.generation.base import MODEL_REGISTRY, GenerationParams
from app.generation.providers import get_generation_provider
from app.prompt_engine.builder import BuiltPrompt, build_prompt, build_validated
from app.prompt_engine.schema import ImageAttributes
from app.services import storage
from app.workers.queue import create_job

log = logging.getLogger(__name__)
Defer = Callable[..., Any]


class PipelineError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------- Stage 1
def start_analysis(db: Session, recon: Reconstruction, provider_name: str, mode: AnalysisMode,
                   model: str | None, defer: Defer, image: Image | None = None) -> Analysis:
    provider = get_vlm_provider(provider_name, model)
    avail = provider.availability()
    if not avail.available:
        raise PipelineError(f"Provider '{provider_name}' unavailable: {avail.reason}")
    img = image or recon.image
    a = Analysis(reconstruction_id=recon.id, image_id=img.id, mode=mode, provider=provider.name,
                 model=provider.model_id, status="pending")
    db.add(a)
    db.flush()
    if provider.execution == "worker":
        job = create_job(db, "analyze", {"analysis_id": a.id, "image_id": img.id, "mode": mode,
                                         "model_id": provider.model_id}, commit=False)
        a.job_id, a.status = job.id, "queued"
        db.commit()
    else:
        db.commit()
        defer(run_inline_analysis, a.id, provider_name, model)
    return a


def run_inline_analysis(analysis_id: str, provider_name: str, model: str | None) -> None:
    with get_sessionmaker()() as db:
        a = db.get(Analysis, analysis_id)
        if a is None:
            return
        a.status = "running"
        db.commit()
        try:
            provider = get_vlm_provider(provider_name, model)
            out = provider.analyze(storage.load_pil(db.get(Image, a.image_id)), a.mode)  # type: ignore[arg-type]
            record_analysis_output(db, a, out.raw_output, out.model, out.duration_s, out.runtime_info)
        except Exception as e:  # noqa: BLE001 — persist any failure on the row
            log.exception("inline analysis failed")
            a.status, a.error = "failed", f"{type(e).__name__}: {e}"
            db.commit()


def record_analysis_output(db: Session, a: Analysis, raw_output: str, model: str, duration_s: float,
                           runtime_info: dict[str, Any] | None = None) -> None:
    a.raw_output, a.model, a.duration_s, a.runtime_info = raw_output, model, duration_s, runtime_info
    try:
        if a.mode == "structured":
            a.result = parse_structured_output(raw_output).model_dump()
        else:
            a.caption = parse_caption_output(raw_output)
        a.status, a.error = "completed", None
    except AnalysisParseError as e:
        a.status, a.error = "failed", f"Could not parse model output: {e}"
    db.commit()


def manual_analysis(db: Session, recon: Reconstruction, attributes: ImageAttributes | None, caption: str | None,
                    annotator: str) -> Analysis:
    """A human annotation — clearly labelled provider='manual' so it is never mistaken for model output."""
    if attributes is None and not caption:
        raise PipelineError("Provide attributes or a caption")
    a = Analysis(reconstruction_id=recon.id, image_id=recon.image_id, mode="structured" if attributes else "caption",
                 provider="manual", model=f"human:{annotator or 'anonymous'}", status="completed",
                 result=attributes.model_dump() if attributes else None, caption=caption, duration_s=None)
    db.add(a)
    db.commit()
    return a


# ----------------------------------------------------------------------------------------------- Stage 2
def _next_version(db: Session, recon_id: str) -> int:
    return (db.scalar(select(func.max(PromptVersion.version)).where(PromptVersion.reconstruction_id == recon_id))
            or 0) + 1


def _save_version(db: Session, recon: Reconstruction, built: BuiltPrompt, source: str, parent_id: str | None,
                  analysis_id: str | None, note: str = "", commit: bool = True) -> PromptVersion:
    pv = PromptVersion(reconstruction_id=recon.id, version=_next_version(db, recon.id), parent_id=parent_id,
                       analysis_id=analysis_id, source=source, template_id=built.template_id,
                       attributes=built.attributes.model_dump(), positive=built.positive, negative=built.negative,
                       validation=built.validation.model_dump() | {"segments": built.segments}, note=note)
    db.add(pv)
    db.flush()
    if commit:
        db.commit()
    return pv


def prompts_from_analysis(db: Session, recon: Reconstruction, analysis: Analysis, template_id: str
                          ) -> list[PromptVersion]:
    if analysis.status != "completed":
        raise PipelineError("Analysis has not completed")
    if analysis.mode == "caption":
        built = build_prompt(ImageAttributes(), "caption_v1", caption=analysis.caption)
        return [_save_version(db, recon, built, "caption", None, analysis.id, "Caption-only baseline")]
    attrs = ImageAttributes.model_validate(analysis.result or {})
    original, validated = build_validated(attrs, template_id)
    v1 = _save_version(db, recon, original, "structured", None, analysis.id, "Generated from analysis", commit=False)
    v2 = _save_version(db, recon, validated, "validated", v1.id, analysis.id,
                       f"Validated: {len(validated.validation.fixes_applied)} automatic fix(es)", commit=False)
    db.commit()
    return [v1, v2]


def edited_version(db: Session, recon: Reconstruction, parent: PromptVersion | None, attributes: ImageAttributes,
                   template_id: str, note: str, source: str = "edited", positive_override: str | None = None,
                   negative_override: str | None = None) -> PromptVersion:
    built = build_prompt(attributes, template_id)
    if positive_override is not None or negative_override is not None:
        # Free-text edits are kept verbatim; validation re-runs against the overridden text.
        from app.prompt_engine.builder import validate

        pos = positive_override if positive_override is not None else built.positive
        neg = negative_override if negative_override is not None else built.negative
        neg_terms = [t.strip() for t in neg.split(",") if t.strip()]
        report = validate(attributes, template_id, pos, [s["text"] for s in built.segments],
                          [s["field"] for s in built.segments], neg_terms)
        built = built.model_copy(update={"positive": pos, "negative": neg, "negative_terms": neg_terms,
                                         "validation": report})
    return _save_version(db, recon, built, source, parent.id if parent else None,
                         parent.analysis_id if parent else None, note)


# ----------------------------------------------------------------------------------------------- Stage 3
def start_generation(db: Session, recon: Reconstruction, pv: PromptVersion, params: GenerationParams,
                     provider_name: str, defer: Defer, condition: str | None = None) -> GenerationRun:
    provider = get_generation_provider(provider_name)
    avail = provider.availability()
    if not avail.available:
        raise PipelineError(f"Provider '{provider_name}' unavailable: {avail.reason}")
    spec = MODEL_REGISTRY[params.model_key]
    if not params.negative_prompt:
        params = params.model_copy(update={"negative_prompt": pv.negative})
    g = GenerationRun(reconstruction_id=recon.id, prompt_version_id=pv.id, condition=condition, provider=provider.name,
                      model_key=spec.key, model_id=spec.hf_id, params=params.as_record(), status="pending")
    db.add(g)
    db.flush()
    if provider.execution == "worker":
        job = create_job(db, "generate", {
            "generation_id": g.id, "model_key": spec.key, "prompt": pv.positive,
            "negative_prompt": params.negative_prompt, "seed": params.seed, "steps": params.steps,
            "cfg_scale": params.cfg_scale, "width": params.width, "height": params.height}, commit=False)
        g.job_id, g.status = job.id, "queued"
        db.commit()
    else:
        db.commit()
        defer(run_inline_generation, g.id, provider_name)
    return g


def run_inline_generation(generation_id: str, provider_name: str) -> None:
    with get_sessionmaker()() as db:
        g = db.get(GenerationRun, generation_id)
        if g is None:
            return
        g.status = "running"
        db.commit()
        try:
            params = GenerationParams.model_validate(g.params)
            out = get_generation_provider(provider_name).generate(g.prompt_version.positive, params)
            record_generation_output(db, g, out.image, out.duration_s, out.runtime_info, defer=lambda f, *a: f(*a))
        except Exception as e:  # noqa: BLE001
            log.exception("inline generation failed")
            g.status, g.error = "failed", f"{type(e).__name__}: {e}"
            db.commit()
            from app.refinement.controller import on_generation_failed

            on_generation_failed(db, g)


def record_generation_output(db: Session, g: GenerationRun, pil, duration_s: float,  # noqa: ANN001
                             runtime_info: dict[str, Any], defer: Defer) -> None:
    img = storage.save_pil(db, pil, "generated", f"gen_{g.id}")
    g.output_image_id, g.duration_s, g.runtime_info, g.status, g.error = img.id, duration_s, runtime_info, \
        "completed", None
    db.commit()
    start_evaluation(db, g, list(DEFAULT_EVALUATION_METRICS), "evaluation", defer)
    from app.refinement.controller import on_generation_completed

    on_generation_completed(db, g, defer)


# ----------------------------------------------------------------------------------------------- Stage 4
def start_evaluation(db: Session, g: GenerationRun, metrics: list[str], role: str, defer: Defer
                     ) -> list[MetricResult]:
    if g.status != "completed" or g.output_image_id is None:
        raise PipelineError("Generation has no output image yet")
    if role not in ("evaluation", "optimization"):
        raise PipelineError("role must be 'evaluation' or 'optimization'")
    recon = db.get(Reconstruction, g.reconstruction_id)
    existing = {(m.metric, m.role) for m in g.metrics if m.status in ("queued", "running", "completed")}
    rows: list[MetricResult] = []
    worker_reqs: list[dict[str, str]] = []
    inline_ids: list[str] = []
    for key in metrics:
        if key not in METRICS:
            raise PipelineError(f"Unknown metric '{key}'")
        if (key, role) in existing:
            continue
        info = METRICS[key]
        m = MetricResult(generation_id=g.id, metric=key, role=role, status="queued",
                         higher_is_better=info.higher_is_better, backend=info.backend)
        db.add(m)
        db.flush()
        rows.append(m)
        if info.execution == "inline":
            inline_ids.append(m.id)
        else:
            worker_reqs.append({"metric_result_id": m.id, "metric": key})
    if worker_reqs:
        job = create_job(db, "evaluate", {"generation_id": g.id, "reference_image_id": recon.image_id,
                                          "generated_image_id": g.output_image_id, "metrics": worker_reqs},
                         commit=False)
        for m in rows:
            if METRICS[m.metric].execution == "worker":
                m.job_id = job.id
    db.commit()
    for mid in inline_ids:
        defer(run_inline_metric, mid)
    return rows


def run_inline_metric(metric_result_id: str) -> None:
    with get_sessionmaker()() as db:
        m = db.get(MetricResult, metric_result_id)
        if m is None:
            return
        g = db.get(GenerationRun, m.generation_id)
        recon = db.get(Reconstruction, g.reconstruction_id)
        metric = get_metric(m.metric)
        avail = metric.availability()
        if not avail.available:
            m.status, m.detail = "unavailable", avail.reason
            db.commit()
            return
        try:
            t0 = time.perf_counter()
            ref = storage.load_pil(db.get(Image, recon.image_id))
            gen = storage.load_pil(db.get(Image, g.output_image_id))
            m.value = metric.compute(ref, gen)
            m.status, m.detail = "completed", f"computed in API process in {time.perf_counter() - t0:.2f}s"
        except Exception as e:  # noqa: BLE001
            m.status, m.detail, m.value = "failed", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}", None
        db.commit()
        from app.refinement.controller import on_metric_completed

        on_metric_completed(db, m)
