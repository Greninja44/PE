"""System status, experiments, demo bundles, and job administration."""
from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.providers import ALLOWED_LOCAL_VLMS, VLM_PROVIDER_NAMES, get_vlm_provider
from app.api import serializers as S
from app.api.deps import DB, get_or_404
from app.config.settings import get_settings
from app.database.models import Experiment, Job, MetricResult, Reconstruction, Worker, utcnow
from app.database.session import as_utc
from app.evaluation.metrics import DEFAULT_EVALUATION_METRICS, DEFAULT_OPTIMIZATION_METRIC, METRICS, get_metric
from app.experiments import tracking
from app.generation.base import MODEL_REGISTRY
from app.generation.providers import GENERATION_PROVIDER_NAMES, get_generation_provider
from app.prompt_engine.templates import TEMPLATES
from app.workers import queue
from app.workers.results import on_job_finalised

router = APIRouter(prefix="/api")
WORKER_ONLINE_WINDOW = timedelta(seconds=90)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _workers(db: Session) -> list[dict[str, Any]]:
    now = utcnow()
    out = []
    for w in db.scalars(select(Worker).where(Worker.revoked.is_(False))):
        seen = as_utc(w.last_seen_at)
        out.append({"id": w.id, "name": w.name, "capabilities": w.capabilities, "last_seen_at": seen,
                    "online": bool(seen and now - seen < WORKER_ONLINE_WINDOW)})
    return out


def _hf_cached(repo_id: str) -> bool | None:
    """True if completed weight files of ``repo_id`` are in the local Hugging Face cache (no network access).
    Config files arrive first, so a snapshot folder alone does not mean the model is usable. Weight files are linked
    into the snapshot only once fully downloaded, so require at least one ``.safetensors`` file there and, for
    sharded checkpoints, every shard named in each ``*.safetensors.index.json``. (Stale ``.incomplete`` blobs from an
    interrupted download are ignored.)"""
    import json
    import os
    from pathlib import Path

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        HF_HUB_CACHE = os.path.join(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"), "hub")
    snaps = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return False
    for idx in snaps.glob("*/**/*.safetensors.index.json"):
        try:
            shards = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        if not all((idx.parent / s).exists() for s in shards):
            return False
    return any(p.exists() for p in snaps.glob("*/**/*.safetensors"))


@router.get("/system")
def system(db: Session = DB) -> dict[str, Any]:
    vlm = []
    for name in VLM_PROVIDER_NAMES:
        p = get_vlm_provider(name)
        vlm.append({"name": name, "execution": p.execution, "description": p.description, "model": p.model_id,
                    **p.availability().model_dump()})
    gen = []
    for name in GENERATION_PROVIDER_NAMES:
        p = get_generation_provider(name)
        gen.append({"name": name, "execution": p.execution, "description": p.description,
                    **p.availability().model_dump()})
    metrics = []
    for key, info in METRICS.items():
        local = get_metric(key).availability()
        metrics.append({"key": key, "label": info.label, "higher_is_better": info.higher_is_better,
                        "execution": info.execution, "backend": info.backend, "range": info.range_note,
                        "description": info.description, "available_in_api_process": local.available,
                        "api_process_reason": local.reason,
                        "default_role": "optimization" if key == DEFAULT_OPTIMIZATION_METRIC else (
                            "evaluation" if key in DEFAULT_EVALUATION_METRICS else "optional")})
    counts = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
    workers = _workers(db)
    fp = tracking.code_fingerprint()
    return {
        "app": get_settings().app_name,
        "vlm_providers": vlm, "local_vlm_models": ALLOWED_LOCAL_VLMS,
        "local_vlm_cached": {m: _hf_cached(m) for m in ALLOWED_LOCAL_VLMS},
        "generation_providers": gen,
        "models": [{"key": s.key, "hf_id": s.hf_id, "label": s.label, "family": s.family,
                    "default_steps": s.default_steps, "default_cfg": s.default_cfg, "default_size": s.default_size,
                    "steps_range": s.steps_range, "cfg_range": s.cfg_range, "max_side": s.max_side,
                    "notes": s.notes, "vram_note": s.vram_note, "est_gb": s.est_gb,
                    "recommended_16gb": s.recommended_16gb, "downloaded": _hf_cached(s.hf_id)}
                   for s in MODEL_REGISTRY.values()],
        "metrics": metrics,
        "templates": [{"id": t.id, "description": t.description} for t in TEMPLATES.values()],
        "jobs": {k: counts.get(k, 0) for k in ("queued", "running", "succeeded", "failed", "cancelled")},
        "workers": workers, "gpu_worker_online": any(w["online"] for w in workers),
        "worker_registration_enabled": bool(get_settings().admin_token),
        "code": {"git_commit": fp["git_commit"], "git_dirty": fp["git_dirty"],
                 "source_sha256": fp["source_sha256"][:12]},
    }


@router.get("/fingerprint")
def fingerprint() -> dict[str, Any]:
    return tracking.code_fingerprint()


@router.get("/stats")
def stats(db: Session = DB) -> dict[str, Any]:
    """Aggregate genuine completed metric values per condition (no imputation of missing values)."""
    rows = db.execute(
        select(MetricResult.metric, MetricResult.role, MetricResult.value, MetricResult.generation_id)
        .where(MetricResult.status == "completed")).all()
    from app.database.models import GenerationRun

    cond = dict(db.execute(select(GenerationRun.id, GenerationRun.condition)).all())
    agg: dict[str, dict[str, list[float]]] = {}
    for metric, role, value, gid in rows:
        if role != "evaluation":
            continue
        agg.setdefault(metric, {}).setdefault(cond.get(gid) or "unlabelled", []).append(value)
    out = {m: {c: {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "max": max(v)} for c, v in d.items()}
           for m, d in agg.items()}
    from app.experiments.comparison import aggregate

    return {"by_metric_and_condition": out, "comparison": aggregate(db),
            "counts": {"reconstructions": db.scalar(select(func.count()).select_from(Reconstruction)),
                       "completed_metrics": len(rows)}}


# ---------------------------------------------------------------------------------------- experiments
class ExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    hypothesis: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


@router.post("/experiments", status_code=201)
def create_experiment(body: ExperimentRequest, db: Session = DB) -> dict[str, Any]:
    e = tracking.new_experiment(db, body.name, body.description, body.hypothesis, body.config)
    return _exp(db, e)


def _exp(db: Session, e: Experiment) -> dict[str, Any]:
    recons = db.scalars(select(Reconstruction).where(Reconstruction.experiment_id == e.id)).all()
    return {"id": e.id, "name": e.name, "description": e.description, "hypothesis": e.hypothesis, "config": e.config,
            "code_fingerprint": e.code_fingerprint, "created_at": e.created_at,
            "reconstructions": [S.reconstruction_summary(r) for r in recons]}


@router.get("/experiments")
def list_experiments(db: Session = DB) -> list[dict[str, Any]]:
    return [_exp(db, e) for e in db.scalars(select(Experiment).order_by(Experiment.created_at.desc()))]


@router.get("/experiments/{eid}")
def get_experiment(eid: str, db: Session = DB) -> dict[str, Any]:
    return _exp(db, get_or_404(db, Experiment, eid))


@router.get("/experiments/{eid}/comparison")
def experiment_comparison(eid: str, db: Session = DB) -> dict[str, Any]:
    from app.experiments.comparison import aggregate

    get_or_404(db, Experiment, eid)
    return aggregate(db, eid)


@router.get("/experiments/{eid}/export.json")
def export_json(eid: str, db: Session = DB) -> PlainTextResponse:
    rec = tracking.experiment_record(db, get_or_404(db, Experiment, eid))
    return PlainTextResponse(json.dumps(rec, indent=2, default=str), media_type="application/json",
                             headers={"Content-Disposition": f'attachment; filename="experiment_{eid[:8]}.json"'})


@router.get("/experiments/{eid}/export.csv")
def export_csv(eid: str, db: Session = DB) -> PlainTextResponse:
    rec = tracking.experiment_record(db, get_or_404(db, Experiment, eid))
    return PlainTextResponse(tracking.experiment_csv(rec), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="experiment_{eid[:8]}.csv"'})


@router.post("/experiments/{eid}/reports", status_code=201)
def write_report(eid: str, db: Session = DB) -> dict[str, str]:
    return tracking.write_report(db, get_or_404(db, Experiment, eid))


@router.get("/reports")
def list_reports() -> list[dict[str, Any]]:
    d = get_settings().reports_dir
    if not d.exists():
        return []
    return [{"name": p.name, "size_bytes": p.stat().st_size} for p in sorted(d.iterdir(), reverse=True)
            if p.suffix in (".json", ".csv")]


@router.get("/reports/{name}")
def get_report(name: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(json|csv)", name):
        raise HTTPException(400, "invalid report name")
    p = get_settings().reports_dir / name
    if not p.is_file():
        raise HTTPException(404, "report not found")
    return FileResponse(p)


# ---------------------------------------------------------------------------------------- demo bundles
@router.get("/demo/bundles")
def demo_bundles() -> list[dict[str, Any]]:
    return tracking.list_demo_bundles()


@router.get("/demo/bundles/{bundle_id}")
def demo_bundle(bundle_id: str) -> dict[str, Any]:
    try:
        return tracking.load_demo_bundle(bundle_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, "bundle not found") from e


@router.get("/demo/bundles/{bundle_id}/files/{name}")
def demo_file(bundle_id: str, name: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", bundle_id) or not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", name):
        raise HTTPException(400, "invalid path")
    p = get_settings().experiments_dir / "demo" / bundle_id / name
    if not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="image/jpeg")


# ------------------------------------------------------------------------------------------------ jobs
@router.get("/jobs")
def list_jobs(status: str | None = None, limit: int = 100, db: Session = DB) -> list[dict[str, Any]]:
    q = select(Job).order_by(Job.created_at.desc()).limit(min(limit, 500))
    if status:
        q = q.where(Job.status == status)
    return [S.job(j) for j in db.scalars(q)]


@router.post("/jobs/{jid}/cancel")
def cancel(jid: str, db: Session = DB) -> dict[str, Any]:
    try:
        j = queue.cancel_job(db, jid)
    except queue.JobError as e:
        raise HTTPException(409, str(e)) from e
    on_job_finalised(db, j)
    return S.job(j)


@router.post("/jobs/{jid}/retry")
def retry(jid: str, db: Session = DB) -> dict[str, Any]:
    try:
        j = queue.retry_job(db, jid)
    except queue.JobError as e:
        raise HTTPException(409, str(e)) from e
    from app.workers.results import on_job_requeued

    on_job_requeued(db, j)
    return S.job(j)


@router.get("/workers")
def workers(db: Session = DB) -> list[dict[str, Any]]:
    return _workers(db)
