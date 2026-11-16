"""Apply job outcomes reported by the GPU worker to the domain tables."""
from __future__ import annotations

import io
from typing import Any

from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.database.models import Analysis, GenerationRun, Job, MetricResult
from app.services import storage
from app.workers.schemas import RESULT_MODELS, AnalyzeResult, EvaluateResult, GenerateResult


class ResultError(ValueError):
    pass


def _immediate(fn, *args):  # noqa: ANN001, ANN202
    return fn(*args)


def on_job_started(db: Session, job: Job) -> None:
    p = job.payload
    if job.type == "analyze" and (a := db.get(Analysis, p["analysis_id"])):
        a.status = "running"
    elif job.type == "generate" and (g := db.get(GenerationRun, p["generation_id"])):
        g.status = "running"
    elif job.type == "evaluate":
        for req in p["metrics"]:
            if (m := db.get(MetricResult, req["metric_result_id"])) and m.status == "queued":
                m.status = "running"
    db.commit()


def validate_result(job: Job, result: dict[str, Any]) -> dict[str, Any]:
    return RESULT_MODELS[job.type].model_validate(result).model_dump()


def apply_job_result(db: Session, job: Job, result: dict[str, Any], image_bytes: bytes | None) -> None:
    from app.services.pipeline import record_analysis_output, record_generation_output

    p = job.payload
    if job.type == "analyze":
        r = AnalyzeResult.model_validate(result)
        a = db.get(Analysis, p["analysis_id"])
        if a is None:
            raise ResultError("analysis row missing")
        record_analysis_output(db, a, r.raw_output, r.model, r.duration_s, r.runtime_info | {"job_id": job.id})
        from app.refinement.controller import on_analysis_completed

        on_analysis_completed(db, a)
    elif job.type == "generate":
        r = GenerateResult.model_validate(result)
        g = db.get(GenerationRun, p["generation_id"])
        if g is None:
            raise ResultError("generation row missing")
        if not image_bytes:
            raise ResultError("generate job completed without an image upload")
        storage.inspect_image(image_bytes)  # validates format/size
        pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        if pil.size != (p["width"], p["height"]):
            raise ResultError(f"uploaded image is {pil.size}, expected {(p['width'], p['height'])}")
        record_generation_output(db, g, pil, r.duration_s, r.runtime_info | {"job_id": job.id}, defer=_immediate)
    elif job.type == "evaluate":
        r = EvaluateResult.model_validate(result)
        allowed = {req["metric_result_id"] for req in p["metrics"]}
        from app.refinement.controller import on_metric_completed

        for mv in r.metrics:
            if mv.metric_result_id not in allowed:
                raise ResultError(f"metric result {mv.metric_result_id} not part of this job")
            m = db.get(MetricResult, mv.metric_result_id)
            if m is None:
                continue
            if mv.status == "completed" and mv.value is None:
                raise ResultError("completed metric must carry a value")
            m.status = mv.status
            m.value = mv.value if mv.status == "completed" else None
            m.backend = mv.backend or m.backend
            m.detail = mv.detail
            db.commit()
            on_metric_completed(db, m)
        # Anything the worker silently skipped is marked failed rather than left pending forever.
        for mid in allowed - {mv.metric_result_id for mv in r.metrics}:
            if (m := db.get(MetricResult, mid)) and m.status in ("queued", "running"):
                m.status, m.detail = "failed", "Worker returned no result for this metric"
        db.commit()


def on_job_finalised(db: Session, job: Job) -> None:
    """Propagate a terminal failure/cancellation to the domain rows so the UI never shows a stale 'queued'."""
    if job.status not in ("failed", "cancelled"):
        return
    msg = f"Job {job.status}: {job.error or 'no detail'}"
    p = job.payload
    if job.type == "analyze" and (a := db.get(Analysis, p["analysis_id"])):
        a.status, a.error = "failed", msg
    elif job.type == "generate" and (g := db.get(GenerationRun, p["generation_id"])):
        g.status, g.error = "failed", msg
        db.commit()
        from app.refinement.controller import on_generation_failed

        on_generation_failed(db, g)
    elif job.type == "evaluate":
        for req in p["metrics"]:
            if (m := db.get(MetricResult, req["metric_result_id"])) and m.status in ("queued", "running"):
                m.status, m.detail, m.value = "failed", msg, None
    db.commit()


def on_job_requeued(db: Session, job: Job) -> None:
    p = job.payload
    if job.type == "analyze" and (a := db.get(Analysis, p["analysis_id"])):
        a.status, a.error = "queued", None
    elif job.type == "generate" and (g := db.get(GenerationRun, p["generation_id"])):
        g.status, g.error = "queued", None
    elif job.type == "evaluate":
        for req in p["metrics"]:
            if (m := db.get(MetricResult, req["metric_result_id"])) and m.status in ("running", "failed"):
                m.status, m.detail = "queued", None
    db.commit()
