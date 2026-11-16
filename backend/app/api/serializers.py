"""Convert ORM rows into JSON-friendly dicts for the API."""
from __future__ import annotations

from typing import Any

from app.database.models import (
    Analysis,
    GenerationRun,
    Image,
    Job,
    MetricResult,
    PromptVersion,
    Reconstruction,
    RefinementSession,
)
from app.evaluation.metrics import METRICS


def image(i: Image | None) -> dict[str, Any] | None:
    if i is None:
        return None
    return {"id": i.id, "kind": i.kind, "filename": i.filename, "width": i.width, "height": i.height,
            "mime": i.mime, "sha256": i.sha256, "size_bytes": i.size_bytes, "url": f"/api/images/{i.id}/file",
            "created_at": i.created_at}


def analysis(a: Analysis) -> dict[str, Any]:
    return {"id": a.id, "reconstruction_id": a.reconstruction_id, "image_id": a.image_id, "mode": a.mode,
            "provider": a.provider, "model": a.model, "status": a.status, "result": a.result, "caption": a.caption,
            "raw_output": a.raw_output, "error": a.error, "duration_s": a.duration_s, "job_id": a.job_id,
            "is_manual": a.provider == "manual", "created_at": a.created_at}


def prompt_version(p: PromptVersion) -> dict[str, Any]:
    return {"id": p.id, "reconstruction_id": p.reconstruction_id, "version": p.version, "parent_id": p.parent_id,
            "analysis_id": p.analysis_id, "source": p.source, "template_id": p.template_id,
            "attributes": p.attributes, "positive": p.positive, "negative": p.negative, "validation": p.validation,
            "note": p.note, "created_at": p.created_at}


def metric(m: MetricResult) -> dict[str, Any]:
    info = METRICS.get(m.metric)
    return {"id": m.id, "metric": m.metric, "label": info.label if info else m.metric, "role": m.role,
            "status": m.status, "value": m.value, "higher_is_better": m.higher_is_better, "backend": m.backend,
            "detail": m.detail, "job_id": m.job_id, "created_at": m.created_at}


def generation(g: GenerationRun) -> dict[str, Any]:
    return {"id": g.id, "reconstruction_id": g.reconstruction_id, "prompt_version_id": g.prompt_version_id,
            "prompt_version": g.prompt_version.version, "positive": g.prompt_version.positive,
            "condition": g.condition, "provider": g.provider, "model_key": g.model_key, "model_id": g.model_id,
            "params": g.params, "status": g.status, "output_image": image(g.output_image), "error": g.error,
            "duration_s": g.duration_s, "runtime_info": g.runtime_info, "job_id": g.job_id,
            "metrics": [metric(m) for m in g.metrics], "created_at": g.created_at}


def refinement_session(s: RefinementSession) -> dict[str, Any]:
    return {"id": s.id, "reconstruction_id": s.reconstruction_id, "base_prompt_version_id": s.base_prompt_version_id,
            "optimization_metric": s.optimization_metric, "max_iterations": s.max_iterations, "patience": s.patience,
            "min_delta": s.min_delta, "generation_config": s.generation_config, "status": s.status,
            "stop_reason": s.stop_reason, "best_iteration_id": s.best_iteration_id, "created_at": s.created_at,
            "iterations": [{"id": i.id, "iteration": i.iteration, "status": i.status, "proposal": i.proposal,
                            "prompt_version_id": i.prompt_version_id, "generation_id": i.generation_id,
                            "optimization_score": i.optimization_score, "decided_at": i.decided_at,
                            "created_at": i.created_at} for i in s.iterations]}


def reconstruction_summary(r: Reconstruction) -> dict[str, Any]:
    return {"id": r.id, "name": r.name, "notes": r.notes, "experiment_id": r.experiment_id, "image": image(r.image),
            "created_at": r.created_at, "n_analyses": len(r.analyses), "n_prompts": len(r.prompt_versions),
            "n_generations": len(r.generations),
            "n_completed_generations": sum(g.status == "completed" for g in r.generations)}


def job(j: Job) -> dict[str, Any]:
    return {"id": j.id, "type": j.type, "status": j.status, "payload": j.payload, "result": j.result,
            "error": j.error, "attempts": j.attempts, "max_attempts": j.max_attempts, "worker_id": j.worker_id,
            "progress": j.progress, "progress_message": j.progress_message, "lease_expires_at": j.lease_expires_at,
            "created_at": j.created_at, "started_at": j.started_at, "finished_at": j.finished_at}
