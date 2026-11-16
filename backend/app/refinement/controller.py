"""Refinement session state machine.

    create_session ─▶ iteration 0 (baseline) ─▶ generate ─▶ score (optimisation metric)
         ┌──────────────────────────────────────────────────────────────┘
         ▼
    propose ─▶ [proposed] ─approve─▶ generating ─▶ scoring ─▶ scored ─▶ policy.decide ─▶ continue | stop
                           └reject─▶ rejected (edits excluded from future proposals)

Proposals always start from the current *best* prompt (hill climbing), so a bad iteration never
contaminates later ones. The optimisation metric is recorded with ``role='optimization'``; the
independent evaluation metrics are computed separately for every generation.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import (
    Analysis,
    GenerationRun,
    MetricResult,
    PromptVersion,
    Reconstruction,
    RefinementIteration,
    RefinementSession,
    utcnow,
)
from app.evaluation.metrics import DEFAULT_EVALUATION_METRICS, METRICS
from app.generation.base import GenerationParams
from app.prompt_engine.schema import ImageAttributes
from app.refinement.policy import decide
from app.refinement.proposer import Edit, Proposal, apply_edits, propose

log = logging.getLogger(__name__)
ACTIVE_ITERATION_STATES = ("proposed", "generating", "scoring")


class RefinementError(ValueError):
    pass


def _iteration_for_generation(db: Session, generation_id: str) -> RefinementIteration | None:
    return db.scalar(select(RefinementIteration).where(RefinementIteration.generation_id == generation_id))


def _execute(db: Session, s: RefinementSession, it: RefinementIteration, pv: PromptVersion, defer) -> None:  # noqa: ANN001
    from app.services.pipeline import start_generation

    recon = db.get(Reconstruction, s.reconstruction_id)
    cfg = s.generation_config
    params = GenerationParams.model_validate(cfg["params"])
    pending: list[tuple[Any, ...]] = []
    g = start_generation(db, recon, pv, params, cfg["provider"], defer=lambda *a: pending.append(a))
    it.generation_id, it.prompt_version_id, it.status = g.id, pv.id, "generating"
    db.commit()
    for call in pending:
        defer(*call)


def create_session(db: Session, recon: Reconstruction, base_pv: PromptVersion, *, optimization_metric: str,
                   max_iterations: int, patience: int, min_delta: float, generation_provider: str,
                   params: GenerationParams, analysis_provider: str | None, analysis_model: str | None,
                   defer) -> RefinementSession:  # noqa: ANN001
    if optimization_metric not in METRICS:
        raise RefinementError(f"Unknown metric '{optimization_metric}'")
    if not 1 <= max_iterations <= 50:
        raise RefinementError("max_iterations must be in [1, 50]")
    if base_pv.reconstruction_id != recon.id:
        raise RefinementError("Prompt version belongs to another reconstruction")
    warnings = []
    if optimization_metric in DEFAULT_EVALUATION_METRICS:
        warnings.append(f"'{optimization_metric}' is also an independent evaluation metric; its evaluation "
                        "results for this session are not independent of the optimisation.")
    s = RefinementSession(reconstruction_id=recon.id, base_prompt_version_id=base_pv.id,
                          optimization_metric=optimization_metric, max_iterations=max_iterations, patience=patience,
                          min_delta=min_delta, status="active",
                          generation_config={"provider": generation_provider, "params": params.as_record(),
                                             "analysis_provider": analysis_provider, "analysis_model": analysis_model,
                                             "warnings": warnings})
    db.add(s)
    db.flush()
    it = RefinementIteration(session_id=s.id, iteration=0, status="approved",
                             proposal={"baseline": True, "edits": [], "rationale": "Baseline prompt"})
    db.add(it)
    db.flush()
    _execute(db, s, it, base_pv, defer)
    return s


def _update_session(db: Session, s: RefinementSession) -> None:
    executed = [i for i in s.iterations if i.status in ("scored", "failed")]
    if not executed:
        return
    info = METRICS[s.optimization_metric]
    scores = [i.optimization_score if i.status == "scored" else None for i in executed]
    d = decide(scores, info.higher_is_better, s.max_iterations, s.patience, s.min_delta)
    s.best_iteration_id = executed[d.best_index].id if d.best_index is not None else None
    busy = any(i.status in ("generating", "scoring") for i in s.iterations)
    if s.status == "active" and d.status != "active" and not busy:
        s.status, s.stop_reason = d.status, d.stop_reason
        for i in s.iterations:  # an outstanding proposal can no longer be executed
            if i.status == "proposed":
                i.status, i.decided_at = "rejected", utcnow()
                i.proposal = (i.proposal or {}) | {"rejection_reason": f"Session ended: {d.stop_reason}"}
    db.commit()


# ---- hooks called by the pipeline ------------------------------------------------------------------------
def on_generation_completed(db: Session, g: GenerationRun, defer) -> None:  # noqa: ANN001
    it = _iteration_for_generation(db, g.id)
    if it is None:
        return
    from app.services.pipeline import PipelineError, start_analysis, start_evaluation

    s = it.session
    it.status = "scoring"
    db.commit()
    start_evaluation(db, g, [s.optimization_metric], "optimization", defer)
    cfg = s.generation_config
    if cfg.get("analysis_provider"):
        try:
            start_analysis(db, db.get(Reconstruction, s.reconstruction_id), cfg["analysis_provider"], "structured",
                           cfg.get("analysis_model"), defer, image=g.output_image)
        except PipelineError as e:
            log.warning("could not analyse generated image: %s", e)


def on_generation_failed(db: Session, g: GenerationRun) -> None:
    it = _iteration_for_generation(db, g.id)
    if it is None:
        return
    it.status = "failed"
    db.commit()
    _update_session(db, it.session)


def on_metric_completed(db: Session, m: MetricResult) -> None:
    if m.role != "optimization":
        return
    it = _iteration_for_generation(db, m.generation_id)
    if it is None or it.session.optimization_metric != m.metric or it.status != "scoring":
        return
    if m.status == "completed" and m.value is not None:
        it.optimization_score, it.status = m.value, "scored"
    elif m.status in ("failed", "unavailable"):
        it.status = "failed"
    else:
        return
    db.commit()
    _update_session(db, it.session)


def on_analysis_completed(db: Session, a: Analysis) -> None:  # noqa: ARG001 — proposals read analyses lazily
    return


# ---- user actions ------------------------------------------------------------------------------------------
def _best(s: RefinementSession) -> RefinementIteration:
    best = next((i for i in s.iterations if i.id == s.best_iteration_id), None)
    if best is None:
        raise RefinementError("No scored iteration yet — wait for the baseline to be generated and scored")
    return best


def _reference_attributes(db: Session, s: RefinementSession) -> ImageAttributes:
    recon = db.get(Reconstruction, s.reconstruction_id)
    a = db.scalar(select(Analysis).where(Analysis.image_id == recon.image_id, Analysis.mode == "structured",
                                         Analysis.status == "completed").order_by(Analysis.created_at.desc()))
    if a and a.result:
        return ImageAttributes.model_validate(a.result)
    return ImageAttributes.model_validate(db.get(PromptVersion, s.base_prompt_version_id).attributes)


def create_proposal(db: Session, s: RefinementSession, max_edits: int = 3) -> RefinementIteration:
    if s.status != "active":
        raise RefinementError(f"Session is {s.status}")
    if any(i.status in ACTIVE_ITERATION_STATES for i in s.iterations):
        raise RefinementError("An iteration is still pending — approve/reject it or wait for scoring")
    best = _best(s)
    best_pv = db.get(PromptVersion, best.prompt_version_id)
    gen = db.get(GenerationRun, best.generation_id)
    gen_attrs = None
    if gen and gen.output_image_id:
        ga = db.scalar(select(Analysis).where(Analysis.image_id == gen.output_image_id, Analysis.mode == "structured",
                                              Analysis.status == "completed").order_by(Analysis.created_at.desc()))
        if ga and ga.result:
            gen_attrs = ImageAttributes.model_validate(ga.result)
    excluded: set[str] = set()
    for i in s.iterations:
        for e in (i.proposal or {}).get("edits", []):
            if i.status == "rejected" or i.status in ("scored", "failed", "generating", "scoring"):
                excluded.add(Edit.model_validate(e).signature())
    p: Proposal = propose(_reference_attributes(db, s), ImageAttributes.model_validate(best_pv.attributes),
                          best_pv.template_id, gen_attrs, excluded, max_edits)
    it = RefinementIteration(session_id=s.id, iteration=max(i.iteration for i in s.iterations) + 1, status="proposed",
                             proposal=p.model_dump() | {"base_prompt_version_id": best_pv.id,
                                                        "base_iteration": best.iteration})
    db.add(it)
    db.commit()
    db.refresh(s)
    return it


def approve(db: Session, it: RefinementIteration, edit_indices: list[int] | None, defer) -> RefinementIteration:  # noqa: ANN001
    from app.services.pipeline import edited_version

    if it.status != "proposed":
        raise RefinementError(f"Iteration is {it.status}, not proposed")
    s = it.session
    p = Proposal.model_validate({k: v for k, v in it.proposal.items() if k in Proposal.model_fields})
    idx = list(range(len(p.edits))) if edit_indices is None else edit_indices
    if any(i < 0 or i >= len(p.edits) for i in idx):
        raise RefinementError("Edit index out of range")
    chosen = [p.edits[i] for i in idx]
    if not chosen:
        raise RefinementError("No edits selected — reject the proposal instead")
    base_pv = db.get(PromptVersion, it.proposal["base_prompt_version_id"])
    attrs = apply_edits(ImageAttributes.model_validate(base_pv.attributes), chosen)
    template = next((e.value for e in chosen if e.op == "template"), base_pv.template_id)
    recon = db.get(Reconstruction, s.reconstruction_id)
    pv = edited_version(db, recon, base_pv, attrs, template, source="refinement",
                        note=f"Refinement iteration {it.iteration}: " + "; ".join(f"{e.op} {e.field}: {e.value}"
                                                                                  for e in chosen))
    it.proposal = it.proposal | {"approved_edit_indices": idx, "edits": [e.model_dump() for e in chosen],
                                 "proposed_edits": [e.model_dump() for e in p.edits]}
    it.status, it.decided_at = "approved", utcnow()
    db.commit()
    _execute(db, s, it, pv, defer)
    return it


def reject(db: Session, it: RefinementIteration, reason: str) -> RefinementIteration:
    if it.status != "proposed":
        raise RefinementError(f"Iteration is {it.status}, not proposed")
    it.status, it.decided_at = "rejected", utcnow()
    it.proposal = (it.proposal or {}) | {"rejection_reason": reason or "Rejected by user"}
    db.commit()
    return it


def stop(db: Session, s: RefinementSession) -> RefinementSession:
    if s.status == "active":
        s.status, s.stop_reason = "stopped", "Stopped by user"
        for i in s.iterations:
            if i.status == "proposed":
                i.status, i.decided_at = "rejected", utcnow()
        db.commit()
    return s
