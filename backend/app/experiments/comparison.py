"""Cross-condition comparison and aggregation without selection bias.

Conditions
----------
* ``A_caption``     plain VLM caption
* ``D_detailed``    detailed prose prompt (``natural_v1`` template over the same structured attributes)
* ``B_structured``  structured, validated tag prompt
* ``C_refined``     iterative refinement started from a prompt (usually B)

Refinement is reported as several *views*, never collapsed into one optimistic number:

* ``C_iter0``          the session's baseline generation (iteration 0, before any edit)
* ``C_refined_final``  the last successfully scored iteration — what the loop actually ended with
* ``C_refined_best``   the iteration with the best *optimisation* score. **Selected post hoc → biased upward**
                       on the optimisation metric and correlated metrics; shown with ``selection_biased: true``.

A generation can belong to several views (e.g. final == best). Views are not partitions, so their n's must not
be added. Paired comparisons only pair generations of the *same reference* with *identical* generation
configuration (model, seed, steps, CFG, resolution) and report wins/losses/ties per metric in the metric's own
direction — not just means.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import GenerationRun, HumanRating, Reconstruction, RefinementSession
from app.evaluation.metrics import METRICS
from app.prompt_engine.validator import approx_clip_tokens

CONDITION_ORDER = ("A_caption", "D_detailed", "B_structured", "C_iter0", "C_refined_final", "C_refined_best")
SELECTION_BIASED = {"C_refined_best"}
PAIRS = (("A_caption", "B_structured"), ("A_caption", "D_detailed"), ("D_detailed", "B_structured"),
         ("B_structured", "C_refined_final"), ("C_iter0", "C_refined_final"))
COMPARISON_METRICS = ("clip_l14", "dinov2_base", "ssim", "lpips_alex", "clip_b32")
CONFIG_KEYS = ("model_key", "seed", "steps", "cfg_scale", "width", "height", "lora")


def includes_warmup(g: GenerationRun) -> bool:
    """First generation after a model load (flagged by the runner; for older rows, a model load inside the same
    job implies it)."""
    ri = g.runtime_info or {}
    return bool(ri.get("first_call_after_load")) or (ri.get("model_load_s") or 0) > 0.5


def config_key(g: GenerationRun) -> tuple[Any, ...]:
    p = g.params or {}
    return tuple([g.model_key] + [p.get(k) for k in CONFIG_KEYS[1:]])


def _refinement_views(db: Session, recon_ids: list[str]) -> dict[str, dict[str, Any]]:
    """generation_id -> {"session_id", "iteration", "views": [...], "iterations_executed"}"""
    out: dict[str, dict[str, Any]] = {}
    if not recon_ids:
        return out
    for s in db.scalars(select(RefinementSession).where(RefinementSession.reconstruction_id.in_(recon_ids))):
        scored = [i for i in s.iterations if i.status == "scored" and i.generation_id]
        executed = [i for i in s.iterations if i.status in ("scored", "failed")]
        final = max(scored, key=lambda i: i.iteration) if scored else None
        for i in s.iterations:
            if not i.generation_id:
                continue
            views = []
            if i.iteration == 0:
                views.append("C_iter0")
            if final is not None and i.id == final.id and i.iteration > 0:
                views.append("C_refined_final")
            if i.id == s.best_iteration_id and i.iteration > 0:
                views.append("C_refined_best")
            out[i.generation_id] = {"session_id": s.id, "iteration": i.iteration, "views": views,
                                    "iterations_executed": len(executed), "session_status": s.status,
                                    "optimization_metric": s.optimization_metric}
    return out


def _ratings(db: Session, gen_ids: list[str]) -> dict[str, list[HumanRating]]:
    out: dict[str, list[HumanRating]] = defaultdict(list)
    if gen_ids:
        for h in db.scalars(select(HumanRating).where(HumanRating.generation_id.in_(gen_ids))):
            out[h.generation_id].append(h)
    return out


def generation_row(g: GenerationRun, ref: dict[str, Any] | None, ratings: list[HumanRating]) -> dict[str, Any]:
    pv = g.prompt_version
    metrics = {m.metric: m for m in g.metrics if m.role == "evaluation"}
    opt = [m for m in g.metrics if m.role == "optimization"]
    groups = ([g.condition] if g.condition else []) + (ref["views"] if ref else [])
    return {
        "generation_id": g.id, "reconstruction_id": g.reconstruction_id, "condition": g.condition,
        "groups": groups, "status": g.status, "created_at": g.created_at,
        "output_image_url": f"/api/images/{g.output_image_id}/file" if g.output_image_id else None,
        "prompt_version": pv.version, "prompt_source": pv.source, "template_id": pv.template_id,
        "prompt": pv.positive, "prompt_chars": len(pv.positive), "prompt_words": len(pv.positive.split()),
        "prompt_clip_tokens_approx": approx_clip_tokens(pv.positive),
        "model_key": g.model_key, "params": {k: (g.params or {}).get(k) for k in CONFIG_KEYS[1:]},
        "generator": f"lora:{(g.params or {}).get('lora')}" if (g.params or {}).get("lora") else "base",
        "config_key": "|".join(str(x) for x in config_key(g)),
        "device": (g.runtime_info or {}).get("device"), "generation_s": g.duration_s,
        "includes_warmup": includes_warmup(g),
        "peak_memory_gb": ((g.runtime_info or {}).get("memory") or {}).get("peak_process_gb"),
        "refinement_session_id": ref["session_id"] if ref else None,
        "refinement_iteration": ref["iteration"] if ref else None,
        "refinement_iterations_executed": ref["iterations_executed"] if ref else None,
        "metrics": {k: {"value": m.value, "status": m.status, "higher_is_better": m.higher_is_better}
                    for k, m in metrics.items()},
        "optimization": [{"metric": m.metric, "value": m.value, "status": m.status} for m in opt],
        "human": {"n": len(ratings), "mean_similarity": (sum(r.similarity for r in ratings) / len(ratings))
                  if ratings else None},
    }


def reconstruction_table(db: Session, r: Reconstruction) -> dict[str, Any]:
    refs = _refinement_views(db, [r.id])
    ratings = _ratings(db, [g.id for g in r.generations])
    rows = [generation_row(g, refs.get(g.id), ratings.get(g.id, [])) for g in r.generations]
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["config_key"]].append(row["generation_id"])
    return {"reconstruction_id": r.id, "rows": rows, "config_groups": dict(groups),
            "metrics": _metric_meta(), "condition_order": CONDITION_ORDER, "selection_biased": sorted(SELECTION_BIASED)}


def _metric_meta() -> dict[str, Any]:
    return {k: {"label": METRICS[k].label, "higher_is_better": METRICS[k].higher_is_better} for k in COMPARISON_METRICS}


def _summary(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "min": min(values), "max": max(values)}


def aggregate(db: Session, experiment_id: str | None = None) -> dict[str, Any]:
    """Per-group summaries of completed *evaluation* metrics, paired win/loss/tie counts, and human ratings."""
    q = select(Reconstruction)
    if experiment_id:
        q = q.where(Reconstruction.experiment_id == experiment_id)
    recons = db.scalars(q).all()
    refs = _refinement_views(db, [r.id for r in recons])
    gens = [g for r in recons for g in r.generations if g.status == "completed"]
    ratings = _ratings(db, [g.id for g in gens])

    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    human: dict[str, list[int]] = defaultdict(list)
    extras: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # (reconstruction, config, group) -> generation (latest wins, deterministic by created_at order)
    cell: dict[tuple[str, tuple[Any, ...], str], GenerationRun] = {}
    for g in sorted(gens, key=lambda x: x.created_at):
        ref = refs.get(g.id)
        groups = ([g.condition] if g.condition else []) + (ref["views"] if ref else [])
        if not groups:
            groups = ["unlabelled"]
        for grp in groups:
            for m in g.metrics:
                if m.role == "evaluation" and m.status == "completed" and m.value is not None:
                    by_group[grp][m.metric].append(m.value)
            human[grp].extend(h.similarity for h in ratings.get(g.id, []))
            if g.duration_s is not None and not includes_warmup(g):
                extras[grp]["generation_s"].append(g.duration_s)  # warm-up runs excluded from timing
            extras[grp]["prompt_clip_tokens_approx"].append(approx_clip_tokens(g.prompt_version.positive))
            if ref and grp.startswith("C_refined"):
                extras[grp]["iterations_executed"].append(ref["iterations_executed"])
            cell[(g.reconstruction_id, config_key(g), grp)] = g

    groups_out = {}
    for grp in sorted(set(by_group) | set(human) | set(extras),
                      key=lambda x: CONDITION_ORDER.index(x) if x in CONDITION_ORDER else 99):
        groups_out[grp] = {
            "selection_biased": grp in SELECTION_BIASED,
            "metrics": {k: _summary(v) for k, v in by_group[grp].items() if v},
            "human_similarity": _summary([float(x) for x in human[grp]]) if human[grp] else None,
            "other": {k: _summary(v) for k, v in extras[grp].items() if v},
        }

    paired = []
    for a, b in PAIRS:
        for metric in COMPARISON_METRICS:
            hib = METRICS[metric].higher_is_better
            wins = losses = ties = 0
            diffs: list[float] = []
            for (rid, cfg, grp), ga in cell.items():
                if grp != a or (rid, cfg, b) not in cell:
                    continue
                gb = cell[(rid, cfg, b)]
                va = next((m.value for m in ga.metrics if m.metric == metric and m.role == "evaluation"
                           and m.status == "completed"), None)
                vb = next((m.value for m in gb.metrics if m.metric == metric and m.role == "evaluation"
                           and m.status == "completed"), None)
                if va is None or vb is None:
                    continue
                d = vb - va
                diffs.append(d)
                better = d > 1e-9 if hib else d < -1e-9
                worse = d < -1e-9 if hib else d > 1e-9
                wins += better
                losses += worse
                ties += not better and not worse
            if diffs:
                paired.append({"a": a, "b": b, "metric": metric, "higher_is_better": hib, "n_pairs": len(diffs),
                               "b_better": wins, "b_worse": losses, "ties": ties,
                               "mean_diff_b_minus_a": sum(diffs) / len(diffs),
                               "selection_biased": a in SELECTION_BIASED or b in SELECTION_BIASED})
    # Base vs LoRA: same reference, same prompt version, identical settings except the adapter.
    by_cfg: dict[tuple[Any, ...], dict[str | None, GenerationRun]] = defaultdict(dict)
    for g in sorted(gens, key=lambda x: x.created_at):
        p = g.params or {}
        k = (g.reconstruction_id, g.prompt_version_id, g.model_key, *[p.get(c) for c in CONFIG_KEYS[1:-1]])
        by_cfg[k][p.get("lora")] = g
    generator_pairs = []
    adapters = sorted({lo for d in by_cfg.values() for lo in d if lo})
    for ad in adapters:
        for metric in COMPARISON_METRICS:
            hib = METRICS[metric].higher_is_better
            diffs = []
            for d in by_cfg.values():
                if None not in d or ad not in d:
                    continue
                vb = next((m.value for m in d[None].metrics if m.metric == metric and m.role == "evaluation"
                           and m.status == "completed"), None)
                vl = next((m.value for m in d[ad].metrics if m.metric == metric and m.role == "evaluation"
                           and m.status == "completed"), None)
                if vb is not None and vl is not None:
                    diffs.append(vl - vb)
            if diffs:
                better = sum((x > 1e-9) if hib else (x < -1e-9) for x in diffs)
                worse = sum((x < -1e-9) if hib else (x > 1e-9) for x in diffs)
                generator_pairs.append({"lora": ad, "metric": metric, "higher_is_better": hib, "n_pairs": len(diffs),
                                        "lora_better": better, "lora_worse": worse,
                                        "ties": len(diffs) - better - worse,
                                        "mean_diff_lora_minus_base": sum(diffs) / len(diffs)})
    return {"groups": groups_out, "paired": paired, "generator_pairs": generator_pairs, "metrics": _metric_meta(), "condition_order": CONDITION_ORDER,
            "n_reconstructions": len(recons), "n_completed_generations": len(gens),
            "notes": ["Values are genuine completed metric results; missing values are excluded, never imputed.",
                      "C_refined_best is selected by the optimisation metric after the fact and is biased upward.",
                      "Views overlap (a generation can be both final and best); do not sum their n.",
                      "Generation time excludes the first run after a model load (one-off warm-up).",
                      "Paired comparisons use only generations of the same reference with identical model, seed, "
                      "steps, CFG and resolution."]}
