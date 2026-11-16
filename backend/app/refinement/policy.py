"""Pure stopping / best-iteration policy for the refinement loop."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    best_index: int | None  # index into the scores list
    stop_reason: str | None  # None = continue
    status: str  # active | stopped_early | max_iterations


def decide(scores: list[float | None], higher_is_better: bool, max_iterations: int, patience: int,
           min_delta: float) -> StopDecision:
    """``scores[0]`` is the baseline; later entries are executed (approved) iterations in order.
    ``None`` = an iteration that failed to produce a score (counts toward the budget, never as best)."""
    best_i: int | None = None
    best: float | None = None
    since_best = 0
    for i, s in enumerate(scores):
        if s is None:
            if best_i is not None:
                since_best += 1
            continue
        better = best is None or (s > best + min_delta if higher_is_better else s < best - min_delta)
        if better:
            best_i, best, since_best = i, s, 0
        else:
            since_best += 1
    executed = len(scores) - 1
    if best_i is not None and since_best >= patience > 0:
        return StopDecision(best_i, f"No improvement > {min_delta} for {since_best} iteration(s) (patience {patience})",
                            "stopped_early")
    if executed >= max_iterations:
        return StopDecision(best_i, f"Reached maximum of {max_iterations} iteration(s)", "max_iterations")
    return StopDecision(best_i, None, "active")
