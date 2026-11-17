"""Fine-tuning run registry (filesystem-backed, git-ignored).

    backend/data/finetune/runs/<run_id>/
        config.json            hyper-parameters, dataset card hash, code fingerprint, device
        status.json            state (preparing|training|completed|stopped|failed), step, losses, timings, memory
        train_log.jsonl        one line per optimisation step (loss, lr, seconds)
        val_log.jsonl          validation loss at fixed noise/timesteps (val split only)
        adapter/pytorch_lora_weights.safetensors     final LoRA (diffusers format)
        checkpoints/step-<n>/pytorch_lora_weights.safetensors
        eval/results.json, eval/*.jpg                base vs LoRA on the held-out test split

Run ids are validated against a strict pattern and resolved only inside the runs directory, so a job payload or API
request can name an adapter but never a path.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.config.settings import BACKEND_DIR

RUNS_DIR = BACKEND_DIR / "data" / "finetune" / "runs"
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ADAPTER_FILE = "pytorch_lora_weights.safetensors"


class RunNotFound(ValueError):
    pass


def runs_dir() -> Path:
    return RUNS_DIR


def run_path(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id or ""):
        raise RunNotFound(f"invalid run id '{run_id}'")
    p = (runs_dir() / run_id).resolve()
    if p.parent != runs_dir().resolve() or not p.is_dir():
        raise RunNotFound(f"run '{run_id}' not found")
    return p


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def adapter_file(run_id: str) -> Path:
    """Final adapter of a completed run (the only one offered for generation)."""
    p = run_path(run_id)
    st = _json(p / "status.json") or {}
    f = p / "adapter" / ADAPTER_FILE
    if st.get("state") != "completed" or not f.is_file():
        raise RunNotFound(f"run '{run_id}' has no completed adapter")
    return f


def adapter_sha256(run_id: str) -> str:
    return hashlib.sha256(adapter_file(run_id).read_bytes()).hexdigest()


def summary(run_id: str) -> dict[str, Any]:
    p = run_path(run_id)
    cfg = _json(p / "config.json") or {}
    st = _json(p / "status.json") or {}
    ev = _json(p / "eval" / "results.json")
    return {"id": run_id, "base_model": cfg.get("base_model"), "config": cfg, "status": st,
            "adapter_available": (p / "adapter" / ADAPTER_FILE).is_file() and st.get("state") == "completed",
            "evaluation": ev and {k: ev.get(k) for k in ("n", "settings", "paired", "summary", "created_at", "notes")}}


def list_runs() -> list[dict[str, Any]]:
    d = runs_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.iterdir(), reverse=True):
        if p.is_dir() and RUN_ID.fullmatch(p.name):
            out.append(summary(p.name))
    return out


def completed_adapters(base_model: str | None = None) -> list[str]:
    return [r["id"] for r in list_runs()
            if r["adapter_available"] and (base_model is None or r["base_model"] == base_model)]


def read_jsonl(p: Path, every: int = 1, limit: int = 5000) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    step = max(every, (len(lines) + limit - 1) // limit) if lines else 1
    return [json.loads(x) for x in lines[::step] if x.strip()]
