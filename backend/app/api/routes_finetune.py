"""Read-only endpoints for fine-tuning: dataset card, runs (config, live status, loss curves) and test-split
evaluation results. Training and evaluation are started from the command line (they need the machine's memory
for tens of minutes); the dashboard shows the exact commands and follows progress here."""
from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.finetune import dataset as D
from app.finetune import registry as R

router = APIRouter(prefix="/api/finetune", tags=["finetune"])
_EVAL_FILE = re.compile(r"^\d{3}_(ref|base|lora)\.jpg$")


@router.get("/dataset")
def dataset_card() -> dict[str, Any]:
    c = D.card()
    return {"built": c is not None, "card": c,
            "raw_parts_present": sorted(p.name for p in D.RAW_DIR.glob("part-*") if p.is_dir())
            if D.RAW_DIR.exists() else []}


@router.get("/runs")
def runs() -> list[dict[str, Any]]:
    return R.list_runs()


@router.get("/runs/{run_id}")
def run(run_id: str) -> dict[str, Any]:
    try:
        p = R.run_path(run_id)
        s = R.summary(run_id)
    except R.RunNotFound as e:
        raise HTTPException(404, str(e)) from e
    return s | {"train_log": R.read_jsonl(p / "train_log.jsonl", limit=600),
                "val_log": R.read_jsonl(p / "val_log.jsonl"),
                "checkpoints": sorted(c.name for c in (p / "checkpoints").iterdir()) if (p / "checkpoints").exists()
                else []}


@router.get("/runs/{run_id}/eval")
def run_eval(run_id: str) -> dict[str, Any]:
    import json

    try:
        p = R.run_path(run_id) / "eval" / "results.json"
    except R.RunNotFound as e:
        raise HTTPException(404, str(e)) from e
    if not p.is_file():
        raise HTTPException(404, "run has not been evaluated yet")
    return json.loads(p.read_text())


@router.get("/runs/{run_id}/files/{name}")
def run_file(run_id: str, name: str) -> FileResponse:
    if not _EVAL_FILE.fullmatch(name):
        raise HTTPException(400, "invalid file name")
    try:
        p = R.run_path(run_id) / "eval" / name
    except R.RunNotFound as e:
        raise HTTPException(404, str(e)) from e
    if not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="image/jpeg")
