"""Share fine-tuning artifacts through git (small, deliberate research artifacts only).

    python scripts/finetune_artifacts.py export   # backend/data/finetune → experiments/finetune (tracked)
    python scripts/finetune_artifacts.py import   # experiments/finetune → backend/data/finetune (local, ignored)

Exported: dataset card + split manifests (prompt lists, CC0), and for each COMPLETED run its config, status, training
and validation logs, final LoRA adapter (a few MB) and evaluation results/images if present. Not exported: DiffusionDB
images (~1.9 GB — re-download with scripts/download_diffusiondb.py), latent caches, intermediate checkpoints.
Machine-specific paths (home directory) are replaced with "~" so no personal paths are published.
Import never overwrites an existing local run or manifest.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "backend" / "data" / "finetune"
SHARED = ROOT / "experiments" / "finetune"
DATASET_FILES = ("dataset_card.json", "train.jsonl", "val.jsonl", "test.jsonl")
RUN_FILES = ("config.json", "status.json", "train_log.jsonl", "val_log.jsonl")
_HOME = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\Users\\\\)[^/\\\"]+")


def _sanitise_text(text: str) -> str:
    return _HOME.sub("~", text)


def _copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_sanitise_text(src.read_text()))


def export() -> int:
    n = 0
    for f in DATASET_FILES:
        src = LOCAL / "diffusiondb" / f
        if src.exists():
            _copy_text(src, SHARED / "dataset" / f)
            n += 1
    for run in sorted((LOCAL / "runs").iterdir()) if (LOCAL / "runs").exists() else []:
        status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        if status.get("state") != "completed":
            print(f"skip {run.name}: state={status.get('state')}")
            continue
        dst = SHARED / "runs" / run.name
        for f in RUN_FILES:
            if (run / f).exists():
                _copy_text(run / f, dst / f)
        ad = run / "adapter" / "pytorch_lora_weights.safetensors"
        (dst / "adapter").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ad, dst / "adapter" / ad.name)
        if (run / "eval" / "results.json").exists():
            (dst / "eval").mkdir(exist_ok=True)
            _copy_text(run / "eval" / "results.json", dst / "eval" / "results.json")
            for img in (run / "eval").glob("*.jpg"):
                shutil.copy2(img, dst / "eval" / img.name)
        print(f"exported {run.name}")
        n += 1
    leaked = [p for p in SHARED.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")
              and _HOME.search(p.read_text())]
    if leaked:
        print(f"personal paths still present in: {leaked}", file=sys.stderr)
        return 1
    print(f"done: {n} item(s) in {SHARED.relative_to(ROOT)}")
    return 0


def import_() -> int:
    n = 0
    for f in DATASET_FILES:
        src, dst = SHARED / "dataset" / f, LOCAL / "diffusiondb" / f
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    for run in sorted((SHARED / "runs").iterdir()) if (SHARED / "runs").exists() else []:
        dst = LOCAL / "runs" / run.name
        if dst.exists():
            print(f"keep existing local run {run.name}")
            continue
        shutil.copytree(run, dst)
        (dst / "checkpoints").mkdir(exist_ok=True)
        print(f"imported {run.name}")
        n += 1
    print(f"done: {n} item(s) imported into {LOCAL.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import"])
    return export() if ap.parse_args().cmd == "export" else import_()


if __name__ == "__main__":
    sys.exit(main())
