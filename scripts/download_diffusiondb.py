"""Download a small, reproducible DiffusionDB (CC0) subset for LoRA fine-tuning.

    backend/.venv/bin/python scripts/download_diffusiondb.py --parts 1 2 3

Fetches ``metadata.parquet`` (prompts, generation settings, NSFW scores; ~195 MB) and the requested image parts
(~600 MB / 1000 images each) into ``backend/data/finetune/diffusiondb/raw/`` (git-ignored) and extracts them.
Resumable: completed parts are skipped without network access. Dataset: poloclub/diffusiondb (Wang et al., 2022).
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "backend" / "data" / "finetune" / "diffusiondb" / "raw"
REPO = "poloclub/diffusiondb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3], help="part numbers 1..2000")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    meta = RAW / "metadata.parquet"
    if not meta.exists():
        print("→ metadata.parquet (~195 MB)", flush=True)
        hf_hub_download(REPO, "metadata.parquet", repo_type="dataset", local_dir=RAW)
    print("✓ metadata.parquet", flush=True)
    for n in a.parts:
        if not 1 <= n <= 2000:
            print(f"part {n} out of range", file=sys.stderr)
            return 2
        name = f"part-{n:06d}"
        dest = RAW / name
        if (dest / f"{name}.json").exists():
            print(f"✓ {name} (already extracted)", flush=True)
            continue
        print(f"→ {name}.zip (~600 MB)", flush=True)
        z = hf_hub_download(REPO, f"images/{name}.zip", repo_type="dataset", local_dir=RAW)
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():  # refuse path traversal in archive members
                target = (dest / m.filename).resolve()
                if dest.resolve() not in target.parents:
                    raise ValueError(f"unsafe path in archive: {m.filename}")
            zf.extractall(dest)
        Path(z).unlink()  # keep only extracted files
        print(f"✓ {name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
