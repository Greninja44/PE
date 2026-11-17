"""Build a filtered, leakage-free DiffusionDB subset.

    python -m app.finetune.dataset build --parts 1 2 3

Filters (all recorded in the dataset card):
* image file present in a downloaded part and decodable;
* native 512×512 (SD 1.5's training resolution — no resizing artefacts);
* ``image_nsfw`` and ``prompt_nsfw`` below conservative thresholds (DiffusionDB's own classifier scores);
* non-empty prompt.

Split: **grouped by normalised prompt text** (lower-case, punctuation and whitespace collapsed). DiffusionDB contains
many images per prompt and many near-identical prompts; splitting per image would leak test prompts into training.
The bucket is derived from SHA-256 of the normalised prompt, so it is deterministic and independent of the parts
downloaded: the same prompt always lands in the same split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config.settings import BACKEND_DIR

DATA_DIR = BACKEND_DIR / "data" / "finetune" / "diffusiondb"
RAW_DIR = DATA_DIR / "raw"
SPLITS = ("train", "val", "test")
_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass
class Filters:
    width: int = 512
    height: int = 512
    max_image_nsfw: float = 0.2
    max_prompt_nsfw: float = 0.2
    min_prompt_chars: int = 3
    split_percent: dict[str, int] = field(default_factory=lambda: {"train": 80, "val": 10, "test": 10})


def normalise_prompt(p: str) -> str:
    return _NON_WORD.sub(" ", p.lower()).strip()


def split_for(prompt: str, pct: dict[str, int]) -> str:
    b = int(hashlib.sha256(normalise_prompt(prompt).encode()).hexdigest()[:8], 16) % 100
    if b < pct["train"]:
        return "train"
    if b < pct["train"] + pct["val"]:
        return "val"
    return "test"


def _read_metadata(raw: Path, parts: list[int]):  # noqa: ANN202
    import pyarrow.parquet as pq

    cols = ["image_name", "prompt", "part_id", "seed", "step", "cfg", "sampler", "width", "height",
            "image_nsfw", "prompt_nsfw"]
    t = pq.read_table(raw / "metadata.parquet", columns=cols,
                      filters=[("part_id", "in", list(parts))])
    return t.to_pylist()


def build(raw: Path = RAW_DIR, out: Path = DATA_DIR, parts: list[int] | None = None,
          filters: Filters | None = None) -> dict[str, Any]:
    from PIL import Image

    f = filters or Filters()
    if parts is None:
        parts = sorted(int(p.name.split("-")[1]) for p in raw.glob("part-*") if p.is_dir())
    if not parts:
        raise FileNotFoundError(f"No extracted DiffusionDB parts in {raw}; run scripts/download_diffusiondb.py")
    rows = _read_metadata(raw, parts)
    reasons: dict[str, int] = {}
    kept: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    seen_sha: set[str] = set()

    def drop(r: str) -> None:
        reasons[r] = reasons.get(r, 0) + 1

    for r in rows:
        path = raw / f"part-{int(r['part_id']):06d}" / r["image_name"]
        prompt = (r["prompt"] or "").strip()
        if not path.is_file():
            drop("image file missing")
            continue
        if (r["width"], r["height"]) != (f.width, f.height):
            drop(f"not {f.width}x{f.height}")
            continue
        if r["image_nsfw"] is None or r["image_nsfw"] >= f.max_image_nsfw:
            drop("image_nsfw above threshold")
            continue
        if r["prompt_nsfw"] is None or r["prompt_nsfw"] >= f.max_prompt_nsfw:
            drop("prompt_nsfw above threshold")
            continue
        if len(prompt) < f.min_prompt_chars:
            drop("empty prompt")
            continue
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_sha:
            drop("duplicate image")
            continue
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            drop("undecodable image")
            continue
        seen_sha.add(sha)
        split = split_for(prompt, f.split_percent)
        kept[split].append({
            "image": path.relative_to(raw).as_posix(), "sha256": sha, "prompt": prompt,
            "prompt_group": hashlib.sha256(normalise_prompt(prompt).encode()).hexdigest()[:16],
            "source_seed": r["seed"], "source_steps": r["step"], "source_cfg": r["cfg"],
            "source_sampler": r["sampler"], "image_nsfw": r["image_nsfw"], "prompt_nsfw": r["prompt_nsfw"],
        })

    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for s in SPLITS:
        kept[s].sort(key=lambda x: x["image"])
        body = "".join(json.dumps(x, sort_keys=True) + "\n" for x in kept[s]).encode()
        (out / f"{s}.jsonl").write_bytes(body)
        hashes[s] = hashlib.sha256(body).hexdigest()
    groups = {s: {x["prompt_group"] for x in kept[s]} for s in SPLITS}
    card = {
        "schema": "pe.finetune.dataset.v1",
        "source": "poloclub/diffusiondb (2M), CC0-1.0 — Wang et al. 2022, 'DiffusionDB: A Large-scale Prompt Gallery "
                  "Dataset for Text-to-Image Generative Models'",
        "note": "DiffusionDB images were generated by Stable Diffusion 1.x from user prompts; they are not photographs.",
        "built_at": datetime.now(UTC).isoformat(), "parts": parts, "filters": asdict(f),
        "candidates": len(rows), "dropped": reasons,
        "counts": {s: len(kept[s]) for s in SPLITS},
        "prompt_groups": {s: len(groups[s]) for s in SPLITS},
        "prompt_group_overlap": {"train∩test": len(groups["train"] & groups["test"]),
                                 "train∩val": len(groups["train"] & groups["val"]),
                                 "val∩test": len(groups["val"] & groups["test"])},
        "split_method": "SHA-256 of normalised prompt → bucket 0..99 (grouped; no prompt shared across splits)",
        "manifest_sha256": hashes,
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    return card


def load_split(split: str, root: Path = DATA_DIR) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    p = root / f"{split}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def card(root: Path = DATA_DIR) -> dict[str, Any] | None:
    p = root / "dataset_card.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.finetune.dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parts", nargs="*", type=int)
    b.add_argument("--max-image-nsfw", type=float, default=0.2)
    b.add_argument("--max-prompt-nsfw", type=float, default=0.2)
    a = ap.parse_args(argv)
    c = build(parts=a.parts, filters=Filters(max_image_nsfw=a.max_image_nsfw, max_prompt_nsfw=a.max_prompt_nsfw))
    print(json.dumps({k: c[k] for k in ("parts", "candidates", "dropped", "counts", "prompt_groups",
                                        "prompt_group_overlap")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
