"""Evaluate a LoRA run against the base model on the held-out TEST split.

    python -m app.finetune.evaluate --run sd15-ddb-r8 --n 40

For each test item: the reference is the DiffusionDB image, the prompt is its original prompt. Base SD 1.5 and
SD 1.5 + LoRA generate with the *same* prompt, seed, steps, CFG and resolution; then CLIP ViT-L/14, DINOv2, SSIM and
LPIPS are computed against the reference. Results are paired per item (LoRA vs base) and reported as wins / losses /
ties plus mean differences — never only the better-looking average. Items are the first ``n`` of the test manifest
(sorted, deterministic); nothing is selected by outcome.

Generation and metrics run sequentially (pipeline unloaded before the metric models load) to fit 16 GB.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.finetune import dataset as D
from app.finetune.registry import adapter_file, adapter_sha256, run_path

METRICS = ("clip_l14", "dinov2_base", "ssim", "lpips_alex")


def paired(base: list[float | None], lora: list[float | None], higher_is_better: bool) -> dict[str, Any]:
    pairs = [(b, l) for b, l in zip(base, lora, strict=True) if b is not None and l is not None]
    better = sum((l > b + 1e-9) if higher_is_better else (l < b - 1e-9) for b, l in pairs)
    worse = sum((l < b - 1e-9) if higher_is_better else (l > b + 1e-9) for b, l in pairs)
    diffs = [l - b for b, l in pairs]
    n = len(pairs)
    mean = sum(diffs) / n if n else None
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5 if n > 1 else None
    return {"n": n, "lora_better": better, "lora_worse": worse, "ties": n - better - worse,
            "mean_base": sum(b for b, _ in pairs) / n if n else None,
            "mean_lora": sum(l for _, l in pairs) / n if n else None, "mean_diff_lora_minus_base": mean,
            "sd_diff": sd, "higher_is_better": higher_is_better}


def run(a: argparse.Namespace) -> int:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image

    from app.evaluation.metrics import METRICS as INFO
    from app.evaluation.metrics import get_metric
    from app.runtime.device import dtype_for, empty_cache, select_device

    rdir = run_path(a.run)
    cfg = json.loads((rdir / "config.json").read_text())
    ad = adapter_file(a.run)
    items = D.load_split("test")[: a.n]
    if not items:
        print("empty test split", file=sys.stderr)
        return 2
    out = rdir / "eval"
    out.mkdir(exist_ok=True)
    dev = select_device(a.device).device
    pipe = StableDiffusionPipeline.from_pretrained(cfg["base_repo"], variant="fp16", torch_dtype=dtype_for(dev, "diffusion"),
                                                   use_safetensors=True).to(dev)
    if dev == "mps":
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.load_lora_weights(str(ad.parent), weight_name=ad.name, adapter_name="ft")
    settings = {"steps": a.steps, "cfg": a.cfg, "size": 512, "seed_rule": f"{a.seed} + item index",
                "negative_prompt": "", "device": dev, "lora_scale": 1.0, "safety_checker": "kept"}
    records = []
    t0 = time.time()
    for i, it in enumerate(items):
        rec: dict[str, Any] = {"index": i, "image": it["image"], "prompt": it["prompt"], "seed": a.seed + i}
        for variant in ("base", "lora"):
            if variant == "base":
                pipe.disable_lora()
            else:
                pipe.enable_lora()
            g = torch.Generator("cpu").manual_seed(a.seed + i)
            res = pipe(it["prompt"], num_inference_steps=a.steps, guidance_scale=a.cfg, width=512, height=512,
                       generator=g)
            img = res.images[0]
            flagged = bool(res.nsfw_content_detected[0]) if res.nsfw_content_detected is not None else False
            name = f"{i:03d}_{variant}.jpg"
            img.save(out / name, quality=90)
            rec[variant] = {"file": name, "nsfw_flagged": flagged}
        with Image.open(D.RAW_DIR / it["image"]) as ref:
            ref.convert("RGB").save(out / f"{i:03d}_ref.jpg", quality=90)
        rec["ref_file"] = f"{i:03d}_ref.jpg"
        records.append(rec)
        print(f"generated {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    gen_s = time.time() - t0
    del pipe
    empty_cache(dev)

    for key in METRICS:
        m = get_metric(key)
        for rec in records:
            ref = Image.open(D.RAW_DIR / rec["image"]).convert("RGB")
            for variant in ("base", "lora"):
                if rec[variant]["nsfw_flagged"]:  # safety-checker black frame: not a valid generation to score
                    rec[variant][key] = None
                    continue
                rec[variant][key] = m.compute(ref, Image.open(out / rec[variant]["file"]).convert("RGB"))
        if hasattr(m, "unload"):
            m.unload()
        empty_cache(dev)
        print(f"metric {key} done", flush=True)

    result = {
        "schema": "pe.finetune.eval.v1", "run_id": a.run, "adapter_sha256": adapter_sha256(a.run),
        "split": "test", "n": len(records), "settings": settings, "generation_seconds": round(gen_s, 1),
        "created_at": datetime.now(UTC).isoformat(),
        "paired": {k: paired([r["base"][k] for r in records], [r["lora"][k] for r in records],
                             INFO[k].higher_is_better) for k in METRICS},
        "nsfw_flagged": {"base": sum(r["base"]["nsfw_flagged"] for r in records),
                         "lora": sum(r["lora"]["nsfw_flagged"] for r in records)},
        "records": records,
        "notes": [
            "Test-split items only (grouped by prompt; no test prompt was used for training or validation).",
            "Reference images are DiffusionDB outputs of Stable Diffusion 1.x, not photographs: a LoRA can improve "
            "here by imitating that distribution without improving reconstruction of real images.",
            "Same prompt, seed, steps, CFG and resolution for base and LoRA; only the adapter differs.",
            "Images flagged by the safety checker are excluded from scoring (counted above).",
        ],
    }
    result["summary"] = {k: {"lora_better": v["lora_better"], "lora_worse": v["lora_worse"], "n": v["n"]}
                         for k, v in result["paired"].items()}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["paired"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.finetune.evaluate")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default=None)
    from app.runtime.power import stay_awake

    with stay_awake("lora evaluation"):
        return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
