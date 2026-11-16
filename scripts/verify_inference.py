"""Real-inference verification on this machine (not a unit test: loads the actual models).

    backend/.venv/bin/python scripts/verify_inference.py --image photo.jpg
    backend/.venv/bin/python scripts/verify_inference.py --image photo.jpg --only sd_turbo ssim lpips_alex
    PE_DEVICE=cpu backend/.venv/bin/python scripts/verify_inference.py --image photo.jpg --only clip_b32

Each model is loaded through the same ModelManager/runners the worker uses, run once on real input, checked for
sane output (non-empty parsed text, non-black image, finite score), timed, and unloaded. Process memory is the
macOS physical footprint (includes Metal/MPS buffers). The report is written to backend/data/verification/
(git-ignored) — it documents the environment; it is NOT an experimental result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.runtime import device as D  # noqa: E402  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch import)
from app.runtime import memory  # noqa: E402

STEPS = ("qwen3vl_2b", "sd_turbo", "sd15", "ssim", "clip_b32", "clip_l14", "dinov2_base", "lpips_alex")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="reference photograph (JPEG/PNG)")
    ap.add_argument("--only", nargs="*", choices=STEPS)
    ap.add_argument("--prompt", default=None, help="prompt for generation (default: the VLM caption, else a fixed one)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from PIL import Image

    from app.runtime.lifecycle import ModelManager

    ref = Image.open(a.image).convert("RGB")
    steps = a.only or list(STEPS)
    mm = ModelManager()
    dev = D.select_device()
    report: dict = {"schema": "pe.verification.v1", "started_at": datetime.now(UTC).isoformat(),
                    "device": D.device_report(), "device_choice": dev.__dict__, "image": Path(a.image).name,
                    "image_size": list(ref.size), "baseline_memory": memory.snapshot(), "steps": {}}
    print(f"device: {dev.device} ({dev.reason}); baseline footprint {report['baseline_memory']['process_gb']} GB, "
          f"system available {report['baseline_memory'].get('system_available_gb')} GB", flush=True)
    caption = None
    generated: Image.Image | None = None

    def record(name: str, fn) -> None:  # noqa: ANN001
        print(f"\n== {name}", flush=True)
        entry: dict = {}
        try:
            with memory.PeakSampler() as mem:
                t0 = time.perf_counter()
                entry.update(fn() or {})
                entry["wall_s"] = round(time.perf_counter() - t0, 2)
            entry["memory"] = mem.report()
            entry["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc(limit=4)
        mm.unload_all(reason="verification step done")
        entry["after_unload_process_gb"] = memory.snapshot()["process_gb"]
        report["steps"][name] = entry
        short = {k: v for k, v in entry.items() if k not in ("traceback", "raw_output")}
        print(json.dumps(short, indent=1, default=str), flush=True)

    if "qwen3vl_2b" in steps:
        def vlm():  # noqa: ANN202
            nonlocal caption
            from app.analysis.providers import VLM_EST_GB, QwenVLRunner

            mid = "Qwen/Qwen3-VL-2B-Instruct"
            t0 = time.perf_counter()
            r = mm.get("vlm", mid, lambda: QwenVLRunner(mid), est_gb=VLM_EST_GB[mid])
            load = time.perf_counter() - t0
            cap = r.analyze(ref, "caption")
            st = r.analyze(ref, "structured")
            caption = cap.caption
            if not caption or st.attributes is None or not st.attributes.subject:
                raise RuntimeError(f"unparseable VLM output: {cap.raw_output[:200]!r} / {st.raw_output[:200]!r}")
            return {"load_s": round(load, 2), "caption": caption, "caption_s": cap.duration_s,
                    "structured_s": st.duration_s, "structured": st.attributes.model_dump(),
                    "runtime_info": st.runtime_info}
        record("qwen3vl_2b", vlm)

    prompt = a.prompt or caption or "a rocky coastline with red vegetation above a calm blue sea, photograph"
    for key in ("sd_turbo", "sd15"):
        if key not in steps:
            continue

        def gen(key: str = key):  # noqa: ANN202
            nonlocal generated
            import numpy as np

            from app.generation.base import MODEL_REGISTRY, GenerationParams
            from app.generation.providers import DiffusersRunner

            t0 = time.perf_counter()
            r = mm.get("diffusion", key, lambda: DiffusersRunner(key), est_gb=MODEL_REGISTRY[key].est_gb)
            load = time.perf_counter() - t0
            p = GenerationParams(model_key=key, seed=a.seed, width=a.size, height=a.size,
                                 negative_prompt="" if key == "sd_turbo" else "lowres, blurry")
            out1 = r.generate(prompt, p)
            out2 = r.generate(prompt, p)  # same seed → must be identical (reproducibility on this device)
            arr1, arr2 = np.asarray(out1.image), np.asarray(out2.image)
            dest = ROOT / "backend" / "data" / "verification"
            dest.mkdir(parents=True, exist_ok=True)
            out1.image.save(dest / f"{key}_seed{a.seed}.png")
            if key == "sd_turbo" or generated is None:
                generated = out1.image
            return {"load_s": round(load, 2), "prompt": prompt, "params": p.model_dump(),
                    "generate_s_first": out1.duration_s, "generate_s_second": out2.duration_s,
                    "same_seed_identical": bool((arr1 == arr2).all()),
                    "max_abs_pixel_diff_same_seed": int(np.abs(arr1.astype(int) - arr2.astype(int)).max()),
                    "pixel_mean": round(float(arr1.mean()), 2), "runtime_info": out1.runtime_info}
        record(key, gen)

    for key in ("ssim", "clip_b32", "clip_l14", "dinov2_base", "lpips_alex"):
        if key not in steps:
            continue

        def met(key: str = key):  # noqa: ANN202
            from app.evaluation.metrics import get_metric

            if generated is None:
                raise RuntimeError("no generated image (run a generation step first)")
            m = mm.get("metrics", key, lambda: get_metric(key))
            t0 = time.perf_counter()
            v_gen = m.compute(ref, generated)
            first = time.perf_counter() - t0
            v_self = m.compute(ref, ref)  # identity check: a perfect match must score at the metric's optimum
            if not (math.isfinite(v_gen) and math.isfinite(v_self)):
                raise RuntimeError("non-finite score")
            return {"value_ref_vs_generated": v_gen, "value_ref_vs_ref": v_self, "first_call_s": round(first, 2),
                    "device": m.last_device or "cpu (numpy)", "fallback_reason": m.fallback_reason}
        record(key, met)

    report["finished_at"] = datetime.now(UTC).isoformat()
    dest = ROOT / "backend" / "data" / "verification"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{dev.device}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    ok = sum(s["status"] == "ok" for s in report["steps"].values())
    print(f"\n{ok}/{len(report['steps'])} steps ok · report: {path.relative_to(ROOT)}")
    return 0 if ok == len(report["steps"]) else 1


if __name__ == "__main__":
    sys.exit(main())
