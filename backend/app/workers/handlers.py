"""Job handlers executed by the inference worker (Apple MPS, NVIDIA CUDA or CPU). A fixed mapping
``job type -> function``; nothing in a payload is ever executed or used as a path/URL.

Models are loaded lazily through :class:`app.runtime.lifecycle.ModelManager`, which keeps only one heavy model
family resident at a time (sequential loading for 16 GB unified memory / 8 GB VRAM). Every job's runtime info
records the device it ran on and the measured peak process memory.
"""
from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from typing import Any

from PIL import Image as PILImage

from app.evaluation.metrics import METRICS, get_metric
from app.generation.base import MODEL_REGISTRY, GenerationParams
from app.runtime.lifecycle import ModelManager  # re-exported: app.workers.handlers.ModelManager
from app.runtime.memory import PeakSampler
from app.workers.schemas import AnalyzePayload, EvaluatePayload, GeneratePayload

__all__ = ["HANDLERS", "HandlerOutput", "ModelManager", "runtime_snapshot"]
log = logging.getLogger("pe.worker")

FetchImage = Callable[[str], PILImage.Image]
Progress = Callable[[float, str], None]
METRIC_EST_GB = {"clip_l14": 1.8, "clip_b32": 0.7, "dinov2_base": 0.5, "lpips_alex": 0.3, "ssim": 0.0}


class HandlerOutput:
    def __init__(self, result: dict[str, Any], image_png: bytes | None = None) -> None:
        self.result = result
        self.image_png = image_png


def handle_analyze(payload: dict[str, Any], fetch: FetchImage, progress: Progress, mm: ModelManager) -> HandlerOutput:
    from app.analysis.providers import VLM_EST_GB, QwenVLRunner

    p = AnalyzePayload.model_validate(payload)
    progress(0.05, "Downloading image")
    img = fetch(p.image_id)
    progress(0.15, f"Loading {p.model_id}")
    with PeakSampler() as mem:
        runner = mm.get("vlm", p.model_id, lambda: QwenVLRunner(p.model_id), est_gb=VLM_EST_GB.get(p.model_id))
        progress(0.4, f"Running VLM on {runner.device}")
        out = runner.analyze(img, p.mode)
    return HandlerOutput({"raw_output": out.raw_output, "model": out.model, "duration_s": out.duration_s,
                          "runtime_info": out.runtime_info | {"memory": mem.report()}})


def handle_generate(payload: dict[str, Any], fetch: FetchImage, progress: Progress, mm: ModelManager
                    ) -> HandlerOutput:
    from app.generation.providers import DiffusersRunner

    p = GeneratePayload.model_validate(payload)
    params = GenerationParams(model_key=p.model_key, seed=p.seed, steps=p.steps, cfg_scale=p.cfg_scale,
                              width=p.width, height=p.height, negative_prompt=p.negative_prompt)
    progress(0.05, f"Loading {p.model_key}")
    with PeakSampler() as mem:
        t0 = time.perf_counter()
        runner = mm.get("diffusion", p.model_key, lambda: DiffusersRunner(p.model_key),
                        est_gb=MODEL_REGISTRY[p.model_key].est_gb)
        load_s = time.perf_counter() - t0
        progress(0.1, f"Denoising on {runner.device}")
        out = runner.generate(p.prompt, params, progress_cb=lambda f: progress(0.1 + 0.85 * f, "Denoising"))
    buf = io.BytesIO()
    out.image.save(buf, format="PNG")
    info = out.runtime_info | {"memory": mem.report(), "model_load_s": round(load_s, 2)}
    return HandlerOutput({"duration_s": out.duration_s, "runtime_info": info}, buf.getvalue())


def handle_evaluate(payload: dict[str, Any], fetch: FetchImage, progress: Progress, mm: ModelManager
                    ) -> HandlerOutput:
    p = EvaluatePayload.model_validate(payload)
    progress(0.05, "Downloading images")
    ref, gen = fetch(p.reference_image_id), fetch(p.generated_image_id)
    results = []
    with PeakSampler() as mem:
        for n, req in enumerate(p.metrics):
            progress(0.1 + 0.85 * n / len(p.metrics), f"Computing {req.metric}")
            backend = METRICS[req.metric].backend
            try:
                metric = mm.get("metrics", req.metric, lambda key=req.metric: get_metric(key),
                                est_gb=METRIC_EST_GB.get(req.metric))
            except Exception as e:  # noqa: BLE001 — a model that cannot load fails only its own metric
                log.exception("metric %s could not be loaded", req.metric)
                results.append({"metric_result_id": req.metric_result_id, "status": "failed", "backend": backend,
                                "detail": f"load failed: {type(e).__name__}: {e}"})
                continue
            avail = metric.availability()
            if not avail.available:
                results.append({"metric_result_id": req.metric_result_id, "status": "unavailable",
                                "backend": backend, "detail": avail.reason})
                continue
            try:
                t0 = time.perf_counter()
                value = metric.compute(ref, gen)
                where = metric.last_device or "cpu (numpy)"
                detail = f"computed on {where} in {time.perf_counter() - t0:.2f}s"
                if metric.fallback_reason:
                    detail += f"; CPU fallback after {metric.fallback_reason}"
                results.append({"metric_result_id": req.metric_result_id, "status": "completed", "value": value,
                                "backend": backend, "detail": detail})
            except Exception as e:  # noqa: BLE001 — one failing metric must not discard the others
                log.exception("metric %s failed", req.metric)
                results.append({"metric_result_id": req.metric_result_id, "status": "failed",
                                "backend": backend, "detail": f"{type(e).__name__}: {e}"})
    return HandlerOutput({"metrics": results, "runtime_info": runtime_snapshot() | {"memory": mem.report()}})


HANDLERS: dict[str, Callable[..., HandlerOutput]] = {
    "analyze": handle_analyze, "generate": handle_generate, "evaluate": handle_evaluate,
}


def runtime_snapshot() -> dict[str, Any]:
    """Device facts (backward-compatible keys: torch, cuda_available, gpu, cuda, capability, vram_gb) plus
    MPS information and the selected device."""
    from app.runtime.device import device_report

    return device_report()
