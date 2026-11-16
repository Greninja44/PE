"""Job handlers executed on the GPU worker. A fixed mapping ``job type -> function``; nothing in a payload
is ever executed or used as a path/URL. Heavy models are loaded lazily and evicted when another family is
needed, so the agent fits in 8 GB of VRAM (RTX 5060 laptop)."""
from __future__ import annotations

import gc
import io
import logging
from collections.abc import Callable
from typing import Any

from PIL import Image as PILImage

from app.evaluation.metrics import METRICS, get_metric
from app.generation.base import GenerationParams
from app.workers.schemas import AnalyzePayload, EvaluatePayload, GeneratePayload

log = logging.getLogger("pe.worker")

FetchImage = Callable[[str], PILImage.Image]
Progress = Callable[[float, str], None]


class HandlerOutput:
    def __init__(self, result: dict[str, Any], image_png: bytes | None = None) -> None:
        self.result = result
        self.image_png = image_png


class ModelManager:
    """Keeps at most one model *category* resident (vlm | diffusion | metrics) to respect VRAM limits."""

    def __init__(self) -> None:
        self._cat: str | None = None
        self._models: dict[str, Any] = {}

    def get(self, category: str, key: str, factory: Callable[[], Any]) -> Any:
        full = f"{category}:{key}"
        if full in self._models:
            return self._models[full]
        # Metric models are small and may co-reside; a VLM or diffusion pipeline always gets the GPU alone.
        if self._cat != category or category != "metrics":
            self.unload_all()
        self._cat = category
        log.info("loading %s", full)
        self._models[full] = factory()
        return self._models[full]

    def unload_all(self) -> None:
        for m in self._models.values():
            if hasattr(m, "unload"):
                try:
                    m.unload()
                except Exception:  # noqa: BLE001
                    log.exception("unload failed")
        self._models.clear()
        self._cat = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def handle_analyze(payload: dict[str, Any], fetch: FetchImage, progress: Progress, mm: ModelManager) -> HandlerOutput:
    from app.analysis.providers import QwenVLRunner

    p = AnalyzePayload.model_validate(payload)
    progress(0.05, "Downloading image")
    img = fetch(p.image_id)
    progress(0.15, f"Loading {p.model_id}")
    runner = mm.get("vlm", p.model_id, lambda: QwenVLRunner(p.model_id))
    progress(0.4, "Running VLM")
    out = runner.analyze(img, p.mode)
    return HandlerOutput({"raw_output": out.raw_output, "model": out.model, "duration_s": out.duration_s,
                          "runtime_info": out.runtime_info})


def handle_generate(payload: dict[str, Any], fetch: FetchImage, progress: Progress, mm: ModelManager
                    ) -> HandlerOutput:
    from app.generation.providers import DiffusersRunner

    p = GeneratePayload.model_validate(payload)
    params = GenerationParams(model_key=p.model_key, seed=p.seed, steps=p.steps, cfg_scale=p.cfg_scale,
                              width=p.width, height=p.height, negative_prompt=p.negative_prompt)
    progress(0.05, f"Loading {p.model_key}")
    runner = mm.get("diffusion", p.model_key, lambda: DiffusersRunner(p.model_key))
    progress(0.1, "Denoising")
    out = runner.generate(p.prompt, params, progress_cb=lambda f: progress(0.1 + 0.85 * f, "Denoising"))
    buf = io.BytesIO()
    out.image.save(buf, format="PNG")
    return HandlerOutput({"duration_s": out.duration_s, "runtime_info": out.runtime_info}, buf.getvalue())


def handle_evaluate(payload: dict[str, Any], fetch: FetchImage, progress: Progress, mm: ModelManager
                    ) -> HandlerOutput:
    p = EvaluatePayload.model_validate(payload)
    progress(0.05, "Downloading images")
    ref, gen = fetch(p.reference_image_id), fetch(p.generated_image_id)
    results = []
    for n, req in enumerate(p.metrics):
        progress(0.1 + 0.85 * n / len(p.metrics), f"Computing {req.metric}")
        metric = mm.get("metrics", req.metric, lambda key=req.metric: get_metric(key))
        avail = metric.availability()
        if not avail.available:
            results.append({"metric_result_id": req.metric_result_id, "status": "unavailable",
                            "backend": METRICS[req.metric].backend, "detail": avail.reason})
            continue
        try:
            value = metric.compute(ref, gen)
            results.append({"metric_result_id": req.metric_result_id, "status": "completed", "value": value,
                            "backend": METRICS[req.metric].backend})
        except Exception as e:  # noqa: BLE001 — one failing metric must not discard the others
            log.exception("metric %s failed", req.metric)
            results.append({"metric_result_id": req.metric_result_id, "status": "failed",
                            "backend": METRICS[req.metric].backend, "detail": f"{type(e).__name__}: {e}"})
    return HandlerOutput({"metrics": results, "runtime_info": runtime_snapshot()})


HANDLERS: dict[str, Callable[..., HandlerOutput]] = {
    "analyze": handle_analyze, "generate": handle_generate, "evaluate": handle_evaluate,
}


def runtime_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        info["torch"] = None
    return info
