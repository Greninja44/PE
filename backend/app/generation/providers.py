"""Concrete generation providers.

* ``gpu_worker``   — Diffusers on the local RTX worker. The API enqueues a job; ``DiffusersRunner`` executes it.
* ``hf_inference`` — Hugging Face hosted text-to-image. Inline HTTP. Remote seed handling is not verifiable,
                     so results are flagged ``reproducible: false`` in runtime_info.
"""
from __future__ import annotations

import io
import time
from typing import Any

import httpx
from PIL import Image as PILImage

from app.analysis.base import Availability
from app.config.settings import get_settings
from app.generation.base import MODEL_REGISTRY, GenerationOutput, GenerationParams, GenerationProvider, ensure_rgb


class GPUWorkerProvider(GenerationProvider):
    name = "gpu_worker"
    execution = "worker"
    description = "Diffusers on the local inference worker (Apple MPS, NVIDIA CUDA or CPU). Fully seeded."

    def availability(self) -> Availability:
        return Availability(available=True, reason="Executes on the GPU worker (job queued until a worker claims it)")


class HFInferenceProvider(GenerationProvider):
    name = "hf_inference"
    execution = "inline"
    description = "Hugging Face hosted inference. Convenience only — seeds are not guaranteed to be honoured."

    def __init__(self, timeout: float = 180.0) -> None:
        s = get_settings()
        self.token = s.hf_token
        self.base = s.hf_base_url.rstrip("/")
        self.timeout = timeout

    def availability(self) -> Availability:
        if not self.token:
            return Availability(available=False, reason="PE_HF_TOKEN is not configured")
        return Availability(available=True, reason="Remote; served models and seed handling vary by provider")

    def generate(self, prompt: str, params: GenerationParams) -> GenerationOutput:
        avail = self.availability()
        if not avail.available:
            raise RuntimeError(avail.reason)
        spec = MODEL_REGISTRY[params.model_key]
        t0 = time.perf_counter()
        body = {"inputs": prompt, "parameters": {
            "negative_prompt": params.negative_prompt, "num_inference_steps": params.steps,
            "guidance_scale": params.cfg_scale, "width": params.width, "height": params.height, "seed": params.seed,
        }}
        r = httpx.post(f"{self.base}/hf-inference/models/{spec.hf_id}", json=body,
                       headers={"Authorization": f"Bearer {self.token}", "Accept": "image/png"}, timeout=self.timeout)
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("image/"):
            raise RuntimeError(f"Unexpected response type: {r.headers.get('content-type')}: {r.text[:300]}")
        img = ensure_rgb(PILImage.open(io.BytesIO(r.content)))
        return GenerationOutput(image=img, duration_s=round(time.perf_counter() - t0, 3), runtime_info={
            "provider": "hf_inference", "model_id": spec.hf_id, "reproducible": False,
            "note": "Remote provider: seed forwarded but not verifiable.",
        })


class DiffusersRunner:
    """Local diffusers execution. Imported and instantiated ONLY in the GPU worker process."""

    def __init__(self, model_key: str, cpu_offload: bool | None = None, device: str | None = None,
                 lora: str | None = None) -> None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_key}' is not in the registry")
        import torch
        from diffusers import AutoPipelineForText2Image

        from app.runtime.device import dtype_for, select_device

        s = get_settings()
        self.torch = torch
        self.spec = MODEL_REGISTRY[model_key]
        self.device = select_device(device).device
        self.dtype = dtype_for(self.device, "diffusion")
        kwargs: dict[str, Any] = {"torch_dtype": self.dtype, "use_safetensors": True}
        # fp16 weight files halve download size and load memory. They are also used for fp32 CPU inference
        # (upcast on load); the original CUDA behaviour (fp32 files for SD 1.5) is preserved.
        self.weights_variant = None
        if not (self.device == "cuda" and self.spec.family == "sd15"):
            self.weights_variant = "fp16"
        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.spec.hf_id, variant=self.weights_variant,
                                                                  **kwargs)
        except (OSError, ValueError):
            if self.weights_variant is None:
                raise
            self.weights_variant = None
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.spec.hf_id, **kwargs)
        # SD 1.5's safety checker is kept; if it blacks out an image, runtime_info records nsfw_flagged=True.
        vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if self.device == "cuda" else 0
        self.offload = cpu_offload if cpu_offload is not None else (
            self.device == "cuda" and self.spec.family == "sdxl" and vram_gb < 12)
        if self.device == "cuda" and self.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        if self.spec.family == "sdxl":
            self.pipe.enable_vae_slicing()
        # Attention slicing lowers peak memory on unified-memory Macs at a small speed cost (diffusers' advice
        # for machines with < 64 GB). "auto" enables it on MPS only.
        self.attention_slicing = s.attention_slicing == "on" or (s.attention_slicing == "auto" and self.device == "mps")
        if self.attention_slicing:
            self.pipe.enable_attention_slicing()
        self.lora, self.lora_sha256 = None, None
        if lora:
            from app.finetune.registry import adapter_file, adapter_sha256

            f = adapter_file(lora)  # validates the id and resolves inside the runs directory only
            self.pipe.load_lora_weights(str(f.parent), weight_name=f.name, adapter_name="ft")
            self.lora, self.lora_sha256 = lora, adapter_sha256(lora)
        self.pipe.set_progress_bar_config(disable=True)
        self._calls = 0

    def generate(self, prompt: str, params: GenerationParams, progress_cb=None) -> GenerationOutput:  # noqa: ANN001
        import diffusers
        import numpy as np

        from app.runtime.device import synchronize

        t0 = time.perf_counter()
        # CUDA keeps its original on-device generator (reproduces earlier RTX runs). MPS/CPU use a CPU generator:
        # the initial noise is then identical on every machine for a given seed, and MPS generators are not
        # guaranteed to be deterministic across macOS/PyTorch versions.
        gen_device = "cpu" if (self.offload or self.device != "cuda") else self.device
        gen = self.torch.Generator(device=gen_device).manual_seed(params.seed)
        kwargs: dict[str, Any] = dict(prompt=prompt, num_inference_steps=params.steps, guidance_scale=params.cfg_scale,
                                      width=params.width, height=params.height, generator=gen)
        if params.negative_prompt and params.cfg_scale and params.cfg_scale > 1.0:
            kwargs["negative_prompt"] = params.negative_prompt
        if progress_cb is not None:
            def _cb(pipe, step, timestep, cb_kwargs):  # noqa: ANN001
                progress_cb((step + 1) / params.steps)
                return cb_kwargs
            kwargs["callback_on_step_end"] = _cb
        with self.torch.inference_mode():
            result = self.pipe(**kwargs)
        synchronize(self.device)
        img = ensure_rgb(result.images[0])
        nsfw = getattr(result, "nsfw_content_detected", None)
        flagged = bool(nsfw[0]) if nsfw is not None else False
        if not flagged and int(np.asarray(img).max()) == 0:
            # Half-precision overflow in the VAE yields NaNs, which post-processing turns into a black frame.
            # Report it as a failure rather than storing a black image as if it were a genuine output.
            raise RuntimeError(f"{self.spec.label} produced an all-black image on {self.device} "
                               f"({str(self.dtype).removeprefix('torch.')}) without an NSFW flag — likely a "
                               "half-precision NaN. Retry with PE_HALF_PRECISION=false.")
        runtime = {
            "provider": "gpu_worker", "model_id": self.spec.hf_id, "device": self.device,
            "torch": self.torch.__version__, "diffusers": diffusers.__version__, "cpu_offload": self.offload,
            "dtype": str(self.dtype).removeprefix("torch."), "reproducible": True,
            "generator_device": gen_device, "weights_variant": self.weights_variant,
            "attention_slicing": self.attention_slicing,
            "negative_prompt_applied": "negative_prompt" in kwargs,
            # The first call after loading includes one-off kernel compilation / warm-up (measured on MPS: ~5 s vs
            # ~0.3 s for SD-Turbo); comparisons of generation time must exclude it.
            "first_call_after_load": self._calls == 0,
            "lora": getattr(self, "lora", None), "lora_sha256": getattr(self, "lora_sha256", None),
        }
        self._calls += 1
        if nsfw is not None:
            runtime["nsfw_flagged"] = flagged
        if self.device == "cuda":
            runtime["gpu"] = self.torch.cuda.get_device_name(0)
            runtime["peak_vram_gb"] = round(self.torch.cuda.max_memory_allocated() / 1e9, 2)
        return GenerationOutput(image=img, duration_s=round(time.perf_counter() - t0, 3), runtime_info=runtime)

    def unload(self) -> None:
        from app.runtime.device import empty_cache

        del self.pipe
        empty_cache(self.device)


GENERATION_PROVIDER_NAMES = ("gpu_worker", "hf_inference")


def get_generation_provider(name: str) -> GenerationProvider:
    if name == "gpu_worker":
        return GPUWorkerProvider()
    if name == "hf_inference":
        return HFInferenceProvider()
    raise ValueError(f"Unknown generation provider '{name}'")
