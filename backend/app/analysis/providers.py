"""Concrete VLM providers.

* ``qwen3vl_local``     — Qwen3-VL-2B via transformers. Runs ONLY on the GPU worker (lazy torch import).
* ``openai_compatible`` — any OpenAI-compatible chat endpoint with image input (vLLM, llama.cpp server,
                          LM Studio, Ollama ``/v1``). Inline: called from the API over HTTP.
* ``hf_router``         — Hugging Face Inference Providers router (OpenAI-compatible). Inline; needs a token.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import time
from typing import Any

import httpx
from PIL import Image as PILImage

from app.analysis.base import (
    CAPTION_INSTRUCTION,
    STRUCTURED_INSTRUCTION,
    AnalysisMode,
    AnalysisOutput,
    Availability,
    VLMProvider,
    parse_caption_output,
    parse_structured_output,
)
from app.config.settings import get_settings

# Models the GPU worker is permitted to load. Job payloads naming anything else are rejected.
ALLOWED_LOCAL_VLMS: dict[str, str] = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL 2B Instruct (default, ~4.5 GB bf16)",
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL 4B Instruct (~9 GB bf16; needs CPU offload on 8 GB; tight on 16 GB Macs)",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL 3B Instruct",
}
# Approximate resident size of the weights in bf16, used for the pre-load memory check (not a measurement).
VLM_EST_GB: dict[str, float] = {"Qwen/Qwen3-VL-2B-Instruct": 4.5, "Qwen/Qwen3-VL-4B-Instruct": 9.0,
                                "Qwen/Qwen2.5-VL-3B-Instruct": 7.5}


def _parse(provider: str, model: str, mode: AnalysisMode, text: str, t0: float,
           runtime: dict[str, Any] | None = None) -> AnalysisOutput:
    out = AnalysisOutput(provider=provider, model=model, mode=mode, raw_output=text,
                         duration_s=round(time.perf_counter() - t0, 3), runtime_info=runtime or {})
    if mode == "structured":
        out.attributes = parse_structured_output(text)
    else:
        out.caption = parse_caption_output(text)
    return out


def _image_data_url(image: PILImage.Image, max_side: int = 1024) -> str:
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Qwen3VLLocalProvider(VLMProvider):
    name = "qwen3vl_local"
    execution = "worker"
    description = "Qwen3-VL executed locally on the GPU worker with transformers."

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id or get_settings().default_vlm_model

    @property
    def model_id(self) -> str:
        return self._model_id

    def availability(self) -> Availability:
        # Availability from the API's point of view = a job can be queued for a worker.
        return Availability(available=True, reason="Executes on the GPU worker (job queued until a worker claims it)")


class QwenVLRunner:
    """The actual local inference, imported and used only inside the GPU worker process."""

    def __init__(self, model_id: str, device: str | None = None, max_image_side: int | None = None) -> None:
        if model_id not in ALLOWED_LOCAL_VLMS:
            raise ValueError(f"Model '{model_id}' is not in the local VLM allow-list")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from app.runtime.device import dtype_for, select_device

        s = get_settings()
        self.torch = torch
        self.model_id = model_id
        choice = select_device(device)
        self.device = choice.device
        self.dtype = dtype_for(self.device, "vlm")
        self.max_image_side = max_image_side or s.vlm_max_image_side
        self.max_new_tokens = s.vlm_max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        # CUDA keeps accelerate's device_map (allows offload on 8 GB cards). MPS/CPU: load then move as a whole —
        # accelerate's "auto" map does not target MPS reliably.
        kwargs: dict[str, Any] = {"dtype": self.dtype, "trust_remote_code": False}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def _prepare(self, image: PILImage.Image) -> tuple[PILImage.Image, list[int]]:
        img = image.convert("RGB")
        original = list(img.size)
        if max(img.size) > self.max_image_side:
            img = img.copy()
            img.thumbnail((self.max_image_side, self.max_image_side), PILImage.Resampling.LANCZOS)
        return img, original

    def analyze(self, image: PILImage.Image, mode: AnalysisMode, max_new_tokens: int | None = None) -> AnalysisOutput:
        from app.runtime.device import synchronize

        t0 = time.perf_counter()
        max_new_tokens = max_new_tokens or self.max_new_tokens
        instruction = STRUCTURED_INSTRUCTION if mode == "structured" else CAPTION_INSTRUCTION
        img, original = self._prepare(image)
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": instruction}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        synchronize(self.device)
        trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        import transformers

        runtime = {"device": str(self.model.device), "dtype": str(self.dtype).removeprefix("torch."),
                   "torch": self.torch.__version__, "transformers": transformers.__version__, "decoding": "greedy",
                   "max_new_tokens": max_new_tokens, "generated_tokens": int(trimmed.shape[1]),
                   "input_tokens": int(inputs["input_ids"].shape[1]), "image_size_original": original,
                   "image_size_model_input": list(img.size), "max_image_side": self.max_image_side}
        if self.device == "cuda":
            runtime["gpu"] = self.torch.cuda.get_device_name(0)
        return _parse("qwen3vl_local", self.model_id, mode, text, t0, runtime)

    def unload(self) -> None:
        from app.runtime.device import empty_cache

        del self.model
        del self.processor
        empty_cache(self.device)


class OpenAICompatibleProvider(VLMProvider):
    name = "openai_compatible"
    execution = "inline"
    description = "Any OpenAI-compatible vision endpoint (vLLM, llama.cpp server, LM Studio, Ollama)."

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 180.0) -> None:
        s = get_settings()
        self.base_url = (base_url if base_url is not None else s.openai_compat_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else s.openai_compat_api_key
        self._model = model or s.openai_compat_model
        self.timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model

    def availability(self) -> Availability:
        if not self.base_url:
            return Availability(available=False, reason="PE_OPENAI_COMPAT_BASE_URL is not configured")
        return Availability(available=True, reason=f"Endpoint {self.base_url}")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def analyze(self, image: PILImage.Image, mode: AnalysisMode) -> AnalysisOutput:
        avail = self.availability()
        if not avail.available:
            raise RuntimeError(avail.reason)
        t0 = time.perf_counter()
        instruction = STRUCTURED_INSTRUCTION if mode == "structured" else CAPTION_INSTRUCTION
        body = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                {"type": "text", "text": instruction},
            ]}],
        }
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        return _parse(self.name, self._model, mode, text, t0, {"endpoint": self.base_url, "temperature": 0})


class HFRouterProvider(OpenAICompatibleProvider):
    name = "hf_router"
    description = "Hugging Face Inference Providers (OpenAI-compatible router). Requires PE_HF_TOKEN."

    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct") -> None:
        s = get_settings()
        super().__init__(base_url=f"{s.hf_base_url}/v1", api_key=s.hf_token, model=model)

    def availability(self) -> Availability:
        if not self.api_key:
            return Availability(available=False, reason="PE_HF_TOKEN is not configured")
        return Availability(available=True, reason="Hugging Face router; served model availability varies")


def get_vlm_provider(name: str, model: str | None = None) -> VLMProvider:
    if name == "qwen3vl_local":
        if model and model not in ALLOWED_LOCAL_VLMS:
            raise ValueError(f"Model '{model}' is not in the local VLM allow-list: {sorted(ALLOWED_LOCAL_VLMS)}")
        return Qwen3VLLocalProvider(model)
    if name == "openai_compatible":
        return OpenAICompatibleProvider(model=model)
    if name == "hf_router":
        return HFRouterProvider(model) if model else HFRouterProvider()
    raise ValueError(f"Unknown VLM provider '{name}'")


VLM_PROVIDER_NAMES = ("qwen3vl_local", "openai_compatible", "hf_router")


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None
