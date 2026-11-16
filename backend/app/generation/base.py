"""Stage 3 — text-to-image provider interface and model registry."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image as PILImage
from pydantic import BaseModel, Field, model_validator

from app.analysis.base import Availability

Execution = Literal["inline", "worker"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    label: str
    family: Literal["sd15", "sd21", "sdxl"]
    default_steps: int
    default_cfg: float
    default_size: int
    steps_range: tuple[int, int]
    cfg_range: tuple[float, float]
    max_side: int
    notes: str = ""
    vram_note: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "sd15": ModelSpec(
        key="sd15", hf_id="stable-diffusion-v1-5/stable-diffusion-v1-5", label="Stable Diffusion 1.5", family="sd15",
        default_steps=30, default_cfg=7.5, default_size=512, steps_range=(1, 150), cfg_range=(0.0, 30.0),
        max_side=1024, notes="CLIP ViT-L/14 text encoder, 77-token window.", vram_note="~3.5 GB fp16",
    ),
    "sd_turbo": ModelSpec(
        key="sd_turbo", hf_id="stabilityai/sd-turbo", label="SD-Turbo", family="sd21",
        default_steps=1, default_cfg=0.0, default_size=512, steps_range=(1, 4), cfg_range=(0.0, 0.0),
        max_side=768, notes="Adversarial diffusion distillation: 1–4 steps, trained without CFG (guidance 0).",
        vram_note="~3 GB fp16",
    ),
    "sdxl": ModelSpec(
        key="sdxl", hf_id="stabilityai/stable-diffusion-xl-base-1.0", label="SDXL 1.0 base", family="sdxl",
        default_steps=30, default_cfg=7.0, default_size=1024, steps_range=(1, 100), cfg_range=(0.0, 20.0),
        max_side=1536, notes="Dual text encoders (CLIP-L + OpenCLIP-bigG). Native 1024×1024.",
        vram_note="~7–8 GB fp16; on an 8 GB RTX 5060 use model CPU offload (enabled automatically)",
    ),
}


class GenerationParams(BaseModel):
    model_key: str = "sd_turbo"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    steps: int | None = None
    cfg_scale: float | None = None
    width: int | None = None
    height: int | None = None
    negative_prompt: str = ""

    @model_validator(mode="after")
    def _apply_defaults_and_validate(self) -> GenerationParams:
        spec = MODEL_REGISTRY.get(self.model_key)
        if spec is None:
            raise ValueError(f"Unknown model '{self.model_key}'. Options: {sorted(MODEL_REGISTRY)}")
        self.steps = spec.default_steps if self.steps is None else self.steps
        self.cfg_scale = spec.default_cfg if self.cfg_scale is None else self.cfg_scale
        self.width = spec.default_size if self.width is None else self.width
        self.height = spec.default_size if self.height is None else self.height
        lo, hi = spec.steps_range
        if not lo <= self.steps <= hi:
            raise ValueError(f"{spec.label}: steps must be in [{lo}, {hi}]")
        clo, chi = spec.cfg_range
        if not clo <= self.cfg_scale <= chi:
            raise ValueError(f"{spec.label}: cfg_scale must be in [{clo}, {chi}]")
        for side in (self.width, self.height):
            if side % 8 or not 256 <= side <= spec.max_side:
                raise ValueError(f"{spec.label}: width/height must be multiples of 8 in [256, {spec.max_side}]")
        return self

    def as_record(self) -> dict[str, Any]:
        return self.model_dump()


class GenerationOutput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    image: Any  # PIL.Image
    duration_s: float
    runtime_info: dict[str, Any] = {}


class GenerationProvider(abc.ABC):
    name: str
    execution: Execution
    description: str = ""

    @abc.abstractmethod
    def availability(self) -> Availability: ...

    def generate(self, prompt: str, params: GenerationParams) -> GenerationOutput:  # pragma: no cover
        raise NotImplementedError(f"{self.name} runs on the GPU worker; enqueue a job instead")


def ensure_rgb(img: PILImage.Image) -> PILImage.Image:
    return img.convert("RGB") if img.mode != "RGB" else img
