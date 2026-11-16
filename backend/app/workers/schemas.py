"""Typed job payloads.

Security model: a job is *data*, never code. ``type`` selects one of a fixed set of handlers on the worker;
payload fields are validated with ``extra="forbid"`` and model identifiers are checked against allow-lists,
so the backend cannot instruct the worker to run shell commands, load arbitrary repositories, or execute
remote code (``trust_remote_code`` is always False).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analysis.providers import ALLOWED_LOCAL_VLMS
from app.evaluation.metrics import METRICS
from app.generation.base import MODEL_REGISTRY

JobType = Literal["analyze", "generate", "evaluate"]
JOB_TYPES: tuple[str, ...] = ("analyze", "generate", "evaluate")
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzePayload(_Strict):
    analysis_id: str
    image_id: str
    mode: Literal["structured", "caption"]
    model_id: str

    @field_validator("model_id")
    @classmethod
    def _allowed(cls, v: str) -> str:
        if v not in ALLOWED_LOCAL_VLMS:
            raise ValueError(f"model '{v}' not in allow-list")
        return v


class GeneratePayload(_Strict):
    generation_id: str
    model_key: str
    prompt: str = Field(max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    seed: int = Field(ge=0, le=2**32 - 1)
    steps: int = Field(ge=1, le=150)
    cfg_scale: float = Field(ge=0, le=30)
    width: int = Field(ge=256, le=1536)
    height: int = Field(ge=256, le=1536)

    @field_validator("model_key")
    @classmethod
    def _allowed(cls, v: str) -> str:
        if v not in MODEL_REGISTRY:
            raise ValueError(f"model '{v}' not in registry")
        return v


class MetricRequest(_Strict):
    metric_result_id: str
    metric: str

    @field_validator("metric")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in METRICS:
            raise ValueError(f"unknown metric '{v}'")
        return v


class EvaluatePayload(_Strict):
    generation_id: str
    reference_image_id: str
    generated_image_id: str
    metrics: list[MetricRequest] = Field(min_length=1)


PAYLOAD_MODELS: dict[str, type[_Strict]] = {
    "analyze": AnalyzePayload, "generate": GeneratePayload, "evaluate": EvaluatePayload,
}


def image_ids_in_payload(job_type: str, payload: dict[str, Any]) -> set[str]:
    if job_type == "analyze":
        return {payload["image_id"]}
    if job_type == "evaluate":
        return {payload["reference_image_id"], payload["generated_image_id"]}
    return set()


# ---- Results reported by the worker -----------------------------------------------------------------
class AnalyzeResult(_Strict):
    raw_output: str = Field(max_length=200_000)
    model: str
    duration_s: float
    runtime_info: dict[str, Any] = {}


class GenerateResult(_Strict):
    duration_s: float
    runtime_info: dict[str, Any] = {}


class MetricValue(_Strict):
    metric_result_id: str
    status: Literal["completed", "failed", "unavailable"]
    value: float | None = None
    backend: str = ""
    detail: str | None = None


class EvaluateResult(_Strict):
    metrics: list[MetricValue]
    runtime_info: dict[str, Any] = {}


RESULT_MODELS: dict[str, type[_Strict]] = {
    "analyze": AnalyzeResult, "generate": GenerateResult, "evaluate": EvaluateResult,
}
