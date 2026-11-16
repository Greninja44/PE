"""Device selection and per-device numeric policy.

Order for ``PE_DEVICE=auto``: CUDA (NVIDIA worker) → MPS (Apple Silicon) → CPU. An explicit request for a
device that is not available falls back to CPU and says so — it never silently claims acceleration.

``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set *before* torch is imported so that the (shrinking) set of operators
without an MPS kernel run on the CPU instead of raising ``NotImplementedError``. Fallbacks are slower but
correct; ``torch_mps_fallback_enabled`` is recorded in every run's runtime info.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

log = logging.getLogger("pe.runtime")
DeviceKind = Literal["cuda", "mps", "cpu"]
Purpose = Literal["vlm", "diffusion", "metric"]


@dataclass(frozen=True)
class DeviceChoice:
    device: DeviceKind
    requested: str
    reason: str


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def select_device(requested: str | None = None) -> DeviceChoice:
    """Resolve ``requested`` (``auto`` | ``mps`` | ``cuda`` | ``cpu``; default from ``PE_DEVICE``)."""
    if requested is None:
        from app.config.settings import get_settings

        requested = get_settings().device
    req = (requested or "auto").lower()
    if req not in ("auto", "mps", "cuda", "cpu"):
        raise ValueError(f"Unknown device '{requested}' (use auto, mps, cuda or cpu)")
    if req == "cpu":
        return DeviceChoice("cpu", req, "CPU requested")
    if req in ("auto", "cuda") and cuda_available():
        return DeviceChoice("cuda", req, "CUDA available")
    if req in ("auto", "mps") and mps_available():
        return DeviceChoice("mps", req, "Apple Metal (MPS) available")
    reason = f"'{req}' requested but not available; using CPU" if req != "auto" else "No GPU backend; using CPU"
    if req != "auto":
        log.warning(reason)
    return DeviceChoice("cpu", req, reason)


def dtype_for(device: str, purpose: Purpose, half: bool | None = None) -> Any:
    """Numeric precision per device and model family.

    * VLM: bfloat16 on CUDA/MPS (Qwen3-VL is trained in bf16; fp16 risks overflow). On CPU bf16 when half
      precision is enabled (halves memory: ~4.3 GB instead of ~8.6 GB for 2B params), else fp32.
    * Diffusion: fp16 on CUDA/MPS, fp32 on CPU (fp16 conv kernels on CPU are slow or missing).
    * Metrics: fp32 everywhere (small models; keeps scores comparable across devices).
    """
    import torch

    if half is None:
        from app.config.settings import get_settings

        half = get_settings().half_precision
    if purpose == "metric" or not half:
        return torch.float32
    if purpose == "vlm":
        return torch.bfloat16
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def empty_cache(device: str | None = None) -> None:
    """Return cached allocator blocks to the OS/driver after a model is released."""
    try:
        import torch
    except ImportError:
        return
    if device in (None, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device in (None, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def synchronize(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg or "insufficient memory" in msg


def device_report() -> dict[str, Any]:
    """Static facts about the accelerator for diagnostics and provenance."""
    info: dict[str, Any] = {"torch": None}
    try:
        import torch
    except ImportError:
        return info
    import platform

    info.update({
        "torch": torch.__version__,
        "machine": platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    })
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    if torch.backends.mps.is_available():
        info["gpu"] = info.get("gpu") or _apple_chip()
        info["mps_recommended_max_gb"] = round(torch.mps.recommended_max_memory() / 1e9, 2)
    info["selected_device"] = select_device().device
    return info


def _apple_chip() -> str:
    import subprocess

    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=2).stdout.strip() or "Apple GPU"
    except (OSError, subprocess.SubprocessError):
        return "Apple GPU"
