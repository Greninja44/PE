"""Stage 4 — independent image-similarity metrics.

Every metric is a separate class with its own availability check. If a metric cannot run (missing
backend, no GPU worker), the result is recorded as ``unavailable``/``queued`` with ``value = None`` —
never a substituted number.

Optimisation vs evaluation
--------------------------
The refinement loop optimises ONE metric (default ``clip_b32``). Independent evaluation uses a
*different* set (default CLIP ViT-L/14, DINOv2, SSIM, LPIPS) so that improvements are not an artefact
of optimising the very number being reported (Goodhart's law).
"""
from __future__ import annotations

import abc
import importlib.util
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image as PILImage

from app.analysis.base import Availability


@dataclass(frozen=True)
class MetricInfo:
    key: str
    label: str
    higher_is_better: bool
    execution: Literal["inline", "worker"]
    backend: str
    range_note: str
    description: str


METRICS: dict[str, MetricInfo] = {
    "clip_l14": MetricInfo("clip_l14", "CLIP similarity (ViT-L/14)", True, "worker",
                           "transformers:openai/clip-vit-large-patch14", "cosine, −1…1",
                           "Semantic similarity of CLIP image embeddings."),
    "clip_b32": MetricInfo("clip_b32", "CLIP similarity (ViT-B/32)", True, "worker",
                           "transformers:openai/clip-vit-base-patch32", "cosine, −1…1",
                           "Lighter CLIP model; default refinement objective."),
    "dinov2_base": MetricInfo("dinov2_base", "DINOv2 similarity", True, "worker",
                              "transformers:facebook/dinov2-base", "cosine, −1…1",
                              "Self-supervised features; sensitive to structure and object identity."),
    "ssim": MetricInfo("ssim", "SSIM", True, "inline", "numpy:gaussian-ssim(σ=1.5,win=11,luma)", "−1…1",
                       "Pixel-level structural similarity after alignment. Strict: penalises any re-layout."),
    "lpips_alex": MetricInfo("lpips_alex", "LPIPS (AlexNet)", False, "worker", "lpips:alex-v0.1", "0…~1, lower=closer",
                             "Learned perceptual distance."),
}

DEFAULT_EVALUATION_METRICS: tuple[str, ...] = ("clip_l14", "dinov2_base", "ssim", "lpips_alex")
DEFAULT_OPTIMIZATION_METRIC = "clip_b32"


def align_pair(reference: PILImage.Image, generated: PILImage.Image, max_side: int = 512
               ) -> tuple[PILImage.Image, PILImage.Image]:
    """Centre-crop the reference to the generated image's aspect ratio, then resize both to a common size
    (generated resolution, capped at ``max_side``). Needed for pixel-aligned metrics (SSIM, LPIPS)."""
    ref, gen = reference.convert("RGB"), generated.convert("RGB")
    gw, gh = gen.size
    target_ar = gw / gh
    rw, rh = ref.size
    if rw / rh > target_ar:
        nw = round(rh * target_ar)
        left = (rw - nw) // 2
        ref = ref.crop((left, 0, left + nw, rh))
    else:
        nh = round(rw / target_ar)
        top = (rh - nh) // 2
        ref = ref.crop((0, top, rw, top + nh))
    scale = min(1.0, max_side / max(gw, gh))
    size = (max(8, round(gw * scale)), max(8, round(gh * scale)))
    return ref.resize(size, PILImage.Resampling.BICUBIC), gen.resize(size, PILImage.Resampling.BICUBIC)


class Metric(abc.ABC):
    key: str

    @property
    def info(self) -> MetricInfo:
        return METRICS[self.key]

    @abc.abstractmethod
    def availability(self) -> Availability: ...

    @abc.abstractmethod
    def compute(self, reference: PILImage.Image, generated: PILImage.Image) -> float: ...


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _filter2d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = len(k)
    v = sliding_window_view(x, n, axis=0) @ k
    return sliding_window_view(v, n, axis=1) @ k


def ssim_gray(x: np.ndarray, y: np.ndarray, data_range: float = 255.0) -> float:
    """Gaussian-window SSIM (Wang et al. 2004), equivalent to scikit-image ``structural_similarity`` with
    ``gaussian_weights=True, sigma=1.5, use_sample_covariance=False``."""
    if x.shape != y.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    k = _gaussian_kernel()
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = _filter2d(x, k), _filter2d(y, k)
    sxx = _filter2d(x * x, k) - mx * mx
    syy = _filter2d(y * y, k) - my * my
    sxy = _filter2d(x * y, k) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


class SSIMMetric(Metric):
    key = "ssim"

    def availability(self) -> Availability:
        return Availability(available=True, reason="Pure NumPy; runs on CPU")

    def compute(self, reference: PILImage.Image, generated: PILImage.Image) -> float:
        r, g = align_pair(reference, generated)
        return ssim_gray(np.asarray(r.convert("L")), np.asarray(g.convert("L")))


def _torch_metric_availability(*modules: str) -> Availability:
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        return Availability(available=False, reason=f"Requires {', '.join(missing)} (installed on the GPU worker)")
    return Availability(available=True, reason="torch backend present")


class _EmbeddingCosine(Metric):
    model_id: str

    def __init__(self, device: str | None = None) -> None:
        self._model = None
        self._processor = None
        self._device = device

    def availability(self) -> Availability:
        return _torch_metric_availability("torch", "transformers")

    def _load(self) -> None:
        raise NotImplementedError

    def _embed(self, img: PILImage.Image):  # noqa: ANN202
        raise NotImplementedError

    def compute(self, reference: PILImage.Image, generated: PILImage.Image) -> float:
        import torch

        if self._model is None:
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._load()
        with torch.inference_mode():
            a = self._embed(reference.convert("RGB"))
            b = self._embed(generated.convert("RGB"))
            return float(torch.nn.functional.cosine_similarity(a, b).item())


class CLIPMetric(_EmbeddingCosine):
    def __init__(self, key: str, device: str | None = None) -> None:
        super().__init__(device)
        self.key = key
        self.model_id = METRICS[key].backend.split(":", 1)[1]

    def _load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self._device).eval()

    def _embed(self, img: PILImage.Image):  # noqa: ANN202
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        return self._model.get_image_features(**inputs).float()


class DINOv2Metric(_EmbeddingCosine):
    key = "dinov2_base"
    model_id = "facebook/dinov2-base"

    def _load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()

    def _embed(self, img: PILImage.Image):  # noqa: ANN202
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        return self._model(**inputs).last_hidden_state[:, 0].float()  # CLS token


class LPIPSMetric(Metric):
    key = "lpips_alex"

    def __init__(self, device: str | None = None) -> None:
        self._net = None
        self._device = device

    def availability(self) -> Availability:
        return _torch_metric_availability("torch", "lpips")

    def compute(self, reference: PILImage.Image, generated: PILImage.Image) -> float:
        import lpips
        import torch

        if self._net is None:
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._net = lpips.LPIPS(net="alex", verbose=False).to(self._device).eval()
        r, g = align_pair(reference, generated)

        def to_t(im: PILImage.Image):  # noqa: ANN202
            arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
            return arr.permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            return float(self._net(to_t(r), to_t(g)).item())


def get_metric(key: str, device: str | None = None) -> Metric:
    if key == "ssim":
        return SSIMMetric()
    if key in ("clip_l14", "clip_b32"):
        return CLIPMetric(key, device)
    if key == "dinov2_base":
        return DINOv2Metric(device)
    if key == "lpips_alex":
        return LPIPSMetric(device)
    raise ValueError(f"Unknown metric '{key}'. Options: {sorted(METRICS)}")
