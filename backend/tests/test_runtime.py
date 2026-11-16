"""Device selection, memory diagnostics, model lifecycle and handler tests for the Apple Silicon migration.
Inference is MOCKED here (fake runners and fake metrics); real-model checks live in scripts/verify_inference.py."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from app.runtime import device as D
from app.runtime import memory
from app.runtime.lifecycle import InsufficientMemory, ModelManager
from tests.conftest import fixture_png


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    from app.config.settings import get_settings

    monkeypatch.delenv("PE_DEVICE", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ------------------------------------------------------------------------------------------------ devices
def test_select_device_cpu_requested():
    assert D.select_device("cpu").device == "cpu"


def test_select_device_prefers_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(D, "cuda_available", lambda: False)
    monkeypatch.setattr(D, "mps_available", lambda: True)
    assert D.select_device("auto").device == "mps"
    monkeypatch.setattr(D, "mps_available", lambda: False)
    c = D.select_device("auto")
    assert c.device == "cpu" and "CPU" in c.reason


def test_select_device_falls_back_when_requested_backend_missing(monkeypatch):
    monkeypatch.setattr(D, "cuda_available", lambda: False)
    monkeypatch.setattr(D, "mps_available", lambda: False)
    c = D.select_device("mps")
    assert c.device == "cpu" and "not available" in c.reason
    c = D.select_device("cuda")
    assert c.device == "cpu"


def test_select_device_cuda_first_for_future_nvidia(monkeypatch):
    monkeypatch.setattr(D, "cuda_available", lambda: True)
    monkeypatch.setattr(D, "mps_available", lambda: True)
    assert D.select_device("auto").device == "cuda"


def test_select_device_from_env_and_rejects_unknown(monkeypatch):
    monkeypatch.setenv("PE_DEVICE", "cpu")
    assert D.select_device().device == "cpu"
    with pytest.raises(ValueError):
        D.select_device("tpu")


def test_dtype_policy():
    torch = pytest.importorskip("torch")
    assert D.dtype_for("mps", "vlm", half=True) == torch.bfloat16
    assert D.dtype_for("mps", "diffusion", half=True) == torch.float16
    assert D.dtype_for("cpu", "diffusion", half=True) == torch.float32
    assert D.dtype_for("mps", "metric", half=True) == torch.float32
    assert D.dtype_for("mps", "diffusion", half=False) == torch.float32


def test_mps_fallback_env_is_set_before_torch_use():
    import os

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_is_oom():
    assert D.is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)"))
    assert not D.is_oom(RuntimeError("shape mismatch"))


# ------------------------------------------------------------------------------------------------- memory
def test_memory_snapshot_and_peak_sampler():
    snap = memory.snapshot()
    assert snap["process_gb"] >= 0 and snap["process_metric"] in ("phys_footprint", "rss", "maxrss")
    with memory.PeakSampler(interval=0.01) as p:
        block = np.ones((64, 1024, 1024), dtype=np.uint8)  # 64 MiB, touched
        block[::4096] = 2
    r = p.report()
    assert r["peak_process_gb"] >= r["start_process_gb"]
    del block


# ---------------------------------------------------------------------------------------------- lifecycle
class FakeModel:
    def __init__(self, name, log):
        self.name, self.log = name, log
        log.append(("load", name))

    def unload(self):
        self.log.append(("unload", self.name))


def test_model_manager_sequential_loading_and_reuse():
    log: list = []
    mm = ModelManager(strict_memory=False)
    a = mm.get("vlm", "q", lambda: FakeModel("q", log))
    assert mm.get("vlm", "q", lambda: FakeModel("q2", log)) is a  # reused, not reloaded
    mm.get("diffusion", "sd", lambda: FakeModel("sd", log))  # different family evicts the VLM first
    assert log == [("load", "q"), ("unload", "q"), ("load", "sd")]
    assert [r["key"] for r in mm.resident()] == ["diffusion:sd"]


def test_model_manager_metrics_coreside_but_evict_heavy_models():
    log: list = []
    mm = ModelManager(strict_memory=False)
    mm.get("diffusion", "sd", lambda: FakeModel("sd", log))
    mm.get("metrics", "clip", lambda: FakeModel("clip", log))
    mm.get("metrics", "dino", lambda: FakeModel("dino", log))
    assert ("unload", "sd") in log and ("unload", "clip") not in log
    assert {r["key"] for r in mm.resident()} == {"metrics:clip", "metrics:dino"}
    mm.unload_all()
    assert mm.resident() == [] and ("unload", "dino") in log


def test_model_manager_idle_unload(monkeypatch):
    log: list = []
    mm = ModelManager(strict_memory=False)
    mm.get("vlm", "q", lambda: FakeModel("q", log))
    assert not mm.unload_if_idle(3600)
    assert mm.unload_if_idle(0.0) is False  # 0 disables idle unloading
    for r in mm._models.values():
        r.last_used -= 10
    assert mm.unload_if_idle(5) and mm.resident() == []


def test_model_manager_memory_check(monkeypatch):
    monkeypatch.setattr(memory, "available_gb", lambda: 2.0)
    log: list = []
    strict = ModelManager(strict_memory=True, margin_gb=1.0)
    with pytest.raises(InsufficientMemory):
        strict.get("vlm", "big", lambda: FakeModel("big", log), est_gb=4.5)
    assert log == []  # nothing was loaded
    lenient = ModelManager(strict_memory=False, margin_gb=1.0)
    lenient.get("vlm", "big", lambda: FakeModel("big", log), est_gb=4.5)
    assert any(e["event"] == "memory_warning" for e in lenient.events)


# ------------------------------------------------------------------------------------------ metric fallback
def _img(seed=0):
    return PILImage.open(io.BytesIO(fixture_png(64, 64, seed)))


def test_torch_metric_falls_back_to_cpu_and_records_it():
    from app.evaluation.metrics import _TorchMetric

    class Flaky(_TorchMetric):
        key = "clip_b32"

        def availability(self):  # noqa: D401
            from app.analysis.base import Availability

            return Availability(available=True)

        def _move_to_cpu(self):
            self.moved = True

        def _score(self, ref, gen):
            if self._device != "cpu":
                raise NotImplementedError("aten::foo is not implemented for MPS")
            return 0.5

    m = Flaky(device="mps")
    m._device = "mps"  # pretend MPS was selected
    assert m.compute(_img(), _img(1)) == 0.5
    assert m.last_device == "cpu" and m.moved and "MPS" in m.fallback_reason


def test_torch_metric_never_reports_nan():
    from app.evaluation.metrics import _TorchMetric

    class Nan(_TorchMetric):
        key = "clip_b32"

        def availability(self):
            from app.analysis.base import Availability

            return Availability(available=True)

        def _score(self, ref, gen):
            return float("nan")

    with pytest.raises(ValueError, match="NaN"):
        Nan(device="cpu").compute(_img(), _img(1))


# ------------------------------------------------------------------------------------- handlers (mocked)
class FakeDiffusersRunner:
    device = "mps"

    def __init__(self, key):
        self.key = key

    def generate(self, prompt, params, progress_cb=None):
        from app.generation.base import GenerationOutput

        for i in range(params.steps):
            progress_cb((i + 1) / params.steps)
        img = PILImage.new("RGB", (params.width, params.height), (120, 60, 30))
        return GenerationOutput(image=img, duration_s=0.01, runtime_info={"device": "mps", "fixture": True})


def test_handle_generate_records_memory_and_device(monkeypatch):
    import app.generation.providers as gp
    from app.workers.handlers import handle_generate

    monkeypatch.setattr(gp, "DiffusersRunner", FakeDiffusersRunner)
    steps = []
    out = handle_generate({"generation_id": "g", "model_key": "sd_turbo", "prompt": "a cat", "negative_prompt": "",
                           "seed": 1, "steps": 2, "cfg_scale": 0.0, "width": 512, "height": 512},
                          fetch=lambda _: None, progress=lambda f, m: steps.append(m), mm=ModelManager())
    info = out.result["runtime_info"]
    assert info["device"] == "mps" and "peak_process_gb" in info["memory"] and "model_load_s" in info
    assert PILImage.open(io.BytesIO(out.image_png)).size == (512, 512)
    assert any("mps" in s for s in steps)


def test_handle_evaluate_isolates_failures(monkeypatch):
    import app.workers.handlers as H

    real = H.get_metric

    def fake_get_metric(key):
        if key == "lpips_alex":
            raise OSError("weights missing")
        return real(key)

    monkeypatch.setattr(H, "get_metric", fake_get_metric)
    imgs = {"r": _img(0), "g": _img(1)}
    out = H.handle_evaluate({"generation_id": "g", "reference_image_id": "r", "generated_image_id": "g",
                             "metrics": [{"metric_result_id": "m1", "metric": "ssim"},
                                         {"metric_result_id": "m2", "metric": "lpips_alex"}]},
                            fetch=imgs.__getitem__, progress=lambda *a: None, mm=ModelManager())
    by = {m["metric_result_id"]: m for m in out.result["metrics"]}
    assert by["m1"]["status"] == "completed" and isinstance(by["m1"]["value"], float)
    assert "cpu (numpy)" in by["m1"]["detail"]
    assert by["m2"]["status"] == "failed" and by["m2"].get("value") is None and "weights missing" in by["m2"]["detail"]


def test_diffusers_runner_rejects_black_nan_output():
    pytest.importorskip("diffusers")
    torch = pytest.importorskip("torch")
    from app.generation.base import MODEL_REGISTRY, GenerationParams
    from app.generation.providers import DiffusersRunner

    class Result:
        images = [PILImage.new("RGB", (512, 512), (0, 0, 0))]
        nsfw_content_detected = [False]

    r = DiffusersRunner.__new__(DiffusersRunner)
    r.torch, r.spec, r.device, r.dtype, r.offload = torch, MODEL_REGISTRY["sd_turbo"], "cpu", torch.float16, False
    r.weights_variant, r.attention_slicing, r.pipe = "fp16", False, lambda **kw: Result()
    with pytest.raises(RuntimeError, match="all-black"):
        r.generate("a cat", GenerationParams(model_key="sd_turbo"))
    Result.nsfw_content_detected = [True]  # safety-checker black frames are recorded, not raised
    out = r.generate("a cat", GenerationParams(model_key="sd_turbo"))
    assert out.runtime_info["nsfw_flagged"] is True and out.runtime_info["generator_device"] == "cpu"


def test_clip_embedding_handles_tensor_and_model_output():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from app.evaluation.metrics import CLIPMetric

    emb = torch.tensor([[1.0, 0.0]])
    for ret in (emb, SimpleNamespace(pooler_output=emb, last_hidden_state=None)):
        m = CLIPMetric("clip_b32", device="cpu")
        m._device = "cpu"
        m._processor = lambda images, return_tensors: SimpleNamespace(to=lambda d: {})
        m._model = SimpleNamespace(get_image_features=lambda **kw: ret)
        assert torch.equal(m._embed(None), emb)
