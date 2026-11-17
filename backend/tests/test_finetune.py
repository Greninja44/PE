"""Fine-tuning: dataset filtering and leakage-free split, run registry safety, real LoRA training mechanics on a tiny
UNet (CPU), paired evaluation statistics, API endpoints and LoRA selection in the generation path.
All images, prompts and scores here are FIXTURES."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from tests.conftest import FIXTURE_ATTRIBUTES
from tests.test_api_flow import W, manual_prompts, worker_complete_generate


# ------------------------------------------------------------------------------------------------ dataset
def _fake_diffusiondb(raw, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (raw / "part-000001").mkdir(parents=True)
    for i, r in enumerate(rows):
        arr = np.full((r.get("h", 512), r.get("w", 512), 3), (i * 37) % 255, dtype=np.uint8)
        Image.fromarray(arr).save(raw / "part-000001" / r["image_name"])
    cols = {k: [r.get(k, d) for r in rows] for k, d in [
        ("image_name", None), ("prompt", ""), ("part_id", 1), ("seed", 1), ("step", 50), ("cfg", 7.0),
        ("sampler", 8), ("width", 512), ("height", 512), ("image_nsfw", 0.01), ("prompt_nsfw", 0.01)]}
    pq.write_table(pa.table(cols), raw / "metadata.parquet")


def test_dataset_filters_and_grouped_split(tmp_path):
    from app.finetune import dataset as D

    rows = [{"image_name": f"{i}.png", "prompt": f"a castle number {i % 30}, oil painting"} for i in range(60)]
    rows += [{"image_name": "big.png", "prompt": "wide", "width": 768, "w": 768},
             {"image_name": "nsfw.png", "prompt": "fine", "image_nsfw": 0.9},
             {"image_name": "p_nsfw.png", "prompt": "bad", "prompt_nsfw": 0.8},
             {"image_name": "missing.png", "prompt": "gone"}]
    raw = tmp_path / "raw"
    _fake_diffusiondb(raw, [r for r in rows if r["image_name"] != "missing.png"])
    import pyarrow as pa
    import pyarrow.parquet as pq  # add a metadata row whose file is absent

    t = pq.read_table(raw / "metadata.parquet").to_pylist() + [{
        "image_name": "missing.png", "prompt": "gone", "part_id": 1, "seed": 1, "step": 50, "cfg": 7.0, "sampler": 8,
        "width": 512, "height": 512, "image_nsfw": 0.0, "prompt_nsfw": 0.0}]
    pq.write_table(pa.Table.from_pylist(t), raw / "metadata.parquet")

    card = D.build(raw=raw, out=tmp_path / "out", parts=[1])
    assert card["dropped"]["not 512x512"] == 1 and card["dropped"]["image_nsfw above threshold"] == 1
    assert card["dropped"]["prompt_nsfw above threshold"] == 1 and card["dropped"]["image file missing"] == 1
    # identical pixel content appears for i and i+... only when (i*37)%255 repeats — none within 60 — so 60 kept
    assert sum(card["counts"].values()) == 60
    assert all(v == 0 for v in card["prompt_group_overlap"].values())  # no prompt shared across splits
    # Same normalised prompt → same split, regardless of case and punctuation.
    assert D.split_for("A Castle number 3, oil painting!", card["filters"]["split_percent"]) == \
        D.split_for("a castle number 3 oil painting", card["filters"]["split_percent"])
    again = D.build(raw=raw, out=tmp_path / "out2", parts=[1])
    assert again["manifest_sha256"] == card["manifest_sha256"]  # deterministic


# ----------------------------------------------------------------------------------------------- registry
@pytest.fixture()
def runs(tmp_path, monkeypatch):
    from app.finetune import registry as R

    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(R, "RUNS_DIR", d)

    def make(run_id, state="completed", adapter=True, base="sd15"):
        p = d / run_id
        (p / "adapter").mkdir(parents=True)
        (p / "config.json").write_text(json.dumps({"base_model": base, "rank": 4, "steps": 10, "lr": 1e-4}))
        (p / "status.json").write_text(json.dumps({"state": state, "step": 10, "total_steps": 10}))
        if adapter:
            (p / "adapter" / R.ADAPTER_FILE).write_bytes(b"FIXTURE-ADAPTER")
        (p / "train_log.jsonl").write_text("".join(json.dumps({"step": i, "loss": 0.1, "loss_ema": 0.1, "lr": 1e-4,
                                                               "seconds": 1}) + "\n" for i in range(1, 11)))
        return p
    return make


def test_registry_validates_ids_and_only_exposes_completed_adapters(runs):
    from app.finetune import registry as R

    runs("good-run")
    runs("half-run", state="stopped")
    for bad in ("../etc", "Good", "", "a/b", "x" * 80):
        with pytest.raises(R.RunNotFound):
            R.run_path(bad)
    assert R.adapter_file("good-run").name == R.ADAPTER_FILE
    with pytest.raises(R.RunNotFound):
        R.adapter_file("half-run")  # stopped runs are never offered for generation
    assert R.completed_adapters("sd15") == ["good-run"]
    assert len(R.adapter_sha256("good-run")) == 64


# -------------------------------------------------------------------------------- real training mechanics
@pytest.fixture()
def tiny_unet():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from diffusers import DDPMScheduler, UNet2DConditionModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
                                attention_head_dim=8, norm_num_groups=8)
    return unet, DDPMScheduler(num_train_timesteps=1000)


def test_lora_training_step_updates_only_adapters(tiny_unet, tmp_path):
    import torch

    from app.finetune.train import add_lora, save_adapter, train_step, validation_loss

    unet, sched = tiny_unet
    params = add_lora(unet, rank=4, alpha=4)
    trainable = {n for n, p in unet.named_parameters() if p.requires_grad}
    assert trainable and all("lora" in n for n in trainable)  # base weights frozen
    assert all(p.dtype == torch.float32 for p in params)
    lat, emb = torch.randn(2, 4, 8, 8), torch.randn(2, 77, 32)
    cache = {"latents": lat, "emb": emb}
    v1 = validation_loss(unet, sched, cache, torch.float32, "cpu", seed=7)
    assert v1 == validation_loss(unet, sched, cache, torch.float32, "cpu", seed=7)  # fixed noise → comparable
    before = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=1e-2)
    loss = train_step(unet, sched, lat[:1], emb[:1], torch.float32, "cpu", torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    assert any(not torch.equal(b, p) for b, p in zip(before, params, strict=True))
    f = save_adapter(unet, tmp_path / "adapter")
    from safetensors.torch import load_file

    keys = load_file(f).keys()
    assert keys and all("lora" in k for k in keys)


def test_paired_statistics_respect_metric_direction():
    from app.finetune.evaluate import paired

    hib = paired([0.5, 0.6, 0.7], [0.6, 0.6, 0.6], higher_is_better=True)
    assert (hib["lora_better"], hib["lora_worse"], hib["ties"], hib["n"]) == (1, 1, 1, 3)
    lib = paired([0.5, 0.6, None], [0.4, 0.7, 0.1], higher_is_better=False)  # None pairs are excluded
    assert (lib["lora_better"], lib["lora_worse"], lib["n"]) == (1, 1, 2)
    assert abs(lib["mean_diff_lora_minus_base"] - 0.0) < 1e-12


# -------------------------------------------------------------------------------------------------- API
def test_finetune_api(client, runs):
    p = runs("demo-run")
    (p / "eval").mkdir()
    (p / "eval" / "000_ref.jpg").write_bytes(b"x")
    (p / "eval" / "results.json").write_text(json.dumps({"n": 1, "paired": {}, "settings": {}, "notes": [],
                                                         "created_at": "now", "summary": {}}))
    lst = client.get("/api/finetune/runs").json()
    assert [r["id"] for r in lst] == ["demo-run"] and lst[0]["adapter_available"] and lst[0]["evaluation"]["n"] == 1
    d = client.get("/api/finetune/runs/demo-run").json()
    assert len(d["train_log"]) == 10
    assert client.get("/api/finetune/runs/demo-run/eval").status_code == 200
    assert client.get("/api/finetune/runs/demo-run/files/000_ref.jpg").status_code == 200
    assert client.get("/api/finetune/runs/demo-run/files/..%2Fconfig.json").status_code in (400, 404)
    assert client.get("/api/finetune/runs/nope/eval").status_code == 404
    assert client.get("/api/finetune/dataset").status_code == 200
    assert [a["id"] for a in client.get("/api/system").json()["lora_adapters"]] == ["demo-run"]


def test_lora_only_with_sd15_and_must_exist(client, recon, runs):
    from app.generation.base import GenerationParams

    with pytest.raises(ValueError):
        GenerationParams(model_key="sd_turbo", lora="demo-run")
    with pytest.raises(ValueError):
        GenerationParams(model_key="sd15", lora="../x")
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    r = client.post(f"/api/reconstructions/{rid}/generations",
                    json={"prompt_version_id": pvs[1]["id"], "params": {"model_key": "sd15", "lora": "not-trained"}})
    assert r.status_code == 422 and "not found" in r.text


def test_base_vs_lora_pairs_in_comparison(client, recon, worker_token, runs):
    runs("demo-run")
    rid = recon["id"]
    _, pvs = manual_prompts(client, rid)
    for lora in (None, "demo-run"):
        g = client.post(f"/api/reconstructions/{rid}/generations", json={
            "prompt_version_id": pvs[1]["id"], "params": {"model_key": "sd15", "seed": 3, "steps": 20, "lora": lora}})
        assert g.status_code == 202, g.text
    for value in (0.70, 0.75):  # FIXTURE: base then LoRA
        job = client.post("/api/worker/jobs/claim", json={"types": ["generate"]}, headers=W(worker_token)).json()
        assert job["payload"]["lora"] in (None, "demo-run")
        worker_complete_generate(client, worker_token, job)
        ev = client.post("/api/worker/jobs/claim", json={"types": ["evaluate"]}, headers=W(worker_token)).json()
        res = [{"metric_result_id": m["metric_result_id"], "status": "completed", "value": value, "backend": "FIXTURE"}
               if m["metric"] == "clip_l14" else
               {"metric_result_id": m["metric_result_id"], "status": "unavailable", "detail": "test"}
               for m in ev["payload"]["metrics"]]
        client.post(f"/api/worker/jobs/{ev['id']}/complete", data={"result": json.dumps({"metrics": res})},
                    headers=W(worker_token))
    table = client.get(f"/api/reconstructions/{rid}/comparison-table").json()
    assert {r["generator"] for r in table["rows"]} == {"base", "lora:demo-run"}
    assert len(table["config_groups"]) == 2  # adapter is part of the settings key
    gp = client.get("/api/stats").json()["comparison"]["generator_pairs"]
    clip = next(p for p in gp if p["metric"] == "clip_l14")
    assert clip["n_pairs"] == 1 and clip["lora_better"] == 1 and abs(clip["mean_diff_lora_minus_base"] - 0.05) < 1e-9
    assert FIXTURE_ATTRIBUTES  # fixtures only


def test_artifact_export_import_roundtrip_and_sanitising(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("fa", Path(__file__).resolve().parents[2] / "scripts" / "finetune_artifacts.py")
    fa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fa)
    assert fa._sanitise_text('"cmd": "/Users/someone/PE/x.py"') == '"cmd": "~/PE/x.py"'
    assert fa._sanitise_text('"cmd": "/home/someone/PE"') == '"cmd": "~/PE"'
    local, shared = tmp_path / "local", tmp_path / "shared"
    run = local / "runs" / "r1"
    (run / "adapter").mkdir(parents=True)
    (run / "config.json").write_text('{"command": "/Users/someone/PE/train.py"}')
    (run / "status.json").write_text('{"state": "completed"}')
    (run / "adapter" / "pytorch_lora_weights.safetensors").write_bytes(b"A")
    stopped = local / "runs" / "r2"
    stopped.mkdir(parents=True)
    (stopped / "status.json").write_text('{"state": "stopped"}')
    monkeypatch.setattr(fa, "LOCAL", local)
    monkeypatch.setattr(fa, "SHARED", shared)
    monkeypatch.setattr(fa, "ROOT", tmp_path)
    assert fa.export() == 0
    assert (shared / "runs" / "r1" / "adapter" / "pytorch_lora_weights.safetensors").read_bytes() == b"A"
    assert "someone" not in (shared / "runs" / "r1" / "config.json").read_text()
    assert not (shared / "runs" / "r2").exists()  # only completed runs are shared
    monkeypatch.setattr(fa, "LOCAL", tmp_path / "other_machine")
    assert fa.import_() == 0
    assert (tmp_path / "other_machine" / "runs" / "r1" / "status.json").exists()
