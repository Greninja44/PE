"""LoRA fine-tuning of the Stable Diffusion 1.5 UNet on the filtered DiffusionDB subset.

    python -m app.finetune.train --steps 1000 --rank 8 --lr 1e-4 --name sd15-ddb-r8

Memory plan for 16 GB unified memory (measured on an M6: ~0.95 s/step, peak ≈ 5.8 GB):
1. Encode every image once with the VAE (latent-distribution mean × scaling factor) and every prompt once with the
   CLIP text encoder; cache both in fp16; unload VAE and text encoder.
2. Train only LoRA matrices (fp32) on the UNet attention projections (to_q/to_k/to_v/to_out.0); the frozen UNet is
   bf16 on MPS/CUDA (fp32 on CPU) with gradient checkpointing; batch 1 with gradient accumulation.

Objective: standard ε-prediction MSE with the model's own DDPM noise schedule. Validation loss is computed on the
val split with *fixed* noise and timesteps so successive values are comparable. The test split is never touched
here. The final adapter is the one published; checkpoints are kept for inspection only (no post-hoc selection).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from app.finetune import dataset as D  # noqa: E402
from app.finetune.registry import ADAPTER_FILE, RUN_ID, runs_dir  # noqa: E402

BASES = {"sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5"}
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


# ------------------------------------------------------------------------------------------------- helpers
def _atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def image_tensor(path: Path, size: int = 512):  # noqa: ANN201
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def encode_prompts(tokenizer, text_encoder, prompts: list[str], device: str):  # noqa: ANN001, ANN201
    import torch

    ids = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        return text_encoder(ids)[0]


def build_cache(items: list[dict[str, Any]], raw: Path, base_repo: str, device: str, cache_file: Path,
                progress=print) -> dict[str, Any]:  # noqa: ANN001
    """Encode images → latents and prompts → embeddings once; returns {"latents": [N,4,64,64] fp16, "emb": [N,77,768]}."""
    import torch
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)
    dt = torch.float32 if device == "cpu" else torch.float16
    vae = AutoencoderKL.from_pretrained(base_repo, subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(device)
    tok = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(base_repo, subfolder="text_encoder", variant="fp16", torch_dtype=dt).to(device)
    lat, emb = [], []
    for i, it in enumerate(items):
        x = image_tensor(raw / it["image"]).unsqueeze(0).to(device)
        with torch.inference_mode():  # VAE in fp32: the SD 1.x VAE overflows in fp16
            lat.append((vae.encode(x).latent_dist.mean * vae.config.scaling_factor).to("cpu", torch.float16))
            emb.append(encode_prompts(tok, te, [it["prompt"]], device).to("cpu", torch.float16))
        if i % 100 == 0:
            progress(f"cache {i}/{len(items)}")
    cache = {"latents": torch.cat(lat), "emb": torch.cat(emb), "sha256": [it["sha256"] for it in items]}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_file)
    del vae, te
    from app.runtime.device import empty_cache

    empty_cache(device)
    return cache


def add_lora(unet, rank: int, alpha: int):  # noqa: ANN001, ANN201
    """Freeze the UNet and attach LoRA adapters; trainable params are kept in fp32 for stable updates."""
    import torch
    from peft import LoraConfig

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=TARGET_MODULES))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.to(torch.float32)
    return params


def train_step(unet, scheduler, latents, emb, dtype, device: str, gen=None):  # noqa: ANN001, ANN201
    """One ε-prediction loss on a batch (latents [B,4,h,w], emb [B,77,d]); returns the loss tensor (not stepped)."""
    import torch

    lat = latents.to(device, dtype)
    noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32).to(device, dtype)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), generator=gen).to(device)
    noisy = scheduler.add_noise(lat, noise, t)
    with torch.autocast(device_type="mps" if device == "mps" else device, dtype=dtype,
                        enabled=dtype != torch.float32 and device in ("mps", "cuda")):
        pred = unet(noisy, t, emb.to(device, dtype)).sample
    target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(lat, noise, t)
    return torch.nn.functional.mse_loss(pred.float(), target.float())


def validation_loss(unet, scheduler, cache, dtype, device: str, max_items: int = 64, seed: int = 1234) -> float:  # noqa: ANN001
    """Mean loss over val items with noise/timesteps fixed by ``seed`` → comparable across evaluations."""
    import torch

    n = min(max_items, cache["latents"].shape[0])
    if n == 0:
        return float("nan")
    gen = torch.Generator().manual_seed(seed)
    unet.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(n):
            total += float(train_step(unet, scheduler, cache["latents"][i:i + 1], cache["emb"][i:i + 1], dtype, device,
                                      gen).item())
    unet.train()
    return total / n


def save_adapter(unet, directory: Path) -> Path:  # noqa: ANN001
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    directory.mkdir(parents=True, exist_ok=True)
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(directory, unet_lora_layers=sd, safe_serialization=True)
    return directory / ADAPTER_FILE


# ---------------------------------------------------------------------------------------------------- main
def run(a: argparse.Namespace) -> int:
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel

    from app.experiments.tracking import code_fingerprint
    from app.runtime import memory
    from app.runtime.device import select_device, synchronize

    if not RUN_ID.fullmatch(a.name):
        print(f"invalid run name '{a.name}' (lower-case letters, digits, - and _)", file=sys.stderr)
        return 2
    rdir = runs_dir() / a.name
    if rdir.exists():
        print(f"run '{a.name}' already exists — choose another --name (runs are never overwritten)", file=sys.stderr)
        return 2
    card = D.card()
    train_items, val_items = D.load_split("train"), D.load_split("val")
    if not card or not train_items:
        print("dataset not built: python -m app.finetune.dataset build", file=sys.stderr)
        return 2
    if a.max_train_items:
        train_items = train_items[:a.max_train_items]
    dev = select_device(a.device).device
    dtype = torch.float32 if dev == "cpu" else torch.bfloat16
    repo = BASES[a.base]
    rdir.mkdir(parents=True)
    (rdir / "checkpoints").mkdir()
    cfg = {"schema": "pe.finetune.run.v1", "run_id": a.name, "base_model": a.base, "base_repo": repo,
           "method": "LoRA on UNet attention (to_q,to_k,to_v,to_out.0); ε-prediction MSE",
           "rank": a.rank, "alpha": a.alpha, "lr": a.lr, "warmup_steps": a.warmup, "steps": a.steps,
           "batch_size": 1, "grad_accum": a.grad_accum, "seed": a.seed, "resolution": 512,
           "device": dev, "frozen_dtype": str(dtype).removeprefix("torch."), "lora_dtype": "float32",
           "gradient_checkpointing": True, "dataset_card": card, "n_train": len(train_items),
           "n_val_used": min(a.val_items, len(val_items)), "created_at": datetime.now(UTC).isoformat(),
           "code_fingerprint": code_fingerprint(),
           "command": "python -m app.finetune.train " + " ".join(sys.argv[1:])}  # portable, no local paths
    _atomic_json(rdir / "config.json", cfg)
    status: dict[str, Any] = {"state": "preparing", "step": 0, "total_steps": a.steps, "message": "Caching latents"}
    _atomic_json(rdir / "status.json", status)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    t_start = time.time()
    try:
        with memory.PeakSampler(interval=0.5) as mem:
            mhash = card["manifest_sha256"]
            cdir = D.DATA_DIR / "cache" / a.base
            train_cache = build_cache(train_items, D.RAW_DIR, repo, dev, cdir / f"train_{mhash['train'][:12]}.pt",
                                      progress=lambda m: _atomic_json(rdir / "status.json", status | {"message": m}))
            if a.max_train_items:
                train_cache = {k: v[:a.max_train_items] for k, v in train_cache.items()}
            val_cache = build_cache(val_items[:a.val_items], D.RAW_DIR, repo, dev,
                                    cdir / f"val_{mhash['val'][:12]}_{a.val_items}.pt")
            unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", variant="fp16", torch_dtype=dtype).to(dev)
            params = add_lora(unet, a.rank, a.alpha)
            unet.enable_gradient_checkpointing()
            unet.train()
            sched = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")
            opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
            lr_at = lambda s: a.lr * min(1.0, (s + 1) / max(1, a.warmup))  # noqa: E731
            gen = torch.Generator().manual_seed(a.seed)
            order = list(range(train_cache["latents"].shape[0]))
            rng = random.Random(a.seed)
            rng.shuffle(order)
            cursor, ema = 0, None
            v0 = validation_loss(unet, sched, val_cache, dtype, dev, a.val_items)
            with open(rdir / "val_log.jsonl", "a") as vf:
                vf.write(json.dumps({"step": 0, "val_loss": v0}) + "\n")
            status.update(state="training", message="Training", val_loss_initial=v0, val_loss=v0)
            log = open(rdir / "train_log.jsonl", "a")  # noqa: SIM115
            t_train = time.time()
            for step in range(1, a.steps + 1):
                if stop["flag"]:
                    break
                t0 = time.time()
                for g in opt.param_groups:
                    g["lr"] = lr_at(step - 1)
                acc = 0.0
                for _ in range(a.grad_accum):
                    if cursor >= len(order):
                        rng.shuffle(order)
                        cursor = 0
                    i = order[cursor]
                    cursor += 1
                    loss = train_step(unet, sched, train_cache["latents"][i:i + 1], train_cache["emb"][i:i + 1],
                                      dtype, dev, gen) / a.grad_accum
                    loss.backward()
                    acc += float(loss.item())
                if not math.isfinite(acc):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                synchronize(dev)
                ema = acc if ema is None else 0.98 * ema + 0.02 * acc
                dt = time.time() - t0
                log.write(json.dumps({"step": step, "loss": acc, "loss_ema": ema, "lr": lr_at(step - 1),
                                      "seconds": round(dt, 3)}) + "\n")
                if step % 10 == 0 or step == a.steps:
                    log.flush()
                    el = time.time() - t_train
                    status.update(step=step, loss_ema=ema, sec_per_step=round(el / step, 3),
                                  eta_s=round(el / step * (a.steps - step)),
                                  epochs=round(step * a.grad_accum / len(order), 2),
                                  memory=memory.snapshot() | {"peak_process_gb": round(mem.peak_bytes / memory.GB, 2)})
                    _atomic_json(rdir / "status.json", status)
                if step % a.val_every == 0 or step == a.steps:
                    v = validation_loss(unet, sched, val_cache, dtype, dev, a.val_items)
                    with open(rdir / "val_log.jsonl", "a") as vf:
                        vf.write(json.dumps({"step": step, "val_loss": v}) + "\n")
                    status["val_loss"] = v
                if step % a.save_every == 0 and step != a.steps:
                    save_adapter(unet, rdir / "checkpoints" / f"step-{step}")
            log.close()
            done = status["step"] == a.steps and not stop["flag"]
            if done:
                save_adapter(unet, rdir / "adapter")
            else:
                save_adapter(unet, rdir / "checkpoints" / f"step-{status['step']}-stopped")
        status.update(state="completed" if done else "stopped",
                      message="Completed" if done else "Stopped by user (adapter not published)",
                      duration_s=round(time.time() - t_start), finished_at=datetime.now(UTC).isoformat(),
                      memory_peak=mem.report())
        _atomic_json(rdir / "status.json", status)
        print(json.dumps({k: status.get(k) for k in ("state", "step", "loss_ema", "val_loss_initial", "val_loss",
                                                       "duration_s", "memory_peak")}, indent=2, default=str))
        return 0 if done else 1
    except Exception as e:  # noqa: BLE001 — record the failure on the run, then re-raise for the CLI exit code
        status.update(state="failed", message=f"{type(e).__name__}: {e}", finished_at=datetime.now(UTC).isoformat())
        _atomic_json(rdir / "status.json", status)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.finetune.train")
    ap.add_argument("--name", default=f"sd15-ddb-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--base", choices=sorted(BASES), default="sd15")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-items", type=int, default=48)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--max-train-items", type=int, default=0, help="debug: limit training images")
    ap.add_argument("--device", default=None, help="auto|mps|cuda|cpu (default PE_DEVICE)")
    from app.runtime.power import stay_awake

    with stay_awake("lora training"):
        return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
