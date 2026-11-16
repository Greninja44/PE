# Apple Silicon (MPS) guide

Target machine: Mac mini, Apple M6, **16 GB unified memory**, macOS 27.2.1. Everything here also applies to other
M-series Macs; with less than 16 GB, use `PE_STRICT_MEMORY_CHECK=true` and SD-Turbo only.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh            # uv (user-local; system Python untouched)
# Node.js LTS from https://nodejs.org, or `brew install node`
scripts/setup_mac.sh                                        # venv (Python 3.12), torch with MPS, npm deps, .env, doctor
backend/.venv/bin/python scripts/download_models.py         # default set; see --list
scripts/dev.sh                                              # API + local worker + dashboard
```

Dependencies: `backend/requirements-mac.txt` (PyPI torch wheels for macOS arm64 include MPS; no CUDA packages and
no extra index URL). `requirements-gpu.txt` remains the NVIDIA/CUDA worker file.

### Model downloads

`scripts/download_models.py` fetches only the files that are loaded (fp16 safetensors for the diffusion models,
no legacy `.ckpt`), into the Hugging Face cache (`~/.cache/huggingface`, override with `HF_HOME`). It is resumable;
rerun it after an interruption. Order: Qwen3-VL-2B → SD-Turbo → CLIP-B/32 → DINOv2 → CLIP-L/14 → SD 1.5, so the
smallest complete pipeline is usable first. If a download fails with a *CAS / xet* error, run it with
`HF_HUB_DISABLE_XET=1`. An optional `HF_TOKEN` raises Hugging Face rate limits.

| Key | Repository | Download |
|---|---|---|
| qwen3vl_2b | Qwen/Qwen3-VL-2B-Instruct | ~4.3 GB |
| sd_turbo | stabilityai/sd-turbo (fp16) | ~2.6 GB |
| clip_b32 | openai/clip-vit-base-patch32 | ~0.6 GB |
| dinov2_base | facebook/dinov2-base | ~0.35 GB |
| clip_l14 | openai/clip-vit-large-patch14 | ~1.7 GB |
| sd15 | stable-diffusion-v1-5/stable-diffusion-v1-5 (fp16) | ~2.6 GB |
| sdxl (optional) | stabilityai/stable-diffusion-xl-base-1.0 | ~7 GB — not recommended on 16 GB |

LPIPS weights ship with the `lpips` package; torchvision downloads the AlexNet backbone (~230 MB) on first use.

## How inference is configured

| Concern | Implementation |
|---|---|
| Device | `app/runtime/device.py`: CUDA → MPS → CPU (`PE_DEVICE` overrides). An unavailable request falls back to CPU and says so. |
| Missing MPS kernels | `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch is imported (ops without MPS kernels run on CPU). Metrics that still raise on MPS are recomputed on CPU; the fallback is written into the metric's detail. |
| Precision | Qwen3-VL bf16 · diffusion fp16 (fp32 on CPU) · metrics fp32. `PE_HALF_PRECISION=false` forces fp32. |
| Invalid output | An all-black image without an NSFW flag (fp16 NaN in the VAE) is raised as a failure, never stored. NaN metric values are rejected. |
| Reproducibility | MPS/CPU use a **CPU** `torch.Generator`, so a seed gives the same initial noise on any machine. The CUDA path keeps its original on-device generator. Runs record `generator_device`, `dtype`, `device`, package versions. |
| Memory | `app/runtime/lifecycle.py`: one model family resident; switching unloads + `torch.mps.empty_cache()`; idle unload after `PE_WORKER_IDLE_UNLOAD_S` (300 s); pre-load check vs free memory; OOM → unload all, job retried with a clear message. |
| Resolution | 512×512 default for SD-Turbo / SD 1.5; reference images downscaled to ≤ `PE_VLM_MAX_IMAGE_SIDE` (768 px) before the VLM. Batch size is always 1. |
| Attention | diffusers attention slicing on MPS (`PE_ATTENTION_SLICING=auto`) to lower peak memory. |
| Diagnostics | `python -m app.workers.agent doctor` runs real tensor ops; the worker heartbeat reports process footprint, free system memory, swap and resident models to the Overview page; each job stores its measured peak footprint. |

Memory figures are the macOS **physical footprint** of the worker process (what Activity Monitor shows as
"Memory"), read via `proc_pid_rusage`; it includes Metal/MPS buffers, unlike RSS. The reader was checked against
the macOS `footprint` tool (same value) and against a 1 GiB MPS allocation (+1.06 GiB).

### MLX

MLX was considered and not adopted: the existing pipeline (transformers Qwen3-VL, diffusers SD, CLIP/DINOv2/LPIPS)
runs on PyTorch MPS, and switching only the VLM to an MLX port would add a second runtime and change the model
implementation behind the recorded results without a measured advantage. It remains future work (quantised
Qwen3-VL to cut memory), to be adopted only with a side-by-side measurement.

## Measured on this machine

Apple M6, 16 GB, macOS 27.2.1, Python 3.12.16, torch 2.15.0 (MPS; recommended max working set 12.71 GB),
transformers 5.23.0, diffusers 0.43.0. Measured 2026-11-16 with `scripts/verify_inference.py` and in-app runs.
"Peak" is the worker's macOS physical footprint (includes Metal buffers), sampled every 0.1 s. Other applications
were open (5.7–9.6 GB system memory free before each run), so these are realistic, not best-case, figures.

| Model / step | Device · dtype | Load | Inference | Peak footprint | After unload |
|---|---|---|---|---|---|
| Qwen3-VL-2B — caption | mps · bf16 | 7.3 s | 0.9–1.0 s (warm) | 5.25 GB | 0.43 GB |
| Qwen3-VL-2B — structured JSON | mps · bf16 | — | 6.9–10.3 s (~360 output tokens) | 5.27 GB | — |
| SD-Turbo 512×512, 1 step | mps · fp16 | 7.1–7.9 s | 0.32 s steady · 5.1–5.5 s first call (warm-up) | 4.98 GB | 0.63 GB |
| SSIM | NumPy (API) | — | < 0.1 s | 0.65 GB | — |
| CLIP ViT-B/32 | mps · fp32 | incl. | 6.7 s first pair (load + 2 images) | 1.58 GB | 0.44 GB |
| CLIP ViT-L/14 | mps · fp32 | incl. | 3.6 s first pair | 2.49 GB | 0.47 GB |
| DINOv2-base | mps · fp32 | incl. | 1.3 s first pair | 0.82 GB | 0.46 GB |
| LPIPS (AlexNet) | mps · fp32 | incl. | 0.24 s | 1.01 GB | 0.90 GB |
| SD 1.5 512×512, 30 steps, CFG 7.5 | mps · fp16 | 5.0 s | 8.0 s steady · 11.8 s first call | 4.49 GB | 0.64 GB |
| SDXL | — | not tested (not recommended on 16 GB) | | | |

Correctness checks performed (not just imports): parsed caption and complete structured JSON from real photographs;
SD-Turbo and SD 1.5 outputs visually coherent and matching the prompt, no fp16 NaN/black frames; **same seed → bit-identical image** on MPS for both (max pixel
difference 0); SSIM(reference, reference) = 1.0; memory returns to < 1 GB after each unload.

Metric correctness on MPS: every metric scores an image against itself at its optimum (CLIP/DINOv2/SSIM = 1.0,
LPIPS = 0.0), runs natively on MPS without CPU fallback, and **agrees with the CPU result to ≤ 3×10⁻⁷**
(CLIP-B/32 1.2e-7, CLIP-L/14 3.0e-7, DINOv2 3.0e-7, LPIPS 0). The same generation (same prompt and seed) scored in
two different processes gave identical metric values.

End-to-end in the app: upload → Qwen caption + structured analysis → A/D/B prompts → SD-Turbo generations →
automatic evaluation → a refinement session (rule proposals using a Qwen analysis of the generated image, human
approval, early stop). The worker kept one model family resident at a time throughout (metrics → VLM → diffusion →
metrics → VLM …, ~0.9–1.2 GB between swaps); all 17 jobs that were not deliberately cancelled succeeded.

Observations that changed the implementation:

* macOS returns freed Metal memory ~0.5 s after `torch.mps.empty_cache()`; measured immediately it still looked
  like 4.4 GB was held. The ModelManager now waits for the footprint to settle before loading the next model.
* The first diffusion call after a load includes ~5 s of one-off warm-up; it is flagged and excluded from
  generation-time comparisons (otherwise whichever condition ran first looked ~16× slower).
* transformers 5 returns CLIP image features as a model-output object; the CLIP metric reads `pooler_output`
  (the same projected embedding as before).

Sequential loading keeps the worst case at one family: VLM (~5.3 GB) **or** SD-Turbo (~5.0 GB), never both, which
leaves room for the OS and a browser on 16 GB.

### Network

Model downloads on the development connection ran at 0.2–1.9 MB/s with a full outage during the session. Parallel
connections did not help (8 streams: 0.26 MB/s total vs 1.44 MB/s single), i.e. the line was the limit. The
download script is resumable, skips completed models offline and fails fast on stalled connections.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor` says CPU instead of MPS | Check `uname -m` is `arm64` (not Rosetta) and that the venv Python is arm64: `backend/.venv/bin/python -c "import platform;print(platform.machine())"`. |
| `MPS backend out of memory` | Close memory-heavy apps (browsers), lower resolution, keep SD-Turbo, set `PE_ATTENTION_SLICING=on`. The worker unloads all models after an OOM and the job is retried. |
| Black generated image reported as failure | fp16 NaN; rerun with `PE_HALF_PRECISION=false` (slower, more memory). |
| Jobs stay *queued* | No worker online: start `scripts/dev.sh` or `python -m app.workers.agent local`. The sidebar shows worker status. |
| `registration failed (401)` | `PE_ADMIN_TOKEN` in `backend/.env` changed; delete `~/.config/pe-worker/token-local` and restart. |
| First run of a model is slow | Weights are read from disk (or downloaded if not cached). The Overview page shows which models are downloaded. |
| Slow downloads / `CAS Client Error` | `HF_HUB_DISABLE_XET=1 backend/.venv/bin/python scripts/download_models.py` (resumes). |
| `sed: 1: ... invalid command code` | Old GNU-style `sed -i`; the scripts now use BSD-compatible forms. |
