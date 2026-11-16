# RTX 5060 laptop setup

The RTX 5060 is a **Blackwell (compute capability 12.0 / sm_120)** GPU with **8 GB VRAM**. It needs:

* NVIDIA driver **R570 or newer**
* PyTorch **≥ 2.7 built for CUDA 12.8+** (`cu128` wheels). Older `cu121`/`cu124` builds fail with
  *"no kernel image is available for execution on the device"*.
* Python 3.12, Node.js 20+ (for the dashboard), Git, ~25 GB free disk for model weights.

## Option A (recommended for tonight): run everything on the laptop

### 1. Get the code

```bash
git clone https://github.com/Greninja44/PE.git
cd PE
```

### 2. Install

**Windows 11 (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_gpu_worker.ps1
```

**Linux / WSL2:**

```bash
scripts/setup_gpu_worker.sh             # also creates backend/.env with a fresh PE_ADMIN_TOKEN
```

Both scripts write `backend/.env` with a random `PE_ADMIN_TOKEN` (read it from that file). To make one by hand: `python -c "import secrets;print(secrets.token_urlsafe(32))"`.

The setup ends with `python -m app.workers.agent doctor`, which must print `✓ GPU ready: NVIDIA GeForce RTX 5060 …`
with `capability: 12.0` and `cuda: 12.8` (or newer).

### 3. Start backend + dashboard

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1          # Windows
```
```bash
scripts/dev.sh                                                     # Linux / WSL2
```

### 4. Register and start the worker (new terminal, in `backend/`)

```powershell
# Windows
.\.venv\Scripts\python.exe -m app.workers.agent register --server http://127.0.0.1:8000 --admin-token <PE_ADMIN_TOKEN> --name rtx5060
.\.venv\Scripts\python.exe -m app.workers.agent run --server http://127.0.0.1:8000
```
```bash
# Linux / WSL2
.venv/bin/python -m app.workers.agent register --server http://127.0.0.1:8000 --admin-token "$PE_ADMIN_TOKEN" --name rtx5060
.venv/bin/python -m app.workers.agent run --server http://127.0.0.1:8000
```

The token is saved to `~/.config/pe-worker/token` (override with `PE_WORKER_TOKEN_FILE` or pass `--token`).
After registration, `scripts\dev.ps1 -Worker` starts all three processes at once.

The sidebar should now show **GPU worker · online**.

### 5. First real run (checklist)

1. **New Reconstruction** → upload a reference image.
2. **Image Analysis** → *Structured JSON* with Qwen3-VL 2B, then *Caption*. The first job downloads ~4.5 GB.
3. **Prompt Studio** → generate versions from the structured analysis (v1 original, v2 validated) and from the caption.
4. **Image Generation** → *Controlled comparison*: pick the caption version (A) and v2 (B), model **SD-Turbo**
   (fast: 1 step, 512²) for a first check, then SD 1.5 or SDXL.
5. **Evaluation** → metrics fill in as the worker computes them (CLIP L/14 ~1.7 GB, DINOv2 ~0.35 GB download).
6. **Refinement** → start from v2 with *Analyse each output with qwen3vl_local*; propose → approve/reject.
7. **Experiments** → *Write report*, then *Export active reconstruction* to create a demo bundle; commit it
   (`git add experiments/reports experiments/prompts experiments/metrics experiments/demo`).

## Model memory on 8 GB

| Model | Download | VRAM (approx.) | Notes |
|---|---|---|---|
| Qwen3-VL-2B-Instruct (bf16) | ~4.5 GB | ~5–6 GB | greedy decoding, max 1024 new tokens |
| SD-Turbo (fp16) | ~2.5 GB | ~3 GB | 1–4 steps, CFG 0 |
| SD 1.5 (safetensors) | ~4 GB | ~3.5 GB fp16 | 512², 30 steps default |
| SDXL base 1.0 (fp16) | ~7 GB | 7–8 GB → **CPU offload enabled automatically** | 1024²; ~2–4× slower with offload |
| CLIP ViT-L/14 + B/32, DINOv2-base, LPIPS | ~2.5 GB total | ~2 GB | co-resident |

The worker unloads one model family before loading another, so analysis, generation and metric jobs can be mixed
in the queue without running out of memory. Expect a few seconds of reload when switching. If you still hit OOM
(e.g. other apps using VRAM), the job is retried after the worker frees memory; close GPU-heavy apps or use SD-Turbo.

Laptop tips: plug in AC power, set Windows power mode to *Best performance*, and make sure the browser/IDE is not
pinned to the dGPU.

## Option B: backend stays in Codespaces, laptop runs only the worker

Use this only if you need the cloud database. Caveats: the dashboard has no login, so a public port exposes it;
Codespaces private ports require GitHub authentication that the worker cannot perform.

1. In Codespaces: set `PE_ADMIN_TOKEN` in `backend/.env`, run `HOST=0.0.0.0 scripts/dev.sh`.
2. In the *Ports* tab set port **8000** visibility to **Public** for the session only; copy its URL
   (`https://<name>-8000.app.github.dev`).
3. On the laptop (after step 2 of Option A, backend venv only):
   `python -m app.workers.agent register --server https://<name>-8000.app.github.dev --admin-token <token>` and
   `python -m app.workers.agent run --server https://<name>-8000.app.github.dev`.
4. Set the port back to Private when finished.

## Moving data between machines

Git carries code, docs and tracked experiment metadata (`reports/`, `prompts/`, `metrics/`, `demo/`). The SQLite DB
and full-resolution images are git-ignored. To move them:

```bash
scripts/backup_data.sh                     # → experiments/bundles/pe-data-<stamp>.tar.gz
# copy the archive, then on the other machine, from the repo root:
tar -xzf pe-data-<stamp>.tar.gz
```

(The Codespace currently holds no research data — only code — so a fresh clone is enough tonight.)

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA error: no kernel image is available` | Reinstall torch from `https://download.pytorch.org/whl/cu128` |
| `doctor` says CUDA not available | Update driver (≥ R570); on WSL2 install the Windows driver only, not a Linux driver |
| Jobs stay *queued* | Worker not running or wrong `--server`; check the sidebar status and worker log |
| `401 Invalid or revoked worker token` | Re-run `register` (the DB was reset or the token file is stale) |
| `503 Worker registration disabled` | Set `PE_ADMIN_TOKEN` in `backend/.env` and restart the API |
| Analysis *failed: Could not parse model output* | Raw output is stored on the analysis card; re-run or annotate manually |
| SD 1.5 image is black | Safety checker triggered; `runtime_info.nsfw_flagged = true` records it |
| HF download slow / rate-limited | `huggingface-cli login` (optional; none of the default models are gated) |
