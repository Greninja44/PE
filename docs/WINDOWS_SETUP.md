# Windows setup (10/11, NVIDIA GPU or CPU-only)

The same code runs on Windows: the device layer picks **CUDA** when an NVIDIA GPU is available and **CPU**
otherwise (Apple MPS is macOS-only). Nothing in this guide changes the system Python.

> These PowerShell scripts were written on macOS and could not be executed there; they use only standard cmdlets.
> If a step fails, the equivalent commands are listed under "Manual setup".

## 1. Prerequisites

* **Git** — https://git-scm.com/download/win
* **Node.js LTS** — https://nodejs.org
* **uv** (Python manager; installs Python 3.12 for the project):
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` — then open a **new**
  PowerShell window.
* **NVIDIA GPU (optional):** a current driver (R570+ for RTX 50-series). Check with `nvidia-smi`.
* ~15 GB free disk for model weights (+ ~2 GB for the fine-tuning dataset).

## 2. Get the code and set up

```powershell
git clone https://github.com/Greninja44/PE.git
cd PE
git checkout macos-apple-silicon-migration
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

The script installs the CUDA 12.8 build of PyTorch if `nvidia-smi` is found, otherwise the CPU build; then the backend
and frontend dependencies, `backend\.env` with a fresh admin token, and runs `doctor` (real tensor ops on the device).

## 3. Models (once, resumable)

```powershell
backend\.venv\Scripts\python.exe scripts\download_models.py
```

If a download fails with a CAS/xet error: `$env:HF_HUB_DISABLE_XET="1"` and run again.

## 4. Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1             # API :8000 + local worker + dashboard
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 -Port 8010  # if port 8000 is taken
```

Open http://localhost:5173. Three windows open (API, dashboard, worker); close them to stop.

Checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
backend\.venv\Scripts\python.exe -m app.workers.agent doctor
backend\.venv\Scripts\python.exe scripts\verify_inference.py --image C:\path\to\photo.jpg
```

## 5. Fine-tuning artifacts from the Mac

The trained LoRA adapters, their logs and the dataset split manifests are in `experiments\finetune\` (git-tracked).
Make them visible to the app:

```powershell
backend\.venv\Scripts\python.exe scripts\finetune_artifacts.py import
```

To re-train or evaluate you also need the DiffusionDB images (~1.9 GB, not in git):

```powershell
cd backend
.venv\Scripts\python.exe ..\scripts\download_diffusiondb.py --parts 1 2 3
.venv\Scripts\python.exe -m app.finetune.dataset build          # produces the same split (deterministic)
.venv\Scripts\python.exe -m app.finetune.evaluate --run sd15-ddb-r8-v1 --n 40
```

## Hardware notes

| GPU | Expectation |
|---|---|
| RTX with ≥ 8 GB VRAM | All models run; one model family is kept on the GPU at a time. LoRA training (bf16 UNet, batch 1, gradient checkpointing) is expected to fit in 8 GB but has only been measured on Apple Silicon. |
| RTX with 4–6 GB | Use SD-Turbo; SD 1.5 should fit; Qwen3-VL-2B (≈4.5 GB bf16) is tight. |
| CPU only | Everything works but slowly (minutes per image with SD 1.5; SD-Turbo is the practical choice). Training on CPU is not practical. |

## Manual setup (if the script fails)

```powershell
cd backend
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or without --index-url for CPU
uv pip install --python .venv\Scripts\python.exe -r requirements-gpu.txt -r requirements-dev.txt
copy .env.example .env      # then set PE_ADMIN_TOKEN to a random string
cd ..\frontend
npm ci
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `running scripts is disabled on this system` | Use `powershell -ExecutionPolicy Bypass -File ...` as shown. |
| `doctor` says CPU on an NVIDIA machine | The CPU wheel was installed; rerun the CUDA install line from "Manual setup". |
| `UnicodeEncodeError` in a console | `$env:PYTHONUTF8="1"` (the scripts set it; the app also replaces unencodable characters). |
| Port 8000 in use | `scripts\dev.ps1 -Port 8010`. |
| `CUDA out of memory` | Close other GPU apps, use SD-Turbo / 512×512; the worker unloads all models after an OOM and retries. |
| Long jobs stall when the laptop sleeps | Training, evaluation and worker jobs request "system required" from Windows while running; plug the laptop in. |
