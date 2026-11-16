# PE — Image-to-Prompt Reconstruction Research Platform

Reverse-engineer a text prompt that makes a diffusion model regenerate a reference image, and
measure — honestly — how close each stage gets.

```
reference image ─▶ 1 Analyse (Qwen3-VL) ─▶ 2 Prompt (templates + validation) ─▶ 3 Generate (SD 1.5 / SD-Turbo / SDXL)
                                                       ▲                                  │
                                                       └── 5 Refine (propose/approve) ◀── 4 Evaluate (CLIP · DINOv2 · SSIM · LPIPS)
```

Three conditions are compared at identical model, seed and parameters:
**A** caption only · **B** structured, validated prompt · **C** structured prompt + iterative refinement.

## Integrity rules (enforced in code)

* **No simulated outputs.** Unconfigured providers are refused (HTTP 422). Worker jobs stay *queued* until a real GPU
  worker executes them. A metric that has not run has `value = null` and a status — never a number.
* **Optimisation ≠ evaluation.** Refinement optimises one metric (default CLIP ViT-B/32); evaluation uses a different
  set (CLIP ViT-L/14, DINOv2, SSIM, LPIPS). Each metric row records its `role`.
* **Full provenance.** Reference SHA-256, prompt lineage, model repo id, seed, steps, CFG, resolution, runtime versions,
  timing and a code fingerprint (git commit + source hash) are stored for every run.
* **Never overwrite reports.** Reports use exclusive file creation with timestamped names.
* **Test fixtures stay in tests.** Synthetic gradient images and hand-picked scores exist only in `backend/tests/`.

## Quick start (Codespaces / any CPU machine)

```bash
scripts/setup_cloud.sh      # Python 3.12 venv + CPU-only deps, npm ci, creates backend/.env with an admin token
scripts/test.sh             # 44 backend tests + frontend type-check/build
scripts/dev.sh              # API on :8000, dashboard on :5173
```

Open http://localhost:5173 (Codespaces forwards the port automatically). API docs: http://localhost:8000/docs.

Without a GPU you can upload images, annotate manually, build and validate prompts, queue jobs, compute SSIM, run
the refinement state machine's control flow, and export experiments. Everything that needs a model waits for the
GPU worker and is labelled as such.

## GPU worker (RTX 5060)

See **[docs/RTX5060_SETUP.md](docs/RTX5060_SETUP.md)**. In short, on the laptop:

```bash
scripts/setup_gpu_worker.sh            # or scripts\setup_gpu_worker.ps1 on Windows
scripts/dev.sh                         # backend + dashboard locally
cd backend && .venv/bin/python -m app.workers.agent register --server http://127.0.0.1:8000 --name rtx5060
.venv/bin/python -m app.workers.agent run --server http://127.0.0.1:8000
```

## Repository layout

```
backend/app/
  analysis/       Stage 1 — VLM provider interface, Qwen3-VL runner, OpenAI-compatible & HF router providers
  prompt_engine/  Stage 2 — attribute schema, fixed templates, validator (contradictions/duplicates/relations)
  generation/     Stage 3 — model registry (SD 1.5, SD-Turbo, SDXL), diffusers runner, HF inference provider
  evaluation/     Stage 4 — CLIP, DINOv2, SSIM (NumPy, verified vs scikit-image), LPIPS
  refinement/     Stage 5 — rule-based proposer, stopping policy, session state machine
  experiments/    tracking, code fingerprint, JSON/CSV export, write-once reports, demo bundles
  workers/        job queue (leases/retries/resume), typed payloads, worker agent + handlers
  services/       storage and pipeline orchestration
  api/            FastAPI routes
  database/       SQLAlchemy models (SQLite)
  config/         settings (env prefix PE_)
frontend/src/     React + TypeScript + Tailwind + Recharts dashboard (11 pages incl. Presentation Mode)
experiments/      originals/ generated/ (media, git-ignored) · prompts/ metrics/ reports/ demo/ (tracked)
docs/             ARCHITECTURE.md · RTX5060_SETUP.md · RECOVERY_AUDIT.md
scripts/          setup, dev, test, GPU worker setup, data backup
```

## Dashboard pages

Overview · New Reconstruction · Image Analysis · Prompt Studio · Image Generation · Evaluation · Refinement ·
Experiments · Research Findings · Methodology · Presentation Mode (`/present`, arrow keys, Esc to exit).

Presentation Mode can show a **live** reconstruction or a **demo bundle** — a self-contained, git-tracked export of a
genuine A/B/C comparison under `experiments/demo/` (Experiments page → *Export active reconstruction*), so results
produced on the GPU laptop can be presented from any machine.

## Configuration

`backend/.env` (see `backend/.env.example`). Key variables: `PE_ADMIN_TOKEN` (required to register workers),
`PE_DATABASE_URL`, `PE_EXPERIMENTS_DIR`, `PE_OPENAI_COMPAT_BASE_URL` / `PE_OPENAI_COMPAT_MODEL` (alternative VLM
endpoint such as llama.cpp, vLLM, LM Studio or Ollama), `PE_HF_TOKEN` (hosted inference).
