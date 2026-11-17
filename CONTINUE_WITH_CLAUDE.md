# Continue this project with Claude (handoff)

Paste the prompt below into Claude Code on the new machine, from inside a clone of this repository on the branch
`macos-apple-silicon-migration`:

```bash
git clone https://github.com/Greninja44/PE.git
cd PE
git checkout macos-apple-silicon-migration
claude
```

---

## Prompt to paste

```
You are continuing my research project "Image-to-Prompt Reconstruction" in this repository
(branch macos-apple-silicon-migration). Read CONTINUE_WITH_CLAUDE.md, README.md, docs/ARCHITECTURE.md,
docs/APPLE_SILICON.md, docs/FINETUNING.md and docs/MACOS_MIGRATION_AUDIT.md before changing anything.

First:
1. Inspect this machine (OS, CPU/GPU, RAM, free disk, Python, Node) and tell me which setup path applies:
   Apple Silicon -> scripts/setup_mac.sh ; NVIDIA GPU -> docs/RTX5060_SETUP.md / scripts/setup_gpu_worker.sh ;
   CPU only -> scripts/setup_cloud.sh. Do not modify the system Python; use uv + Python 3.12.
2. Run the test suite (scripts/test.sh) and report the result before making changes.
3. Download only the models needed (backend/.venv/bin/python scripts/download_models.py; it is resumable),
   then verify real inference with scripts/verify_inference.py on a real photograph. Do not claim a model works
   because it imports - run it.

Then continue the remaining work listed in CONTINUE_WITH_CLAUDE.md under "Remaining work", in that order.

Rules (non-negotiable research integrity):
- Never fabricate, simulate, back-fill or "improve" results. A metric that has not run has no value.
- Keep measured results, model/rule suggestions, human ratings, hypotheses and future work clearly separated.
- The earlier 12-image pilot found structured prompts did NOT consistently beat captions and refinement gains may
  be selection bias. Do not hide or rewrite that. Report the FINAL refinement iteration, not the post-hoc best.
- Fine-tuning: never use the test split for training or checkpoint choice; report paired wins/losses, not only means;
  state that DiffusionDB references are SD 1.x outputs, not photographs.
- Do not commit model weights, datasets, databases, .env or tokens. Do not force-push or delete branches.
- Commit in small validated steps on this branch and ask me before pushing.
```

---

## Project state (at handoff)

**Research question.** Does a structured reconstructed prompt (and iterative refinement) reproduce a reference image
more faithfully than an ordinary caption, with model/seed/settings held fixed? Conditions: A caption · D detailed
prose · B structured · C refined (final iteration). Metrics: CLIP ViT-L/14 ↑, DINOv2 ↑, SSIM ↑, LPIPS ↓, plus blind
human ratings (separate table).

**Architecture.** FastAPI API (no torch) + SQLite + job queue; an inference worker process runs all models
(`python -m app.workers.agent local` on the same machine, or a remote NVIDIA worker); React dashboard.
`app/runtime/` handles device selection (CUDA → MPS → CPU), precision, one-model-family-at-a-time memory management,
memory measurement and keeping macOS awake during long jobs.

**Verified on the original Mac (Apple M6, 16 GB, MPS)** — see docs/APPLE_SILICON.md for numbers:
Qwen3-VL-2B (analysis), SD-Turbo and SD 1.5 (generation, bit-identical for the same seed), CLIP-B/32, CLIP-L/14,
DINOv2, LPIPS (match CPU to ≤ 3e-7), SSIM; full in-app run upload → analysis → A/D/B → evaluation → refinement.
76 backend tests pass; frontend builds.

**Fine-tuning (LoRA on SD 1.5, DiffusionDB CC0)** — see docs/FINETUNING.md:
- Dataset pipeline, trainer, evaluator, API (`/api/finetune/*`), Fine-tuning dashboard page and LoRA selector for
  SD 1.5 generation are implemented and tested (tiny-UNet training mechanics tested on CPU).
- Real training verified on MPS: run `sd15-ddb-r8-v1` (rank 8, lr 1e-4, 500 steps × 4 accumulation, 38 min, peak
  7.5 GB): validation loss 0.1496 → 0.1488 (almost flat). Run `sd15-ddb-r16-v2` (rank 16, lr 5e-4) was still
  training at handoff. **These runs and the dataset live only on the original Mac** (git-ignored); on a new machine
  they must be re-created.
- **`app/finetune/evaluate.py` has NOT yet been run on real models.** Its statistics function is unit-tested, but
  the end-to-end evaluation is unverified.

**Environment notes.**
- If port 8000 is taken (it was by Docker on the original Mac): `PE_API_PORT=8010 scripts/dev.sh`.
- Slow/unstable internet: downloads are resumable; `HF_HUB_DISABLE_XET=1` helped against CDN errors.
- The UI could not be screenshotted from the agent session; it was checked by type-checking, route and API checks.
  Look at it in a browser.

## Remaining work

1. Set up and verify the environment on the new machine (see prompt). On NVIDIA, CUDA is selected automatically.
2. Re-create fine-tuning data and runs:
   `scripts/download_diffusiondb.py --parts 1 2 3` → `python -m app.finetune.dataset build` →
   `python -m app.finetune.train --name sd15-ddb-r8-v1 --steps 500 --rank 8 --lr 1e-4` (and
   `--name sd15-ddb-r16-v2 --rank 16 --alpha 16 --lr 5e-4`). The split is deterministic, so it will match.
3. Run `python -m app.finetune.evaluate --run <run> --n 40` for each run; fix anything that fails; check the
   Fine-tuning page; add the measured results to docs/FINETUNING.md ("Runs and results").
4. Use the LoRA selector on real reference photographs (A/D/B protocol, base vs LoRA pairs appear automatically in
   Compare & Rate / Findings).
5. Visual check of all dashboard pages (Overview, Compare & Rate, Findings, Fine-tuning, Presentation Mode) and
   fix layout issues.
6. Research next steps: a larger image set with several seeds per image, blind ratings by more than one rater,
   paired statistical tests once n is adequate.
7. Open a pull request from `macos-apple-silicon-migration` into `main` when the owner approves.
