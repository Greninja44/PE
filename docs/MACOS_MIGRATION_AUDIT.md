# Apple Silicon migration — initial audit (2026-11-16)

Audited before any change, on a Mac mini (Apple M6, 16 GB unified memory, macOS 27.2.1). The repository was
cloned fresh to `~/Projects/PE` from `origin = https://github.com/Greninja44/PE.git`; only branch `main`,
5 commits (`a345c49` … `7298b24`), clean working tree.

## Baseline (before modification)

| Check | Result |
|---|---|
| `pytest` (Python 3.12.x via uv, CPU deps only) | **44 passed** |
| `npm run build` (tsc + vite) | **passes** (single 680 kB JS chunk warning only) |
| Experiment data in repo | **None.** `experiments/{originals,generated,prompts,metrics,reports}` contain only `.gitkeep`; no `experiments/demo/` bundles; no notebooks |
| Earlier implementation | `docs/RECOVERY_AUDIT.md` records that commit `51c5848` (original implementation) is unrecoverable; the current code is a rebuild |

**Research-integrity note.** The earlier 12-image pilot (structured prompts did *not* consistently outperform
captions; refinement gains possibly due to selection bias) is **not present as data in this repository** — no
images, prompts, metrics or reports. It can therefore only be shown as a *researcher-reported prior finding*,
never recomputed or re-plotted. If the raw files exist elsewhere, they should be imported unchanged.

## 1. What already works (verified by tests or code reading)

* FastAPI backend, SQLite via SQLAlchemy, content-addressed image storage, upload validation.
* Prompt engine: 4 fixed templates (`structured_v1`, `subject_first_v1`, `natural_v1`, `caption_v1`), validator
  (contradictions, duplicates, relations, CLIP-window overflow), immutable prompt versions with lineage.
* Job queue with leases/retries/cancel/resume; worker agent; typed payloads; model allow-lists.
* Refinement state machine (rule-based proposer, human approve/reject, patience/min-delta stopping), with
  optimisation metric (`clip_b32`) kept separate from evaluation metrics (CLIP-L/14, DINOv2, SSIM, LPIPS).
* SSIM in NumPy (no torch), verified in tests.
* Experiment tracking: code fingerprint, JSON/CSV export, write-once reports, demo bundles.
* React dashboard: 11 pages incl. Presentation Mode; Findings page shows only genuine completed metrics.

## 2. Incomplete

* No **human evaluation** at all (only automatic metrics).
* Conditions are A caption / B structured / C refined. A separate **detailed prose prompt** condition
  (user's comparison "B") is not a recorded condition, although the `natural_v1` template can produce one.
* No cross-condition **comparison table** with generation time, prompt length and iteration count.
* No memory diagnostics; no idle model unloading; no local (same-machine) worker workflow.
* No model download tooling (weights would be pulled whole, including multi-GB legacy `.ckpt` files if
  `snapshot_download` were used naïvely).

## 3. Broken / will break on this Mac

* All model inference assumes **CUDA**: `QwenVLRunner` uses `device_map="auto"` + bf16; `DiffusersRunner` picks
  `cuda` else `cpu` with fp32 (MPS never used); metrics pick `cuda` else `cpu`; `ModelManager` only empties
  the CUDA cache; `agent doctor` fails unless CUDA is present; runtime snapshot reports only CUDA.
* `scripts/setup_cloud.sh` uses GNU `sed -i` (fails on BSD/macOS `sed`).
* README/setup assume a separate RTX 5060 laptop and manual worker registration.

## 4. Needs modification for macOS

* Device abstraction (MPS → CPU fallback, CUDA kept for the future), `PYTORCH_ENABLE_MPS_FALLBACK=1`,
  dtype policy per device, CPU-seeded generators for reproducibility, MPS cache release.
* Sequential model lifecycle sized for 16 GB: one heavy family resident, explicit unload, idle unload,
  memory check before load, diagnostics reported to the UI.
* A one-command local stack: API + local inference worker + dashboard.
* macOS dependency file (PyPI torch wheels include MPS; no CUDA index).

## 5. Should remain unchanged

* Database schema for existing tables and their semantics (new tables only additive).
* Provider names stored in rows (`gpu_worker`, `qwen3vl_local`), condition keys `A_caption`/`B_structured`/
  `C_refined`, metric keys, template ids, export schema `pe.experiment.v1`.
* Integrity rules: no simulated outputs, `value = NULL` unless computed, optimisation ≠ evaluation, write-once
  reports, fixtures only in tests.
* The API process never imports torch (keeps the server light; the worker can be restarted to free memory).
* Default VLM stays **Qwen/Qwen3-VL-2B-Instruct**; generation models SD 1.5 / SD-Turbo stay; SDXL stays
  registered but is flagged as not recommended on 16 GB.
* Existing CUDA/RTX worker path and docs (kept for future NVIDIA use).

## 6. Migration sequence

1. Environment: uv + Python 3.12 venv, Node LTS (user-local), baseline tests. *(done before this file)*
2. Device + memory runtime module; adapt VLM, diffusion, metric runners; worker doctor/snapshot.
3. Local worker mode (auto-register with admin token, same machine), idle unload, memory diagnostics in heartbeat.
4. Real inference verification on MPS for each model, with measured time and memory.
5. Research features: detailed-prose condition, comparison endpoint/view, human ratings (separate table).
6. Frontend: worker/model status, comparison view, human evaluation, findings labels (measured / suggestion /
   human / hypothesis / future work / reported prior finding).
7. Tests (mocked inference), macOS scripts, README/docs, incremental commits (no push).
