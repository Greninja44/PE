"""LoRA fine-tuning of Stable Diffusion 1.5 on a filtered DiffusionDB (CC0) subset.

Pipeline: ``dataset`` (filter + leakage-free grouped split) → ``train`` (cached latents/text embeddings, LoRA on the
UNet attention projections, MPS/CUDA/CPU) → ``evaluate`` (base vs LoRA on the held-out test split, paired metrics).
Runs live in ``backend/data/finetune/runs/<run_id>/`` (git-ignored); the API and dashboard read them from there.
"""
