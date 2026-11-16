"""Download only the model files the pipeline actually loads (no legacy .ckpt, no duplicate precisions).

    backend/.venv/bin/python scripts/download_models.py                 # default set (~9 GB)
    backend/.venv/bin/python scripts/download_models.py --only sd_turbo clip_b32
    backend/.venv/bin/python scripts/download_models.py --list

Weights go to the Hugging Face cache (``HF_HOME``, default ``~/.cache/huggingface``), never into the repo.
"""
from __future__ import annotations

import argparse
import sys

CONFIGS = ["*.json", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab*"]

# key -> (repo id, allow_patterns, approx size, in default set). Ordered so the smallest complete pipeline
# (VLM + SD-Turbo + metrics) is available first on slow connections.
MODELS: dict[str, tuple[str, list[str], str, bool]] = {
    "qwen3vl_2b": ("Qwen/Qwen3-VL-2B-Instruct", ["*.safetensors", *CONFIGS, "*.jinja"], "~4.3 GB", True),
    "sd_turbo": ("stabilityai/sd-turbo",
                 ["*/diffusion_pytorch_model.fp16.safetensors", "*/model.fp16.safetensors", "*/*.json", "*/*.txt",
                  "model_index.json"], "~2.6 GB", True),
    # This repo publishes only pytorch_model.bin on its main branch (no safetensors).
    "clip_b32": ("openai/clip-vit-base-patch32", ["pytorch_model.bin", *CONFIGS], "~0.6 GB", True),
    "dinov2_base": ("facebook/dinov2-base", ["model.safetensors", *CONFIGS], "~0.35 GB", True),
    "clip_l14": ("openai/clip-vit-large-patch14", ["model.safetensors", *CONFIGS], "~1.7 GB", True),
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5",
             ["*/diffusion_pytorch_model.fp16.safetensors", "*/model.fp16.safetensors", "*/*.json", "*/*.txt",
              "model_index.json"], "~2.6 GB (incl. safety checker)", True),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0",
             ["*/diffusion_pytorch_model.fp16.safetensors", "*/model.fp16.safetensors", "*/*.json", "*/*.txt",
              "model_index.json"], "~7 GB — not recommended on 16 GB machines", False),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help=f"subset of: {', '.join(MODELS)}")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (repo, _, size, default) in MODELS.items():
            print(f"{k:12s} {repo:48s} {size:40s} {'default' if default else 'optional'}")
        return 0
    import os
    from pathlib import Path

    # Fail fast on stalled connections (the default waits a long time on a hung TLS handshake); the outer retry
    # resumes partial files. Must be set before huggingface_hub is imported.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import snapshot_download

    keys = a.only or [k for k, v in MODELS.items() if v[3]]
    unknown = set(keys) - set(MODELS)
    if unknown:
        print(f"unknown model keys: {sorted(unknown)}", file=sys.stderr)
        return 2
    for k in keys:
        repo, patterns, size, _ = MODELS[k]
        # Already complete locally? Skip without any network request.
        try:
            local = snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
            if any(w for ext in ("*.safetensors", "*.bin") for w in Path(local).rglob(ext)):
                print(f"✓ {k}: already downloaded", flush=True)
                continue
        except Exception:  # noqa: BLE001 — not (fully) cached yet
            pass
        print(f"→ {k}: {repo} ({size})", flush=True)
        path = snapshot_download(repo, allow_patterns=patterns)

        weights = [w for ext in ("*.safetensors", "*.bin") for w in Path(path).rglob(ext)]
        if not weights:  # configs alone are not a usable model — never report success without weights
            print(f"  ✗ {k}: no weight files matched {patterns}", file=sys.stderr, flush=True)
            return 1
        print(f"  ✓ {path} ({len(weights)} weight file(s))", flush=True)
    print("LPIPS (AlexNet) weights ship with the `lpips` package; torchvision fetches the AlexNet backbone (~230 MB) "
          "on first use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
