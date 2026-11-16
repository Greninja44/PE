#!/usr/bin/env bash
# GPU worker setup for Linux / WSL2 with an NVIDIA RTX 50-series GPU (Blackwell, sm_120).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
nvidia-smi >/dev/null || { echo "nvidia-smi not found: install an NVIDIA driver >= 570 first"; exit 1; }
PY="${PYTHON:-python3.12}"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
# Blackwell needs CUDA 12.8+ builds of PyTorch (>= 2.7).
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements-gpu.txt -r requirements-dev.txt
[ -f .env ] || { cp .env.example .env; sed -i "s|^PE_ADMIN_TOKEN=.*|PE_ADMIN_TOKEN=$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(32))')|; s|^PE_DATABASE_URL=.*|PE_DATABASE_URL=sqlite:///./data/pe.db|" .env; echo "created backend/.env with a fresh admin token"; }
(cd "$ROOT/frontend" && npm ci --no-audit --no-fund)
.venv/bin/python -m app.workers.agent doctor
