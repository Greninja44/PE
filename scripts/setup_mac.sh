#!/usr/bin/env bash
# Apple Silicon setup: Python 3.12 venv via uv (system Python untouched), MPS-enabled PyTorch from PyPI,
# frontend dependencies, backend/.env with a fresh admin token, and an MPS self-test.
# Model weights are NOT downloaded here; run scripts/download_models.py afterwards.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || echo "⚠ Not macOS arm64 — use scripts/setup_cloud.sh or setup_gpu_worker.sh instead."
command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v npm >/dev/null || { echo "Node.js/npm not found. Install Node LTS from https://nodejs.org (or: brew install node)"; exit 1; }

cd "$ROOT/backend"
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt -r requirements-dev.txt
if [ ! -f .env ]; then
  cp .env.example .env
  TOKEN=$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(32))')
  # BSD sed (macOS) needs an explicit (empty) backup suffix.
  sed -i '' "s|^PE_ADMIN_TOKEN=.*|PE_ADMIN_TOKEN=${TOKEN}|" .env
  sed -i '' "s|^PE_OPENAI_COMPAT_BASE_URL=.*|PE_OPENAI_COMPAT_BASE_URL=|" .env
  echo "created backend/.env with a fresh admin token"
fi
(cd "$ROOT/frontend" && npm ci --no-audit --no-fund)
.venv/bin/python -m app.workers.agent doctor | tail -3
echo
echo "Next:  backend/.venv/bin/python scripts/download_models.py   # ~9 GB, once"
echo "       scripts/dev.sh                                          # API + local MPS worker + dashboard"
