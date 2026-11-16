#!/usr/bin/env bash
# CPU-only development setup (Codespaces or any machine). Installs NO CUDA packages or model weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
PY="${PYTHON:-python3.12}"
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv --python "$PY" .venv
  uv pip install --python .venv/bin/python -r requirements-dev.txt
else
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements-dev.txt
fi
[ -f .env ] || { cp .env.example .env; sed -i "s|^PE_ADMIN_TOKEN=.*|PE_ADMIN_TOKEN=$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(32))')|" .env; sed -i "s|^PE_DATABASE_URL=.*|PE_DATABASE_URL=sqlite:///./data/pe.db|" .env; echo "created backend/.env with a fresh admin token"; }
cd "$ROOT/frontend"
npm ci --no-audit --no-fund
echo "Setup complete. Run: scripts/dev.sh"
