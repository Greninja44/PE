#!/usr/bin/env bash
# Start the API (:8000), a local inference worker (Apple MPS / CUDA / CPU) and the dashboard (:5173).
# Ctrl+C stops all three. Set PE_NO_WORKER=1 to start without the local worker (e.g. when a remote GPU
# worker is used instead).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
cd "$ROOT/backend"
.venv/bin/uvicorn app.main:app --host "$HOST" --port 8000 --reload --reload-dir app &
PIDS+=($!)
if [ "${PE_NO_WORKER:-0}" != "1" ]; then
  if .venv/bin/python -c "import torch" 2>/dev/null; then
    .venv/bin/python -m app.workers.agent local --server "http://127.0.0.1:8000" &
    PIDS+=($!)
  else
    echo "⚠ torch not installed in backend/.venv — starting without a local worker (scripts/setup_mac.sh)"
  fi
fi
cd "$ROOT/frontend"
npx vite --host "$HOST" --port 5173
