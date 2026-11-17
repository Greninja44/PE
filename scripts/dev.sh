#!/usr/bin/env bash
# Start the API, a local inference worker (Apple MPS / CUDA / CPU) and the dashboard (:5173).
# Ctrl+C stops all three.
#   PE_API_PORT=8010 scripts/dev.sh   # API on another port (default 8000), e.g. if Docker already uses 8000
#   PE_NO_WORKER=1 scripts/dev.sh     # no local worker (e.g. a remote GPU worker is used instead)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PE_API_PORT:-8000}"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "✗ Port $PORT is already in use by: $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | awk 'NR==2{print $1" (pid "$2")"}')"
  echo "  Start on another port, e.g.:  PE_API_PORT=8010 scripts/dev.sh"
  exit 1
fi
export PE_API_URL="http://127.0.0.1:$PORT"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
cd "$ROOT/backend"
.venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" --reload --reload-dir app &
PIDS+=($!)
if [ "${PE_NO_WORKER:-0}" != "1" ]; then
  if .venv/bin/python -c "import torch" 2>/dev/null; then
    .venv/bin/python -m app.workers.agent local --server "$PE_API_URL" &
    PIDS+=($!)
  else
    echo "⚠ torch not installed in backend/.venv — starting without a local worker (scripts/setup_mac.sh)"
  fi
fi
echo "API: $PE_API_URL/docs · Dashboard: http://localhost:5173"
cd "$ROOT/frontend"
npx vite --host "$HOST" --port 5173
