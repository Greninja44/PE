#!/usr/bin/env bash
# Start the API (port 8000) and the dashboard (port 5173). Ctrl+C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
.venv/bin/uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port 8000 --reload &
API=$!
trap 'kill $API 2>/dev/null' EXIT
cd "$ROOT/frontend"
npx vite --host "${HOST:-127.0.0.1}" --port 5173
