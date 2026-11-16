#!/usr/bin/env bash
# Backend unit/API tests + frontend type-check and production build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$ROOT/backend" && .venv/bin/python -m pytest -p no:warnings "$@")
(cd "$ROOT/frontend" && npm run build)
echo "All checks passed."
