#!/usr/bin/env bash
# Archive runtime data that git does NOT track (SQLite DB + uploaded/generated images) for transfer.
# Restore on the other machine with:  tar -xzf <archive> -C <repo root>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p experiments/bundles
OUT="experiments/bundles/pe-data-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
PATHS=()
[ -d backend/data ] && PATHS+=(backend/data)
PATHS+=(experiments/originals experiments/generated)
tar -czf "$OUT" "${PATHS[@]}"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
