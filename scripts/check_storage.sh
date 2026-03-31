#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_dir "$STWM_ROOT/manifests"
REPORT="$STWM_ROOT/manifests/storage_report_$(slug_now).txt"

{
  echo "Generated: $(timestamp)"
  echo "STWM_ROOT: $STWM_ROOT"
  echo
  echo "[filesystem]"
  df -h "$STWM_ROOT" || true
  echo
  echo "[top-level directories]"
  find "$STWM_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 \
    | xargs -0 du -sh 2>/dev/null \
    | sort -h
  echo
  echo "[data/raw]"
  find "$STWM_ROOT/data/raw" -mindepth 1 -maxdepth 1 -type d -print0 \
    | xargs -0 du -sh 2>/dev/null \
    | sort -h || true
  echo
  echo "[models/checkpoints]"
  find "$STWM_ROOT/models/checkpoints" -mindepth 1 -maxdepth 1 -type d -print0 \
    | xargs -0 du -sh 2>/dev/null \
    | sort -h || true
} > "$REPORT"

echo "Storage report written to $REPORT"
