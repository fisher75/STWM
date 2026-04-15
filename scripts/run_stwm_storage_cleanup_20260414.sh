#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chen034/workspace/stwm
LOG=/home/chen034/workspace/stwm/logs/stwm_storage_cleanup_20260414.log
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
source /home/chen034/miniconda3/etc/profile.d/conda.sh
conda activate stwm
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
python code/stwm/tools/audit_stwm_storage_20260414.py >> "$LOG" 2>&1
