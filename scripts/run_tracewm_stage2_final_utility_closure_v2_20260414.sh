#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chen034/workspace/stwm
LOG=/home/chen034/workspace/stwm/logs/stage2_final_utility_closure_v2_20260414.log
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
source /home/chen034/miniconda3/etc/profile.d/conda.sh
conda activate stwm
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
python code/stwm/tools/run_tracewm_stage2_mechanism_ablation_closure_v2_20260414.py --mode all >> "$LOG" 2>&1
