#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chen034/workspace/stwm
source /home/chen034/miniconda3/etc/profile.d/conda.sh
conda activate stwm
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python:eval}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
cd "$ROOT"
python code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v6_20260414.py "$@"
