#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chen034/workspace/stwm
cd "$ROOT"
source /home/chen034/miniconda3/etc/profile.d/conda.sh
conda activate stwm
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
python code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v6_20260414.py "$@"
