#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

cd "${ROOT}"

"${PYTHON_BIN}" code/stwm/tools/run_stage2_mechanism_ablation_fix_20260415.py --mode all &
MECH_PID=$!

"${PYTHON_BIN}" code/stwm/tools/build_stage2_state_identifiability_protocol_20260415.py
"${PYTHON_BIN}" code/stwm/tools/run_stage2_state_identifiability_eval_20260415.py

wait "${MECH_PID}"

"${PYTHON_BIN}" code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v7_20260415.py
"${PYTHON_BIN}" code/stwm/tools/run_stage2_top_tier_closure_20260415.py --mode all
