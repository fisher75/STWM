#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
mkdir -p logs
SESSION=stwm_ostf_v33_7_identity_belief_calibration_smoke_20260509
LOG=logs/stwm_ostf_v33_7_identity_belief_calibration_smoke_20260509.log
tmux new-session -d -s "${SESSION}" "bash scripts/run_ostf_v33_7_identity_belief_calibration_smoke_20260509.sh 2>&1 | tee ${LOG}"
echo "${SESSION}"
echo "${LOG}"
