#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code

${PY} code/stwm/tools/audit_ostf_v33_7_identity_training_forensics_20260509.py
${PY} code/stwm/tools/build_ostf_v33_7_h32_m128_complete_targets_20260509.py
${PY} code/stwm/tools/build_ostf_v33_7_hard_identity_train_masks_20260509.py

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} ${PY} code/stwm/tools/train_ostf_v33_7_identity_belief_calibration_20260509.py \
  --experiment-name v33_7_identity_belief_m128_h32_seed42 \
  --steps "${V33_7_STEPS:-1800}" \
  --batch-size "${V33_7_BATCH:-32}" \
  --num-workers "${V33_7_WORKERS:-2}" \
  --write-main-summary

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} ${PY} code/stwm/tools/eval_ostf_v33_7_identity_belief_calibration_20260509.py
${PY} code/stwm/tools/write_ostf_v33_7_identity_belief_decision_20260509.py
