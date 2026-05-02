#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
export PYTHONPATH=/raid/chen034/workspace/stwm/code
export STWM_PROC_TITLE=python
export STWM_PROC_TITLE_MODE=generic
export PYTHONUNBUFFERED=1

LOG_DIR=outputs/logs/stwm_fstf_scaling_v12_20260502
mkdir -p "$LOG_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
/home/chen034/miniconda3/envs/stwm/bin/python \
  code/stwm/tools/build_stwm_fstf_rawframe_rollout_visualization_v12_20260502.py \
  --device cuda \
  --output reports/stwm_fstf_visualization_v12_20260502.json \
  --doc docs/STWM_FSTF_VISUALIZATION_V12_20260502.md \
  > "$LOG_DIR/visualization_v12.log" 2>&1
