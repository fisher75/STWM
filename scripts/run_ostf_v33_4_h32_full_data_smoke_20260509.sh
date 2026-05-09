#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} "$PY" code/stwm/tools/train_ostf_v33_3_structured_semantic_identity_20260509.py \
  --experiment-name v33_4_structured_semantic_identity_m128_h32_seed42_full_data_smoke \
  --batch-size 32 \
  --steps 1200 \
  --eval-interval 600
