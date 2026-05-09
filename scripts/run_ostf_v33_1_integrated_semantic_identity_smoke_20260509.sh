#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=/home/chen034/miniconda3/envs/stwm/bin/python
export PYTHONPATH=code
: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES

$PY code/stwm/tools/audit_ostf_v33_1_artifact_truth_20260509.py
$PY code/stwm/tools/write_ostf_v33_semantic_identity_target_construction_decision_20260509.py
$PY code/stwm/tests/test_ostf_v33_1_sidecar_dataset_contract_20260509.py
$PY code/stwm/tools/train_ostf_v33_1_integrated_semantic_identity_20260509.py \
  --experiment-name v33_1_integrated_m128_h32_seed42_smoke \
  --m-points 128 \
  --horizon 32 \
  --seed 42 \
  --steps 700 \
  --eval-interval 250 \
  --batch-size 32 \
  --max-train-items 512 \
  --max-eval-items 256
$PY code/stwm/tools/eval_ostf_v33_1_integrated_semantic_identity_20260509.py
$PY code/stwm/tools/render_ostf_v33_1_identity_field_visualizations_20260509.py
$PY code/stwm/tools/write_ostf_v33_1_integrated_decision_20260509.py
