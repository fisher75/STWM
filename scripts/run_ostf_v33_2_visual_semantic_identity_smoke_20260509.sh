#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=/home/chen034/miniconda3/envs/stwm/bin/python
export PYTHONPATH=code
: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES

$PY code/stwm/tools/preflight_ostf_v33_visual_teacher_semantic_prototypes_20260509.py
$PY code/stwm/tools/audit_ostf_v33_2_artifact_truth_and_claims_20260509.py
$PY code/stwm/tools/build_ostf_v33_2_pointodyssey_visual_teacher_prototypes_20260509.py --m-points 128 --horizon 32 --max-samples-per-split 64 --teacher-name clip_vit_b32_local
$PY code/stwm/tools/build_ostf_v33_2_hard_identity_semantic_subsets_20260509.py --teacher-name clip_vit_b32_local --max-items 128
$PY code/stwm/tools/train_ostf_v33_2_visual_semantic_identity_20260509.py \
  --experiment-name v33_2_visual_semantic_identity_m128_h32_seed42_smoke \
  --m-points 128 \
  --horizon 32 \
  --seed 42 \
  --steps 1000 \
  --eval-interval 500 \
  --batch-size 16 \
  --max-train-items 64 \
  --max-eval-items 64
$PY code/stwm/tools/eval_ostf_v33_2_visual_semantic_identity_20260509.py
$PY code/stwm/tools/render_ostf_v33_2_semantic_identity_trace_field_20260509.py
$PY code/stwm/tools/write_ostf_v33_2_visual_semantic_identity_decision_20260509.py
