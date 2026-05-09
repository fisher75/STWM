#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code

$PY code/stwm/tools/audit_ostf_v33_3_artifact_truth_20260509.py
$PY code/stwm/tools/write_ostf_v33_3_claim_boundary_20260509.py
$PY code/stwm/tools/build_ostf_v33_3_semantic_prototype_vocabulary_20260509.py
$PY code/stwm/tools/build_ostf_v33_3_semantic_prototype_targets_20260509.py
$PY code/stwm/tools/build_ostf_v33_3_balanced_hard_identity_semantic_subset_20260509.py
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY code/stwm/tools/train_ostf_v33_3_structured_semantic_identity_20260509.py "$@"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY code/stwm/tools/eval_ostf_v33_3_structured_semantic_identity_20260509.py
$PY code/stwm/tools/render_ostf_v33_3_structured_semantic_trace_field_20260509.py
$PY code/stwm/tools/write_ostf_v33_3_structured_semantic_identity_decision_20260509.py
