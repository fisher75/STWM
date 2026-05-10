#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code:${PYTHONPATH:-}

$PY code/stwm/tools/audit_ostf_v33_9_v33_8_training_truth_20260510.py
$PY code/stwm/tools/audit_ostf_v33_9_semantic_gate_forensics_20260510.py

$PY code/stwm/tools/train_ostf_v33_9_fresh_expanded_identity_semantic_20260510.py --candidate v33_9_v33_6_global_contrastive_fresh_seed42 "$@"
$PY code/stwm/tools/train_ostf_v33_9_fresh_expanded_identity_semantic_20260510.py --candidate v33_9_v33_7_no_fused_logits_fresh_seed42 "$@"
$PY code/stwm/tools/train_ostf_v33_9_fresh_expanded_identity_semantic_20260510.py --candidate v33_9_v33_7_full_identity_belief_fresh_seed42 "$@"
$PY code/stwm/tools/train_ostf_v33_9_fresh_expanded_identity_semantic_20260510.py --aggregate-only

$PY code/stwm/tools/eval_ostf_v33_9_fresh_expanded_identity_semantic_20260510.py
$PY code/stwm/tools/render_ostf_v33_9_world_model_diagnostics_20260510.py
$PY code/stwm/tools/write_ostf_v33_9_decision_20260510.py
