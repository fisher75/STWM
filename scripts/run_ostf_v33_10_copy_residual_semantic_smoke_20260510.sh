#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code:${PYTHONPATH:-}

$PY code/stwm/tools/audit_ostf_v33_10_semantic_gate_failure_20260510.py
$PY code/stwm/tools/build_ostf_v33_10_semantic_nontrivial_baselines_20260510.py
$PY code/stwm/tools/build_ostf_v33_10_copy_residual_semantic_targets_20260510.py

$PY code/stwm/tools/train_ostf_v33_10_copy_residual_semantic_20260510.py --write-main-summary "$@"
$PY code/stwm/tools/eval_ostf_v33_10_copy_residual_semantic_20260510.py

$PY code/stwm/tools/train_ostf_v33_10_copy_residual_semantic_20260510.py --experiment-name v33_10_no_copy_prior_control --no-copy-prior --steps 800 --batch-size 64
$PY code/stwm/tools/eval_ostf_v33_10_copy_residual_semantic_20260510.py --candidate v33_10_no_copy_prior_control --summary-path reports/stwm_ostf_v33_10_no_copy_prior_eval_summary_20260510.json --decision-path reports/stwm_ostf_v33_10_no_copy_prior_eval_decision_20260510.json --doc-path docs/STWM_OSTF_V33_10_NO_COPY_PRIOR_EVAL_20260510.md

$PY code/stwm/tools/train_ostf_v33_10_copy_residual_semantic_20260510.py --experiment-name v33_10_no_change_gate_control --no-change-gate --steps 800 --batch-size 64
$PY code/stwm/tools/eval_ostf_v33_10_copy_residual_semantic_20260510.py --candidate v33_10_no_change_gate_control --summary-path reports/stwm_ostf_v33_10_no_change_gate_eval_summary_20260510.json --decision-path reports/stwm_ostf_v33_10_no_change_gate_eval_decision_20260510.json --doc-path docs/STWM_OSTF_V33_10_NO_CHANGE_GATE_EVAL_20260510.md

$PY code/stwm/tools/write_ostf_v33_10_copy_residual_semantic_ablation_20260510.py
$PY code/stwm/tools/render_ostf_v33_10_copy_residual_semantic_diagnostics_20260510.py
$PY code/stwm/tools/write_ostf_v33_10_copy_residual_semantic_decision_20260510.py
