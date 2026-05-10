#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}

"$PY" code/stwm/tools/audit_ostf_v33_11_v33_10_forensics_20260510.py
"$PY" code/stwm/tools/build_ostf_v33_11_true_semantic_hard_protocol_20260510.py
"$PY" code/stwm/tools/build_ostf_v33_11_semantic_baseline_bank_20260510.py
"$PY" code/stwm/tools/build_ostf_v33_11_copy_residual_semantic_targets_20260510.py
"$PY" code/stwm/tools/eval_ostf_v33_11_oracle_gate_upper_bound_20260510.py --batch-size 32 --num-workers 0
"$PY" code/stwm/tools/train_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510.py --write-main-summary --steps 1500 --batch-size 32 --num-workers 0
"$PY" code/stwm/tools/eval_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510.py --batch-size 32 --num-workers 0
"$PY" code/stwm/tools/render_ostf_v33_11_semantic_identity_diagnostics_20260510.py --batch-size 32 --num-workers 0
"$PY" code/stwm/tools/write_ostf_v33_11_identity_preserving_copy_residual_ablation_20260510.py
"$PY" code/stwm/tools/write_ostf_v33_11_decision_20260510.py
