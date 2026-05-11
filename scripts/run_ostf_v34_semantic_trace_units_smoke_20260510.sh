#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
"$PY" code/stwm/tools/audit_ostf_v34_no_drift_route_20260510.py
"$PY" code/stwm/tools/build_ostf_v34_semantic_measurement_bank_20260510.py
"$PY" code/stwm/tools/train_ostf_v34_semantic_trace_units_20260510.py "$@"
"$PY" code/stwm/tools/eval_ostf_v34_semantic_trace_units_20260510.py
"$PY" code/stwm/tools/compare_ostf_v34_against_teacher_prototype_baselines_20260510.py
"$PY" code/stwm/tools/render_ostf_v34_semantic_trace_unit_diagnostics_20260510.py
"$PY" code/stwm/tools/write_ostf_v34_semantic_trace_unit_decision_20260510.py
