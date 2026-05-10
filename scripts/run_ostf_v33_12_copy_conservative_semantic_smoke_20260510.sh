#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}
export PYTHONPATH=code
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}

"$PY" code/stwm/tools/train_ostf_v33_12_copy_conservative_semantic_20260510.py "$@"
"$PY" code/stwm/tools/eval_ostf_v33_12_copy_conservative_semantic_20260510.py
