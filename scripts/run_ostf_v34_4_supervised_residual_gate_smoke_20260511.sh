#!/usr/bin/env bash
set -euo pipefail

cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}

"$PY" code/stwm/tools/train_ostf_v34_4_supervised_residual_gate_20260511.py "$@"
"$PY" code/stwm/tools/eval_ostf_v34_4_supervised_residual_gate_20260511.py
