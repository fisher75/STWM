#!/usr/bin/env bash
set -euo pipefail

cd /raid/chen034/workspace/stwm
PY=${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}

"$PY" code/stwm/tools/audit_ostf_v34_3_v34_2_bottleneck_diagnosis_20260511.py
"$PY" code/stwm/tools/train_ostf_v34_3_pointwise_unit_residual_20260511.py "$@"
"$PY" code/stwm/tools/eval_ostf_v34_3_pointwise_unit_residual_20260511.py
"$PY" code/stwm/tools/render_ostf_v34_3_pointwise_unit_residual_visualizations_20260511.py
"$PY" code/stwm/tools/write_ostf_v34_3_decision_20260511.py
