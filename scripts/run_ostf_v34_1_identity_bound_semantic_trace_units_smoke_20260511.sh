#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
export PYTHONPATH=code
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/audit_ostf_v34_1_semantic_trace_unit_forensics_20260511.py
/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/eval_ostf_v34_1_unit_intervention_probe_20260511.py --batch-size 32 --num-workers 0
/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/build_ostf_v34_1_unit_identity_binding_targets_20260511.py
/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/train_ostf_v34_1_identity_bound_semantic_trace_units_20260511.py --steps 1500 --batch-size 32 --num-workers 0
/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511.py --batch-size 32 --num-workers 0
/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/render_ostf_v34_1_unit_loadbearing_visualizations_20260511.py --batch-size 32 --num-workers 0
/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/write_ostf_v34_1_decision_20260511.py
