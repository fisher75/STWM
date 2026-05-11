#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
export PYTHONPATH=code
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/audit_ostf_v34_2_v34_1_failure_attribution_20260511.py
CUDA_VISIBLE_DEVICES="${V34_2_DUAL_GPU:-0}" /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/train_ostf_v34_2_dual_source_semantic_trace_units_20260511.py --steps 1500 --batch-size 32 --num-workers 0
CUDA_VISIBLE_DEVICES="${V34_2_POINTWISE_GPU:-1}" /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/train_ostf_v34_2_pointwise_no_unit_baseline_20260511.py --steps 1500 --batch-size 32 --num-workers 0
CUDA_VISIBLE_DEVICES="${V34_2_POINTWISE_GPU:-1}" /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/eval_ostf_v34_2_pointwise_no_unit_baseline_20260511.py --batch-size 32 --num-workers 0
CUDA_VISIBLE_DEVICES="${V34_2_DUAL_GPU:-0}" /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/eval_ostf_v34_2_dual_source_semantic_trace_units_20260511.py --batch-size 32 --num-workers 0
CUDA_VISIBLE_DEVICES="${V34_2_DUAL_GPU:-0}" /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/render_ostf_v34_2_dual_source_unit_visualizations_20260511.py --batch-size 32 --num-workers 0
/home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/write_ostf_v34_2_decision_20260511.py
