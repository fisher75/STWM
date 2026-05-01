#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
mkdir -p reports docs outputs/logs/stwm_fstf_scaling_v11_20260502
"${PY}" code/stwm/tools/materialize_stwm_fstf_scaling_cache_v11_20260502.py
"${PY}" code/stwm/tools/aggregate_stwm_fstf_scaling_v11_20260502.py
echo "[v11-horizon] H16/H24 jobs blocked until matching future feature/prototype target caches are materialized; no fake horizon eval launched."
