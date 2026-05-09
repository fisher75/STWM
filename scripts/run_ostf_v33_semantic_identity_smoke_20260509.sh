#!/usr/bin/env bash
set -euo pipefail
ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" code/stwm/tools/train_ostf_v33_semantic_identity_head_20260509.py --m-points 128 --horizon 32 --seed 42 --steps 200 --batch-size 32
