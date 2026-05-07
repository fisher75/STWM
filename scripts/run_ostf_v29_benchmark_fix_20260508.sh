#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"

cd "$ROOT"

echo "[V29] preflight from V28"
"$PY" code/stwm/tools/write_ostf_v29_preflight_from_v28_20260508.py

echo "[V29] prior dominance diagnosis"
"$PY" code/stwm/tools/diagnose_ostf_v29_prior_dominance_20260508.py

echo "[V29] anti-prior hardbench manifest"
"$PY" code/stwm/tools/build_ostf_v29_antiprior_hardbench_20260508.py

echo "[V29] existing priors/models eval"
"$PY" code/stwm/tools/eval_ostf_v29_antiprior_existing_models_20260508.py

echo "[V29] benchmark decision"
"$PY" code/stwm/tools/write_ostf_v29_benchmark_decision_20260508.py

echo "[V29] complete"
