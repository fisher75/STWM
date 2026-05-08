#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"

cd "$ROOT"

echo "[V30] refresh V29 preflight spelling"
"$PY" code/stwm/tools/write_ostf_v29_preflight_from_v28_20260508.py

echo "[V30] refresh V29 decision logic"
"$PY" code/stwm/tools/write_ostf_v29_benchmark_decision_20260508.py

echo "[V30] V29 bugfix audit"
"$PY" code/stwm/tools/write_ostf_v30_v29_bugfix_audit_20260508.py

echo "[V30] external GT data-root audit"
"$PY" code/stwm/tools/audit_ostf_v30_external_gt_data_roots_20260508.py

echo "[V30] PointOdyssey external GT cache"
"$PY" code/stwm/tools/build_ostf_v30_pointodyssey_gt_cache_20260508.py

echo "[V30] TAP-Vid external GT cache"
"$PY" code/stwm/tools/build_ostf_v30_tapvid_gt_cache_20260508.py

echo "[V30] TAPVid-3D external GT cache"
"$PY" code/stwm/tools/build_ostf_v30_tapvid3d_gt_cache_20260508.py

echo "[V30] external GT anti-prior benchmark"
"$PY" code/stwm/tools/build_ostf_v30_external_gt_antiprior_benchmark_20260508.py

echo "[V30] external GT priors/existing eval"
"$PY" code/stwm/tools/eval_ostf_v30_external_gt_priors_and_existing_20260508.py

echo "[V30] external GT decision"
"$PY" code/stwm/tools/write_ostf_v30_external_gt_decision_20260508.py

echo "[V30] complete"
