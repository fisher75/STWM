#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"

"${PYTHON_BIN}" code/stwm/tools/audit_stwm_fstf_visualization_artifacts_v13_20260502.py "$@"
