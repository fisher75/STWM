#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"

"$PY" code/stwm/tools/audit_ostf_v33_8_h32_m128_reachable_coverage_20260510.py
"$PY" code/stwm/tools/build_ostf_v33_8_full_visual_teacher_prototypes_20260510.py --max-samples-per-split 0 --batch-size "${V33_8_TEACHER_BATCH:-128}"
"$PY" code/stwm/tools/build_ostf_v33_8_semantic_prototype_vocabulary_20260510.py
"$PY" code/stwm/tools/build_ostf_v33_8_semantic_prototype_targets_20260510.py
"$PY" code/stwm/tools/build_ostf_v33_8_complete_h32_m128_targets_20260510.py
"$PY" code/stwm/tools/build_ostf_v33_8_split_matched_hard_masks_20260510.py
"$PY" code/stwm/tools/train_ostf_v33_8_ablation_safe_identity_semantic_20260510.py --steps "${V33_8_STEPS:-1500}" --batch-size "${V33_8_BATCH:-32}" --skip-existing
"$PY" code/stwm/tools/eval_ostf_v33_8_ablation_safe_identity_semantic_20260510.py --batch-size "${V33_8_EVAL_BATCH:-32}"
"$PY" code/stwm/tools/render_ostf_v33_8_semantic_identity_diagnostics_20260510.py
"$PY" code/stwm/tools/write_ostf_v33_8_decision_20260510.py
