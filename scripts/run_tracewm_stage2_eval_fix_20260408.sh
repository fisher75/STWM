#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage2_eval_fix_${DATE_TAG}.log"

EVAL_FIX_PROTOCOL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE2_EVAL_FIX_PROTOCOL_${DATE_TAG}.md"
HARDENING_COMPARISON_JSON="$WORK_ROOT/reports/stage2_semantic_hardening_comparison_${DATE_TAG}.json"
HARDENING_CORE_JSON="$WORK_ROOT/reports/stage2_semantic_hardening_core_${DATE_TAG}.json"
HARDENING_BURST_JSON="$WORK_ROOT/reports/stage2_semantic_hardening_core_plus_burst_${DATE_TAG}.json"

EVAL_FIX_CORE_JSON="$WORK_ROOT/reports/stage2_eval_fix_core_${DATE_TAG}.json"
EVAL_FIX_BURST_JSON="$WORK_ROOT/reports/stage2_eval_fix_core_plus_burst_${DATE_TAG}.json"
EVAL_FIX_COMPARISON_JSON="$WORK_ROOT/reports/stage2_eval_fix_comparison_${DATE_TAG}.json"
EVAL_FIX_RESULTS_MD="$WORK_ROOT/docs/STAGE2_EVAL_FIX_RESULTS_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage2-eval-fix] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-eval-fix] python=$PYTHON_BIN"

for f in \
  "$EVAL_FIX_PROTOCOL_DOC" \
  "$HARDENING_COMPARISON_JSON" \
  "$HARDENING_CORE_JSON" \
  "$HARDENING_BURST_JSON"
do
  if [[ ! -f "$f" ]]; then
    echo "[stage2-eval-fix] missing_required_file=$f"
    exit 2
  fi
done

echo "[stage2-eval-fix] step=validate_hardening_facts"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

cmp_payload = json.loads(Path("$HARDENING_COMPARISON_JSON").read_text(encoding="utf-8"))

mainline = str(cmp_payload.get("current_mainline_semantic_source", ""))
if mainline != "crop_visual_encoder":
    raise SystemExit(f"unexpected current_mainline_semantic_source: {mainline}")

if not bool(cmp_payload.get("frozen_boundary_kept_correct", False)):
    raise SystemExit("frozen_boundary_kept_correct must be true")

if bool(cmp_payload.get("core_plus_burst_better_than_core_only", True)):
    raise SystemExit("hardening fact violated: core+burst should not be better than core-only")

if str(cmp_payload.get("next_step_choice", "")) != "do_one_more_stage2_eval_fix":
    raise SystemExit("hardening next_step_choice must remain do_one_more_stage2_eval_fix")

print("[stage2-eval-fix] hardening_mainline_semantic_source=crop_visual_encoder")
print("[stage2-eval-fix] hardening_frozen_boundary_kept_correct=True")
print("[stage2-eval-fix] hardening_conclusion_core_only_better=True")
print("[stage2-eval-fix] hardening_next_step_choice=do_one_more_stage2_eval_fix")
PY

echo "[stage2-eval-fix] step=normalize_core_summary"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2_stage2/tools/stage2_eval_protocol.py" \
  normalize-run-summary \
  --input-run-json "$HARDENING_CORE_JSON" \
  --output-run-json "$EVAL_FIX_CORE_JSON" \
  --run-label stage2_smalltrain_cropenc_core

echo "[stage2-eval-fix] step=normalize_core_plus_burst_summary"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2_stage2/tools/stage2_eval_protocol.py" \
  normalize-run-summary \
  --input-run-json "$HARDENING_BURST_JSON" \
  --output-run-json "$EVAL_FIX_BURST_JSON" \
  --run-label stage2_smalltrain_cropenc_core_plus_burst

echo "[stage2-eval-fix] step=build_eval_fix_comparison"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2_stage2/tools/stage2_eval_protocol.py" \
  compare-runs \
  --core-run-json "$EVAL_FIX_CORE_JSON" \
  --core-plus-burst-run-json "$EVAL_FIX_BURST_JSON" \
  --comparison-json "$EVAL_FIX_COMPARISON_JSON" \
  --results-md "$EVAL_FIX_RESULTS_MD" \
  --round-name stage2_eval_fix_20260408 \
  --results-title "Stage2 Eval-Fix Results"

echo "[stage2-eval-fix] step=validate_decision_fields"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

payload = json.loads(Path("$EVAL_FIX_COMPARISON_JSON").read_text(encoding="utf-8"))

allowed_mainline = {
    "stage2_core_cropenc",
    "stage2_core_plus_burst_cropenc",
    "invalid_comparison",
}
allowed_next = {
    "continue_stage2_training_core_only",
    "do_one_targeted_burst_fix",
    "fix_comparison_first",
}

mainline = str(payload.get("final_recommended_mainline", ""))
if mainline not in allowed_mainline:
    raise SystemExit(f"unexpected final_recommended_mainline: {mainline}")

next_choice = str(payload.get("next_step_choice", ""))
if next_choice not in allowed_next:
    raise SystemExit(f"unexpected next_step_choice: {next_choice}")

if mainline == "invalid_comparison" and bool(payload.get("can_continue_stage2_training", False)):
    raise SystemExit("invalid comparison cannot continue stage2 training")

print(f"[stage2-eval-fix] final_recommended_mainline={mainline}")
print(f"[stage2-eval-fix] can_continue_stage2_training={bool(payload.get('can_continue_stage2_training', False))}")
print(f"[stage2-eval-fix] next_step_choice={next_choice}")
PY

echo "[stage2-eval-fix] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-eval-fix] core_summary_json=$EVAL_FIX_CORE_JSON"
echo "[stage2-eval-fix] core_plus_burst_summary_json=$EVAL_FIX_BURST_JSON"
echo "[stage2-eval-fix] comparison_json=$EVAL_FIX_COMPARISON_JSON"
echo "[stage2-eval-fix] results_md=$EVAL_FIX_RESULTS_MD"
