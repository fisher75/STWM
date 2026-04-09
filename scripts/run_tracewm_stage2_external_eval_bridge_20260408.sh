#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage2_external_eval_bridge_${DATE_TAG}.log"

PROTOCOL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE2_EXTERNAL_EVAL_BRIDGE_PROTOCOL_${DATE_TAG}.md"
CORE_MAINLINE_FINAL_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_final_${DATE_TAG}.json"
CORE_MAINLINE_RAW_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_raw_${DATE_TAG}.json"
STAGE2_CONTRACT_JSON="$WORK_ROOT/reports/stage2_bootstrap_data_contract_${DATE_TAG}.json"

PRIMARY_CKPT="$WORK_ROOT/outputs/checkpoints/stage2_core_mainline_train_${DATE_TAG}/best.pt"
SECONDARY_CKPT="$WORK_ROOT/outputs/checkpoints/stage2_core_mainline_train_${DATE_TAG}/latest.pt"

BRIDGE_JSON="$WORK_ROOT/reports/stage2_external_eval_bridge_${DATE_TAG}.json"
RESULTS_MD="$WORK_ROOT/docs/STAGE2_EXTERNAL_EVAL_BRIDGE_RESULTS_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage2-external-eval-bridge] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-external-eval-bridge] python=$PYTHON_BIN"

for f in \
  "$PROTOCOL_DOC" \
  "$CORE_MAINLINE_FINAL_JSON" \
  "$CORE_MAINLINE_RAW_JSON" \
  "$STAGE2_CONTRACT_JSON" \
  "$PRIMARY_CKPT"
do
  if [[ ! -f "$f" ]]; then
    echo "[stage2-external-eval-bridge] missing_required_file=$f"
    exit 2
  fi
done

echo "[stage2-external-eval-bridge] step=validate_frozen_mainline_facts"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

payload = json.loads(Path("$CORE_MAINLINE_FINAL_JSON").read_text(encoding="utf-8"))

if str(payload.get("current_mainline_semantic_source", "")) != "crop_visual_encoder":
    raise SystemExit("current_mainline_semantic_source must be crop_visual_encoder")
if not bool(payload.get("frozen_boundary_kept_correct", False)):
    raise SystemExit("frozen_boundary_kept_correct must be true")
if str(payload.get("next_step_choice", "")) != "freeze_stage2_core_mainline":
    raise SystemExit("next_step_choice must be freeze_stage2_core_mainline")

datasets_eval = payload.get("datasets_bound_for_eval", []) if isinstance(payload.get("datasets_bound_for_eval", []), list) else []
ds = {str(x).strip().lower() for x in datasets_eval}
if ds != {"vspw", "vipseg"}:
    raise SystemExit(f"datasets_bound_for_eval must remain core-only vspw+vipseg, got={sorted(ds)}")

print("[stage2-external-eval-bridge] current_mainline_semantic_source=crop_visual_encoder")
print("[stage2-external-eval-bridge] frozen_boundary_kept_correct=True")
print("[stage2-external-eval-bridge] next_step_choice=freeze_stage2_core_mainline")
print("[stage2-external-eval-bridge] datasets_bound_for_eval=vspw,vipseg")
PY

echo "[stage2-external-eval-bridge] step=run_external_eval_bridge"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2_stage2/tools/stage2_external_eval_bridge.py" \
  --core-mainline-final-json "$CORE_MAINLINE_FINAL_JSON" \
  --core-mainline-raw-json "$CORE_MAINLINE_RAW_JSON" \
  --stage2-contract-path "$STAGE2_CONTRACT_JSON" \
  --checkpoint-under-test "$PRIMARY_CKPT" \
  --secondary-checkpoint "$SECONDARY_CKPT" \
  --bridge-json "$BRIDGE_JSON" \
  --results-md "$RESULTS_MD" \
  --tap-style-payload-npz "$WORK_ROOT/reports/stage2_external_eval_bridge_tap_style_payload_${DATE_TAG}.npz" \
  --tap-style-secondary-payload-npz "$WORK_ROOT/reports/stage2_external_eval_bridge_tap_style_payload_latest_${DATE_TAG}.npz" \
  --max-eval-batches 8

echo "[stage2-external-eval-bridge] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-external-eval-bridge] bridge_json=$BRIDGE_JSON"
echo "[stage2-external-eval-bridge] results_md=$RESULTS_MD"
