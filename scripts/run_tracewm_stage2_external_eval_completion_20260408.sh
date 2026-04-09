#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage2_external_eval_completion_${DATE_TAG}.log"
PROTOCOL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE2_EXTERNAL_EVAL_COMPLETION_PROTOCOL_${DATE_TAG}.md"
CORE_MAINLINE_FINAL_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_final_${DATE_TAG}.json"
CORE_MAINLINE_RAW_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_raw_${DATE_TAG}.json"
STAGE2_CONTRACT_JSON="$WORK_ROOT/reports/stage2_bootstrap_data_contract_${DATE_TAG}.json"

PRIMARY_CKPT="$WORK_ROOT/outputs/checkpoints/stage2_core_mainline_train_${DATE_TAG}/best.pt"
SECONDARY_CKPT="$WORK_ROOT/outputs/checkpoints/stage2_core_mainline_train_${DATE_TAG}/latest.pt"

COMPLETION_JSON="$WORK_ROOT/reports/stage2_external_eval_completion_${DATE_TAG}.json"
RESULTS_MD="$WORK_ROOT/docs/STAGE2_EXTERNAL_EVAL_COMPLETION_RESULTS_${DATE_TAG}.md"
TAPNET_PYTHON="/home/chen034/workspace/data/.venv_tapvid3d_repair_20260406_py310/bin/python"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage2-external-eval-completion] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-external-eval-completion] python=$PYTHON_BIN"
echo "[stage2-external-eval-completion] tapnet_python=$TAPNET_PYTHON"

for f in \
  "$PROTOCOL_DOC" \
  "$CORE_MAINLINE_FINAL_JSON" \
  "$CORE_MAINLINE_RAW_JSON" \
  "$STAGE2_CONTRACT_JSON" \
  "$PRIMARY_CKPT"
do
  if [[ ! -f "$f" ]]; then
    echo "[stage2-external-eval-completion] missing_required_file=$f"
    exit 2
  fi
done

echo "[stage2-external-eval-completion] step=validate_frozen_mainline_facts"
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

print("[stage2-external-eval-completion] current_mainline_semantic_source=crop_visual_encoder")
print("[stage2-external-eval-completion] frozen_boundary_kept_correct=True")
print("[stage2-external-eval-completion] next_step_choice=freeze_stage2_core_mainline")
print("[stage2-external-eval-completion] datasets_bound_for_eval=vspw,vipseg")
PY

echo "[stage2-external-eval-completion] step=run_completion_round"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2_stage2/tools/stage2_external_eval_bridge.py" \
  --core-mainline-final-json "$CORE_MAINLINE_FINAL_JSON" \
  --core-mainline-raw-json "$CORE_MAINLINE_RAW_JSON" \
  --stage2-contract-path "$STAGE2_CONTRACT_JSON" \
  --checkpoint-under-test "$PRIMARY_CKPT" \
  --secondary-checkpoint "$SECONDARY_CKPT" \
  --completion-json "$COMPLETION_JSON" \
  --results-md "$RESULTS_MD" \
  --tap-style-proxy-payload-npz "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_proxy_payload_${DATE_TAG}.npz" \
  --tap-style-secondary-proxy-payload-npz "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_proxy_payload_latest_${DATE_TAG}.npz" \
  --tap-style-official-payload-npz "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_official_payload_${DATE_TAG}.npz" \
  --tap-style-secondary-official-payload-npz "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_official_payload_latest_${DATE_TAG}.npz" \
  --tap-style-export-report-json "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_export_${DATE_TAG}.json" \
  --tap-style-secondary-export-report-json "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_export_latest_${DATE_TAG}.json" \
  --tap-style-official-eval-json "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_eval_${DATE_TAG}.json" \
  --tap-style-secondary-official-eval-json "$WORK_ROOT/reports/stage2_external_eval_completion_tap_style_eval_latest_${DATE_TAG}.json" \
  --tapnet-python "$TAPNET_PYTHON" \
  --tap3d-dataset-root "/home/chen034/workspace/data/tapvid3d/minival_dataset" \
  --max-eval-batches 8

echo "[stage2-external-eval-completion] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-external-eval-completion] completion_json=$COMPLETION_JSON"
echo "[stage2-external-eval-completion] results_md=$RESULTS_MD"
