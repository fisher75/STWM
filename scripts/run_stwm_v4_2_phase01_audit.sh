#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
AUDIT_ROOT="${STWM_V4_2_PHASE01_AUDIT_ROOT:-$STWM_ROOT/outputs/audits/stwm_v4_2_phase01_${STAMP}}"
TRAIN_MANIFEST="${STWM_V4_2_REAL_MANIFEST:-$STWM_ROOT/manifests/realsplits/stwm_v4_2_vspw_vipseg_train_v1.json}"
WARMUP_STEPS="${STWM_V4_2_PHASE01_WARMUP_STEPS:-120}"
CHECKPOINT_INTERVAL="${STWM_V4_2_PHASE01_CHECKPOINT_INTERVAL:-50}"

LANE0_GPUS="${STWM_V4_2_PHASE01_LANE0_GPUS:-0,1,3,7}"
LANE1_GPUS="${STWM_V4_2_PHASE01_LANE1_GPUS:-2,4,5,6}"

POLL_SECONDS="${STWM_V4_2_PHASE01_POLL_SECONDS:-10}"
MAX_MEM_USED_MIB="${STWM_V4_2_PHASE01_MAX_MEM_USED_MIB:-90000}"
MAX_UTIL_PERCENT="${STWM_V4_2_PHASE01_MAX_UTIL_PERCENT:-98}"
TIMEOUT_SECONDS="${STWM_V4_2_PHASE01_TIMEOUT_SECONDS:-600}"
MIN_DISK_FREE_GB="${STWM_V4_2_PHASE01_MIN_DISK_FREE_GB:-50}"

SEED="${STWM_V4_2_PHASE01_SEED:-42}"

LANE0_RUN="full_v4_2_1b"
LANE1_RUN="full_v4_2"

LANE0_DIR="$AUDIT_ROOT/lane0"
LANE1_DIR="$AUDIT_ROOT/lane1"
LANE0_TRAIN_ROOT="$AUDIT_ROOT/training_lane0"
LANE1_TRAIN_ROOT="$AUDIT_ROOT/training_lane1"

AUDIT_SUMMARY_JSON="$AUDIT_ROOT/phase1_audit_summary.json"
MANIFEST_REPORT_JSON="$AUDIT_ROOT/real_train_manifest_report.json"
BUDGET_220M_JSON="$AUDIT_ROOT/budget_220m.json"
BUDGET_1B_JSON="$AUDIT_ROOT/budget_1b.json"

mkdir -p "$AUDIT_ROOT" "$LANE0_DIR" "$LANE1_DIR"

echo "[phase0/1] build/refresh real train manifest"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
  python "$STWM_ROOT/code/stwm/tools/build_stwm_v4_2_real_train_manifest.py" \
    --data-root "$STWM_ROOT/data/external" \
    --output-manifest "$TRAIN_MANIFEST" \
    --output-report "$MANIFEST_REPORT_JSON" \
    --max-frames 64 >/dev/null

echo "[phase0/1] compute fixed budgets (effective_batch=16 for both 220M and 1B)"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
  python "$STWM_ROOT/code/stwm/tools/compute_stwm_v4_2_real_budget.py" \
    --manifest "$TRAIN_MANIFEST" \
    --effective-batch 16 \
    --target-epochs 3 \
    --min-optimizer-steps 5000 \
    --max-optimizer-steps 8000 \
    --output-json "$BUDGET_220M_JSON" >/dev/null
cp -f "$BUDGET_220M_JSON" "$BUDGET_1B_JSON"

run_lane_bg() {
  local lane_name="$1"
  local scale="$2"
  local run_name="$3"
  local lane_dir="$4"
  local train_root="$5"
  local candidate_gpus="$6"

  echo "[phase0/1] start ${lane_name}: scale=${scale} run=${run_name} seed=${SEED}"

  bash "$SCRIPT_DIR/gpu_auto_claim_run.sh" \
    --prefer-gpus 1 \
    --min-gpus 1 \
    --poll-seconds "$POLL_SECONDS" \
    --max-mem-used-mib "$MAX_MEM_USED_MIB" \
    --max-utilization "$MAX_UTIL_PERCENT" \
    --candidate-gpus "$candidate_gpus" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    -- bash "$SCRIPT_DIR/run_stwm_v4_2_phase01_lane_job.sh" \
      --scale "$scale" \
      --seed "$SEED" \
      --run-name "$run_name" \
      --lane-dir "$lane_dir" \
      --train-root "$train_root" \
      --warmup-steps "$WARMUP_STEPS" \
      --checkpoint-interval "$CHECKPOINT_INTERVAL"
}

set +e
run_lane_bg "lane0" "1b" "$LANE0_RUN" "$LANE0_DIR" "$LANE0_TRAIN_ROOT" "$LANE0_GPUS" &
pid0=$!
run_lane_bg "lane1" "220m" "$LANE1_RUN" "$LANE1_DIR" "$LANE1_TRAIN_ROOT" "$LANE1_GPUS" &
pid1=$!

wait "$pid0"
rc0=$?
wait "$pid1"
rc1=$?
set -e

if (( rc0 != 0 || rc1 != 0 )); then
  echo "[phase0/1] warmup lane failed: lane0_rc=${rc0} lane1_rc=${rc1}" >&2
  exit 3
fi

echo "[phase0/1] warmup complete; verify resume by extending +2 optimizer steps"
resume_check_lane() {
  local lane_dir="$1"
  local scale="$2"
  local run_name="$3"
  local train_root="$4"
  local candidate_gpus="$5"

  local resume_target_steps
  resume_target_steps=$(( WARMUP_STEPS + 2 ))

  bash "$SCRIPT_DIR/gpu_auto_claim_run.sh" \
    --prefer-gpus 1 \
    --min-gpus 1 \
    --poll-seconds "$POLL_SECONDS" \
    --max-mem-used-mib "$MAX_MEM_USED_MIB" \
    --max-utilization "$MAX_UTIL_PERCENT" \
    --candidate-gpus "$candidate_gpus" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    -- bash -lc "
      set -euo pipefail
      STWM_V4_2_REAL_FORCE_STEPS=${resume_target_steps} \
      STWM_V4_2_REAL_RUNS=${run_name} \
      STWM_V4_2_CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL} \
      STWM_V4_2_MILESTONE_INTERVAL=0 \
        bash ${SCRIPT_DIR}/run_stwm_v4_2_real_train_seed.sh --scale ${scale} --seed ${SEED} ${train_root}
    "

  local run_dir="$train_root/seed_${SEED}/${run_name}"
  cp -f "$run_dir/train_log.jsonl" "$lane_dir/train_log.jsonl" 2>/dev/null || true
  cp -f "$run_dir/mini_val_summary.json" "$lane_dir/mini_val_summary.json" 2>/dev/null || true
  mkdir -p "$lane_dir/checkpoints"
  cp -f "$run_dir/checkpoints/latest.pt" "$lane_dir/checkpoints/latest.pt" 2>/dev/null || true
  cp -f "$run_dir/checkpoints/best.pt" "$lane_dir/checkpoints/best.pt" 2>/dev/null || true

  python - "$run_dir/mini_val_summary.json" "$WARMUP_STEPS" > "$lane_dir/resume_check.json" <<'PY'
import json
import sys

summary_path = sys.argv[1]
warmup_steps = int(sys.argv[2])
summary = json.loads(open(summary_path, "r", encoding="utf-8").read())
start_step = int(summary.get("resume", {}).get("start_step", 0))
resolved_steps = int(summary.get("budget", {}).get("resolved_optimizer_steps", summary.get("steps", 0)))
out = {
    "summary_path": summary_path,
    "start_step": start_step,
    "resolved_steps": resolved_steps,
    "warmup_steps": warmup_steps,
    "resume_verified": bool(start_step >= warmup_steps and resolved_steps >= warmup_steps + 2),
}
print(json.dumps(out, indent=2))
PY
}

resume_check_lane "$LANE0_DIR" "1b" "$LANE0_RUN" "$LANE0_TRAIN_ROOT" "$LANE0_GPUS"
resume_check_lane "$LANE1_DIR" "220m" "$LANE1_RUN" "$LANE1_TRAIN_ROOT" "$LANE1_GPUS"

echo "[phase0/1] summarize IO/throughput audit"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
  python "$STWM_ROOT/code/stwm/tools/summarize_stwm_v4_2_io_audit.py" \
    --audit-root "$AUDIT_ROOT" \
    --min-disk-free-gb "$MIN_DISK_FREE_GB" \
    --output-json "$AUDIT_SUMMARY_JSON" >/dev/null

echo "[phase0/1] done"
echo "  audit_root:      $AUDIT_ROOT"
echo "  train_manifest:  $TRAIN_MANIFEST"
echo "  manifest_report: $MANIFEST_REPORT_JSON"
echo "  budget_220m:     $BUDGET_220M_JSON"
echo "  budget_1b:       $BUDGET_1B_JSON"
echo "  phase1_summary:  $AUDIT_SUMMARY_JSON"