#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${STWM_PROTOCOL_V2_QUEUE_ROOT:-$REPO_ROOT/outputs/queue/stwm_protocol_v2}"
QUEUE_DIR="$QUEUE_ROOT/d0_eval"

CHECKPOINT_DIR="${STWM_D0_CHECKPOINT_DIR:-$REPO_ROOT/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints}"
MANIFEST_JSON="${STWM_D0_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_main_v1.json}"
MODEL_PRESET="${STWM_D0_MODEL_PRESET:-prototype_220m_v4_2}"
PRESET_FILE="${STWM_D0_PRESET_FILE:-$REPO_ROOT/code/stwm/configs/model_presets_v4_2.json}"
DATA_ROOT="${STWM_D0_DATA_ROOT:-$REPO_ROOT/data/external}"
CANDIDATES="${STWM_D0_CANDIDATES:-best.pt,latest.pt}"
MAX_CLIPS="${STWM_D0_MAX_CLIPS:-8}"

JOB_NAME="${STWM_D0_JOB_NAME:-d0_protocol_best_dryrun_220m_seed42_full_v4_2}"
NOTES="D0 Class-A detached dryrun on completed 220m seed42 full_v4_2"
RESUME_HINT="If interrupted, re-submit this same D0 job; script is idempotent for latest timestamped dryrun dir."

bash "$REPO_ROOT/scripts/protocol_v2_queue_submit.sh" \
  --queue-dir "$QUEUE_DIR" \
  --job-name "$JOB_NAME" \
  --class-type A \
  --workdir "$REPO_ROOT" \
  --notes "$NOTES" \
  --resume-hint "$RESUME_HINT" \
  -- \
  bash "$REPO_ROOT/scripts/run_stwm_v4_2_protocol_best_dryrun.sh" \
    "$CHECKPOINT_DIR" \
    "$MANIFEST_JSON" \
    "$MODEL_PRESET" \
    "$PRESET_FILE" \
    "$DATA_ROOT" \
    "$CANDIDATES" \
    "$MAX_CLIPS"
