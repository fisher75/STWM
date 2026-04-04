#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${STWM_PROTOCOL_V2_QUEUE_ROOT:-$REPO_ROOT/outputs/queue/stwm_protocol_v2_frontend_default_v1}"
QUEUE_DIR="$QUEUE_ROOT/d1_train"

TRAIN_SCRIPT="$REPO_ROOT/code/stwm/trainers/train_stwm_v4_2_real.py"
TRAIN_MANIFEST="${STWM_D1_TRAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/train_v2.json}"
PROTOCOL_MAIN_MANIFEST="${STWM_D1_PROTOCOL_MAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_main_v1.json}"
PROTOCOL_EVENTFUL_MANIFEST="${STWM_D1_PROTOCOL_EVENTFUL_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_eventful_v1.json}"
MODEL_PRESET="${STWM_D1_MODEL_PRESET:-prototype_220m_v4_2}"
PRESET_FILE="${STWM_D1_PRESET_FILE:-$REPO_ROOT/code/stwm/configs/model_presets_v4_2.json}"
DATA_ROOT="${STWM_D1_DATA_ROOT:-$REPO_ROOT/data/external}"

DATA_MODE="${STWM_D1_DATA_MODE:-frontend_cache}"
FRONTEND_CACHE_DIR="${STWM_D1_FRONTEND_CACHE_DIR:-$REPO_ROOT/data/cache/frontend_cache_protocol_v2_full_v1}"
FRONTEND_CACHE_INDEX="${STWM_D1_FRONTEND_CACHE_INDEX:-$FRONTEND_CACHE_DIR/index.json}"
FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY="${STWM_D1_FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY:-8}"

SEED="${STWM_D1_SEED:-123}"
STEPS="${STWM_D1_STEPS:-2000}"
SAMPLE_LIMIT="${STWM_D1_SAMPLE_LIMIT:-0}"

LSEM_10="${STWM_D1_LSEM_10:-0.5}"
PROTOCOL_EVAL_INTERVAL="${STWM_D1_PROTOCOL_EVAL_INTERVAL:-500}"
CHECKPOINT_INTERVAL="${STWM_D1_CHECKPOINT_INTERVAL:-500}"

# Two-path residual proxy: keep alpha and enable gated branch as in seed42 best.
OBJECT_BIAS_ALPHA="${STWM_TWO_PATH_OBJECT_BIAS_ALPHA:-0.50}"
OBJECT_BIAS_GATE_THRESHOLD="${STWM_TWO_PATH_OBJECT_BIAS_GATE_THRESHOLD:-0.5}"

RUN_NAME="${STWM_TWO_PATH_RUN_NAME:-two_path_residual_seed123_challenge_v1}"
PREFERRED_GPU="${STWM_D1_PREFERRED_GPU_ALL:-}"

OUT_ROOT="${STWM_TWO_PATH_SEED123_OUT_ROOT:-$REPO_ROOT/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1}"
OUT_DIR="$OUT_ROOT/seed_${SEED}/${RUN_NAME}"

SUBMIT_JSON="${STWM_TWO_PATH_SEED123_SUBMIT_JSON:-$REPO_ROOT/reports/stwm_two_path_residual_seed123_submit_v1.json}"

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$SUBMIT_JSON")"

notes="Two-path residual minimal seed123 challenge v1 | data_mode=${DATA_MODE}"
resume_hint="Resume with same output_dir and --auto-resume; compare against seed123 baseline/alpha/gated/wo_object_bias by official rule"

cmd=(
  env "PYTHONPATH=$REPO_ROOT/code:${PYTHONPATH:-}"
  conda run --no-capture-output -n stwm
  python "$TRAIN_SCRIPT"
  --data-root "$DATA_ROOT"
  --manifest "$TRAIN_MANIFEST"
  --output-dir "$OUT_DIR"
  --run-name "$RUN_NAME"
  --seed "$SEED"
  --steps "$STEPS"
  --target-epochs 0
  --min-optimizer-steps 0
  --max-optimizer-steps 0
  --sample-limit "$SAMPLE_LIMIT"
  --model-preset "$MODEL_PRESET"
  --preset-file "$PRESET_FILE"
  --use-teacher-priors
  --save-checkpoint
  --checkpoint-dir-name checkpoints
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --milestone-interval 0
  --auto-resume
  --micro-batch-per-gpu 2
  --grad-accum 8
  --num-workers 12
  --prefetch-factor 2
  --persistent-workers
  --pin-memory
  --bf16
  --activation-checkpointing
  --lambda-traj 1.0
  --lambda-vis 0.25
  --lambda-sem "$LSEM_10"
  --lambda-reid 0.25
  --lambda-query 0.25
  --lambda-reconnect 0.1
  --gradient-audit-interval 0
  --protocol-eval-interval "$PROTOCOL_EVAL_INTERVAL"
  --protocol-eval-manifest "$PROTOCOL_MAIN_MANIFEST"
  --protocol-eval-dataset all
  --protocol-eval-max-clips 0
  --protocol-eval-seed "$SEED"
  --protocol-eval-obs-steps 8
  --protocol-eval-pred-steps 8
  --protocol-eval-run-name protocol_val_main
  --protocol-diagnostics-manifest "$PROTOCOL_EVENTFUL_MANIFEST"
  --protocol-diagnostics-dataset all
  --protocol-diagnostics-max-clips 0
  --protocol-diagnostics-run-name protocol_val_eventful
  --protocol-version v2_4_detached_frozen
  --protocol-best-checkpoint-name best_protocol_main.pt
  --protocol-best-selection-name best_protocol_main_selection.json
)

if [[ "$DATA_MODE" == "frontend_cache" ]]; then
  cmd+=(
    --data-mode frontend_cache
    --frontend-cache-dir "$FRONTEND_CACHE_DIR"
    --frontend-cache-index "$FRONTEND_CACHE_INDEX"
    --frontend-cache-max-shards-in-memory "$FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY"
  )
elif [[ "$DATA_MODE" == "raw" ]]; then
  cmd+=(--data-mode raw)
else
  echo "unsupported STWM_D1_DATA_MODE=$DATA_MODE (expected: frontend_cache|raw)" >&2
  exit 2
fi

cmd+=(
  --object-bias-alpha "$OBJECT_BIAS_ALPHA"
  --object-bias-gated
  --object-bias-gate-threshold "$OBJECT_BIAS_GATE_THRESHOLD"
)

submit_args=(
  --queue-dir "$QUEUE_DIR"
  --job-name "$RUN_NAME"
  --class-type B
  --workdir "$REPO_ROOT"
  --notes "$notes"
  --resume-hint "$resume_hint"
)
if [[ -n "$PREFERRED_GPU" ]]; then
  submit_args+=(--preferred-gpu "$PREFERRED_GPU")
fi

submit_output="$(bash "$REPO_ROOT/scripts/protocol_v2_queue_submit.sh" "${submit_args[@]}" -- "${cmd[@]}")"
echo "$submit_output"

job_id="$(echo "$submit_output" | sed -n 's/^  job_id:[[:space:]]*//p' | tail -n 1)"
status_file="$(echo "$submit_output" | sed -n 's/^  status_file:[[:space:]]*//p' | tail -n 1)"
main_log="$(echo "$submit_output" | sed -n 's/^  main_log:[[:space:]]*//p' | tail -n 1)"

python - "$SUBMIT_JSON" "$RUN_NAME" "$job_id" "$status_file" "$main_log" "$OUT_DIR" "$QUEUE_DIR" <<'PY'
from pathlib import Path
import json
import sys
import time

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
obj = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "run_name": sys.argv[2],
    "job_id": sys.argv[3],
    "status_file": sys.argv[4],
    "main_log": sys.argv[5],
    "output_dir": sys.argv[6],
    "queue_dir": sys.argv[7],
}
out.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
PY

echo "[two-path-seed123-submit] submit_json=$SUBMIT_JSON"
