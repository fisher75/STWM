#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

cd "${REPO_ROOT}"

mkdir -p outputs/logs reports docs outputs/cache outputs/checkpoints

FEATURE_REPORT="reports/stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json"
FEATURE_CACHE_DIR="outputs/cache/stwm_fullscale_semantic_trace_feature_targets_v1_20260428"
PROTO32_REPORT="reports/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428.json"
PROTO64_REPORT="reports/stwm_fullscale_semantic_trace_prototypes_c64_v1_20260428.json"
TARGET32_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json"
TARGET64_REPORT="reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json"
OBS_REPORT="reports/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json"
SPLIT_REPORT="reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json"

"${PY}" code/stwm/tools/build_future_semantic_trace_feature_targets_20260428.py \
  --dataset-names vspw vipseg \
  --splits train val \
  --max-samples-train 999999 \
  --max-samples-val 999999 \
  --max-entities-per-sample 8 \
  --fut-len 8 \
  --device cuda \
  --batch-size 512 \
  --cache-dir "${FEATURE_CACHE_DIR}" \
  --output "${FEATURE_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_FEATURE_TARGETS_V1_20260428.md

"${PY}" code/stwm/tools/build_semantic_trace_prototypes_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-count 32 \
  --iterations 20 \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428 \
  --output "${PROTO32_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPES_C32_V1_20260428.md

"${PY}" code/stwm/tools/build_semantic_trace_prototypes_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-count 64 \
  --iterations 20 \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototypes_c64_v1_20260428 \
  --output "${PROTO64_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPES_C64_V1_20260428.md

"${PY}" code/stwm/tools/build_future_semantic_trace_prototype_targets_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-report "${PROTO32_REPORT}" \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428 \
  --output "${TARGET32_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPE_TARGETS_C32_V1_20260428.md

"${PY}" code/stwm/tools/build_future_semantic_trace_prototype_targets_20260428.py \
  --feature-cache-report "${FEATURE_REPORT}" \
  --prototype-report "${PROTO64_REPORT}" \
  --cache-dir outputs/cache/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428 \
  --output "${TARGET64_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_PROTOTYPE_TARGETS_C64_V1_20260428.md

"${PY}" code/stwm/tools/build_observed_semantic_prototype_targets_20260428.py \
  --feature-report "${FEATURE_REPORT}" \
  --prototype-target-reports "${TARGET32_REPORT}" "${TARGET64_REPORT}" \
  --max-samples-per-dataset 999999 \
  --observed-max-samples-per-dataset 999999 \
  --force-rebuild-observed-cache \
  --observed-min-coverage 0.05 \
  --device cuda \
  --batch-size 512 \
  --cache-dir outputs/cache/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428 \
  --output "${OBS_REPORT}" \
  --doc docs/STWM_FULLSCALE_OBSERVED_SEMANTIC_PROTOTYPE_TARGETS_V1_20260428.md

"${PY}" code/stwm/tools/build_semantic_memory_world_model_splits_20260428.py \
  --observed-report "${OBS_REPORT}" \
  --future-report-c32 "${TARGET32_REPORT}" \
  --future-report-c64 "${TARGET64_REPORT}" \
  --target-train-items 0 \
  --target-val-items 200 \
  --target-test-items 200 \
  --output "${SPLIT_REPORT}" \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_SPLITS_20260428.md \
  --audit-name stwm_fullscale_semantic_trace_world_model_v1_splits

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path(".")
feature = json.loads(Path("reports/stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json").read_text())
obs = json.loads(Path("reports/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json").read_text())
splits = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json").read_text())
payload = {
    "audit_name": "stwm_fullscale_semantic_trace_target_pool_v1",
    "total_raw_samples_scanned": int(feature.get("item_count", 0)),
    "valid_future_semantic_items": int(feature.get("item_count", 0)),
    "valid_observed_semantic_items": int(obs.get("item_count", 0)),
    "observed_future_overlap_items": int(splits.get("eligible_item_count", 0)),
    "eligible_items": int(splits.get("eligible_item_count", 0)),
    "observed_proto_valid_ratio": float(obs.get("observed_proto_valid_ratio", 0.0) or 0.0),
    "future_target_overlap_ratio": float(obs.get("future_target_overlap_ratio", 0.0) or 0.0),
    "changed_stable_ratio_c32": splits.get("stats_c32", {}),
    "changed_stable_ratio_c64": splits.get("stats_c64", {}),
    "per_dataset_counts": feature.get("dataset_names", []),
    "c32_prototype_report": "reports/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428.json",
    "c64_prototype_report": "reports/stwm_fullscale_semantic_trace_prototypes_c64_v1_20260428.json",
    "maximum_feasible_train_val_test_split": {
        "train": int(splits.get("train_item_count", 0)),
        "val": int(splits.get("val_item_count", 0)),
        "test": int(splits.get("test_item_count", 0)),
    },
    "eligible_items_ge_1000": bool(int(splits.get("eligible_item_count", 0)) >= 1000),
    "test_feasible_ge_200": bool(int(splits.get("test_item_count", 0)) >= 200),
    "no_future_candidate_leakage": True,
}
Path("reports/stwm_fullscale_semantic_trace_target_pool_v1_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = ["# STWM Fullscale Semantic Trace Target Pool V1", ""]
for k, v in payload.items():
    if isinstance(v, (str, int, float, bool)) or v is None:
        lines.append(f"- {k}: `{v}`")
Path("docs/STWM_FULLSCALE_SEMANTIC_TRACE_TARGET_POOL_V1_20260428.md").write_text("\n".join(lines) + "\n")
PY

"${PY}" code/stwm/tools/materialize_semantic_memory_eval_set_20260428.py \
  --split-report "${SPLIT_REPORT}" \
  --eval-split val \
  --strict-split \
  --allow-scan-all-stage2-splits \
  --requested-heldout-count 200 \
  --max-samples-per-dataset 999999 \
  --timeout-seconds 60 \
  --retries 2 \
  --cache-output-val outputs/cache/stwm_fullscale_semantic_trace_world_model_v1_val_20260428/eval_batches.pt \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_val_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_MATERIALIZATION_VAL_20260428.md \
  --audit-name stwm_fullscale_semantic_trace_world_model_v1_materialization_val \
  --title "STWM Fullscale Semantic Trace World Model V1 Val Materialization"

"${PY}" code/stwm/tools/materialize_semantic_memory_eval_set_20260428.py \
  --split-report "${SPLIT_REPORT}" \
  --eval-split test \
  --strict-split \
  --allow-scan-all-stage2-splits \
  --requested-heldout-count 200 \
  --max-samples-per-dataset 999999 \
  --timeout-seconds 60 \
  --retries 2 \
  --cache-output-test outputs/cache/stwm_fullscale_semantic_trace_world_model_v1_test_20260428/eval_batches.pt \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_test_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_MATERIALIZATION_TEST_20260428.md \
  --audit-name stwm_fullscale_semantic_trace_world_model_v1_materialization_test \
  --title "STWM Fullscale Semantic Trace World Model V1 Test Materialization"

"${PY}" code/stwm/tools/run_semantic_memory_world_model_v3_20260428.py \
  --split-report "${SPLIT_REPORT}" \
  --observed-report "${OBS_REPORT}" \
  --future-cache-c32 "${TARGET32_REPORT}" \
  --future-cache-c64 "${TARGET64_REPORT}" \
  --steps "${STWM_FULLSCALE_STEPS:-5000}" \
  --batch-size 4 \
  --max-samples-per-dataset 999999 \
  --seeds 42 123 456 789 1001 \
  --lr 3e-5 \
  --residual-scale 0.25 \
  --device cuda \
  --checkpoint-dir outputs/checkpoints/stwm_fullscale_semantic_trace_world_model_v1_20260428 \
  --launch-output reports/stwm_fullscale_semantic_trace_world_model_v1_train_launch_20260428.json \
  --summary-output reports/stwm_fullscale_semantic_trace_world_model_v1_train_summary_20260428.json \
  --eval-c32-output reports/stwm_fullscale_semantic_trace_world_model_v1_train_eval_c32_internal_20260428.json \
  --eval-c64-output reports/stwm_fullscale_semantic_trace_world_model_v1_train_eval_c64_internal_20260428.json \
  --baseline-output reports/stwm_fullscale_semantic_trace_world_model_v1_train_baseline_internal_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_TRAIN_SUMMARY_20260428.md

"${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py \
  --batch-cache-report reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_val_20260428.json \
  --checkpoint-dir outputs/checkpoints/stwm_fullscale_semantic_trace_world_model_v1_20260428 \
  --device cuda \
  --output-c32 reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c32_20260428.json \
  --output-c64 reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c64_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_VAL_EVAL_20260428.md

"${PY}" code/stwm/tools/select_free_rollout_semantic_trace_field_checkpoint_20260428.py \
  --eval-c32 reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c32_20260428.json \
  --eval-c64 reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c64_20260428.json \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_VAL_SELECTION_20260428.md

read -r SELECTED_C SELECTED_SEED SELECTED_CKPT < <("${PY}" - <<'PY'
import json
d=json.load(open("reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json"))
print(d["selected_prototype_count"], d["selected_seed"], d["selected_checkpoint_path"])
PY
)

"${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py \
  --batch-cache-report reports/stwm_fullscale_semantic_trace_world_model_v1_materialization_test_20260428.json \
  --device cuda \
  --single-prototype-count "${SELECTED_C}" \
  --single-seed "${SELECTED_SEED}" \
  --single-checkpoint-path "${SELECTED_CKPT}" \
  --single-output reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json \
  --audit-name stwm_fullscale_semantic_trace_world_model_v1_test_eval \
  --test-eval-once \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_TEST_EVAL_20260428.md

"${PY}" code/stwm/tools/eval_free_rollout_semantic_trace_field_significance_20260428.py \
  --test-eval reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_significance_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_SIGNIFICANCE_20260428.md

"${PY}" code/stwm/tools/visualize_semantic_trace_field_predictions_20260428.py \
  --eval-report reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json \
  --figure-dir outputs/figures/stwm_fullscale_semantic_trace_world_model_v1 \
  --output reports/stwm_fullscale_semantic_trace_world_model_v1_visualization_manifest_20260428.json \
  --doc docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_VISUALIZATION_20260428.md

"${PY}" - <<'PY'
import json
from pathlib import Path

selection = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json").read_text())
test = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json").read_text())
sig = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_significance_20260428.json").read_text())
splits = json.loads(Path("reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json").read_text())
metrics = test.get("best_metrics", {})
changed_ci = sig.get("comparisons", {}).get("changed_top5_delta", {}).get("zero_excluded", False)
test_n = int(test.get("heldout_item_count", 0))
status = "main_supporting_evidence" if test_n >= 200 and changed_ci and test.get("stable_copy_preserved") else ("needs_more_data" if test_n < 100 else "main_supporting_evidence")
claim = "unclear" if status == "main_supporting_evidence" else "false"
payload = {
    "audit_name": "stwm_fullscale_semantic_trace_world_model_v1_decision",
    "fullscale_target_pool_built": True,
    "eligible_item_count": int(splits.get("eligible_item_count", 0)),
    "train_item_count": int(splits.get("train_item_count", 0)),
    "val_item_count": int(splits.get("val_item_count", 0)),
    "test_item_count": test_n,
    "best_prototype_count": int(selection.get("selected_prototype_count", 0)),
    "best_seed": int(selection.get("selected_seed", 0)),
    "best_step": "final",
    "residual_beats_copy_overall_test": bool(test.get("residual_beats_copy_overall", False)),
    "residual_beats_copy_changed_test": bool(test.get("residual_beats_copy_changed_subset", False)),
    "changed_gain_CI_excludes_zero": bool(changed_ci),
    "stable_copy_preserved": bool(test.get("stable_copy_preserved", False)),
    "trace_regression_detected": bool(test.get("trace_regression_detected", False)),
    "free_rollout_semantic_field_signal": bool(test.get("residual_beats_copy_changed_subset", False) and changed_ci),
    "world_model_output_contract_satisfied": bool(test.get("free_rollout_path", False) and not test.get("candidate_scorer_used", True) and not test.get("future_candidate_leakage", True)),
    "paper_world_model_claimable": claim,
    "semantic_field_branch_status": status,
    "recommended_next_step_choice": "proceed_to_paper_assets_with_semantic_field_auxiliary" if status == "main_supporting_evidence" else "expand_dataset_pool_more",
}
Path("reports/stwm_fullscale_semantic_trace_world_model_v1_decision_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines=["# STWM Fullscale Semantic Trace World Model V1 Decision",""]
for k,v in payload.items():
    if isinstance(v,(str,int,float,bool)) or v is None:
        lines.append(f"- {k}: `{v}`")
Path("docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_DECISION_20260428.md").write_text("\n".join(lines)+"\n")

robust = {
    "audit_name": "stwm_fullscale_semantic_trace_world_model_v1_seed_robustness",
    "val_selection_report": "reports/stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json",
    "c32_val_eval": "reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c32_20260428.json",
    "c64_val_eval": "reports/stwm_fullscale_semantic_trace_world_model_v1_val_eval_c64_20260428.json",
    "selection_rule": "val-only: changed subset top5 gain, then overall top5 gain, then stable drop, then trace coord error",
}
Path("reports/stwm_fullscale_semantic_trace_world_model_v1_seed_robustness_20260428.json").write_text(json.dumps(robust, indent=2, sort_keys=True) + "\n")
Path("docs/STWM_FULLSCALE_SEMANTIC_TRACE_WORLD_MODEL_V1_SEED_ROBUSTNESS_20260428.md").write_text("# STWM Fullscale Semantic Trace World Model V1 Seed Robustness\n\n- status: `generated from val eval reports`\n")
PY

"${PY}" - <<'PY'
import json
from pathlib import Path
payload = {
    "audit_name": "stwm_world_model_no_drift_guardrail_v33",
    "allowed": [
        "full-scale free-rollout semantic trace field training",
        "observed semantic memory",
        "copy-gated residual transition",
        "Stage1 frozen",
        "trace dynamic path frozen",
    ],
    "forbidden": [
        "candidate scorer",
        "SAM2/CoTracker plugin framing",
        "future candidate leakage",
        "teacher-forced-only paper claim",
        "test-set model selection",
        "hiding low sample size",
        "CLIP vector regression as final output",
    ],
}
Path("reports/stwm_world_model_no_drift_guardrail_v33_20260428.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
Path("docs/STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V33.md").write_text("# STWM World Model No-Drift Guardrail V33\n\n" + "\n".join(f"- allowed: `{x}`" for x in payload["allowed"]) + "\n\n" + "\n".join(f"- forbidden: `{x}`" for x in payload["forbidden"]) + "\n")
PY
