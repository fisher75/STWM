#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_14_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_DECISION_20260510.md"


def load(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def as_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def main() -> int:
    failure = load("reports/stwm_ostf_v33_14_v33_13_target_failure_audit_20260510.json")
    prepare = load("reports/stwm_ostf_v33_14_real_teacher_model_prepare_20260510.json")
    cache = load("reports/stwm_ostf_v33_14_teacher_feature_cache_build_20260510.json")
    vocab = load("reports/stwm_ostf_v33_14_teacher_prototype_vocab_sweep_20260510.json")
    targets = load("reports/stwm_ostf_v33_14_teacher_semantic_target_build_20260510.json")
    probe = load("reports/stwm_ostf_v33_14_teacher_target_space_probe_sweep_20260510.json")
    train = load("reports/stwm_ostf_v33_14_stronger_teacher_model_train_summary_20260510.json")
    eval_dec = load("reports/stwm_ostf_v33_14_stronger_teacher_model_eval_decision_20260510.json")
    viz = load("reports/stwm_ostf_v33_14_teacher_target_visualization_manifest_20260510.json")

    available_teachers = prepare.get("available_teachers") or [
        row.get("teacher_name") or row.get("teacher")
        for row in prepare.get("teacher_rows", prepare.get("rows", []))
        if row.get("forward_dryrun_passed")
    ]
    available_teachers = [str(x) for x in available_teachers if x]
    forward_ok = as_bool(prepare.get("stronger_teacher_forward_dryrun_passed") or available_teachers)
    cache_built = as_bool(cache.get("teacher_feature_cache_built") or cache.get("cache_built") or cache.get("rows"))
    vocab_done = as_bool(vocab.get("teacher_prototype_vocab_sweep_done") or vocab.get("rows"))
    targets_built = as_bool(targets.get("teacher_semantic_targets_built") or targets.get("rows"))
    probe_done = as_bool(probe.get("target_space_probe_sweep_done") or probe.get("rows"))
    learnable = as_bool(probe.get("target_space_learnability_passed"))
    ready = as_bool(probe.get("ready_for_model_training"))
    training_ran = as_bool(train.get("fresh_training_completed") or train.get("completed") or train.get("v33_14_model_training_ran"))

    if not forward_ok:
        next_step = "fix_teacher_model_availability"
    elif not (cache_built and vocab_done and targets_built):
        next_step = "build_more_teacher_features"
    elif not probe_done:
        next_step = "build_more_teacher_features"
    elif not learnable:
        next_step = "build_teacher_ensemble_targets"
    elif ready and not training_ran:
        next_step = "train_v33_14_copy_residual_model_on_best_teacher"
    elif training_ran and not (
        as_bool(eval_dec.get("stable_preservation_not_degraded_top5"))
        and as_bool(eval_dec.get("changed_top5_beats_strongest_baseline"))
        and as_bool(eval_dec.get("semantic_hard_top5_beats_strongest_baseline"))
        and not as_bool(eval_dec.get("identity_regressed_vs_v33_9"))
        and not as_bool(eval_dec.get("trajectory_degraded"))
    ):
        next_step = "fix_v33_14_model_loss"
    elif training_ran:
        next_step = "run_v33_14_seed123_replication"
    else:
        next_step = "build_teacher_ensemble_targets"

    payload = {
        "generated_at_utc": utc_now(),
        "v33_13_target_failure_audit_done": as_bool(failure),
        "stronger_teacher_model_prepare_done": as_bool(prepare),
        "stronger_teacher_forward_dryrun_passed": forward_ok,
        "available_teachers": available_teachers,
        "teacher_feature_cache_built": cache_built,
        "teacher_prototype_vocab_sweep_done": vocab_done,
        "teacher_semantic_targets_built": targets_built,
        "target_space_probe_sweep_done": probe_done,
        "best_teacher_by_val": probe.get("best_teacher_by_val") or vocab.get("best_teacher_by_val"),
        "best_aggregation_by_val": probe.get("best_aggregation_by_val") or vocab.get("best_aggregation_by_val"),
        "best_K_by_val": probe.get("best_K_by_val") or vocab.get("best_K_by_val"),
        "best_probe_by_val": probe.get("best_probe_by_val"),
        "target_space_learnability_passed": learnable,
        "changed_signal_positive": as_bool(probe.get("changed_signal_positive")),
        "semantic_hard_signal_positive": as_bool(probe.get("semantic_hard_signal_positive")),
        "clip_b32_k256_beaten": as_bool(probe.get("clip_b32_k256_beaten")),
        "ready_for_model_training": ready,
        "v33_14_model_training_ran": training_ran,
        "stable_preservation_not_degraded_top5": eval_dec.get("stable_preservation_not_degraded_top5", "not_run") if training_ran else "not_run",
        "changed_top5_beats_strongest_baseline": eval_dec.get("changed_top5_beats_strongest_baseline", "not_run") if training_ran else "not_run",
        "semantic_hard_top5_beats_strongest_baseline": eval_dec.get("semantic_hard_top5_beats_strongest_baseline", "not_run") if training_ran else "not_run",
        "identity_regressed_vs_v33_9": eval_dec.get("identity_regressed_vs_v33_9", "not_run") if training_ran else "not_run",
        "trajectory_degraded": eval_dec.get("trajectory_degraded", "not_run") if training_ran else "not_run",
        "visualization_ready": as_bool(viz.get("visualization_ready")),
        "teacher_features_are_offline_supervision_only": True,
        "stage2_mainline_remains_trace_conditioned_semantic_trace_units": True,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.14 Decision",
        payload,
        [
            "v33_13_target_failure_audit_done",
            "stronger_teacher_forward_dryrun_passed",
            "available_teachers",
            "teacher_feature_cache_built",
            "teacher_prototype_vocab_sweep_done",
            "teacher_semantic_targets_built",
            "target_space_probe_sweep_done",
            "best_teacher_by_val",
            "best_aggregation_by_val",
            "best_K_by_val",
            "target_space_learnability_passed",
            "changed_signal_positive",
            "semantic_hard_signal_positive",
            "clip_b32_k256_beaten",
            "ready_for_model_training",
            "v33_14_model_training_ran",
            "visualization_ready",
            "integrated_identity_field_claim_allowed",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
