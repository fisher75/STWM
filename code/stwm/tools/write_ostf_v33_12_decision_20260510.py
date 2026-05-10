#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_12_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_12_DECISION_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def metric(payload: dict[str, Any], key: str, default: Any = "not_run") -> Any:
    return payload.get(key, default) if payload else default


def main() -> int:
    truth = load("reports/stwm_ostf_v33_12_v33_11_result_truth_audit_20260510.json")
    oracle = load("reports/stwm_ostf_v33_12_v33_11_oracle_decomposition_20260510.json")
    clip = load("reports/stwm_ostf_v33_12_clip_k32_target_space_audit_20260510.json")
    teacher = load("reports/stwm_ostf_v33_12_semantic_teacher_candidate_build_20260510.json")
    sweep = load("reports/stwm_ostf_v33_12_semantic_target_space_sweep_20260510.json")
    train = load("reports/stwm_ostf_v33_12_copy_conservative_semantic_train_summary_20260510.json")
    eval_dec = load("reports/stwm_ostf_v33_12_copy_conservative_semantic_eval_decision_20260510.json")
    viz = load("reports/stwm_ostf_v33_12_visualization_manifest_20260510.json")

    true_oracle_done = bool(oracle.get("uses_v33_11_checkpoint"))
    old_oracle_not_v3311 = bool(truth.get("v33_11_oracle_not_actually_run"))
    target_ready = bool(sweep.get("target_space_ready_for_training"))
    training_ran = bool(train.get("fresh_training_completed") or eval_dec)
    clip_sufficient = bool(clip.get("clip_b32_target_space_sufficient"))
    stronger_built = bool(teacher.get("stronger_teacher_candidates_built"))
    visualization_ready = bool(viz.get("visualization_ready") and not viz.get("placeholder_only", True))

    stable_ok = metric(eval_dec, "stable_preservation_not_degraded_top5")
    changed_ok = metric(eval_dec, "changed_top5_beats_strongest_baseline")
    hard_ok = metric(eval_dec, "semantic_hard_top5_beats_strongest_baseline")
    identity_regressed = metric(eval_dec, "identity_regressed_vs_v33_9")
    trajectory_degraded = metric(eval_dec, "trajectory_degraded")

    if not true_oracle_done:
        next_step = "fix_target_space_generation"
    elif not stronger_built:
        next_step = "fix_teacher_availability"
    elif not target_ready:
        available = teacher.get("teacher_available", {})
        if available and sum(1 for v in available.values() if v) <= 1:
            next_step = "build_even_stronger_teacher_ensemble"
        else:
            next_step = "fix_target_space_generation"
    elif not training_ran:
        next_step = "train_v33_12_copy_conservative_semantic_on_best_target_space"
    elif trajectory_degraded is True or identity_regressed is True or stable_ok is not True or changed_ok is not True or hard_ok is not True:
        next_step = "fix_copy_conservative_loss"
    elif not visualization_ready:
        next_step = "fix_target_space_generation"
    else:
        next_step = "run_v33_12_seed123_replication"

    payload = {
        "generated_at_utc": utc_now(),
        "v33_11_oracle_not_actually_run": bool(old_oracle_not_v3311 and not true_oracle_done),
        "v33_11_prior_oracle_artifact_used_wrong_checkpoint": old_oracle_not_v3311,
        "v33_12_true_v33_11_oracle_done": true_oracle_done,
        "clip_k32_target_space_sufficient": clip_sufficient,
        "teacher_jitter_suspected": bool(clip.get("teacher_jitter_suspected")),
        "K32_too_coarse_suspected": bool(clip.get("K32_too_coarse_suspected")),
        "sample_frequency_baseline_too_strong": bool(clip.get("sample_frequency_baseline_too_strong")),
        "stronger_teacher_candidates_built": stronger_built,
        "best_teacher_by_val": sweep.get("best_teacher_by_val"),
        "best_aggregation_by_val": sweep.get("best_aggregation_by_val"),
        "best_K_by_val": sweep.get("best_K_by_val"),
        "target_space_oracle_passes": bool(sweep.get("target_space_oracle_passes")),
        "target_space_ready_for_training": target_ready,
        "v33_12_training_ran": training_ran,
        "stable_preservation_not_degraded_top5": stable_ok,
        "stable_wrong_update_rate": metric(eval_dec, "stable_wrong_update_rate"),
        "changed_top5_beats_strongest_baseline": changed_ok,
        "semantic_hard_top5_beats_strongest_baseline": hard_ok,
        "identity_regressed_vs_v33_9": identity_regressed,
        "trajectory_degraded": trajectory_degraded,
        "visualization_ready": visualization_ready,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
        "source_reports": {
            "result_truth": "reports/stwm_ostf_v33_12_v33_11_result_truth_audit_20260510.json",
            "oracle_decomposition": "reports/stwm_ostf_v33_12_v33_11_oracle_decomposition_20260510.json",
            "clip_k32_target_space": "reports/stwm_ostf_v33_12_clip_k32_target_space_audit_20260510.json",
            "teacher_candidates": "reports/stwm_ostf_v33_12_semantic_teacher_candidate_build_20260510.json",
            "target_space_sweep": "reports/stwm_ostf_v33_12_semantic_target_space_sweep_20260510.json",
            "visualization": "reports/stwm_ostf_v33_12_visualization_manifest_20260510.json",
        },
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.12 Decision",
        payload,
        [
            "v33_11_oracle_not_actually_run",
            "v33_12_true_v33_11_oracle_done",
            "clip_k32_target_space_sufficient",
            "teacher_jitter_suspected",
            "K32_too_coarse_suspected",
            "sample_frequency_baseline_too_strong",
            "stronger_teacher_candidates_built",
            "best_teacher_by_val",
            "best_aggregation_by_val",
            "best_K_by_val",
            "target_space_oracle_passes",
            "target_space_ready_for_training",
            "v33_12_training_ran",
            "stable_preservation_not_degraded_top5",
            "changed_top5_beats_strongest_baseline",
            "semantic_hard_top5_beats_strongest_baseline",
            "identity_regressed_vs_v33_9",
            "trajectory_degraded",
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
