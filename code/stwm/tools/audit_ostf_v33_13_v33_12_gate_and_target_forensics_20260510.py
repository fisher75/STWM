#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_13_v33_12_gate_and_target_forensics_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_V33_12_GATE_AND_TARGET_FORENSICS_20260510.md"


def read(path: str) -> str:
    p = ROOT / path
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    v3312_model = read("code/stwm/modules/ostf_v33_12_copy_conservative_semantic_world_model.py")
    v3311_model = read("code/stwm/modules/ostf_v33_11_identity_preserving_copy_residual_semantic_world_model.py")
    train = read("code/stwm/tools/train_ostf_v33_12_copy_conservative_semantic_20260510.py")
    sweep_code = read("code/stwm/tools/eval_ostf_v33_12_semantic_target_space_sweep_20260510.py")
    teacher_code = read("code/stwm/tools/build_ostf_v33_12_stronger_semantic_teacher_candidates_20260510.py")
    train_report = load("reports/stwm_ostf_v33_12_copy_conservative_semantic_train_summary_20260510.json")
    eval_dec = load("reports/stwm_ostf_v33_12_copy_conservative_semantic_eval_decision_20260510.json")
    clip_audit = load("reports/stwm_ostf_v33_12_clip_k32_target_space_audit_20260510.json")
    decision = load("reports/stwm_ostf_v33_12_decision_20260510.json")
    teacher = load("reports/stwm_ostf_v33_12_semantic_teacher_candidate_build_20260510.json")

    gate_raw_probability = 'raw_gate = out["semantic_change_gate"]' in v3312_model and '"semantic_change_gate": gate' in v3311_model
    double_sigmoid = "semantic_change_gate_raw" in train and ".sigmoid().mean()" in train
    train_eval_mismatch = "if self.training and self.residual_update_budget" in v3312_model
    residual_budget_train_only = train_eval_mismatch and "residual_update_budget" in v3312_model
    onehot_oracle = "oracle_logits = np.log(onehot(target" in sweep_code or "changed_oracle_top5" in sweep_code
    only_clip = bool(teacher.get("teacher_available", {}).get("clip_vit_b32_local")) and sum(1 for v in teacher.get("teacher_available", {}).values() if v) == 1
    stronger_built = bool(teacher.get("stronger_teacher_candidates_built"))
    best_teacher_overstated = only_clip and decision.get("best_teacher_by_val") == "clip_vit_b32_local"
    stable_failed = eval_dec.get("stable_preservation_not_degraded_top5") is False
    changed_hard_failed = eval_dec.get("changed_top5_beats_strongest_baseline") is False or eval_dec.get("semantic_hard_top5_beats_strongest_baseline") is False
    locations = {
        "probability_misnamed_raw": "code/stwm/modules/ostf_v33_12_copy_conservative_semantic_world_model.py:forward raw_gate = out['semantic_change_gate']",
        "double_sigmoid_loss": "code/stwm/tools/train_ostf_v33_12_copy_conservative_semantic_20260510.py:loss_fn semantic_change_gate_raw.sigmoid()",
        "train_eval_gate_mismatch": "code/stwm/modules/ostf_v33_12_copy_conservative_semantic_world_model.py:forward training-only residual_update_budget clip",
        "onehot_oracle": "code/stwm/tools/eval_ostf_v33_12_semantic_target_space_sweep_20260510.py onehot(target) oracle",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "semantic_change_gate_raw_is_probability_not_logit": bool(gate_raw_probability),
        "double_sigmoid_gate_loss_detected": bool(double_sigmoid),
        "train_eval_gate_mismatch_detected": bool(train_eval_mismatch),
        "residual_update_budget_train_only": bool(residual_budget_train_only),
        "target_space_oracle_is_onehot_future_target": bool(onehot_oracle),
        "target_space_learnability_not_proven": bool(onehot_oracle),
        "stronger_teacher_candidates_actually_built": bool(stronger_built),
        "only_clip_b32_available": bool(only_clip),
        "best_teacher_claim_overstated": bool(best_teacher_overstated),
        "stable_preservation_failed": bool(stable_failed),
        "changed_hard_failed_vs_strongest_baseline": bool(changed_hard_failed),
        "clip_k32_target_space_sufficient": bool(clip_audit.get("clip_b32_target_space_sufficient")),
        "v33_12_checkpoint_path": train_report.get("checkpoint_path"),
        "exact_code_locations": locations,
        "recommended_fix": "repair_gate_logits_train_eval_contract_and_run_observed_context_target_space_probe",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.13 V33.12 Gate and Target Forensics",
        payload,
        [
            "semantic_change_gate_raw_is_probability_not_logit",
            "double_sigmoid_gate_loss_detected",
            "train_eval_gate_mismatch_detected",
            "residual_update_budget_train_only",
            "target_space_oracle_is_onehot_future_target",
            "target_space_learnability_not_proven",
            "stronger_teacher_candidates_actually_built",
            "only_clip_b32_available",
            "best_teacher_claim_overstated",
            "stable_preservation_failed",
            "changed_hard_failed_vs_strongest_baseline",
            "recommended_fix",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
