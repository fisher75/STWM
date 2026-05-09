#!/usr/bin/env python3
from __future__ import annotations

import json

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_5_protocol_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_5_PROTOCOL_DECISION_20260509.md"


def load(rel: str) -> dict:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    subset = load("reports/stwm_ostf_v33_5_split_matched_hard_subset_20260509.json")
    rerun = load("reports/stwm_ostf_v33_5_protocol_rerun_decision_20260509.json")
    manifest_full = bool(rerun.get("manifest_full_coverage_ok", False))
    available_ratio = float(rerun.get("available_ratio", 0.0) or 0.0)
    split_matched = bool(subset.get("split_matched_hard_subset_built", False))
    stable = bool(subset.get("hard_subset_sampling_stable_ready", False))
    val_test_agree = bool(rerun.get("val_test_agree", False))
    identity_stable = bool(rerun.get("identity_signal_stable", False))
    sem_rank = bool(rerun.get("semantic_ranking_signal_stable", False))
    sem_top1 = bool(rerun.get("semantic_top1_copy_beaten", False))
    traj_deg = bool(rerun.get("trajectory_degraded", True))
    if not manifest_full or available_ratio < 0.95:
        next_step = "fix_manifest_dataset_coverage"
    elif not split_matched or not stable or not val_test_agree:
        next_step = "fix_split_matched_hard_subset"
    elif not identity_stable:
        next_step = "fix_identity_contrastive_loss"
    elif not sem_rank:
        next_step = "fix_semantic_prototype_loss"
    else:
        next_step = "run_v33_5_h32_full_data_smoke"
    payload = {
        "generated_at_utc": utc_now(),
        "manifest_full_coverage_ok": manifest_full,
        "available_ratio": available_ratio,
        "split_matched_hard_subset_built": split_matched,
        "hard_subset_sampling_stable": stable,
        "val_test_agree": val_test_agree,
        "identity_signal_stable": identity_stable,
        "semantic_ranking_signal_stable": sem_rank,
        "semantic_top1_signal_positive": sem_top1,
        "trajectory_degraded": traj_deg,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.5 Protocol Decision", payload, ["manifest_full_coverage_ok", "available_ratio", "split_matched_hard_subset_built", "hard_subset_sampling_stable", "val_test_agree", "identity_signal_stable", "semantic_ranking_signal_stable", "semantic_top1_signal_positive", "trajectory_degraded", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
