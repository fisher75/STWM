#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_round1_claim_boundary_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_ROUND1_CLAIM_BOUNDARY_AUDIT_20260508.md"


def read_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def semantic_id_valid_ratio(max_files: int = 500) -> float:
    files = list((ROOT / "outputs/cache/stwm_ostf_v30_external_gt/pointodyssey").rglob("*.npz"))[:max_files]
    vals = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        vals.append(float(np.asarray(z.get("semantic_id", -1)).item() >= 0))
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    decision = read_json("reports/stwm_ostf_v30_external_gt_round1_decision_20260508.json")
    summary = read_json("reports/stwm_ostf_v30_external_gt_round1_summary_20260508.json")
    boot = read_json("reports/stwm_ostf_v30_external_gt_round1_bootstrap_20260508.json")
    prior = read_json("reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    h32 = boot.get("comparisons", {}).get("v30_extgt_m128_h32_seed42_vs_last_observed_copy_motion_minFDE_K", {})
    h64 = boot.get("comparisons", {}).get("v30_extgt_m128_h64_seed42_vs_last_observed_copy_motion_minFDE_K", {})
    semantic_ratio = semantic_id_valid_ratio()
    missrate32_h32 = boot.get("comparisons", {}).get("v30_extgt_m128_h32_seed42_vs_last_observed_copy_motion_MissRate@32", {})
    missrate32_h64 = boot.get("comparisons", {}).get("v30_extgt_m128_h64_seed42_vs_last_observed_copy_motion_MissRate@32", {})
    m512_note = {
        "m512_h32_present": "v30_extgt_m512_h32_seed42" in summary.get("runs", {}),
        "m512_h64_present": "v30_extgt_m512_h64_seed42" in summary.get("runs", {}),
        "interpretation": "optional_pilot_not_density_scaling_conclusion",
    }
    payload = {
        "audit_name": "stwm_ostf_v30_round1_claim_boundary_audit",
        "generated_at_utc": utc_now(),
        "round1_trajectory_positive": bool(
            decision.get("v30_h32_beats_strongest_prior")
            and decision.get("v30_h64_beats_strongest_prior")
            and h32.get("zero_excluded")
            and h64.get("zero_excluded")
        ),
        "round1_semantic_target_available": bool(semantic_ratio > 0.0),
        "semantic_id_valid_ratio": semantic_ratio,
        "semantic_loss_present": False,
        "semantic_load_bearing_interpretable": False,
        "semantic_not_tested_not_failed": True,
        "dense_trajectory_field_claim_allowed_preliminary": bool(decision.get("v30_h32_beats_strongest_prior") and decision.get("v30_h64_beats_strongest_prior")),
        "semantic_trace_field_claim_allowed": False,
        "m512_result_boundary": m512_note,
        "missrate32_saturation_status": {
            "h32_motion": "saturated_or_non_discriminative" if not missrate32_h32.get("zero_excluded") else "discriminative",
            "h64_motion": "saturated_or_non_discriminative" if not missrate32_h64.get("zero_excluded") else "discriminative",
        },
        "primary_secondary_metric_when_missrate32_saturates": "threshold_auc_endpoint_16_32_64_128",
        "strongest_prior_reported_round1": {
            "h32": decision.get("strongest_prior_h32"),
            "h64": decision.get("strongest_prior_h64"),
            "prior_suite_overall": prior.get("strongest_causal_prior_by_val_minFDE"),
        },
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 Round-1 Claim Boundary Audit",
        payload,
        [
            "round1_trajectory_positive",
            "round1_semantic_target_available",
            "semantic_id_valid_ratio",
            "semantic_loss_present",
            "semantic_load_bearing_interpretable",
            "semantic_not_tested_not_failed",
            "dense_trajectory_field_claim_allowed_preliminary",
            "semantic_trace_field_claim_allowed",
            "m512_result_boundary",
            "missrate32_saturation_status",
            "primary_secondary_metric_when_missrate32_saturates",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
