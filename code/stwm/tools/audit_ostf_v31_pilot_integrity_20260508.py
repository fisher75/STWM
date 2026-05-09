#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v31_pilot_integrity_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V31_PILOT_INTEGRITY_AUDIT_20260508.md"


INPUTS = {
    "compression_audit": "reports/stwm_ostf_v31_v30_field_compression_audit_20260508.json",
    "code_audit": "reports/stwm_ostf_v31_model_code_audit_20260508.json",
    "smoke": "reports/stwm_ostf_v31_field_preserving_smoke_summary_20260508.json",
    "pilot_summary": "reports/stwm_ostf_v31_field_preserving_pilot_summary_20260508.json",
    "pilot_bootstrap": "reports/stwm_ostf_v31_field_preserving_pilot_bootstrap_20260508.json",
    "pilot_decision": "reports/stwm_ostf_v31_field_preserving_pilot_decision_20260508.json",
    "semantic_broadcast": "reports/stwm_ostf_v31_semantic_broadcasting_readiness_20260508.json",
}


def _load(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    payloads = {name: _load(rel) for name, rel in INPUTS.items()}
    missing = [rel for rel in INPUTS.values() if not (ROOT / rel).exists()]
    summary = payloads["pilot_summary"]
    decision = payloads["pilot_decision"]
    code = payloads["code_audit"]
    compression = payloads["compression_audit"]
    semantic = payloads["semantic_broadcast"]
    runs = summary.get("runs", {})
    same_seed_same_mh = True
    for name, rec in runs.items():
        if not str(name).startswith("v31_field_m"):
            same_seed_same_mh = False
        if rec.get("v30_same_seed_all") is None:
            same_seed_same_mh = False
    partial_positive = bool(
        decision.get("v31_m128_beats_v30_m128_h32")
        and not decision.get("v31_m128_beats_v30_m128_h64")
        and decision.get("v31_m128_beats_v30_m128_h96")
    )
    m512_partial = bool(
        decision.get("v31_m512_beats_v31_m128", {}).get("h32")
        and not decision.get("v31_m512_beats_v31_m128", {}).get("h64")
        and decision.get("v31_m512_beats_v31_m128", {}).get("h96")
    )
    checks = {
        "inputs_present": not missing,
        "v30_confirmed_object_token_rollout_baseline": bool(compression.get("v30_can_be_used_as_object_token_rollout_baseline")),
        "v31_preserves_point_tokens_before_rollout": bool(code.get("preserves_point_tokens_before_rollout")),
        "main_rollout_state_shape_is_BMHD": str(code.get("main_rollout_state_shape")) == "[B,M,H,D]",
        "smoke_passed": bool(payloads["smoke"].get("smoke_passed")),
        "pilot_completed_6_of_6": int(summary.get("completed_run_count") or 0) == 6 and not summary.get("missing_runs"),
        "v31_vs_v30_same_seed_same_mh": bool(same_seed_same_mh),
        "seed42_result_partial_positive_not_full_dominance": bool(partial_positive),
        "density_recovery_only_seed42_pilot": bool(m512_partial and decision.get("density_scaling_recovered_with_v31")),
        "semantic_not_tested_not_failed": bool(semantic.get("semantic_status") == "not_tested_not_failed" and decision.get("semantic_not_tested_not_failed")),
        "field_tokens_load_bearing_not_yet_proven_by_ablation": "explicit ablation not part" in str(decision.get("field_tokens_load_bearing")),
    }
    audit_passed = all(
        checks[k]
        for k in [
            "inputs_present",
            "v31_preserves_point_tokens_before_rollout",
            "main_rollout_state_shape_is_BMHD",
            "smoke_passed",
            "pilot_completed_6_of_6",
            "v31_vs_v30_same_seed_same_mh",
            "seed42_result_partial_positive_not_full_dominance",
            "semantic_not_tested_not_failed",
            "field_tokens_load_bearing_not_yet_proven_by_ablation",
        ]
    )
    payload = {
        "generated_at_utc": utc_now(),
        "input_paths": INPUTS,
        "missing_inputs": missing,
        "checks": checks,
        "audit_passed": bool(audit_passed),
        "v31_multiseed_allowed": bool(audit_passed),
        "required_caveat": "V31 seed42 is partial positive; multiseed and field-interaction ablation are required before promotion.",
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V31 Pilot Integrity Audit",
        payload,
        ["audit_passed", "v31_multiseed_allowed", "checks", "required_caveat"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if audit_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
