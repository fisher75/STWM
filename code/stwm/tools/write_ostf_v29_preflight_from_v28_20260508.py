#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v29_benchmark_utils_20260508 import ROOT, load_json, utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v29_prefight_from_v28_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V29_PREFLIGHT_FROM_V28_20260508.md"

REQUIRED = [
    "reports/stwm_ostf_v28_multiseed_decision_20260507.json",
    "reports/stwm_ostf_v28_multiseed_summary_20260507.json",
    "reports/stwm_ostf_v28_multiseed_bootstrap_20260507.json",
    "reports/stwm_ostf_v28_stronger_prior_eval_20260507.json",
    "reports/stwm_ostf_v28_stronger_prior_bootstrap_20260507.json",
    "reports/stwm_ostf_v28_contract_and_claims_audit_20260507.json",
    "code/stwm/tools/eval_ostf_lastobs_residual_v28_multiseed_20260507.py",
    "code/stwm/tools/eval_ostf_v28_stronger_priors_20260507.py",
    "code/stwm/tools/ostf_lastobs_v28_common_20260502.py",
]


def _json_complete(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, {}
    return bool(payload), payload


def main() -> int:
    files: dict[str, Any] = {}
    all_ok = True
    loaded: dict[str, dict[str, Any]] = {}
    for rel in REQUIRED:
        path = ROOT / rel
        info = {"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        if rel.endswith(".json"):
            ok, payload = _json_complete(path)
            info["json_parse_ok"] = ok
            loaded[rel] = payload
            all_ok = all_ok and ok
        else:
            info["nonempty"] = bool(path.exists() and path.stat().st_size > 0)
            all_ok = all_ok and bool(info["nonempty"])
        files[rel] = info

    decision = loaded.get("reports/stwm_ostf_v28_multiseed_decision_20260507.json", {})
    summary = loaded.get("reports/stwm_ostf_v28_multiseed_summary_20260507.json", {})
    contract = loaded.get("reports/stwm_ostf_v28_contract_and_claims_audit_20260507.json", {})
    v28_strict_reason = contract.get("object_dense_semantic_trace_field_claim_allowed_strict_reason")
    semantic_trace_decision = decision.get("semantic_trace_dynamics_load_bearing")
    semantic_trace_contract = contract.get("semantic_trace_dynamics_load_bearing")
    payload = {
        "audit_name": "stwm_ostf_v29_prefight_from_v28",
        "generated_at_utc": utc_now(),
        "root": str(ROOT),
        "required_artifacts_complete": all_ok,
        "files": files,
        "completed_run_count": decision.get("completed_run_count", summary.get("completed_run_count")),
        "expected_run_count": decision.get("expected_run_count", summary.get("expected_run_count")),
        "v28_partial": bool(decision.get("partial", summary.get("partial", True))),
        "strongest_causal_prior_H64": decision.get("strongest_causal_prior_H64"),
        "H64_beats_last_visible_copy_hard_minFDE": bool(decision.get("H64_beats_last_visible_copy_hard_minFDE")),
        "H64_MissRate32_improves": bool(decision.get("H64_MissRate32_improves_vs_last_visible_and_visibility_aware_damped")),
        "dense_points_load_bearing": decision.get("dense_points_load_bearing"),
        "semantic_field_load_bearing": decision.get("semantic_field_load_bearing", contract.get("semantic_field_load_bearing")),
        "semantic_trace_dynamics_load_bearing": semantic_trace_decision,
        "semantic_trace_dynamics_load_bearing_contract_audit": semantic_trace_contract,
        "semantic_trace_dynamics_internal_inconsistency": bool(
            semantic_trace_decision is not None
            and semantic_trace_contract is not None
            and str(semantic_trace_decision) != str(semantic_trace_contract)
        ),
        "object_dense_semantic_trace_field_claim_allowed_strict": bool(
            contract.get("object_dense_semantic_trace_field_claim_allowed_strict", False)
        ),
        "why_V28_strict_claim_is_false": v28_strict_reason
        or (
            "V28 does not beat last_visible_copy on H64 hard minFDE/MissRate@32, "
            "and strict semantic/dense-shape/visibility requirements are not all satisfied."
        ),
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V29 Preflight From V28",
        payload,
        [
            "required_artifacts_complete",
            "completed_run_count",
            "expected_run_count",
            "v28_partial",
            "strongest_causal_prior_H64",
            "H64_beats_last_visible_copy_hard_minFDE",
            "H64_MissRate32_improves",
            "dense_points_load_bearing",
            "semantic_field_load_bearing",
            "semantic_trace_dynamics_load_bearing",
            "semantic_trace_dynamics_load_bearing_contract_audit",
            "semantic_trace_dynamics_internal_inconsistency",
            "object_dense_semantic_trace_field_claim_allowed_strict",
            "why_V28_strict_claim_is_false",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
