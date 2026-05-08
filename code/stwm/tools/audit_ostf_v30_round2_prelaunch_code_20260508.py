#!/usr/bin/env python3
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_round2_prelaunch_code_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_ROUND2_PRELAUNCH_CODE_AUDIT_20260508.md"

FILES = [
    "code/stwm/datasets/ostf_v30_external_gt_dataset_20260508.py",
    "code/stwm/modules/ostf_external_gt_world_model_v30.py",
    "code/stwm/tools/ostf_v30_external_gt_metrics_20260508.py",
    "code/stwm/tools/train_ostf_external_gt_v30_20260508.py",
    "code/stwm/tools/eval_ostf_external_gt_v30_20260508.py",
    "code/stwm/tools/eval_ostf_v30_external_gt_prior_suite_20260508.py",
    "code/stwm/tools/aggregate_ostf_external_gt_v30_20260508.py",
    "code/stwm/tools/audit_ostf_v30_round1_claim_boundary_20260508.py",
]


def read_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _compile_status() -> dict[str, Any]:
    out = {}
    for rel in FILES:
        try:
            py_compile.compile(str(ROOT / rel), doraise=True)
            out[rel] = {"py_compile_passed": True}
        except Exception as exc:
            out[rel] = {"py_compile_passed": False, "error": str(exc)}
    return out


def _uid_uniqueness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    uid_keys = [f"{r.get('uid')}|H{r.get('H')}|M{r.get('M')}" for r in rows]
    item_keys = [str(r.get("item_key") or f"{r.get('uid')}|H{r.get('H')}|M{r.get('M')}|{r.get('cache_path','')}") for r in rows]
    return {
        "row_count": len(rows),
        "uid_h_m_unique": len(set(uid_keys)) == len(uid_keys),
        "item_key_unique": len(set(item_keys)) == len(item_keys),
        "item_key_present_ratio": sum(1 for r in rows if r.get("item_key")) / max(len(rows), 1),
    }


def main() -> int:
    claim = read_json("reports/stwm_ostf_v30_round1_claim_boundary_audit_20260508.json")
    decision = read_json("reports/stwm_ostf_v30_external_gt_round1_decision_20260508.json")
    prior = read_json("reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    compile_status = _compile_status()
    seed42_h32 = read_json("reports/stwm_ostf_v30_external_gt_runs/v30_extgt_m128_h32_seed42.json")
    seed42_h64 = read_json("reports/stwm_ostf_v30_external_gt_runs/v30_extgt_m128_h64_seed42.json")
    h32_unique = _uid_uniqueness(seed42_h32.get("test_item_rows", []))
    h64_unique = _uid_uniqueness(seed42_h64.get("test_item_rows", []))
    strongest_h32 = decision.get("strongest_prior_h32")
    strongest_h64 = decision.get("strongest_prior_h64")
    tie_note = "not_identical"
    try:
        lv32 = prior["splits"]["val"]["last_visible_copy"]["by_horizon"]["H32"]["minFDE"]
        lo32 = prior["splits"]["val"]["last_observed_copy"]["by_horizon"]["H32"]["minFDE"]
        lv64 = prior["splits"]["val"]["last_visible_copy"]["by_horizon"]["H64"]["minFDE"]
        lo64 = prior["splits"]["val"]["last_observed_copy"]["by_horizon"]["H64"]["minFDE"]
        if abs(float(lv32) - float(lo32)) < 1e-6 and abs(float(lv64) - float(lo64)) < 1e-6:
            tie_note = "last_observed_copy_or_last_visible_copy_tie"
    except Exception:
        tie_note = "unknown"
    fatal = []
    if not all(rec["py_compile_passed"] for rec in compile_status.values()):
        fatal.append("py_compile_failure")
    if not h32_unique["uid_h_m_unique"] or not h64_unique["uid_h_m_unique"]:
        fatal.append("seed42_uid_h_m_not_unique")
    for name, payload in [("h32", seed42_h32), ("h64", seed42_h64)]:
        if not payload.get("train_loss_decreased"):
            fatal.append(f"{name}_seed42_train_loss_not_decreased")
        if not payload.get("test_item_rows"):
            fatal.append(f"{name}_seed42_missing_eval_item_rows")
    payload = {
        "audit_name": "stwm_ostf_v30_round2_prelaunch_code_audit",
        "generated_at_utc": utc_now(),
        "compile_status": compile_status,
        "fatal_issue_found": bool(fatal),
        "fatal_issues": fatal,
        "semantic_status": {
            "semantic_load_bearing": "not_tested" if claim.get("semantic_not_tested_not_failed") else "tested",
            "semantic_not_tested_not_failed": claim.get("semantic_not_tested_not_failed"),
            "semantic_id_valid_ratio": claim.get("semantic_id_valid_ratio"),
        },
        "strongest_prior_naming": {
            "h32": strongest_h32,
            "h64": strongest_h64,
            "tie_status": tie_note,
            "report_name_if_tie": "last_observed_copy_or_last_visible_copy_tie",
        },
        "bootstrap_pairing_rule": "item_key=uid+H+M+cache_path when present; legacy uid+H+M fallback only for old seed42 rows",
        "item_uid_uniqueness": {"h32_seed42": h32_unique, "h64_seed42": h64_unique},
        "seed42_eval_validation": {
            "h32_train_loss_decreased": seed42_h32.get("train_loss_decreased"),
            "h64_train_loss_decreased": seed42_h64.get("train_loss_decreased"),
            "h32_eval_item_row_count": len(seed42_h32.get("test_item_rows", [])),
            "h64_eval_item_row_count": len(seed42_h64.get("test_item_rows", [])),
        },
        "missrate32_saturation": claim.get("missrate32_saturation_status"),
        "threshold_auc_metric_key": "threshold_auc_endpoint_16_32_64_128",
        "ready_to_launch_round2": not fatal,
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 Round-2 Prelaunch Code Audit",
        payload,
        [
            "fatal_issue_found",
            "fatal_issues",
            "semantic_status",
            "strongest_prior_naming",
            "bootstrap_pairing_rule",
            "item_uid_uniqueness",
            "seed42_eval_validation",
            "missrate32_saturation",
            "threshold_auc_metric_key",
            "ready_to_launch_round2",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if not fatal else 2


if __name__ == "__main__":
    raise SystemExit(main())
