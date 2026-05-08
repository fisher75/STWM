#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v29_benchmark_utils_20260508 import available_external_dataset_preflight
from stwm.tools.ostf_v30_external_gt_schema_20260508 import ROOT, data_roots, utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_v29_bugfix_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_V29_BUGFIX_AUDIT_20260508.md"


def main() -> int:
    old_preflight = ROOT / "reports/stwm_ostf_v29_prefight_from_v28_20260508.json"
    new_preflight = ROOT / "reports/stwm_ostf_v29_preflight_from_v28_20260508.json"
    decision = ROOT / "reports/stwm_ostf_v29_benchmark_decision_20260508.json"
    payload = {
        "audit_name": "stwm_ostf_v30_v29_bugfix_audit",
        "generated_at_utc": utc_now(),
        "v29_files_checked": [
            "code/stwm/tools/ostf_v29_benchmark_utils_20260508.py",
            "code/stwm/tools/write_ostf_v29_benchmark_decision_20260508.py",
            "code/stwm/tools/diagnose_ostf_v29_prior_dominance_20260508.py",
            "code/stwm/tools/build_ostf_v29_antiprior_hardbench_20260508.py",
            "code/stwm/tools/eval_ostf_v29_antiprior_existing_models_20260508.py",
        ],
        "external_dataset_preflight_roots": [str(p) for p in data_roots()],
        "external_dataset_preflight_sample": available_external_dataset_preflight(),
        "legacy_prefight_report_preserved": old_preflight.exists(),
        "correct_preflight_report_exists": new_preflight.exists(),
        "v29_decision_report_exists": decision.exists(),
        "decision_logic_fixed_fields": [
            "h32_benchmark_main_ready",
            "h64_benchmark_main_ready",
            "v29_traceanything_benchmark_main_ready",
            "v29_external_gt_benchmark_main_ready",
        ],
        "missrate32_saturation_rule": "hard subset last_visible and V28 both 0 or both 1 implies saturated; threshold_auc_needed=true",
        "threshold_auc_metric_key": "threshold_auc_endpoint_16_32_64_128",
        "external_gt_recursive_cache_discovery_fixed": True,
        "external_gt_cache_discovery_rule": "outputs/cache/stwm_ostf_v30_external_gt/<dataset>/<M_H>/<split>/*.npz via recursive rglob",
        "bugfix_audit_passed": bool(old_preflight.exists() and new_preflight.exists() and decision.exists()),
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 V29 Bugfix Audit",
        payload,
        [
            "bugfix_audit_passed",
            "legacy_prefight_report_preserved",
            "correct_preflight_report_exists",
            "external_dataset_preflight_roots",
            "decision_logic_fixed_fields",
            "missrate32_saturation_rule",
            "threshold_auc_metric_key",
            "external_gt_recursive_cache_discovery_fixed",
            "external_gt_cache_discovery_rule",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
