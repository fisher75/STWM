#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import AGG_CACHE_REPORT, ROOT, load_external_sample, utc_now
from stwm.tools.ostf_v29_benchmark_utils_20260508 import load_json


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_decision_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_DECISION_20260508.md"
AUDIT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_data_root_audit_20260508.json"
MANIFEST_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_antiprior_manifest_20260508.json"
EVAL_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_existing_eval_20260508.json"


def _best_prior(eval_payload: dict) -> str | None:
    best = (None, float("inf"))
    for name, payload in eval_payload.get("priors", {}).items():
        val = payload.get("subsets", {}).get("motion", {}).get("minFDE")
        if val is None:
            val = payload.get("all", {}).get("minFDE")
        if val is not None and float(val) < best[1]:
            best = (name, float(val))
    return best[0]


def main() -> int:
    audit = load_json(AUDIT_PATH)
    cache = load_json(AGG_CACHE_REPORT)
    manifest = load_json(MANIFEST_PATH)
    eval_payload = load_json(EVAL_PATH)
    datasets = audit.get("datasets", {})
    pointodyssey_complete = datasets.get("pointodyssey", {}).get("completeness_status") == "complete"
    tapvid_complete = datasets.get("tapvid", {}).get("completeness_status") == "complete"
    tapvid3d_complete = datasets.get("tapvid3d", {}).get("completeness_status") == "complete"
    h32_ready = bool(manifest.get("h32_external_gt_main_ready"))
    h64_ready = bool(manifest.get("h64_external_gt_main_ready"))
    h96_ready = bool(manifest.get("h96_external_gt_main_ready"))
    best_prior = _best_prior(eval_payload)
    last_visible_still = best_prior == "last_visible_copy"
    external_ready = bool(cache.get("external_gt_cache_ready") and (h32_ready or h64_ready or h96_ready))
    existing_v28_compatible = bool(eval_payload.get("existing_v28_compatible", False))
    if external_ready and not last_visible_still:
        next_step = "train_v30_external_gt_h32_h64"
    elif not audit.get("summary", {}).get("external_gt_data_available"):
        next_step = "download_or_authorize_PointOdyssey_TAPVid3D"
    elif not cache.get("external_gt_cache_ready") and audit.get("summary", {}).get("external_gt_data_available"):
        next_step = "fix_external_gt_adapter"
    elif not external_ready and audit.get("summary", {}).get("external_gt_data_available"):
        next_step = "fix_external_gt_adapter"
    else:
        next_step = "expand_traceanything_cache_to_1k_motion_clips"
    payload = {
        "decision_name": "stwm_ostf_v30_external_gt_decision",
        "generated_at_utc": utc_now(),
        "external_gt_data_available": bool(audit.get("summary", {}).get("external_gt_data_available")),
        "pointodyssey_complete": bool(pointodyssey_complete),
        "tapvid_complete": bool(tapvid_complete),
        "tapvid3d_complete": bool(tapvid3d_complete),
        "external_gt_cache_ready": bool(cache.get("external_gt_cache_ready")),
        "h32_external_gt_main_ready": h32_ready,
        "h64_external_gt_main_ready": h64_ready,
        "h96_external_gt_main_ready": h96_ready,
        "last_visible_prior_still_dominates": bool(last_visible_still),
        "strongest_external_gt_prior": best_prior,
        "benchmark_requires_training_new_model": bool(external_ready and not existing_v28_compatible),
        "existing_v28_compatible": existing_v28_compatible,
        "recommended_next_step": next_step,
        "decision_rationale": (
            "External GT cache/benchmark is ready enough and strongest prior is no longer last_visible; train a V30 adapter."
            if next_step == "train_v30_external_gt_h32_h64"
            else "Do not train yet: either the adapter/benchmark is not main-ready or analytic last-visible still dominates."
        ),
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 External GT Decision",
        payload,
        [
            "external_gt_data_available",
            "pointodyssey_complete",
            "tapvid_complete",
            "tapvid3d_complete",
            "external_gt_cache_ready",
            "h32_external_gt_main_ready",
            "h64_external_gt_main_ready",
            "h96_external_gt_main_ready",
            "last_visible_prior_still_dominates",
            "existing_v28_compatible",
            "recommended_next_step",
            "decision_rationale",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
