#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import (
    EXTERNAL_MANIFEST_DIR,
    ROOT,
    aggregate_metric_rows,
    load_external_sample,
    paired_bootstrap,
    point_metrics,
    prior_predictions,
    utc_now,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_existing_eval_20260508.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_existing_bootstrap_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_EXISTING_EVAL_20260508.md"
PRIORS = (
    "last_observed_copy",
    "last_visible_copy",
    "visibility_aware_damped",
    "visibility_aware_cv",
    "fixed_affine",
    "median_object_anchor_copy",
    "constant_velocity",
)


def _load_manifest(name: str) -> list[dict]:
    path = EXTERNAL_MANIFEST_DIR / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("entries", [])


def _row_for(sample, prior_name: str, pred, manifest_entry: dict) -> dict:
    row = {
        "uid": manifest_entry["uid"],
        "dataset": sample.dataset,
        "H": int(sample.fut_points.shape[1]),
        "M": int(sample.obs_points.shape[0]),
        "coordinate_system": sample.coordinate_system,
    }
    row.update(point_metrics(sample, pred))
    for tag in manifest_entry.get("v30_subset_tags", []):
        row[f"v30_{tag}"] = True
    return row


def _eval_entries(entries: list[dict]) -> dict[str, list[dict]]:
    rows = {name: [] for name in PRIORS}
    for entry in entries:
        sample = load_external_sample(ROOT / entry["cache_path"])
        preds = prior_predictions(sample, gamma=0.0)
        for name in PRIORS:
            rows[name].append(_row_for(sample, name, preds[name], entry))
    return rows


def _payload_for_rows(rows: list[dict]) -> dict:
    datasets = sorted({r["dataset"] for r in rows})
    horizons = sorted({int(r["H"]) for r in rows})
    return {
        "all": aggregate_metric_rows(rows),
        "by_dataset": {ds: aggregate_metric_rows(rows, dataset=ds) for ds in datasets},
        "by_horizon": {f"H{h}": aggregate_metric_rows(rows, horizon=h) for h in horizons},
        "subsets": {
            "motion": aggregate_metric_rows(rows, subset_key="v30_motion"),
            "occlusion_reappearance": aggregate_metric_rows(rows, subset_key="v30_occlusion_reappearance"),
            "nonlinear_large_disp": aggregate_metric_rows(rows, subset_key="v30_nonlinear_large_disp"),
            "long_gap": aggregate_metric_rows(rows, subset_key="v30_long_gap"),
        },
    }


def main() -> int:
    test_entries = _load_manifest("test")
    rows_by_prior = _eval_entries(test_entries)
    eval_payload = {
        "eval_name": "stwm_ostf_v30_external_gt_existing_eval",
        "generated_at_utc": utc_now(),
        "entry_count": len(test_entries),
        "priors": {name: _payload_for_rows(rows) for name, rows in rows_by_prior.items()},
        "existing_v28_compatible": False,
        "existing_v28_incompatibility_reason": (
            "V28 consumes TraceAnything-derived object trace state with semantic memory; external GT cache schema has "
            "PointOdyssey/TAPVid point-field GT and lacks the trained V28 input semantic/context tensors. Evaluation is "
            "therefore restricted to analytic priors until a V30 model adapter is trained."
        ),
        "metrics": [
            "minADE",
            "minFDE",
            "MissRate@16/32/64/128",
            "threshold_auc_endpoint_16_32_64_128",
            "PCK@8/16/32/64",
            "visibility_F1",
            "relative_deformation_layout_error",
        ],
    }
    boot = {"bootstrap_name": "stwm_ostf_v30_external_gt_existing_bootstrap", "generated_at_utc": utc_now(), "comparisons": {}}
    for prior in ["last_observed_copy", "visibility_aware_damped", "visibility_aware_cv", "fixed_affine", "constant_velocity"]:
        for metric, higher in [
            ("minFDE", False),
            ("MissRate@32", False),
            ("MissRate@64", False),
            ("MissRate@128", False),
            ("threshold_auc_endpoint_16_32_64_128", True),
        ]:
            boot["comparisons"][f"last_visible_copy_vs_{prior}_{metric}"] = paired_bootstrap(
                rows_by_prior["last_visible_copy"], rows_by_prior[prior], metric, higher
            )
    dump_json(REPORT_PATH, eval_payload)
    dump_json(BOOT_PATH, boot)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 External GT Existing Eval",
        {
            "entry_count": len(test_entries),
            "existing_v28_compatible": False,
            "existing_v28_incompatibility_reason": eval_payload["existing_v28_incompatibility_reason"],
            "bootstrap_path": str(BOOT_PATH.relative_to(ROOT)),
        },
        ["entry_count", "existing_v28_compatible", "existing_v28_incompatibility_reason", "bootstrap_path"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
