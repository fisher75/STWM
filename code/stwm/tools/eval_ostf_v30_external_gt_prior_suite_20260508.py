#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import aggregate_report, item_row, paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import (
    EXTERNAL_MANIFEST_DIR,
    load_external_sample,
    prior_predictions,
    utc_now,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_PRIOR_SUITE_20260508.md"
PRIORS = (
    "last_visible_copy",
    "last_observed_copy",
    "visibility_aware_cv",
    "visibility_aware_damped",
    "fixed_affine",
    "median_object_anchor_copy",
)


def _entries(split: str) -> list[dict[str, Any]]:
    path = EXTERNAL_MANIFEST_DIR / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8")).get("entries", [])


def _rows_for(entries: list[dict[str, Any]], gamma: float) -> dict[str, list[dict[str, Any]]]:
    rows = {name: [] for name in PRIORS}
    oracle_rows = []
    for entry in entries:
        sample = load_external_sample(ROOT / entry["cache_path"])
        preds = prior_predictions(sample, gamma=gamma)
        per_prior = []
        for name in PRIORS:
            row = item_row(
                uid=entry["uid"],
                dataset=sample.dataset,
                horizon=sample.fut_points.shape[1],
                m_points=sample.obs_points.shape[0],
                cache_path=str(entry["cache_path"]),
                fut_points=sample.fut_points,
                fut_vis=sample.fut_vis,
                pred=preds[name],
                tags=entry.get("v30_subset_tags", []),
            )
            row["prior_name"] = name
            rows[name].append(row)
            per_prior.append(row)
        best = min(per_prior, key=lambda r: float(r["minFDE"]))
        oracle = dict(best)
        oracle["prior_name"] = "oracle_best_prior"
        oracle_rows.append(oracle)
    rows["oracle_best_prior"] = oracle_rows
    return rows


def _choose_gamma() -> tuple[float, dict[str, float]]:
    candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    scores = {}
    for gamma in candidates:
        rows = _rows_for(_entries("val"), gamma)["visibility_aware_damped"]
        scores[str(gamma)] = float(aggregate_report(rows)["all"]["minFDE"] or 1e9)
    best = min(candidates, key=lambda g: scores[str(g)])
    return float(best), scores


def main() -> int:
    gamma, gamma_scores = _choose_gamma()
    split_payload = {}
    test_rows_by_prior = {}
    for split in ("train", "val", "test"):
        rows_by_prior = _rows_for(_entries(split), gamma)
        split_payload[split] = {name: aggregate_report(rows) for name, rows in rows_by_prior.items()}
        if split == "test":
            test_rows_by_prior = rows_by_prior
    strongest = min(PRIORS, key=lambda name: float(split_payload["val"][name]["all"]["minFDE"] or 1e9))
    boot = {}
    for name in PRIORS:
        if name == strongest:
            continue
        for metric, higher in [
            ("minFDE", False),
            ("MissRate@32", False),
            ("MissRate@64", False),
            ("MissRate@128", False),
            ("threshold_auc_endpoint_16_32_64_128", True),
        ]:
            boot[f"{strongest}_vs_{name}_{metric}"] = paired_bootstrap(
                test_rows_by_prior[strongest], test_rows_by_prior[name], metric, higher_better=higher
            )
    payload = {
        "report_name": "stwm_ostf_v30_external_gt_prior_suite",
        "generated_at_utc": utc_now(),
        "val_selected_damped_gamma": gamma,
        "damped_gamma_val_minFDE_scores": gamma_scores,
        "strongest_causal_prior_by_val_minFDE": strongest,
        "metrics": [
            "minADE/minFDE",
            "endpoint_L1",
            "MissRate@16/32/64/128",
            "threshold_auc_endpoint_16_32_64_128",
            "PCK@8/16/32/64",
            "visibility_F1",
            "relative_deformation_layout_error",
        ],
        "splits": split_payload,
        "test_item_rows_by_prior": test_rows_by_prior,
        "paired_bootstrap_test": boot,
        "oracle_best_prior_is_diagnostic_only": True,
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 External GT Prior Suite",
        payload,
        [
            "val_selected_damped_gamma",
            "strongest_causal_prior_by_val_minFDE",
            "oracle_best_prior_is_diagnostic_only",
            "metrics",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
