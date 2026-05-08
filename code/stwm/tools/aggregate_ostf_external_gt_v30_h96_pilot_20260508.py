#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SEEDS = [42, 123]
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY = ROOT / "reports/stwm_ostf_v30_external_gt_h96_pilot_summary_20260508.json"
BOOT = ROOT / "reports/stwm_ostf_v30_external_gt_h96_pilot_bootstrap_20260508.json"
DECISION = ROOT / "reports/stwm_ostf_v30_external_gt_h96_pilot_decision_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_H96_PILOT_DECISION_20260508.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strongest_prior() -> str:
    prior = read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    best = "last_observed_copy"
    score = None
    for name, payload in prior.get("splits", {}).get("val", {}).items():
        if name == "oracle_best_prior":
            continue
        val = payload.get("by_horizon", {}).get("H96", {}).get("minFDE")
        if val is not None and (score is None or float(val) < score):
            best, score = name, float(val)
    return best


def prior_rows(prior_name: str) -> list[dict[str, Any]]:
    prior = read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    return [r for r in prior.get("test_item_rows_by_prior", {}).get(prior_name, []) if int(r.get("H", 0)) == 96 and int(r.get("M", 0)) == 128]


def main() -> int:
    prior_name = strongest_prior()
    p_rows = prior_rows(prior_name)
    per_seed: list[dict[str, Any]] = []
    boot: dict[str, Any] = {}
    positives = 0
    motion_positives = 0
    completed = 0
    for seed in SEEDS:
        name = f"v30_extgt_m128_h96_seed{seed}"
        path = RUN_DIR / f"{name}.json"
        if not path.exists():
            per_seed.append({"seed": seed, "completed": False, "missing": True, "report_path": str(path.relative_to(ROOT))})
            continue
        payload = read_json(path)
        rows = payload.get("test_item_rows", [])
        comp = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False)
        motion = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False, subset_key="v30_motion")
        auc = paired_bootstrap(rows, p_rows, "threshold_auc_endpoint_16_32_64_128", higher_better=True)
        boot[f"{name}_vs_{prior_name}_minFDE_K"] = comp
        boot[f"{name}_vs_{prior_name}_motion_minFDE_K"] = motion
        boot[f"{name}_vs_{prior_name}_threshold_auc_endpoint_16_32_64_128"] = auc
        positive = bool((comp.get("mean_delta") or 0) > 0)
        motion_positive = bool((motion.get("mean_delta") or 0) > 0)
        positives += int(positive)
        motion_positives += int(motion_positive)
        completed += int(bool(payload.get("completed")))
        per_seed.append(
            {
                "seed": seed,
                "completed": bool(payload.get("completed")),
                "report_path": str(path.relative_to(ROOT)),
                "delta_minFDE_vs_strongest_prior": comp.get("mean_delta"),
                "motion_delta_minFDE_vs_strongest_prior": motion.get("mean_delta"),
                "threshold_auc_delta": auc.get("mean_delta"),
                "minFDE_K": payload.get("test_metrics", {}).get("all", {}).get("minFDE_K"),
                "motion_minFDE_K": payload.get("test_metrics", {}).get("subsets", {}).get("motion", {}).get("minFDE_K"),
                "MissRate@64": payload.get("test_metrics", {}).get("all", {}).get("MissRate@64"),
                "MissRate@128": payload.get("test_metrics", {}).get("all", {}).get("MissRate@128"),
                "visibility_F1": payload.get("test_metrics", {}).get("all", {}).get("visibility_F1"),
                "relative_deformation_layout_error": payload.get("test_metrics", {}).get("all", {}).get("relative_deformation_layout_error"),
                "train_loss_decreased": bool(payload.get("train_loss_decreased")),
                "positive_vs_prior": positive,
                "motion_positive_vs_prior": motion_positive,
            }
        )
    pooled_model_rows = []
    pooled_prior_rows = []
    for seed in SEEDS:
        path = RUN_DIR / f"v30_extgt_m128_h96_seed{seed}.json"
        if not path.exists():
            continue
        rows = read_json(path).get("test_item_rows", [])
        for row in rows:
            r = dict(row)
            r["bootstrap_pair_key"] = f"seed{seed}|{row.get('uid')}|H{row.get('H')}|M{row.get('M')}"
            pooled_model_rows.append(r)
        for row in p_rows:
            r = dict(row)
            r["bootstrap_pair_key"] = f"seed{seed}|{row.get('uid')}|H{row.get('H')}|M{row.get('M')}"
            pooled_prior_rows.append(r)
    pooled = {
        "h96_pooled_seed_item_minFDE_K": paired_bootstrap(pooled_model_rows, pooled_prior_rows, "minFDE_K", higher_better=False),
        "h96_pooled_seed_item_motion_minFDE_K": paired_bootstrap(pooled_model_rows, pooled_prior_rows, "minFDE_K", higher_better=False, subset_key="v30_motion"),
        "h96_pooled_seed_item_threshold_auc": paired_bootstrap(pooled_model_rows, pooled_prior_rows, "threshold_auc_endpoint_16_32_64_128", higher_better=True),
    }
    pilot_bootstrap_positive = bool(pooled["h96_pooled_seed_item_minFDE_K"].get("zero_excluded") and (pooled["h96_pooled_seed_item_minFDE_K"].get("mean_delta") or 0) > 0)
    motion_bootstrap_positive = bool(pooled["h96_pooled_seed_item_motion_minFDE_K"].get("zero_excluded") and (pooled["h96_pooled_seed_item_motion_minFDE_K"].get("mean_delta") or 0) > 0)
    pilot_positive = bool(positives == 2 and motion_positives == 2 and pilot_bootstrap_positive and motion_bootstrap_positive)
    deltas = [float(x["delta_minFDE_vs_strongest_prior"]) for x in per_seed if x.get("delta_minFDE_vs_strongest_prior") is not None]
    summary = {
        "summary_name": "stwm_ostf_v30_external_gt_h96_pilot_summary",
        "generated_at_utc": utc_now(),
        "strongest_prior_H96": prior_name,
        "completed_run_count": completed,
        "expected_run_count": len(SEEDS),
        "per_seed": per_seed,
        "mean_delta_minFDE": float(statistics.mean(deltas)) if deltas else None,
        "semantic_not_tested_not_failed": True,
    }
    decision = {
        "decision_name": "stwm_ostf_v30_external_gt_h96_pilot_decision",
        "generated_at_utc": utc_now(),
        "strongest_prior_H96": prior_name,
        "h96_seed42_positive": any(x.get("seed") == 42 and x.get("positive_vs_prior") for x in per_seed),
        "h96_seed123_positive": any(x.get("seed") == 123 and x.get("positive_vs_prior") for x in per_seed),
        "h96_pilot_positive_seed_count": positives,
        "h96_pilot_motion_positive_seed_count": motion_positives,
        "h96_pilot_bootstrap_positive": pilot_bootstrap_positive,
        "h96_pilot_motion_bootstrap_positive": motion_bootstrap_positive,
        "h96_long_horizon_positive_preliminary": pilot_positive,
        "h96_needs_model_improvement": not pilot_positive,
        "semantic_not_tested_not_failed": True,
        "next_step_choice": "run_v30_h96_5seed" if pilot_positive else ("improve_v30_long_horizon_residual_modes" if positives >= 1 else "fix_h96_dataset_or_metrics"),
    }
    dump_json(SUMMARY, summary)
    dump_json(BOOT, {"bootstrap_name": "stwm_ostf_v30_external_gt_h96_pilot_bootstrap", "generated_at_utc": utc_now(), "per_seed_item_bootstrap": boot, "pooled_seed_item_bootstrap": pooled})
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V30 External GT H96 Pilot Decision",
        decision,
        [
            "strongest_prior_H96",
            "h96_seed42_positive",
            "h96_seed123_positive",
            "h96_pilot_positive_seed_count",
            "h96_pilot_bootstrap_positive",
            "h96_long_horizon_positive_preliminary",
            "semantic_not_tested_not_failed",
            "next_step_choice",
        ],
    )
    print(DECISION.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
