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


SEEDS = [42, 123, 456, 789, 2026]
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY = ROOT / "reports/stwm_ostf_v30_external_gt_h96_multiseed_summary_20260508.json"
BOOT = ROOT / "reports/stwm_ostf_v30_external_gt_h96_multiseed_bootstrap_20260508.json"
DECISION = ROOT / "reports/stwm_ostf_v30_external_gt_h96_multiseed_decision_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_H96_MULTISEED_DECISION_20260508.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prior_name_and_rows() -> tuple[str, list[dict[str, Any]]]:
    prior = read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    best = "last_observed_copy"
    score = None
    for name, payload in prior.get("splits", {}).get("val", {}).items():
        if name == "oracle_best_prior":
            continue
        val = payload.get("by_horizon", {}).get("H96", {}).get("minFDE")
        if val is not None and (score is None or float(val) < score):
            best, score = name, float(val)
    rows = [r for r in prior.get("test_item_rows_by_prior", {}).get(best, []) if int(r.get("H", 0)) == 96 and int(r.get("M", 0)) == 128]
    return best, rows


def main() -> int:
    prior_name, p_rows = prior_name_and_rows()
    per_seed: list[dict[str, Any]] = []
    per_seed_boot: dict[str, Any] = {}
    pooled_model: list[dict[str, Any]] = []
    pooled_prior: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"v30_extgt_m128_h96_seed{seed}"
        path = RUN_DIR / f"{name}.json"
        if not path.exists():
            per_seed.append({"seed": seed, "completed": False, "missing": True})
            continue
        payload = read_json(path)
        rows = payload.get("test_item_rows", [])
        comp = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False)
        motion = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False, subset_key="v30_motion")
        auc = paired_bootstrap(rows, p_rows, "threshold_auc_endpoint_16_32_64_128", higher_better=True)
        per_seed_boot[f"{name}_vs_{prior_name}_minFDE_K"] = comp
        per_seed_boot[f"{name}_vs_{prior_name}_motion_minFDE_K"] = motion
        per_seed_boot[f"{name}_vs_{prior_name}_threshold_auc"] = auc
        per_seed.append(
            {
                "seed": seed,
                "completed": bool(payload.get("completed")),
                "report_path": str(path.relative_to(ROOT)),
                "delta_minFDE_vs_strongest_prior": comp.get("mean_delta"),
                "motion_delta_minFDE_vs_strongest_prior": motion.get("mean_delta"),
                "threshold_auc_delta": auc.get("mean_delta"),
                "positive_vs_prior": bool((comp.get("mean_delta") or 0) > 0),
                "motion_positive_vs_prior": bool((motion.get("mean_delta") or 0) > 0),
                "train_loss_decreased": bool(payload.get("train_loss_decreased")),
                "minFDE_K": payload.get("test_metrics", {}).get("all", {}).get("minFDE_K"),
                "motion_minFDE_K": payload.get("test_metrics", {}).get("subsets", {}).get("motion", {}).get("minFDE_K"),
                "MissRate@64": payload.get("test_metrics", {}).get("all", {}).get("MissRate@64"),
                "MissRate@128": payload.get("test_metrics", {}).get("all", {}).get("MissRate@128"),
                "visibility_F1": payload.get("test_metrics", {}).get("all", {}).get("visibility_F1"),
                "relative_deformation_layout_error": payload.get("test_metrics", {}).get("all", {}).get("relative_deformation_layout_error"),
            }
        )
        for row in rows:
            r = dict(row)
            r["bootstrap_pair_key"] = f"seed{seed}|{row.get('uid')}|H{row.get('H')}|M{row.get('M')}"
            pooled_model.append(r)
        for row in p_rows:
            r = dict(row)
            r["bootstrap_pair_key"] = f"seed{seed}|{row.get('uid')}|H{row.get('H')}|M{row.get('M')}"
            pooled_prior.append(r)

    deltas = [float(x["delta_minFDE_vs_strongest_prior"]) for x in per_seed if x.get("delta_minFDE_vs_strongest_prior") is not None]
    motion_deltas = [float(x["motion_delta_minFDE_vs_strongest_prior"]) for x in per_seed if x.get("motion_delta_minFDE_vs_strongest_prior") is not None]
    positives = sum(1 for x in per_seed if x.get("positive_vs_prior"))
    motion_positives = sum(1 for x in per_seed if x.get("motion_positive_vs_prior"))
    pooled = {
        "h96_pooled_seed_item_minFDE_K": paired_bootstrap(pooled_model, pooled_prior, "minFDE_K", higher_better=False),
        "h96_pooled_seed_item_motion_minFDE_K": paired_bootstrap(pooled_model, pooled_prior, "minFDE_K", higher_better=False, subset_key="v30_motion"),
        "h96_pooled_seed_item_threshold_auc": paired_bootstrap(pooled_model, pooled_prior, "threshold_auc_endpoint_16_32_64_128", higher_better=True),
    }
    robust = bool(
        positives >= 4
        and motion_positives >= 4
        and pooled["h96_pooled_seed_item_minFDE_K"].get("zero_excluded")
        and (pooled["h96_pooled_seed_item_minFDE_K"].get("mean_delta") or 0) > 0
        and pooled["h96_pooled_seed_item_motion_minFDE_K"].get("zero_excluded")
        and (pooled["h96_pooled_seed_item_motion_minFDE_K"].get("mean_delta") or 0) > 0
    )
    summary = {
        "summary_name": "stwm_ostf_v30_external_gt_h96_multiseed_summary",
        "generated_at_utc": utc_now(),
        "strongest_prior_H96": prior_name,
        "expected_run_count": len(SEEDS),
        "completed_run_count": sum(1 for x in per_seed if x.get("completed")),
        "per_seed": per_seed,
        "seed_mean_delta_minFDE": float(statistics.mean(deltas)) if deltas else None,
        "seed_std_delta_minFDE": float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0,
        "seed_mean_motion_delta_minFDE": float(statistics.mean(motion_deltas)) if motion_deltas else None,
        "semantic_not_tested_not_failed": True,
    }
    decision = {
        "decision_name": "stwm_ostf_v30_external_gt_h96_multiseed_decision",
        "generated_at_utc": utc_now(),
        "strongest_prior_H96": prior_name,
        "h96_positive_seed_count": int(positives),
        "h96_motion_positive_seed_count": int(motion_positives),
        "h96_item_bootstrap_positive": bool(pooled["h96_pooled_seed_item_minFDE_K"].get("zero_excluded") and (pooled["h96_pooled_seed_item_minFDE_K"].get("mean_delta") or 0) > 0),
        "h96_motion_bootstrap_positive": bool(pooled["h96_pooled_seed_item_motion_minFDE_K"].get("zero_excluded") and (pooled["h96_pooled_seed_item_motion_minFDE_K"].get("mean_delta") or 0) > 0),
        "h96_long_horizon_robust": robust,
        "trajectory_world_model_remains_robust_through_H96": robust,
        "semantic_not_tested_not_failed": True,
        "semantic_trace_field_claim_allowed": False,
        "next_step_choice": "run_v30_m512_m1024_density_scaling" if robust else "improve_v30_long_horizon_residual_modes",
    }
    dump_json(SUMMARY, summary)
    dump_json(BOOT, {"bootstrap_name": "stwm_ostf_v30_external_gt_h96_multiseed_bootstrap", "generated_at_utc": utc_now(), "per_seed_item_bootstrap": per_seed_boot, "pooled_seed_item_bootstrap": pooled})
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V30 External GT H96 Multiseed Decision",
        decision,
        [
            "strongest_prior_H96",
            "h96_positive_seed_count",
            "h96_motion_positive_seed_count",
            "h96_item_bootstrap_positive",
            "h96_motion_bootstrap_positive",
            "trajectory_world_model_remains_robust_through_H96",
            "semantic_not_tested_not_failed",
            "next_step_choice",
        ],
    )
    print(DECISION.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
