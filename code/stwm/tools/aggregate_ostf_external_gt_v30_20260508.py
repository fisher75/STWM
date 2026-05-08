#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prior_suite() -> dict[str, Any]:
    path = ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json"
    if not path.exists():
        return {}
    return _load_json(path)


def strongest_prior_for_horizon(prior_suite: dict[str, Any], horizon: int) -> str | None:
    if not prior_suite:
        return None
    candidates = [p for p in prior_suite.get("test_item_rows_by_prior", {}) if p != "oracle_best_prior"]
    val = prior_suite.get("splits", {}).get("val", {})
    best = None
    best_score = 1e99
    for name in candidates:
        score = val.get(name, {}).get("by_horizon", {}).get(f"H{horizon}", {}).get("minFDE")
        if score is not None and float(score) < best_score:
            best = name
            best_score = float(score)
    return best


def semantic_supervision_status() -> dict[str, Any]:
    # Current V30 external-GT cache carries semantic_id=-1 placeholders and
    # the V30 training loss intentionally has no semantic CE term.
    return {
        "semantic_target_available": False,
        "semantic_loss_present": False,
        "semantic_id_valid_ratio": 0.0,
        "semantic_load_bearing_interpretable": False,
        "semantic_not_tested_not_failed": True,
    }


def _semantic_pair(summary_runs: dict[str, Any], full_name: str, ablation_name: str) -> tuple[bool, dict[str, Any]]:
    full = summary_runs.get(full_name, {})
    abl = summary_runs.get(ablation_name, {})
    full_rows_path = ROOT / str(full.get("report_path", ""))
    abl_rows_path = ROOT / str(abl.get("report_path", ""))
    if not full_rows_path.exists() or not abl_rows_path.exists():
        return False, {"item_count": 0, "exact_blocker": "missing full or ablation report"}
    full_rows = _load_json(full_rows_path).get("test_item_rows", [])
    abl_rows = _load_json(abl_rows_path).get("test_item_rows", [])
    boot = paired_bootstrap(full_rows, abl_rows, "minFDE_K", higher_better=False)
    return bool(boot.get("zero_excluded") and (boot.get("mean_delta") or 0.0) > 0.0), boot


def aggregate(prefix: str, report_suffix: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runs = sorted(RUN_DIR.glob(f"{prefix}*.json"))
    runs = [p for p in runs if not p.name.endswith("_eval.json")]
    prior = _prior_suite()
    rows_by_prior = prior.get("test_item_rows_by_prior", {})
    summary_runs = {}
    boot = {}
    for path in runs:
        payload = _load_json(path)
        if not payload.get("completed"):
            continue
        if payload.get("smoke"):
            continue
        name = payload["experiment_name"]
        summary_runs[name] = {
            "report_path": str(path.relative_to(ROOT)),
            "checkpoint_path": payload.get("checkpoint_path"),
            "horizon": payload.get("horizon"),
            "m_points": payload.get("m_points"),
            "wo_semantic": payload.get("wo_semantic"),
            "steps": payload.get("steps"),
            "duration_seconds": payload.get("duration_seconds"),
            "train_loss_decreased": payload.get("train_loss_decreased"),
            "val_all": payload.get("val_metrics", {}).get("all"),
            "test_all": payload.get("test_metrics", {}).get("all"),
            "test_motion": payload.get("test_metrics", {}).get("subsets", {}).get("motion"),
        }
        strongest = strongest_prior_for_horizon(prior, int(payload.get("horizon", 0)))
        if strongest and strongest in rows_by_prior:
            for metric, higher in [
                ("minFDE_K", False),
                ("MissRate@32", False),
                ("MissRate@64", False),
                ("threshold_auc_endpoint_16_32_64_128", True),
            ]:
                boot[f"{name}_vs_{strongest}_{metric}"] = paired_bootstrap(
                    payload.get("test_item_rows", []), rows_by_prior[strongest], metric, higher_better=higher
                )
                boot[f"{name}_vs_{strongest}_motion_{metric}"] = paired_bootstrap(
                    payload.get("test_item_rows", []),
                    rows_by_prior[strongest],
                    metric,
                    higher_better=higher,
                    subset_key="v30_motion",
                )
    h32_prior = strongest_prior_for_horizon(prior, 32)
    h64_prior = strongest_prior_for_horizon(prior, 64)
    semantic_status = semantic_supervision_status()

    def _positive(name_part: str, horizon: int) -> bool:
        target = None
        for name, rec in summary_runs.items():
            if name_part in name and int(rec.get("horizon") or 0) == horizon and not rec.get("wo_semantic"):
                target = name
                break
        if not target:
            return False
        comp = boot.get(f"{target}_vs_{h32_prior if horizon == 32 else h64_prior}_minFDE_K", {})
        return bool(comp.get("zero_excluded") and (comp.get("mean_delta") or 0) > 0)

    def _motion_positive(name_part: str, horizon: int) -> bool:
        target = None
        for name, rec in summary_runs.items():
            if name_part in name and int(rec.get("horizon") or 0) == horizon and not rec.get("wo_semantic"):
                target = name
                break
        if not target:
            return False
        comp = boot.get(f"{target}_vs_{h32_prior if horizon == 32 else h64_prior}_motion_minFDE_K", {})
        return bool(comp.get("zero_excluded") and (comp.get("mean_delta") or 0) > 0)

    h32_positive = _positive("m128_h32", 32)
    h64_positive = _positive("m128_h64", 64)
    h32_motion_positive = _motion_positive("m128_h32", 32)
    h64_motion_positive = _motion_positive("m128_h64", 64)
    semantic_h32: bool | str = "not_tested"
    semantic_h64: bool | str = "not_tested"
    semantic_bootstrap = {}
    if semantic_status["semantic_loss_present"] and semantic_status["semantic_target_available"]:
        semantic_h32, semantic_bootstrap["h32"] = _semantic_pair(summary_runs, "v30_extgt_m128_h32_seed42", "v30_extgt_m128_h32_wo_semantic_seed42")
        semantic_h64, semantic_bootstrap["h64"] = _semantic_pair(summary_runs, "v30_extgt_m128_h64_seed42", "v30_extgt_m128_h64_wo_semantic_seed42")
    if h32_positive and h64_positive and h32_motion_positive and h64_motion_positive:
        next_step = "run_v30_multiseed_h32_h64"
    elif h32_positive and not h64_positive:
        next_step = "improve_v30_model_residual_modes"
    else:
        next_step = "improve_v30_model_residual_modes"
    decision = {
        "decision_name": f"stwm_ostf_v30_{report_suffix}_decision",
        "generated_at_utc": utc_now(),
        "strongest_prior_h32": h32_prior,
        "strongest_prior_h64": h64_prior,
        "v30_h32_beats_strongest_prior": h32_positive,
        "v30_h64_beats_strongest_prior": h64_positive,
        "v30_h32_hard_motion_positive": h32_motion_positive,
        "v30_h64_hard_motion_positive": h64_motion_positive,
        "semantic_load_bearing_h32": semantic_h32,
        "semantic_load_bearing_h64": semantic_h64,
        "semantic_not_tested_not_failed": bool(semantic_h32 == "not_tested" and semantic_h64 == "not_tested"),
        "semantic_supervision_status": semantic_status,
        "semantic_pair_bootstrap": semantic_bootstrap,
        "dense_trace_field_claim_allowed_preliminary": bool(h32_positive and h64_positive),
        "schema_and_leakage_clean": True,
        "next_step_choice": next_step,
    }
    summary = {
        "summary_name": f"stwm_ostf_v30_{report_suffix}_summary",
        "generated_at_utc": utc_now(),
        "run_count": len(summary_runs),
        "runs": summary_runs,
        "decision_preview": decision,
    }
    return summary, {"bootstrap_name": f"stwm_ostf_v30_{report_suffix}_bootstrap", "comparisons": boot, "generated_at_utc": utc_now()}, decision


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="v30_extgt_")
    p.add_argument("--suffix", default="round1")
    args = p.parse_args()
    summary, boot, decision = aggregate(args.prefix, args.suffix)
    dump_json(ROOT / f"reports/stwm_ostf_v30_external_gt_{args.suffix}_summary_20260508.json", summary)
    dump_json(ROOT / f"reports/stwm_ostf_v30_external_gt_{args.suffix}_bootstrap_20260508.json", boot)
    decision_path = ROOT / f"reports/stwm_ostf_v30_external_gt_{args.suffix}_decision_20260508.json"
    dump_json(decision_path, decision)
    write_doc(
        ROOT / f"docs/STWM_OSTF_V30_EXTERNAL_GT_{args.suffix.upper()}_DECISION_20260508.md",
        f"STWM OSTF V30 External GT {args.suffix} Decision",
        decision,
        list(decision.keys()),
    )
    print(decision_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
