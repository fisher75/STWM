#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SEEDS = [42, 123, 456, 789, 2026]
HORIZONS = [32, 64]
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_round2_multiseed_summary_v2_20260508.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_round2_multiseed_bootstrap_v2_20260508.json"
DECISION_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_round2_multiseed_decision_v2_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_ROUND2_MULTISEED_DECISION_V2_20260508.md"


def read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prior_suite() -> dict[str, Any]:
    return read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")


def strongest_prior_for_horizon(prior: dict[str, Any], horizon: int) -> str:
    val = prior.get("splits", {}).get("val", {})
    best = None
    best_score = 1e99
    for name, payload in val.items():
        if name == "oracle_best_prior":
            continue
        score = payload.get("by_horizon", {}).get(f"H{horizon}", {}).get("minFDE")
        if score is not None and float(score) < best_score:
            best = name
            best_score = float(score)
    return str(best or "last_observed_copy")


def std(vals: list[float]) -> float:
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def mean(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(statistics.mean(vals)) if vals else None


def base_pair_key(row: dict[str, Any]) -> str:
    # The prior-suite rows were generated before cache_path/item_key was added,
    # so pooled seed-aware pairing must use the UID/H/M key shared by both sides.
    return f"{row.get('uid')}|H{row.get('H')}|M{row.get('M')}"


def seed_aware_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new_row = dict(row)
        new_row["seed"] = int(seed)
        new_row["bootstrap_pair_key"] = f"seed{int(seed)}|{base_pair_key(row)}"
        out.append(new_row)
    return out


def seed_level_ci(vals: list[float], n_boot: int = 2000) -> dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"seed_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    import numpy as np

    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(123)
    means = [float(arr[rng.integers(0, arr.size, size=arr.size)].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5]).tolist()
    return {
        "seed_count": int(arr.size),
        "mean_delta": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0 or hi < 0),
    }


def sign_test(positive_count: int, total_count: int) -> dict[str, Any]:
    if total_count <= 0:
        return {"positive_count": 0, "total_count": 0, "two_sided_p_under_0p5": None}
    tail = sum(math.comb(total_count, i) for i in range(positive_count, total_count + 1)) / (2**total_count)
    p = min(1.0, 2.0 * min(tail, 1.0 - tail + math.comb(total_count, positive_count) / (2**total_count)))
    return {
        "positive_count": int(positive_count),
        "total_count": int(total_count),
        "two_sided_p_under_0p5": float(p),
        "all_positive": bool(positive_count == total_count),
    }


def main() -> int:
    prior = prior_suite()
    prior_rows = prior.get("test_item_rows_by_prior", {})
    runs: dict[str, Any] = {}
    seed_rows: dict[str, list[dict[str, Any]]] = {"H32": [], "H64": []}
    pooled_model_rows: dict[str, list[dict[str, Any]]] = {"H32": [], "H64": []}
    pooled_prior_rows: dict[str, list[dict[str, Any]]] = {"H32": [], "H64": []}
    boot: dict[str, Any] = {}
    seed_delta: dict[str, list[float]] = {"H32": [], "H64": []}
    seed_motion_delta: dict[str, list[float]] = {"H32": [], "H64": []}
    seed_positive: dict[str, list[bool]] = {"H32": [], "H64": []}
    train_loss_decreased: dict[str, list[bool]] = {"H32": [], "H64": []}
    strongest = {32: strongest_prior_for_horizon(prior, 32), 64: strongest_prior_for_horizon(prior, 64)}

    for horizon in HORIZONS:
        hkey = f"H{horizon}"
        prior_name = strongest[horizon]
        prior_h_rows = [r for r in prior_rows.get(prior_name, []) if int(r.get("H", 0)) == horizon and int(r.get("M", 0)) == 128]
        for seed in SEEDS:
            name = f"v30_extgt_m128_h{horizon}_seed{seed}"
            path = RUN_DIR / f"{name}.json"
            if not path.exists():
                runs[name] = {"completed": False, "missing": True}
                continue
            payload = read_json(path)
            rows = payload.get("test_item_rows", [])
            runs[name] = {
                "completed": bool(payload.get("completed")),
                "report_path": str(path.relative_to(ROOT)),
                "checkpoint_path": payload.get("checkpoint_path"),
                "seed": seed,
                "horizon": horizon,
                "train_loss_decreased": bool(payload.get("train_loss_decreased")),
                "eval_item_row_count": len(rows),
                "all": payload.get("test_metrics", {}).get("all"),
                "motion": payload.get("test_metrics", {}).get("subsets", {}).get("motion"),
                "visibility_F1": payload.get("test_metrics", {}).get("all", {}).get("visibility_F1"),
                "relative_deformation_layout_error": payload.get("test_metrics", {}).get("all", {}).get("relative_deformation_layout_error"),
            }
            train_loss_decreased[hkey].append(bool(payload.get("train_loss_decreased")))
            comp_all = paired_bootstrap(rows, prior_h_rows, "minFDE_K", higher_better=False)
            comp_motion = paired_bootstrap(rows, prior_h_rows, "minFDE_K", higher_better=False, subset_key="v30_motion")
            comp_auc = paired_bootstrap(rows, prior_h_rows, "threshold_auc_endpoint_16_32_64_128", higher_better=True)
            boot[f"{name}_vs_{prior_name}_minFDE_K"] = comp_all
            boot[f"{name}_vs_{prior_name}_motion_minFDE_K"] = comp_motion
            boot[f"{name}_vs_{prior_name}_threshold_auc_endpoint_16_32_64_128"] = comp_auc
            delta = float(comp_all.get("mean_delta") or 0.0)
            mdelta = float(comp_motion.get("mean_delta") or 0.0)
            seed_delta[hkey].append(delta)
            seed_motion_delta[hkey].append(mdelta)
            seed_positive[hkey].append(delta > 0)
            seed_rows[hkey].append(
                {
                    "seed": seed,
                    "minFDE_K": payload.get("test_metrics", {}).get("all", {}).get("minFDE_K"),
                    "motion_minFDE_K": payload.get("test_metrics", {}).get("subsets", {}).get("motion", {}).get("minFDE_K"),
                    "threshold_auc_endpoint_16_32_64_128": payload.get("test_metrics", {}).get("all", {}).get("threshold_auc_endpoint_16_32_64_128"),
                    "MissRate@64": payload.get("test_metrics", {}).get("all", {}).get("MissRate@64"),
                    "MissRate@128": payload.get("test_metrics", {}).get("all", {}).get("MissRate@128"),
                    "visibility_F1": payload.get("test_metrics", {}).get("all", {}).get("visibility_F1"),
                    "relative_deformation_layout_error": payload.get("test_metrics", {}).get("all", {}).get("relative_deformation_layout_error"),
                    "delta_minFDE_vs_strongest_prior": delta,
                    "motion_delta_minFDE_vs_strongest_prior": mdelta,
                    "positive_vs_prior": delta > 0,
                    "motion_positive_vs_prior": mdelta > 0,
                    "train_loss_decreased": bool(payload.get("train_loss_decreased")),
                }
            )
            pooled_model_rows[hkey].extend(seed_aware_rows(rows, seed))
            pooled_prior_rows[hkey].extend(seed_aware_rows(prior_h_rows, seed))

    pooled = {}
    for horizon in HORIZONS:
        hkey = f"H{horizon}"
        prior_name = strongest[horizon]
        pooled[f"{hkey}_item_bootstrap_minFDE_K"] = paired_bootstrap(
            pooled_model_rows[hkey], pooled_prior_rows[hkey], "minFDE_K", higher_better=False
        )
        pooled[f"{hkey}_motion_bootstrap_minFDE_K"] = paired_bootstrap(
            pooled_model_rows[hkey], pooled_prior_rows[hkey], "minFDE_K", higher_better=False, subset_key="v30_motion"
        )
        pooled[f"{hkey}_item_bootstrap_threshold_auc"] = paired_bootstrap(
            pooled_model_rows[hkey],
            pooled_prior_rows[hkey],
            "threshold_auc_endpoint_16_32_64_128",
            higher_better=True,
        )
        pooled[f"{hkey}_strongest_prior"] = prior_name
        pooled[f"{hkey}_pairing_mode"] = "seed|uid|H|M"

    h32_seed_count = sum(seed_positive["H32"])
    h64_seed_count = sum(seed_positive["H64"])
    h32_item_positive = bool(pooled["H32_item_bootstrap_minFDE_K"].get("zero_excluded") and (pooled["H32_item_bootstrap_minFDE_K"].get("mean_delta") or 0) > 0)
    h64_item_positive = bool(pooled["H64_item_bootstrap_minFDE_K"].get("zero_excluded") and (pooled["H64_item_bootstrap_minFDE_K"].get("mean_delta") or 0) > 0)
    h32_motion_positive = bool(pooled["H32_motion_bootstrap_minFDE_K"].get("zero_excluded") and (pooled["H32_motion_bootstrap_minFDE_K"].get("mean_delta") or 0) > 0)
    h64_motion_positive = bool(pooled["H64_motion_bootstrap_minFDE_K"].get("zero_excluded") and (pooled["H64_motion_bootstrap_minFDE_K"].get("mean_delta") or 0) > 0)
    robust = bool(h32_seed_count >= 4 and h64_seed_count >= 4 and h32_item_positive and h64_item_positive and h32_motion_positive and h64_motion_positive)
    next_step = "run_v30_h96_seed42_then_multiseed" if robust else ("improve_v30_residual_modes" if h32_seed_count >= 4 else "fix_external_gt_dataset_or_metrics")
    seed_sign_tests = {
        "H32_all_minFDE": sign_test(int(h32_seed_count), len(seed_positive["H32"])),
        "H64_all_minFDE": sign_test(int(h64_seed_count), len(seed_positive["H64"])),
        "H32_motion_minFDE": sign_test(int(sum(v > 0 for v in seed_motion_delta["H32"])), len(seed_motion_delta["H32"])),
        "H64_motion_minFDE": sign_test(int(sum(v > 0 for v in seed_motion_delta["H64"])), len(seed_motion_delta["H64"])),
    }
    seed_mean_ci = {
        "H32_delta_minFDE": seed_level_ci(seed_delta["H32"]),
        "H64_delta_minFDE": seed_level_ci(seed_delta["H64"]),
        "H32_motion_delta_minFDE": seed_level_ci(seed_motion_delta["H32"]),
        "H64_motion_delta_minFDE": seed_level_ci(seed_motion_delta["H64"]),
    }

    summary = {
        "summary_name": "stwm_ostf_v30_external_gt_round2_multiseed_summary_v2",
        "generated_at_utc": utc_now(),
        "seeds": SEEDS,
        "expected_run_count": len(SEEDS) * len(HORIZONS),
        "completed_run_count": int(sum(1 for r in runs.values() if r.get("completed"))),
        "runs": runs,
        "per_seed": seed_rows,
        "seed_level_mean_std": {
            "H32_delta_minFDE_mean": mean(seed_delta["H32"]),
            "H32_delta_minFDE_std": std(seed_delta["H32"]),
            "H64_delta_minFDE_mean": mean(seed_delta["H64"]),
            "H64_delta_minFDE_std": std(seed_delta["H64"]),
            "H32_motion_delta_minFDE_mean": mean(seed_motion_delta["H32"]),
            "H32_motion_delta_minFDE_std": std(seed_motion_delta["H32"]),
            "H64_motion_delta_minFDE_mean": mean(seed_motion_delta["H64"]),
            "H64_motion_delta_minFDE_std": std(seed_motion_delta["H64"]),
        },
        "seed_level_mean_delta_CI": seed_mean_ci,
        "seed_level_sign_test": seed_sign_tests,
        "train_loss_decreased_rate": {
            "H32": float(sum(train_loss_decreased["H32"]) / max(len(train_loss_decreased["H32"]), 1)),
            "H64": float(sum(train_loss_decreased["H64"]) / max(len(train_loss_decreased["H64"]), 1)),
        },
        "pointodyssey_only": True,
        "semantic_status": "not_tested_due_absent_semantic_target_loss",
    }
    decision = {
        "decision_name": "stwm_ostf_v30_external_gt_round2_multiseed_decision_v2",
        "generated_at_utc": utc_now(),
        "h32_positive_seed_count": int(h32_seed_count),
        "h64_positive_seed_count": int(h64_seed_count),
        "h32_seed_mean_delta_minFDE": mean(seed_delta["H32"]),
        "h64_seed_mean_delta_minFDE": mean(seed_delta["H64"]),
        "h32_seed_std_delta_minFDE": std(seed_delta["H32"]),
        "h64_seed_std_delta_minFDE": std(seed_delta["H64"]),
        "h32_item_bootstrap_positive": h32_item_positive,
        "h64_item_bootstrap_positive": h64_item_positive,
        "h32_motion_bootstrap_positive": h32_motion_positive,
        "h64_motion_bootstrap_positive": h64_motion_positive,
        "seed_level_sign_test": seed_sign_tests,
        "seed_level_mean_delta_CI": seed_mean_ci,
        "trajectory_world_model_claim_preliminary": robust,
        "semantic_trace_field_claim_allowed": False,
        "semantic_not_tested_not_failed": True,
        "result_is_pointodyssey_only": True,
        "next_step_choice": next_step,
    }
    dump_json(SUMMARY_PATH, summary)
    dump_json(
        BOOT_PATH,
        {
            "bootstrap_name": "stwm_ostf_v30_external_gt_round2_multiseed_bootstrap_v2",
            "generated_at_utc": utc_now(),
            "pairing_fix": "pooled bootstraps pair by seed|uid|H|M because prior-suite rows lack cache_path/item_key; repeated test items across seeds are no longer collapsed",
            "per_seed_item_bootstrap": boot,
            "pooled_seed_item_bootstrap": pooled,
            "seed_level_sign_test": seed_sign_tests,
            "seed_level_mean_delta_CI": seed_mean_ci,
        },
    )
    dump_json(DECISION_PATH, decision)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 External GT Round-2 Multiseed Decision",
        decision,
        [
            "h32_positive_seed_count",
            "h64_positive_seed_count",
            "h32_seed_mean_delta_minFDE",
            "h64_seed_mean_delta_minFDE",
            "h32_item_bootstrap_positive",
            "h64_item_bootstrap_positive",
            "h32_motion_bootstrap_positive",
            "h64_motion_bootstrap_positive",
            "trajectory_world_model_claim_preliminary",
            "semantic_trace_field_claim_allowed",
            "semantic_not_tested_not_failed",
            "result_is_pointodyssey_only",
            "next_step_choice",
        ],
    )
    print(DECISION_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
