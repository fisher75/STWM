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


SEEDS = [42, 123, 456]
HORIZONS = [32, 64, 96]
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY = ROOT / "reports/stwm_ostf_v30_density_m512_pilot_summary_20260508.json"
BOOT = ROOT / "reports/stwm_ostf_v30_density_m512_pilot_bootstrap_20260508.json"
DECISION = ROOT / "reports/stwm_ostf_v30_density_m512_pilot_decision_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_M512_PILOT_DECISION_20260508.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strongest_prior(horizon: int) -> str:
    prior = read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    best = "last_observed_copy"
    score = None
    for name, payload in prior.get("splits", {}).get("val", {}).items():
        if name == "oracle_best_prior":
            continue
        val = payload.get("by_horizon", {}).get(f"H{horizon}", {}).get("minFDE")
        if val is not None and (score is None or float(val) < score):
            best, score = name, float(val)
    return best


def prior_rows(prior_name: str, horizon: int) -> list[dict[str, Any]]:
    prior = read_json(ROOT / "reports/stwm_ostf_v30_external_gt_prior_suite_20260508.json")
    return [r for r in prior.get("test_item_rows_by_prior", {}).get(prior_name, []) if int(r.get("H", 0)) == horizon and int(r.get("M", 0)) == 512]


def run_payload(m: int, horizon: int, seed: int) -> dict[str, Any] | None:
    path = RUN_DIR / f"v30_extgt_m{m}_h{horizon}_seed{seed}.json"
    if not path.exists():
        return None
    payload = read_json(path)
    payload["_report_path"] = str(path.relative_to(ROOT))
    return payload


def metric(payload: dict[str, Any] | None, subset: str = "all", key: str = "minFDE_K") -> float | None:
    if payload is None:
        return None
    if subset == "all":
        val = payload.get("test_metrics", {}).get("all", {}).get(key)
    else:
        val = payload.get("test_metrics", {}).get("subsets", {}).get(subset, {}).get(key)
    return float(val) if val is not None else None


def main() -> int:
    per_h: dict[str, list[dict[str, Any]]] = {}
    boot: dict[str, Any] = {}
    decision_counts: dict[str, int] = {}
    beats_m128: dict[str, bool] = {}
    failure_modes: list[str] = []

    for h in HORIZONS:
        hkey = f"H{h}"
        prior_name = strongest_prior(h)
        p_rows = prior_rows(prior_name, h)
        per_h[hkey] = []
        deltas = []
        m128_deltas = []
        for seed in SEEDS:
            name = f"v30_extgt_m512_h{h}_seed{seed}"
            payload = run_payload(512, h, seed)
            ref = run_payload(128, h, seed)
            if not payload:
                per_h[hkey].append({"seed": seed, "completed": False, "missing": True})
                continue
            rows = payload.get("test_item_rows", [])
            comp = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False)
            motion = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False, subset_key="v30_motion")
            auc = paired_bootstrap(rows, p_rows, "threshold_auc_endpoint_16_32_64_128", higher_better=True)
            boot[f"{name}_vs_{prior_name}_minFDE_K"] = comp
            boot[f"{name}_vs_{prior_name}_motion_minFDE_K"] = motion
            boot[f"{name}_vs_{prior_name}_threshold_auc"] = auc
            delta = float(comp.get("mean_delta") or 0.0)
            deltas.append(delta)
            m512_minfde = metric(payload, "all", "minFDE_K")
            m128_minfde = metric(ref, "all", "minFDE_K")
            m512_motion = metric(payload, "motion", "minFDE_K")
            m128_motion = metric(ref, "motion", "minFDE_K")
            m512_auc = metric(payload, "all", "threshold_auc_endpoint_16_32_64_128")
            m128_auc = metric(ref, "all", "threshold_auc_endpoint_16_32_64_128")
            m512_m128_delta = (m512_minfde - m128_minfde) if m512_minfde is not None and m128_minfde is not None else None
            if m512_m128_delta is not None:
                m128_deltas.append(m512_m128_delta)
            per_h[hkey].append(
                {
                    "seed": seed,
                    "completed": bool(payload.get("completed")),
                    "report_path": payload.get("_report_path"),
                    "reused_from_round1": bool(payload.get("steps") != 4000),
                    "steps": payload.get("steps"),
                    "batch_size": payload.get("batch_size"),
                    "grad_accum_steps": payload.get("grad_accum_steps"),
                    "effective_batch_size": payload.get("effective_batch_size"),
                    "train_loss_decreased": payload.get("train_loss_decreased"),
                    "point_valid_ratio_last": (payload.get("train_loss_last") or {}).get("point_valid_ratio"),
                    "point_encoder_activation_norm_last": (payload.get("train_loss_last") or {}).get("point_encoder_activation_norm"),
                    "delta_minFDE_vs_strongest_prior": delta,
                    "motion_delta_minFDE_vs_strongest_prior": motion.get("mean_delta"),
                    "threshold_auc_delta_vs_prior": auc.get("mean_delta"),
                    "m512_minFDE_K": m512_minfde,
                    "m128_same_seed_minFDE_K": m128_minfde,
                    "m512_minus_m128_minFDE_K": m512_m128_delta,
                    "m512_motion_minFDE_K": m512_motion,
                    "m128_same_seed_motion_minFDE_K": m128_motion,
                    "m512_minus_m128_motion_minFDE_K": (m512_motion - m128_motion) if m512_motion is not None and m128_motion is not None else None,
                    "m512_threshold_auc": m512_auc,
                    "m128_same_seed_threshold_auc": m128_auc,
                    "m512_minus_m128_threshold_auc": (m512_auc - m128_auc) if m512_auc is not None and m128_auc is not None else None,
                    "m512_MissRate@64": metric(payload, "all", "MissRate@64"),
                    "m512_MissRate@128": metric(payload, "all", "MissRate@128"),
                    "m128_same_seed_MissRate@64": metric(ref, "all", "MissRate@64"),
                    "m128_same_seed_MissRate@128": metric(ref, "all", "MissRate@128"),
                    "m512_relative_deformation_layout_error": metric(payload, "all", "relative_deformation_layout_error"),
                    "m128_same_seed_relative_deformation_layout_error": metric(ref, "all", "relative_deformation_layout_error"),
                }
            )
        decision_counts[f"m512_h{h}_positive_vs_prior_seed_count"] = sum(1 for x in per_h[hkey] if (x.get("delta_minFDE_vs_strongest_prior") or 0) > 0)
        beats_m128[f"H{h}"] = bool(m128_deltas and statistics.mean(m128_deltas) < 0)
        if m128_deltas and statistics.mean(m128_deltas) >= 0:
            failure_modes.append("pooling_bottleneck")
        if any((x.get("effective_batch_size") or 0) < 8 for x in per_h[hkey] if x.get("completed")):
            failure_modes.append("effective_batch_too_small")
        if any(not bool(x.get("train_loss_decreased")) for x in per_h[hkey] if x.get("completed")):
            failure_modes.append("insufficient_steps")

    positive_vs_prior_all = all(decision_counts.get(f"m512_h{h}_positive_vs_prior_seed_count", 0) >= 2 for h in HORIZONS)
    beats_2of3 = sum(1 for v in beats_m128.values() if v) >= 2
    if positive_vs_prior_all and beats_2of3:
        next_step = "run_m1024_smoke_then_pilot"
        failure_mode = "none"
        density_prelim = True
    elif positive_vs_prior_all:
        next_step = "fix_density_pooling_or_training"
        failure_mode = "mixed" if len(set(failure_modes)) > 1 else (failure_modes[0] if failure_modes else "metric_insensitive")
        density_prelim = False
    else:
        next_step = "fix_density_pooling_or_training"
        failure_mode = "mixed" if failure_modes else "cache_or_manifest_issue"
        density_prelim = False

    summary = {
        "summary_name": "stwm_ostf_v30_density_m512_pilot_summary",
        "generated_at_utc": utc_now(),
        "seeds": SEEDS,
        "horizons": HORIZONS,
        "per_horizon": per_h,
        "m512_vs_m128_beats_by_horizon": beats_m128,
        "positive_vs_prior_all_horizons": positive_vs_prior_all,
    }
    decision = {
        "decision_name": "stwm_ostf_v30_density_m512_pilot_decision",
        "generated_at_utc": utc_now(),
        **decision_counts,
        "m512_beats_m128_h32": beats_m128.get("H32", False),
        "m512_beats_m128_h64": beats_m128.get("H64", False),
        "m512_beats_m128_h96": beats_m128.get("H96", False),
        "density_scaling_positive_preliminary": density_prelim,
        "density_scaling_failure_mode": failure_mode,
        "next_step_choice": next_step,
        "semantic_not_tested_not_failed": True,
    }
    dump_json(SUMMARY, summary)
    dump_json(BOOT, {"bootstrap_name": "stwm_ostf_v30_density_m512_pilot_bootstrap", "generated_at_utc": utc_now(), "per_seed_bootstrap": boot})
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V30 Density M512 Pilot Decision",
        decision,
        [
            "m512_h32_positive_vs_prior_seed_count",
            "m512_h64_positive_vs_prior_seed_count",
            "m512_h96_positive_vs_prior_seed_count",
            "m512_beats_m128_h32",
            "m512_beats_m128_h64",
            "m512_beats_m128_h96",
            "density_scaling_positive_preliminary",
            "density_scaling_failure_mode",
            "next_step_choice",
        ],
    )
    print(DECISION.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
