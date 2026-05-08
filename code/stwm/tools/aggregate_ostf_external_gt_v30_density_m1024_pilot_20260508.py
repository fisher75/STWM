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
HORIZONS = [32, 64, 96]
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY = ROOT / "reports/stwm_ostf_v30_density_m1024_pilot_summary_20260508.json"
BOOT = ROOT / "reports/stwm_ostf_v30_density_m1024_pilot_bootstrap_20260508.json"
DECISION = ROOT / "reports/stwm_ostf_v30_density_m1024_pilot_decision_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_M1024_PILOT_DECISION_20260508.md"


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
    return [r for r in prior.get("test_item_rows_by_prior", {}).get(prior_name, []) if int(r.get("H", 0)) == horizon and int(r.get("M", 0)) == 1024]


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
    beats_m512: dict[str, bool] = {}
    positive_prior_counts: dict[str, int] = {}
    for h in HORIZONS:
        hkey = f"H{h}"
        prior_name = strongest_prior(h)
        p_rows = prior_rows(prior_name, h)
        per_h[hkey] = []
        m512_deltas = []
        for seed in SEEDS:
            name = f"v30_extgt_m1024_h{h}_seed{seed}"
            payload = run_payload(1024, h, seed)
            ref = run_payload(512, h, seed)
            if payload is None:
                per_h[hkey].append({"seed": seed, "completed": False, "missing": True})
                continue
            rows = payload.get("test_item_rows", [])
            comp = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False)
            motion = paired_bootstrap(rows, p_rows, "minFDE_K", higher_better=False, subset_key="v30_motion")
            boot[f"{name}_vs_{prior_name}_minFDE_K"] = comp
            boot[f"{name}_vs_{prior_name}_motion_minFDE_K"] = motion
            m1024 = metric(payload)
            m512 = metric(ref)
            delta512 = (m1024 - m512) if m1024 is not None and m512 is not None else None
            if delta512 is not None:
                m512_deltas.append(delta512)
            per_h[hkey].append(
                {
                    "seed": seed,
                    "completed": bool(payload.get("completed")),
                    "report_path": payload.get("_report_path"),
                    "delta_minFDE_vs_strongest_prior": comp.get("mean_delta"),
                    "motion_delta_minFDE_vs_strongest_prior": motion.get("mean_delta"),
                    "m1024_minFDE_K": m1024,
                    "m512_same_seed_minFDE_K": m512,
                    "m1024_minus_m512_minFDE_K": delta512,
                    "m1024_motion_minFDE_K": metric(payload, "motion"),
                    "m512_same_seed_motion_minFDE_K": metric(ref, "motion"),
                    "batch_size": payload.get("batch_size"),
                    "grad_accum_steps": payload.get("grad_accum_steps"),
                    "effective_batch_size": payload.get("effective_batch_size"),
                    "train_loss_decreased": payload.get("train_loss_decreased"),
                    "point_valid_ratio_last": (payload.get("train_loss_last") or {}).get("point_valid_ratio"),
                }
            )
        positive_prior_counts[f"m1024_h{h}_positive_vs_prior_seed_count"] = sum(1 for x in per_h[hkey] if (x.get("delta_minFDE_vs_strongest_prior") or 0) > 0)
        beats_m512[hkey] = bool(m512_deltas and statistics.mean(m512_deltas) < 0)
    m1024_beats_m512 = sum(1 for x in beats_m512.values() if x) >= 2
    summary = {
        "summary_name": "stwm_ostf_v30_density_m1024_pilot_summary",
        "generated_at_utc": utc_now(),
        "per_horizon": per_h,
        "m1024_beats_m512_by_horizon": beats_m512,
        "semantic_not_tested_not_failed": True,
    }
    decision = {
        "decision_name": "stwm_ostf_v30_density_m1024_pilot_decision",
        "generated_at_utc": utc_now(),
        **positive_prior_counts,
        "m1024_beats_m512_h32": beats_m512.get("H32", False),
        "m1024_beats_m512_h64": beats_m512.get("H64", False),
        "m1024_beats_m512_h96": beats_m512.get("H96", False),
        "m1024_beats_m512": m1024_beats_m512,
        "semantic_not_tested_not_failed": True,
        "next_step_choice": "run_m512_m1024_full_multiseed" if m1024_beats_m512 else "fix_density_aware_pooling",
    }
    dump_json(SUMMARY, summary)
    dump_json(BOOT, {"bootstrap_name": "stwm_ostf_v30_density_m1024_pilot_bootstrap", "generated_at_utc": utc_now(), "per_seed_bootstrap": boot})
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V30 Density M1024 Pilot Decision",
        decision,
        [
            "m1024_h32_positive_vs_prior_seed_count",
            "m1024_h64_positive_vs_prior_seed_count",
            "m1024_h96_positive_vs_prior_seed_count",
            "m1024_beats_m512_h32",
            "m1024_beats_m512_h64",
            "m1024_beats_m512_h96",
            "m1024_beats_m512",
            "next_step_choice",
        ],
    )
    print(DECISION.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
