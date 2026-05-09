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


V32_DIR = ROOT / "reports/stwm_ostf_v32_recurrent_field_runs"
V31_DIR = ROOT / "reports/stwm_ostf_v31_field_preserving_runs"
V30_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY_PATH = ROOT / "reports/stwm_ostf_v32_recurrent_field_pilot_summary_20260509.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v32_recurrent_field_pilot_bootstrap_20260509.json"
DECISION_PATH = ROOT / "reports/stwm_ostf_v32_recurrent_field_pilot_decision_20260509.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V32_RECURRENT_FIELD_PILOT_DECISION_20260509.md"

PILOT = [
    (128, 32),
    (128, 64),
    (128, 96),
    (512, 32),
    (512, 64),
    (512, 96),
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _metric(payload: dict[str, Any], metric: str, subset: str | None = None) -> float | None:
    if subset:
        val = payload.get("test_metrics", {}).get("subsets", {}).get(subset, {}).get(metric)
    else:
        val = payload.get("test_metrics", {}).get("all", {}).get(metric)
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _delta_lower_better(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b - a)


def _delta_higher_better(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def _v32_name(m: int, h: int) -> str:
    return f"v32_rf_m{m}_h{h}_seed42"


def _v31_name(m: int, h: int) -> str:
    return f"v31_field_m{m}_h{h}_seed42"


def _v30_name(m: int, h: int) -> str:
    return f"v30_extgt_m{m}_h{h}_seed42"


def _strongest_prior(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
    best_name = None
    best_metric: dict[str, Any] = {}
    best = 1e99
    for name, rec in payload.get("test_prior_metrics", {}).items():
        val = rec.get("all", {}).get("minFDE")
        if val is not None and float(val) < best:
            best = float(val)
            best_name = name
            best_metric = rec
    rows = payload.get("test_prior_item_rows", {}).get(best_name or "", [])
    return best_name, best_metric, rows


def _stats(vals: list[float]) -> dict[str, Any]:
    return {
        "count": len(vals),
        "mean": float(statistics.mean(vals)) if vals else None,
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None,
        "positive_count": int(sum(v > 0 for v in vals)),
    }


def _bootstrap(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str, higher: bool) -> dict[str, Any]:
    return paired_bootstrap(rows_a, rows_b, metric, higher_better=higher)


def main() -> int:
    runs: dict[str, Any] = {}
    missing = []
    boot: dict[str, Any] = {"generated_at_utc": utc_now(), "comparisons": {}}
    v32_vs_v30_m128 = []
    v32_vs_v31_m128 = []
    v32_vs_v30_m512 = []
    v32_vs_v31_m512 = []
    density_deltas = []
    for m, h in PILOT:
        name = _v32_name(m, h)
        run = _load(V32_DIR / f"{name}.json")
        if not run.get("completed"):
            missing.append(name)
            continue
        v30 = _load(V30_DIR / f"{_v30_name(m, h)}.json")
        v31 = _load(V31_DIR / f"{_v31_name(m, h)}.json")
        prior_name, prior_metric, prior_rows = _strongest_prior(run)
        d30 = _delta_lower_better(_metric(run, "minFDE_K"), _metric(v30, "minFDE_K"))
        d31 = _delta_lower_better(_metric(run, "minFDE_K"), _metric(v31, "minFDE_K"))
        if m == 128 and d30 is not None:
            v32_vs_v30_m128.append(d30)
        if m == 128 and d31 is not None:
            v32_vs_v31_m128.append(d31)
        if m == 512 and d30 is not None:
            v32_vs_v30_m512.append(d30)
        if m == 512 and d31 is not None:
            v32_vs_v31_m512.append(d31)
        row = {
            "report_path": str((V32_DIR / f"{name}.json").relative_to(ROOT)),
            "checkpoint_path": run.get("checkpoint_path"),
            "M": m,
            "H": h,
            "seed": 42,
            "duration_seconds": run.get("duration_seconds"),
            "gpu_peak_memory_mib": run.get("gpu_peak_memory_mib"),
            "batch_size": run.get("batch_size"),
            "effective_batch_size": run.get("effective_batch_size"),
            "train_loss_decreased": run.get("train_loss_decreased"),
            "test_all": run.get("test_metrics", {}).get("all"),
            "test_motion": run.get("test_metrics", {}).get("subsets", {}).get("motion"),
            "strongest_prior": prior_name,
            "strongest_prior_all": prior_metric.get("all"),
            "v30_same_seed_all": v30.get("test_metrics", {}).get("all"),
            "v31_same_seed_all": v31.get("test_metrics", {}).get("all"),
            "v32_minus_v30_minFDE_K_positive": d30,
            "v32_minus_v31_minFDE_K_positive": d31,
            "v32_minus_prior_minFDE_K_positive": _delta_lower_better(_metric(run, "minFDE_K"), prior_metric.get("all", {}).get("minFDE")),
            "motion_minFDE_K": _metric(run, "minFDE_K", "motion"),
            "threshold_auc_endpoint_16_32_64_128": _metric(run, "threshold_auc_endpoint_16_32_64_128"),
            "MissRate@64": _metric(run, "MissRate@64"),
            "MissRate@128": _metric(run, "MissRate@128"),
            "relative_deformation_layout_error": _metric(run, "relative_deformation_layout_error"),
            "visibility_F1": _metric(run, "visibility_F1"),
            "recurrent_diagnostics": {
                "field_state_norm": run.get("train_loss_last", {}).get("field_state_norm"),
                "recurrent_delta_norm": run.get("train_loss_last", {}).get("recurrent_delta_norm"),
                "global_motion_norm": run.get("train_loss_last", {}).get("global_motion_norm"),
                "recurrent_loop_steps": run.get("train_loss_last", {}).get("recurrent_loop_steps"),
                "global_motion_prior_active": run.get("train_loss_last", {}).get("global_motion_prior_active"),
            },
        }
        runs[name] = row
        for other_name, other in [("v30", v30), ("v31", v31)]:
            if other.get("test_item_rows"):
                for metric, higher in [
                    ("minFDE_K", False),
                    ("threshold_auc_endpoint_16_32_64_128", True),
                    ("MissRate@64", False),
                    ("MissRate@128", False),
                    ("relative_deformation_layout_error", False),
                    ("visibility_F1", True),
                ]:
                    boot["comparisons"][f"{name}_vs_{other_name}_{metric}"] = _bootstrap(
                        run.get("test_item_rows", []), other.get("test_item_rows", []), metric, higher
                    )
        if prior_rows:
            for metric, higher in [
                ("minFDE_K", False),
                ("threshold_auc_endpoint_16_32_64_128", True),
                ("MissRate@64", False),
                ("MissRate@128", False),
            ]:
                boot["comparisons"][f"{name}_vs_{prior_name}_{metric}"] = _bootstrap(
                    run.get("test_item_rows", []), prior_rows, metric, higher
                )

    m512_vs_m128: dict[str, Any] = {}
    for h in (32, 64, 96):
        r128 = _load(V32_DIR / f"{_v32_name(128, h)}.json")
        r512 = _load(V32_DIR / f"{_v32_name(512, h)}.json")
        d = _delta_lower_better(_metric(r512, "minFDE_K"), _metric(r128, "minFDE_K"))
        if d is not None:
            density_deltas.append(d)
        m512_vs_m128[f"H{h}"] = {
            "m512_minFDE_K": _metric(r512, "minFDE_K"),
            "m128_minFDE_K": _metric(r128, "minFDE_K"),
            "m512_minus_m128_positive": d,
        }
        if r128.get("test_item_rows") and r512.get("test_item_rows"):
            boot["comparisons"][f"v32_m512_vs_m128_H{h}_minFDE_K"] = _bootstrap(
                r512.get("test_item_rows", []), r128.get("test_item_rows", []), "minFDE_K", False
            )

    smoke = _load(ROOT / "reports/stwm_ostf_v32_recurrent_field_smoke_summary_20260509.json")
    v32_m128_beats_v30 = sum(d > 0 for d in v32_vs_v30_m128)
    v32_m128_beats_v31 = sum(d > 0 for d in v32_vs_v31_m128)
    density_positive = sum(d > 0 for d in density_deltas)
    recurrent_positive = bool(v32_m128_beats_v30 >= 2 and v32_m128_beats_v31 >= 2)
    density_recovered = bool(density_positive >= 2)
    # "Matches V30" is intentionally stricter than "has one near-tie"; the
    # V32 decision rule only allows multiseed if V32 is broadly competitive
    # with the established V30 baseline while improving density behavior.
    v32_m128_matches_v30 = sum(d >= -2.0 for d in v32_vs_v30_m128) >= 2
    if recurrent_positive:
        next_step = "run_v32_m128_m512_multiseed"
    elif v32_m128_matches_v30 and density_recovered:
        next_step = "run_v32_m128_m512_multiseed"
    elif not missing and (v32_m128_beats_v30 or v32_m128_beats_v31):
        next_step = "ablate_v32_global_motion_prior_and_recurrent_field"
    elif missing:
        next_step = "improve_v32_efficiency_or_training"
    else:
        next_step = "keep_v30_m128_main_move_to_semantic_identity_targets"

    summary = {
        "generated_at_utc": utc_now(),
        "expected_run_count": len(PILOT),
        "completed_run_count": len(runs),
        "missing_runs": missing,
        "runs": runs,
        "v32_vs_v30_m128_delta_stats": _stats(v32_vs_v30_m128),
        "v32_vs_v31_m128_delta_stats": _stats(v32_vs_v31_m128),
        "v32_vs_v30_m512_delta_stats": _stats(v32_vs_v30_m512),
        "v32_vs_v31_m512_delta_stats": _stats(v32_vs_v31_m512),
        "v32_m512_vs_m128": m512_vs_m128,
        "v32_m512_vs_m128_delta_stats": _stats(density_deltas),
    }
    decision = {
        "generated_at_utc": utc_now(),
        "v32_smoke_passed": bool(smoke.get("smoke_passed")),
        "v32_m128_beats_v30_m128_h32": bool(runs.get("v32_rf_m128_h32_seed42", {}).get("v32_minus_v30_minFDE_K_positive", 0) > 0),
        "v32_m128_beats_v30_m128_h64": bool(runs.get("v32_rf_m128_h64_seed42", {}).get("v32_minus_v30_minFDE_K_positive", 0) > 0),
        "v32_m128_beats_v30_m128_h96": bool(runs.get("v32_rf_m128_h96_seed42", {}).get("v32_minus_v30_minFDE_K_positive", 0) > 0),
        "v32_m128_beats_v31_m128_h32": bool(runs.get("v32_rf_m128_h32_seed42", {}).get("v32_minus_v31_minFDE_K_positive", 0) > 0),
        "v32_m128_beats_v31_m128_h64": bool(runs.get("v32_rf_m128_h64_seed42", {}).get("v32_minus_v31_minFDE_K_positive", 0) > 0),
        "v32_m128_beats_v31_m128_h96": bool(runs.get("v32_rf_m128_h96_seed42", {}).get("v32_minus_v31_minFDE_K_positive", 0) > 0),
        "v32_m512_beats_v30_m512_h32": bool(runs.get("v32_rf_m512_h32_seed42", {}).get("v32_minus_v30_minFDE_K_positive", 0) > 0),
        "v32_m512_beats_v30_m512_h64": bool(runs.get("v32_rf_m512_h64_seed42", {}).get("v32_minus_v30_minFDE_K_positive", 0) > 0),
        "v32_m512_beats_v30_m512_h96": bool(runs.get("v32_rf_m512_h96_seed42", {}).get("v32_minus_v30_minFDE_K_positive", 0) > 0),
        "v32_m512_beats_v31_m512_h32": bool(runs.get("v32_rf_m512_h32_seed42", {}).get("v32_minus_v31_minFDE_K_positive", 0) > 0),
        "v32_m512_beats_v31_m512_h64": bool(runs.get("v32_rf_m512_h64_seed42", {}).get("v32_minus_v31_minFDE_K_positive", 0) > 0),
        "v32_m512_beats_v31_m512_h96": bool(runs.get("v32_rf_m512_h96_seed42", {}).get("v32_minus_v31_minFDE_K_positive", 0) > 0),
        "v32_m512_beats_v32_m128": {h: row.get("m512_minus_m128_positive") for h, row in m512_vs_m128.items()},
        "recurrent_field_dynamics_positive": recurrent_positive,
        "v32_m128_matches_v30_for_multiseed_gate": bool(v32_m128_matches_v30),
        "global_motion_prior_needed": "not_ablation_tested_in_v32_seed42_pilot",
        "density_scaling_recovered_with_v32": density_recovered,
        "semantic_not_tested_not_failed": True,
        "recommended_next_step": next_step,
    }
    dump_json(SUMMARY_PATH, summary)
    dump_json(BOOT_PATH, boot)
    dump_json(DECISION_PATH, decision)
    write_doc(
        DOC_PATH,
        "STWM OSTF V32 Recurrent Field Pilot Decision",
        decision,
        list(decision.keys()),
    )
    print(DECISION_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
