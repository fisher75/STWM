#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


V31_DIR = ROOT / "reports/stwm_ostf_v31_field_preserving_runs"
V30_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY_PATH = ROOT / "reports/stwm_ostf_v31_field_preserving_pilot_summary_20260508.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v31_field_preserving_pilot_bootstrap_20260508.json"
DECISION_PATH = ROOT / "reports/stwm_ostf_v31_field_preserving_pilot_decision_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V31_FIELD_PRESERVING_PILOT_DECISION_20260508.md"


PILOT_NAMES = [
    "v31_field_m128_h32_seed42",
    "v31_field_m128_h64_seed42",
    "v31_field_m128_h96_seed42",
    "v31_field_m512_h32_seed42",
    "v31_field_m512_h64_seed42",
    "v31_field_m512_h96_seed42",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _v30_name(m: int, h: int) -> str:
    return f"v30_extgt_m{m}_h{h}_seed42"


def _metric(payload: dict[str, Any], metric: str, subset: str | None = None) -> float | None:
    if subset:
        val = payload.get("test_metrics", {}).get("subsets", {}).get(subset, {}).get(metric)
    else:
        val = payload.get("test_metrics", {}).get("all", {}).get(metric)
    return float(val) if val is not None else None


def _lt(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and a < b


def _strongest_prior_name(payload: dict[str, Any]) -> str | None:
    prior_metrics = payload.get("test_prior_metrics", {})
    best_name = None
    best = 1e99
    for name, rec in prior_metrics.items():
        val = rec.get("all", {}).get("minFDE")
        if val is not None and float(val) < best:
            best = float(val)
            best_name = name
    return best_name


def main() -> int:
    runs: dict[str, Any] = {}
    boot: dict[str, Any] = {"generated_at_utc": utc_now(), "comparisons": {}}
    missing = []
    for name in PILOT_NAMES:
        path = V31_DIR / f"{name}.json"
        payload = _load(path)
        if not payload:
            missing.append(name)
            continue
        m = int(payload.get("m_points") or 0)
        h = int(payload.get("horizon") or 0)
        v30_payload = _load(V30_DIR / f"{_v30_name(m, h)}.json")
        strongest_prior = _strongest_prior_name(payload)
        strongest_prior_metrics = payload.get("test_prior_metrics", {}).get(strongest_prior or "", {})
        runs[name] = {
            "report_path": str(path.relative_to(ROOT)),
            "checkpoint_path": payload.get("checkpoint_path"),
            "m_points": m,
            "horizon": h,
            "duration_seconds": payload.get("duration_seconds"),
            "train_loss_decreased": payload.get("train_loss_decreased"),
            "field_preserving_rollout": payload.get("field_preserving_rollout"),
            "test_all": payload.get("test_metrics", {}).get("all"),
            "test_motion": payload.get("test_metrics", {}).get("subsets", {}).get("motion"),
            "strongest_prior": strongest_prior,
            "strongest_prior_all": strongest_prior_metrics.get("all"),
            "v30_same_seed_all": v30_payload.get("test_metrics", {}).get("all"),
            "v30_same_seed_motion": v30_payload.get("test_metrics", {}).get("subsets", {}).get("motion"),
            "v31_beats_v30_minFDE_K": _lt(_metric(payload, "minFDE_K"), _metric(v30_payload, "minFDE_K")),
            "v31_beats_v30_motion_minFDE_K": _lt(_metric(payload, "minFDE_K", "motion"), _metric(v30_payload, "minFDE_K", "motion")),
            "v31_beats_strongest_prior_minFDE_K": _lt(_metric(payload, "minFDE_K"), strongest_prior_metrics.get("all", {}).get("minFDE")),
            "v31_beats_strongest_prior_motion_minFDE_K": _lt(_metric(payload, "minFDE_K", "motion"), strongest_prior_metrics.get("subsets", {}).get("motion", {}).get("minFDE")),
        }
        if v30_payload.get("test_item_rows"):
            for metric, higher in [
                ("minFDE_K", False),
                ("threshold_auc_endpoint_16_32_64_128", True),
                ("MissRate@64", False),
                ("MissRate@128", False),
                ("relative_deformation_layout_error", False),
                ("visibility_F1", True),
            ]:
                boot["comparisons"][f"{name}_vs_v30_{metric}"] = paired_bootstrap(
                    payload.get("test_item_rows", []), v30_payload.get("test_item_rows", []), metric, higher_better=higher
                )
        if strongest_prior:
            prior_rows = payload.get("test_prior_item_rows", {}).get(strongest_prior, [])
            for metric, higher in [
                ("minFDE_K", False),
                ("threshold_auc_endpoint_16_32_64_128", True),
                ("MissRate@64", False),
                ("MissRate@128", False),
            ]:
                boot["comparisons"][f"{name}_vs_{strongest_prior}_{metric}"] = paired_bootstrap(
                    payload.get("test_item_rows", []), prior_rows, metric, higher_better=higher
                )
    v31_m128_beats = {}
    v31_m512_beats = {}
    for h in (32, 64, 96):
        r128 = runs.get(f"v31_field_m128_h{h}_seed42", {})
        r512 = runs.get(f"v31_field_m512_h{h}_seed42", {})
        v31_m128_beats[f"h{h}"] = bool(r128.get("v31_beats_v30_minFDE_K"))
        v31_m512_beats[f"h{h}"] = bool(r512.get("v31_beats_v30_minFDE_K"))
    m512_vs_m128 = {}
    for h in (32, 64, 96):
        r128 = _load(V31_DIR / f"v31_field_m128_h{h}_seed42.json")
        r512 = _load(V31_DIR / f"v31_field_m512_h{h}_seed42.json")
        m512_vs_m128[f"h{h}"] = _lt(_metric(r512, "minFDE_K"), _metric(r128, "minFDE_K"))
    m128_beats_count = sum(bool(v) for v in v31_m128_beats.values())
    m512_beats_m128_count = sum(bool(v) for v in m512_vs_m128.values())
    field_positive = m128_beats_count >= 2 or (m128_beats_count >= 1 and m512_beats_m128_count >= 2)
    density_recovered = m512_beats_m128_count >= 2
    if m128_beats_count >= 2:
        next_step = "run_v31_m128_m512_multiseed"
    elif density_recovered:
        next_step = "run_v31_m128_m512_multiseed"
    elif not missing and any(bool(v) for v in v31_m128_beats.values()):
        next_step = "improve_v31_field_interaction"
    elif not missing:
        next_step = "keep_v30_m128_main_move_to_semantic_identity_targets"
    else:
        next_step = "improve_v31_field_interaction"
    summary = {
        "generated_at_utc": utc_now(),
        "expected_run_count": len(PILOT_NAMES),
        "completed_run_count": len(runs),
        "missing_runs": missing,
        "runs": runs,
    }
    decision = {
        "generated_at_utc": utc_now(),
        "v31_field_preserving_smoke_passed": bool(_load(ROOT / "reports/stwm_ostf_v31_field_preserving_smoke_summary_20260508.json").get("smoke_passed")),
        "v31_m128_beats_v30_m128_h32": v31_m128_beats.get("h32", False),
        "v31_m128_beats_v30_m128_h64": v31_m128_beats.get("h64", False),
        "v31_m128_beats_v30_m128_h96": v31_m128_beats.get("h96", False),
        "v31_m512_beats_v30_m512_h32": v31_m512_beats.get("h32", False),
        "v31_m512_beats_v30_m512_h64": v31_m512_beats.get("h64", False),
        "v31_m512_beats_v30_m512_h96": v31_m512_beats.get("h96", False),
        "v31_m512_beats_v31_m128": m512_vs_m128,
        "field_preserving_rollout_positive": bool(field_positive),
        "density_scaling_recovered_with_v31": bool(density_recovered),
        "semantic_not_tested_not_failed": True,
        "field_tokens_load_bearing": "architecture_preserved; explicit ablation not part of completed pilot" if not any("no_field" in n for n in runs) else "see ablation rows",
        "recommended_next_step": next_step,
    }
    dump_json(SUMMARY_PATH, summary)
    dump_json(BOOT_PATH, boot)
    dump_json(DECISION_PATH, decision)
    write_doc(
        DOC_PATH,
        "STWM OSTF V31 Field-Preserving Pilot Decision",
        decision,
        list(decision.keys()),
    )
    print(DECISION_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
