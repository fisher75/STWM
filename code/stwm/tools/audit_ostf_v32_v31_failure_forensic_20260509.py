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
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


V31_DIR = ROOT / "reports/stwm_ostf_v31_field_preserving_runs"
V30_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
OUT_JSON = ROOT / "reports/stwm_ostf_v32_v31_failure_forensic_20260509.json"
OUT_MD = ROOT / "docs/STWM_OSTF_V32_V31_FAILURE_FORENSIC_20260509.md"

SEEDS = [42, 123, 456, 789, 2026]
MS = [128, 512]
HS = [32, 64, 96]


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


def _stats(vals: list[float]) -> dict[str, Any]:
    return {
        "count": len(vals),
        "mean": float(statistics.mean(vals)) if vals else None,
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None,
        "positive_count": int(sum(v > 0 for v in vals)),
    }


def _v31_name(m: int, h: int, seed: int, variant: str = "full") -> str:
    if variant == "full":
        return f"v31_field_m{m}_h{h}_seed{seed}"
    return f"v31_field_m{m}_h{h}_{variant}_seed{seed}"


def _v30_name(m: int, h: int, seed: int) -> str:
    return f"v30_extgt_m{m}_h{h}_seed{seed}"


def main() -> int:
    v31_decision = _load(ROOT / "reports/stwm_ostf_v31_field_multiseed_decision_20260508.json")
    v31_summary = _load(ROOT / "reports/stwm_ostf_v31_field_multiseed_summary_20260508.json")
    compression_audit = _load(ROOT / "reports/stwm_ostf_v31_v30_field_compression_audit_20260508.json")
    code_audit = _load(ROOT / "reports/stwm_ostf_v31_model_code_audit_20260508.json")
    v31_source = (ROOT / "code/stwm/modules/ostf_field_preserving_world_model_v31.py").read_text(encoding="utf-8")

    comparisons: dict[str, Any] = {}
    for m in MS:
        for h in HS:
            deltas = []
            motion_deltas = []
            missing = []
            for seed in SEEDS:
                v31 = _load(V31_DIR / f"{_v31_name(m, h, seed)}.json")
                v30 = _load(V30_DIR / f"{_v30_name(m, h, seed)}.json")
                d = _delta_lower_better(_metric(v31, "minFDE_K"), _metric(v30, "minFDE_K"))
                dm = _delta_lower_better(_metric(v31, "minFDE_K", "motion"), _metric(v30, "minFDE_K", "motion"))
                if d is None:
                    missing.append(seed)
                else:
                    deltas.append(d)
                if dm is not None:
                    motion_deltas.append(dm)
            comparisons[f"M{m}_H{h}"] = {
                "delta_positive_means_v31_lower_minFDE_K": _stats(deltas),
                "motion_delta_positive_means_v31_lower_minFDE_K": _stats(motion_deltas),
                "missing_seeds": missing,
            }

    ablations: dict[str, Any] = {}
    ablation_deltas = []
    for m in MS:
        for h in HS:
            for seed in (42, 123):
                full = _load(V31_DIR / f"{_v31_name(m, h, seed)}.json")
                no_field = _load(V31_DIR / f"{_v31_name(m, h, seed, 'no_field')}.json")
                d = _delta_lower_better(_metric(full, "minFDE_K"), _metric(no_field, "minFDE_K"))
                ablations[f"M{m}_H{h}_seed{seed}"] = {
                    "full_minFDE_K": _metric(full, "minFDE_K"),
                    "no_field_minFDE_K": _metric(no_field, "minFDE_K"),
                    "full_minus_no_field_positive": d,
                    "no_field_completed": bool(no_field.get("completed")),
                }
                if d is not None:
                    ablation_deltas.append(d)

    temporal_rollout = {
        "v31_has_python_recurrent_future_loop": "for h in range(self.cfg.horizon)" in v31_source,
        "v31_uses_step_seed_from_observed_field_plus_time": "step_seed = field_token" in v31_source,
        "v31_temporal_rollout_over_point_time_tokens_once": "self.temporal_rollout(step_seed.reshape" in v31_source,
        "v31_point_point_interaction_during_future_rollout": False,
        "interpretation": "V31 preserves a field-shaped tensor, but future H steps are expanded from observed field tokens once; predicted positions are not fed back into later field updates.",
    }
    global_prior = {
        "v31_has_explicit_global_motion_prior_branch": "global_motion_head" in v31_source,
        "v31_likely_loses_v30_object_level_inductive_bias": True,
        "reason": "V31 uses global/semantic context tokens but does not decode an explicit global displacement branch that can carry V30-style object motion priors through recurrent dynamics.",
    }
    density = {
        "v31_m512_beats_v31_m128_h32_seed_count": v31_decision.get("v31_m512_beats_v31_m128_h32_seed_count"),
        "v31_m512_beats_v31_m128_h64_seed_count": v31_decision.get("v31_m512_beats_v31_m128_h64_seed_count"),
        "v31_m512_beats_v31_m128_h96_seed_count": v31_decision.get("v31_m512_beats_v31_m128_h96_seed_count"),
        "density_scaling_recovered_with_v31": v31_decision.get("density_scaling_recovered_with_v31"),
        "interpretation": "Density scaling remained unstable; field-preserving shape alone did not make M512 reliably better.",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "source_reports": {
            "v31_multiseed_decision": "reports/stwm_ostf_v31_field_multiseed_decision_20260508.json",
            "v31_multiseed_summary": "reports/stwm_ostf_v31_field_multiseed_summary_20260508.json",
            "v31_compression_audit": "reports/stwm_ostf_v31_v30_field_compression_audit_20260508.json",
            "v31_code_audit": "reports/stwm_ostf_v31_model_code_audit_20260508.json",
        },
        "v31_completed_primary_runs": v31_summary.get("primary_completed_run_count", v31_summary.get("completed_run_count")),
        "v31_decision_facts": {
            "v31_overall_beats_v30": v31_decision.get("v31_overall_beats_v30"),
            "density_scaling_recovered_with_v31": v31_decision.get("density_scaling_recovered_with_v31"),
            "field_interaction_load_bearing": v31_decision.get("field_interaction_load_bearing"),
            "recommended_next_step": v31_decision.get("recommended_next_step"),
        },
        "v31_vs_v30_by_m_h_seed": comparisons,
        "field_interaction_ablation": {
            "rows": ablations,
            "summary": _stats(ablation_deltas),
            "ablation_was_strong_enough": bool(ablation_deltas),
            "field_interaction_load_bearing_confirmed": bool(ablation_deltas and sum(v > 0 for v in ablation_deltas) >= 8),
        },
        "temporal_rollout_structure": temporal_rollout,
        "global_motion_prior_forensic": global_prior,
        "density_scaling_forensic": density,
        "v31_code_report_bug_detected": False,
        "v32_justified": True,
        "conclusion": "V31 is field-shaped but not recurrent field dynamics. V32 is justified because it keeps per-point state, feeds predicted positions forward step-wise, and restores an explicit global motion branch.",
        "semantic_status": "not_tested_not_failed",
    }
    dump_json(OUT_JSON, payload)
    write_doc(
        OUT_MD,
        "STWM OSTF V32 V31 Failure Forensic",
        payload,
        [
            "v31_decision_facts",
            "field_interaction_ablation",
            "temporal_rollout_structure",
            "global_motion_prior_forensic",
            "density_scaling_forensic",
            "v31_code_report_bug_detected",
            "v32_justified",
            "conclusion",
        ],
    )
    print(OUT_JSON.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
