#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SUMMARY = ROOT / "reports/stwm_ostf_v30_density_pooling_pilot_summary_20260508.json"
BOOT = ROOT / "reports/stwm_ostf_v30_density_pooling_pilot_bootstrap_20260508.json"
DECISION = ROOT / "reports/stwm_ostf_v30_density_pooling_pilot_decision_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_POOLING_PILOT_DECISION_20260508.md"
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
MODES = ("mean", "moments", "induced_attention", "hybrid_moments_attention")
NEW_MODES = ("moments", "induced_attention", "hybrid_moments_attention")
MS = (512, 1024)
HS = (32, 64, 96)
SEEDS = (42, 123)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def exp_name(m: int, h: int, seed: int, mode: str) -> str:
    if mode == "mean":
        return f"v30_extgt_m{m}_h{h}_seed{seed}"
    return f"v30_extgt_density_pool_{mode}_m{m}_h{h}_seed{seed}"


def report_for(m: int, h: int, seed: int, mode: str) -> dict[str, Any]:
    name = exp_name(m, h, seed, mode)
    path = RUN_DIR / f"{name}.json"
    report = load(path)
    allm = report.get("test_metrics", {}).get("all", {})
    motion = report.get("test_metrics", {}).get("subsets", {}).get("motion", {})
    return {
        "experiment_name": name,
        "report_path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "completed": bool(report.get("completed")),
        "M": m,
        "H": h,
        "seed": seed,
        "pooling_mode": mode,
        "minFDE_K": allm.get("minFDE_K"),
        "motion_minFDE_K": motion.get("minFDE_K"),
        "threshold_auc_endpoint_16_32_64_128": allm.get("threshold_auc_endpoint_16_32_64_128"),
        "relative_deformation_layout_error": allm.get("relative_deformation_layout_error"),
        "visibility_F1": allm.get("visibility_F1"),
        "train_loss_decreased": report.get("train_loss_decreased"),
        "effective_batch_size": report.get("effective_batch_size"),
        "batch_size": report.get("batch_size"),
        "grad_accum_steps": report.get("grad_accum_steps"),
        "density_attention_entropy": report.get("train_loss_last", {}).get("density_attention_entropy"),
        "object_token_norm": report.get("train_loss_last", {}).get("object_token_norm"),
        "test_item_rows": report.get("test_item_rows", []),
    }


def better(a: dict[str, Any], b: dict[str, Any], metric: str = "minFDE_K") -> bool:
    av, bv = a.get(metric), b.get(metric)
    return av is not None and bv is not None and float(av) < float(bv)


def mean(vals: list[float]) -> float | None:
    clean = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def main() -> int:
    rows = [report_for(m, h, seed, mode) for m in MS for h in HS for seed in SEEDS for mode in MODES]
    by = {(r["M"], r["H"], r["seed"], r["pooling_mode"]): r for r in rows}
    comparisons = []
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for m in MS:
        for h in HS:
            mode_scores = {}
            for mode in MODES:
                vals = [by[(m, h, seed, mode)].get("minFDE_K") for seed in SEEDS if by[(m, h, seed, mode)].get("completed")]
                mode_scores[mode] = mean(vals)
            best_mode = min((k for k, v in mode_scores.items() if v is not None), key=lambda k: mode_scores[k], default=None)
            best[(m, h)] = {"mode": best_mode, "mode_scores_minFDE_K": mode_scores}
            for seed in SEEDS:
                mean_row = by[(m, h, seed, "mean")]
                for mode in NEW_MODES:
                    mode_row = by[(m, h, seed, mode)]
                    comparisons.append(
                        {
                            "M": m,
                            "H": h,
                            "seed": seed,
                            "pooling_mode": mode,
                            "mode_completed": mode_row["completed"],
                            "mean_completed": mean_row["completed"],
                            "delta_minFDE_K_vs_mean": (float(mean_row["minFDE_K"]) - float(mode_row["minFDE_K"])) if mean_row.get("minFDE_K") is not None and mode_row.get("minFDE_K") is not None else None,
                            "delta_motion_minFDE_K_vs_mean": (float(mean_row["motion_minFDE_K"]) - float(mode_row["motion_minFDE_K"])) if mean_row.get("motion_minFDE_K") is not None and mode_row.get("motion_minFDE_K") is not None else None,
                            "delta_threshold_auc_vs_mean": (float(mode_row["threshold_auc_endpoint_16_32_64_128"]) - float(mean_row["threshold_auc_endpoint_16_32_64_128"])) if mean_row.get("threshold_auc_endpoint_16_32_64_128") is not None and mode_row.get("threshold_auc_endpoint_16_32_64_128") is not None else None,
                        }
                    )
    m512_beats_m128 = {}
    m1024_beats_m512 = {}
    for h in HS:
        m512_mode = best[(512, h)]["mode"]
        m1024_mode = best[(1024, h)]["mode"]
        m512_wins = []
        m1024_wins = []
        for seed in SEEDS:
            m128 = load(RUN_DIR / f"v30_extgt_m128_h{h}_seed{seed}.json")
            m128_val = m128.get("test_metrics", {}).get("all", {}).get("minFDE_K")
            if m512_mode:
                m512_wins.append(better(by[(512, h, seed, m512_mode)], {"minFDE_K": m128_val}))
            if m1024_mode and m512_mode:
                m1024_wins.append(better(by[(1024, h, seed, m1024_mode)], by[(512, h, seed, m512_mode)]))
        m512_beats_m128[f"H{h}"] = bool(m512_wins and sum(m512_wins) >= max(1, len(m512_wins) // 2 + 1))
        m1024_beats_m512[f"H{h}"] = bool(m1024_wins and sum(m1024_wins) >= max(1, len(m1024_wins) // 2 + 1))
    boot = {"bootstrap_name": "stwm_ostf_v30_density_pooling_pilot_bootstrap", "generated_at_utc": utc_now(), "comparisons": {}}
    for m in MS:
        for h in HS:
            mode = best[(m, h)]["mode"]
            if not mode or mode == "mean":
                continue
            best_rows = []
            mean_rows = []
            for seed in SEEDS:
                best_rows.extend(by[(m, h, seed, mode)].get("test_item_rows", []))
                mean_rows.extend(by[(m, h, seed, "mean")].get("test_item_rows", []))
            boot["comparisons"][f"M{m}_H{h}_{mode}_vs_mean_minFDE_K"] = paired_bootstrap(best_rows, mean_rows, "minFDE_K", higher_better=False)
            boot["comparisons"][f"M{m}_H{h}_{mode}_vs_mean_threshold_auc"] = paired_bootstrap(best_rows, mean_rows, "threshold_auc_endpoint_16_32_64_128", higher_better=True)
    completed = [r for r in rows if r["completed"]]
    payload = {
        "summary_name": "stwm_ostf_v30_density_pooling_pilot_summary",
        "generated_at_utc": utc_now(),
        "expected_run_count_including_reused_mean": len(rows),
        "completed_run_count_including_reused_mean": len(completed),
        "missing_or_failed": [{k: r[k] for k in ("experiment_name", "report_path", "M", "H", "seed", "pooling_mode", "completed")} for r in rows if not r["completed"]],
        "rows": [{k: v for k, v in r.items() if k != "test_item_rows"} for r in rows],
        "pooling_vs_mean": comparisons,
        "best_by_M_H": {f"M{m}_H{h}": v for (m, h), v in best.items()},
        "m512_best_pooling_beats_m128": m512_beats_m128,
        "m1024_best_pooling_beats_m512": m1024_beats_m512,
    }
    pooling_fix_improves_m512 = any(
        c.get("delta_minFDE_K_vs_mean") is not None and c["delta_minFDE_K_vs_mean"] > 0
        for c in comparisons
        if c["M"] == 512
    )
    pooling_fix_improves_m1024 = any(
        c.get("delta_minFDE_K_vs_mean") is not None and c["delta_minFDE_K_vs_mean"] > 0
        for c in comparisons
        if c["M"] == 1024
    )
    m512_win_count = sum(m512_beats_m128.values())
    m1024_win_count = sum(m1024_beats_m512.values())
    density_recovered = bool(m512_win_count >= 2 and m1024_win_count >= 2)
    if density_recovered:
        remaining = "none"
        next_step = "run_best_pooling_m512_m1024_5seed"
    elif pooling_fix_improves_m512 and not pooling_fix_improves_m1024:
        remaining = "effective_batch_too_small"
        next_step = "improve_density_pooling_v2"
    elif not pooling_fix_improves_m512 and not pooling_fix_improves_m1024:
        remaining = "pooling_bottleneck_not_fixed"
        next_step = "keep_M128_main_move_to_semantic_identity_targets"
    else:
        remaining = "mixed"
        next_step = "improve_density_pooling_v2"
    decision = {
        "decision_name": "stwm_ostf_v30_density_pooling_pilot_decision",
        "generated_at_utc": utc_now(),
        "best_pooling_mode_m512_h32": best[(512, 32)]["mode"],
        "best_pooling_mode_m512_h64": best[(512, 64)]["mode"],
        "best_pooling_mode_m512_h96": best[(512, 96)]["mode"],
        "best_pooling_mode_m1024_h32": best[(1024, 32)]["mode"],
        "best_pooling_mode_m1024_h64": best[(1024, 64)]["mode"],
        "best_pooling_mode_m1024_h96": best[(1024, 96)]["mode"],
        "pooling_fix_improves_m512": bool(pooling_fix_improves_m512),
        "pooling_fix_improves_m1024": bool(pooling_fix_improves_m1024),
        "m512_beats_m128_after_pooling_fix": bool(m512_win_count >= 2),
        "m1024_beats_m512_after_pooling_fix": bool(m1024_win_count >= 2),
        "density_scaling_recovered": density_recovered,
        "remaining_failure_mode": remaining,
        "semantic_remains_not_tested": True,
        "recommended_next_step": next_step,
    }
    dump_json(SUMMARY, payload)
    dump_json(BOOT, boot)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V30 Density Pooling Pilot Decision",
        decision,
        [
            "best_pooling_mode_m512_h32",
            "best_pooling_mode_m512_h64",
            "best_pooling_mode_m512_h96",
            "best_pooling_mode_m1024_h32",
            "best_pooling_mode_m1024_h64",
            "best_pooling_mode_m1024_h96",
            "pooling_fix_improves_m512",
            "pooling_fix_improves_m1024",
            "density_scaling_recovered",
            "remaining_failure_mode",
            "recommended_next_step",
        ],
    )
    print(SUMMARY.relative_to(ROOT))
    print(DECISION.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
