#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_metrics_20260508 import paired_bootstrap
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


V31_DIR = ROOT / "reports/stwm_ostf_v31_field_preserving_runs"
V30_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
SUMMARY_PATH = ROOT / "reports/stwm_ostf_v31_field_multiseed_summary_20260508.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v31_field_multiseed_bootstrap_20260508.json"
DECISION_PATH = ROOT / "reports/stwm_ostf_v31_field_multiseed_decision_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V31_FIELD_MULTISEED_DECISION_20260508.md"

SEEDS = [42, 123, 456, 789, 2026]
MS = [128, 512]
HS = [32, 64, 96]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_name(m: int, h: int, seed: int, variant: str = "full") -> str:
    if variant == "full":
        return f"v31_field_m{m}_h{h}_seed{seed}"
    return f"v31_field_m{m}_h{h}_{variant}_seed{seed}"


def _v30_name(m: int, h: int, seed: int) -> str:
    return f"v30_extgt_m{m}_h{h}_seed{seed}"


def _metric(payload: dict[str, Any], metric: str, subset: str | None = None) -> float | None:
    if subset:
        val = payload.get("test_metrics", {}).get("subsets", {}).get(subset, {}).get(metric)
    else:
        val = payload.get("test_metrics", {}).get("all", {}).get(metric)
    return float(val) if val is not None and math.isfinite(float(val)) else None


def _delta_lower_better(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b - a)


def _mean_std(vals: list[float]) -> dict[str, Any]:
    return {
        "count": len(vals),
        "mean": float(statistics.mean(vals)) if vals else None,
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None,
    }


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _gpu_snapshot() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(float(parts[2])),
                "memory_free_mib": int(float(parts[3])),
                "gpu_utilization_percent": int(float(parts[4])),
            }
        )
    return rows


def _rows_with_key(rows: list[dict[str, Any]], seed: int, key_mode: str = "full") -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        if key_mode == "uid_h":
            r["bootstrap_pair_key"] = f"seed{seed}|{row.get('uid')}|H{row.get('H')}"
        else:
            r["bootstrap_pair_key"] = f"seed{seed}|{row.get('item_key')}"
        out.append(r)
    return out


def _strongest_prior(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    best_name = None
    best_metric: dict[str, Any] = {}
    best = 1e99
    for name, rec in payload.get("test_prior_metrics", {}).items():
        val = rec.get("all", {}).get("minFDE")
        if val is not None and float(val) < best:
            best = float(val)
            best_name = name
            best_metric = rec
    return best_name, best_metric


def main() -> int:
    summary_runs: dict[str, Any] = {}
    missing_primary = []
    missing_ablation = []
    boot: dict[str, Any] = {"generated_at_utc": utc_now(), "comparisons": {}}

    for m in MS:
        for h in HS:
            for seed in SEEDS:
                name = _run_name(m, h, seed)
                path = V31_DIR / f"{name}.json"
                run = _load(path)
                if not run.get("completed"):
                    missing_primary.append(name)
                    continue
                v30 = _load(V30_DIR / f"{_v30_name(m, h, seed)}.json")
                prior_name, prior_metrics = _strongest_prior(run)
                summary_runs[name] = {
                    "report_path": str(path.relative_to(ROOT)),
                    "checkpoint_path": run.get("checkpoint_path"),
                    "M": m,
                    "H": h,
                    "seed": seed,
                    "variant": "full",
                    "duration_seconds": run.get("duration_seconds"),
                    "batch_size": run.get("batch_size"),
                    "grad_accum_steps": run.get("grad_accum_steps"),
                    "effective_batch_size": run.get("effective_batch_size"),
                    "cuda_visible_devices": run.get("cuda_visible_devices"),
                    "train_loss_decreased": run.get("train_loss_decreased"),
                    "test_all": run.get("test_metrics", {}).get("all"),
                    "test_motion": run.get("test_metrics", {}).get("subsets", {}).get("motion"),
                    "strongest_prior": prior_name,
                    "strongest_prior_all": prior_metrics.get("all"),
                    "v30_same_seed_available": bool(v30),
                    "v30_same_seed_all": v30.get("test_metrics", {}).get("all"),
                    "v31_minus_v30_minFDE_K_positive": _delta_lower_better(_metric(run, "minFDE_K"), _metric(v30, "minFDE_K")),
                    "v31_minus_prior_minFDE_K_positive": _delta_lower_better(_metric(run, "minFDE_K"), prior_metrics.get("all", {}).get("minFDE")),
                }
                if v30.get("test_item_rows"):
                    for metric, higher in [
                        ("minFDE_K", False),
                        ("threshold_auc_endpoint_16_32_64_128", True),
                        ("MissRate@64", False),
                        ("MissRate@128", False),
                        ("relative_deformation_layout_error", False),
                        ("visibility_F1", True),
                    ]:
                        boot["comparisons"][f"{name}_vs_v30_{metric}"] = paired_bootstrap(
                            _rows_with_key(run.get("test_item_rows", []), seed),
                            _rows_with_key(v30.get("test_item_rows", []), seed),
                            metric,
                            higher_better=higher,
                        )
                if prior_name:
                    prior_rows = run.get("test_prior_item_rows", {}).get(prior_name, [])
                    for metric, higher in [
                        ("minFDE_K", False),
                        ("threshold_auc_endpoint_16_32_64_128", True),
                        ("MissRate@64", False),
                        ("MissRate@128", False),
                    ]:
                        boot["comparisons"][f"{name}_vs_{prior_name}_{metric}"] = paired_bootstrap(
                            _rows_with_key(run.get("test_item_rows", []), seed),
                            _rows_with_key(prior_rows, seed),
                            metric,
                            higher_better=higher,
                        )

    # M512 versus M128 at same horizon/seed.
    density_counts: dict[str, int] = {}
    density_deltas: dict[str, list[float]] = {}
    for h in HS:
        count = 0
        deltas = []
        rows512_all = []
        rows128_all = []
        for seed in SEEDS:
            r128 = _load(V31_DIR / f"{_run_name(128, h, seed)}.json")
            r512 = _load(V31_DIR / f"{_run_name(512, h, seed)}.json")
            d = _delta_lower_better(_metric(r512, "minFDE_K"), _metric(r128, "minFDE_K"))
            if d is not None:
                deltas.append(d)
                count += int(d > 0)
                rows512_all.extend(_rows_with_key(r512.get("test_item_rows", []), seed, key_mode="uid_h"))
                rows128_all.extend(_rows_with_key(r128.get("test_item_rows", []), seed, key_mode="uid_h"))
        density_counts[f"H{h}"] = count
        density_deltas[f"H{h}"] = deltas
        if rows512_all and rows128_all:
            boot["comparisons"][f"v31_m512_vs_m128_H{h}_minFDE_K"] = paired_bootstrap(rows512_all, rows128_all, "minFDE_K", higher_better=False)

    v31_v30_counts: dict[str, int] = {}
    v31_v30_delta_stats: dict[str, Any] = {}
    for m in MS:
        for h in HS:
            deltas = []
            count = 0
            available = 0
            for seed in SEEDS:
                run = _load(V31_DIR / f"{_run_name(m, h, seed)}.json")
                v30 = _load(V30_DIR / f"{_v30_name(m, h, seed)}.json")
                d = _delta_lower_better(_metric(run, "minFDE_K"), _metric(v30, "minFDE_K"))
                if d is not None:
                    available += 1
                    deltas.append(d)
                    count += int(d > 0)
            v31_v30_counts[f"M{m}_H{h}"] = count
            v31_v30_delta_stats[f"M{m}_H{h}"] = {"available_seed_count": available, **_mean_std(deltas)}

    # Field interaction ablations: full versus no_field_interaction for seed42/123.
    ablation: dict[str, Any] = {}
    ablation_positive = 0
    ablation_available = 0
    for m in MS:
        for h in HS:
            for seed in (42, 123):
                full = _load(V31_DIR / f"{_run_name(m, h, seed)}.json")
                nf_name = _run_name(m, h, seed, "no_field")
                nf = _load(V31_DIR / f"{nf_name}.json")
                if not nf.get("completed"):
                    missing_ablation.append(nf_name)
                    continue
                d = _delta_lower_better(_metric(full, "minFDE_K"), _metric(nf, "minFDE_K"))
                ablation_available += 1
                ablation_positive += int(d is not None and d > 0)
                ablation[nf_name] = {
                    "full_minFDE_K": _metric(full, "minFDE_K"),
                    "no_field_minFDE_K": _metric(nf, "minFDE_K"),
                    "full_minus_no_field_positive": d,
                    "full_motion_minFDE_K": _metric(full, "minFDE_K", "motion"),
                    "no_field_motion_minFDE_K": _metric(nf, "minFDE_K", "motion"),
                    "no_field_report_path": str((V31_DIR / f"{nf_name}.json").relative_to(ROOT)),
                }
                boot["comparisons"][f"{nf_name}_full_vs_no_field_minFDE_K"] = paired_bootstrap(
                    _rows_with_key(full.get("test_item_rows", []), seed),
                    _rows_with_key(nf.get("test_item_rows", []), seed),
                    "minFDE_K",
                    higher_better=False,
                )
    field_load_bearing = bool(ablation_available >= 6 and ablation_positive >= max(4, ablation_available // 2 + 1))

    v31_m128_beats_v30_h32 = v31_v30_counts.get("M128_H32", 0)
    v31_m128_beats_v30_h64 = v31_v30_counts.get("M128_H64", 0)
    v31_m128_beats_v30_h96 = v31_v30_counts.get("M128_H96", 0)
    v31_m512_beats_v30_h32 = v31_v30_counts.get("M512_H32", 0)
    v31_m512_beats_v30_h64 = v31_v30_counts.get("M512_H64", 0)
    v31_m512_beats_v30_h96 = v31_v30_counts.get("M512_H96", 0)
    v31_overall = sum(c >= 4 for c in [v31_m128_beats_v30_h32, v31_m128_beats_v30_h64, v31_m128_beats_v30_h96]) >= 2
    density_recovered = sum(density_counts.get(f"H{h}", 0) >= 4 for h in HS) >= 2
    if not field_load_bearing:
        next_step = "improve_v31_field_interaction"
    elif v31_overall and density_recovered:
        next_step = "promote_v31_to_main_and_run_m1024_efficient_attention"
    elif not v31_overall:
        next_step = "keep_v30_m128_main_move_to_semantic_identity_targets"
    else:
        next_step = "improve_v31_field_interaction"

    seed_level_summary: dict[str, Any] = {}
    for m in MS:
        for h in HS:
            rows = []
            for seed in SEEDS:
                run = _load(V31_DIR / f"{_run_name(m, h, seed)}.json")
                if not run.get("completed"):
                    continue
                all_metrics = run.get("test_metrics", {}).get("all", {})
                motion_metrics = run.get("test_metrics", {}).get("subsets", {}).get("motion", {})
                rows.append(
                    {
                        "seed": seed,
                        "minFDE_K": _safe_float(all_metrics.get("minFDE_K")),
                        "motion_minFDE_K": _safe_float(motion_metrics.get("minFDE_K")),
                        "MissRate@32": _safe_float(all_metrics.get("MissRate@32")),
                        "MissRate@64": _safe_float(all_metrics.get("MissRate@64")),
                        "MissRate@128": _safe_float(all_metrics.get("MissRate@128")),
                        "threshold_auc_endpoint_16_32_64_128": _safe_float(
                            all_metrics.get("threshold_auc_endpoint_16_32_64_128")
                        ),
                        "relative_deformation_layout_error": _safe_float(
                            all_metrics.get("relative_deformation_layout_error")
                        ),
                        "visibility_F1": _safe_float(all_metrics.get("visibility_F1")),
                        "duration_seconds": _safe_float(run.get("duration_seconds")),
                        "batch_size": run.get("batch_size"),
                        "effective_batch_size": run.get("effective_batch_size"),
                        "cuda_visible_devices": run.get("cuda_visible_devices"),
                        "train_loss_decreased": run.get("train_loss_decreased"),
                    }
                )
            seed_level_summary[f"M{m}_H{h}"] = {
                "rows": rows,
                "minFDE_K": _mean_std([r["minFDE_K"] for r in rows if r["minFDE_K"] is not None]),
                "motion_minFDE_K": _mean_std(
                    [r["motion_minFDE_K"] for r in rows if r["motion_minFDE_K"] is not None]
                ),
                "MissRate@32": _mean_std([r["MissRate@32"] for r in rows if r["MissRate@32"] is not None]),
                "visibility_F1": _mean_std([r["visibility_F1"] for r in rows if r["visibility_F1"] is not None]),
            }

    ablation_seed_summary: dict[str, Any] = {}
    for m in MS:
        for h in HS:
            rows = []
            for seed in (42, 123):
                nf = _load(V31_DIR / f"{_run_name(m, h, seed, 'no_field')}.json")
                if not nf.get("completed"):
                    continue
                all_metrics = nf.get("test_metrics", {}).get("all", {})
                rows.append(
                    {
                        "seed": seed,
                        "minFDE_K": _safe_float(all_metrics.get("minFDE_K")),
                        "MissRate@32": _safe_float(all_metrics.get("MissRate@32")),
                        "visibility_F1": _safe_float(all_metrics.get("visibility_F1")),
                        "duration_seconds": _safe_float(nf.get("duration_seconds")),
                        "batch_size": nf.get("batch_size"),
                        "effective_batch_size": nf.get("effective_batch_size"),
                        "cuda_visible_devices": nf.get("cuda_visible_devices"),
                        "train_loss_decreased": nf.get("train_loss_decreased"),
                    }
                )
            ablation_seed_summary[f"M{m}_H{h}"] = {
                "rows": rows,
                "minFDE_K": _mean_std([r["minFDE_K"] for r in rows if r["minFDE_K"] is not None]),
            }

    summary = {
        "generated_at_utc": utc_now(),
        "expected_primary_run_count": len(SEEDS) * len(MS) * len(HS),
        "completed_primary_run_count": len(summary_runs),
        "missing_primary_runs": missing_primary,
        "expected_ablation_run_count": 2 * len(MS) * len(HS),
        "completed_ablation_run_count": ablation_available,
        "missing_ablation_runs": missing_ablation,
        "runs": summary_runs,
        "v31_vs_v30_positive_seed_counts": v31_v30_counts,
        "v31_vs_v30_delta_stats": v31_v30_delta_stats,
        "density_m512_vs_m128_positive_seed_counts": density_counts,
        "density_m512_vs_m128_delta_stats": {k: _mean_std(v) for k, v in density_deltas.items()},
        "seed_level_summary": seed_level_summary,
        "field_ablation": ablation,
        "field_ablation_seed_summary": ablation_seed_summary,
        "missing_field_ablation_runs": missing_ablation,
        "runtime_memory_note": "Per-run peak GPU memory was not logged by the V31 trainer; current nvidia-smi snapshot is included for live occupancy only.",
        "gpu_snapshot_after_aggregation": _gpu_snapshot(),
        "semantic_not_tested_not_failed": True,
    }
    decision = {
        "generated_at_utc": utc_now(),
        "v31_m128_beats_v30_h32_seed_count": v31_m128_beats_v30_h32,
        "v31_m128_beats_v30_h64_seed_count": v31_m128_beats_v30_h64,
        "v31_m128_beats_v30_h96_seed_count": v31_m128_beats_v30_h96,
        "v31_m512_beats_v30_h32_seed_count": v31_m512_beats_v30_h32,
        "v31_m512_beats_v30_h64_seed_count": v31_m512_beats_v30_h64,
        "v31_m512_beats_v30_h96_seed_count": v31_m512_beats_v30_h96,
        "v31_m512_beats_v31_m128_h32_seed_count": density_counts.get("H32", 0),
        "v31_m512_beats_v31_m128_h64_seed_count": density_counts.get("H64", 0),
        "v31_m512_beats_v31_m128_h96_seed_count": density_counts.get("H96", 0),
        "field_interaction_load_bearing": field_load_bearing,
        "field_ablation_available_count": ablation_available,
        "field_ablation_positive_count": ablation_positive,
        "v31_overall_beats_v30": bool(v31_overall),
        "density_scaling_recovered_with_v31": bool(density_recovered),
        "semantic_not_tested_not_failed": True,
        "recommended_next_step": next_step,
    }
    dump_json(SUMMARY_PATH, summary)
    dump_json(BOOT_PATH, boot)
    dump_json(DECISION_PATH, decision)
    write_doc(
        DOC_PATH,
        "STWM OSTF V31 Field Multiseed Decision",
        decision,
        list(decision.keys()),
    )
    print(DECISION_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
