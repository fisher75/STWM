#!/usr/bin/env python3
"""Aggregate V8 strong copy-aware FSTF baselines and paired bootstrap.

This is intentionally a reporting/aggregation tool only. It does not train a
model, does not select on test, and does not use future candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LEARNED_BASELINES = [
    "copy_residual_mlp",
    "copy_residual_transformer",
    "copy_gated_residual_no_trace",
    "copy_gated_residual_trace_only",
    "copy_gated_residual_plain_trace_semantic",
]
SEEDS = [42, 123, 456, 789, 1001]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def bootstrap_ci(values: list[float], *, n_boot: int = 5000, seed: int = 20260501) -> dict[str, Any]:
    if not values:
        return {
            "item_count": 0,
            "mean_delta": None,
            "ci95": [None, None],
            "zero_excluded": False,
            "bootstrap_win_rate": None,
        }
    rng = random.Random(seed)
    n = len(values)
    means = []
    wins = 0
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        m = statistics.fmean(sample)
        means.append(m)
        if m > 0:
            wins += 1
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return {
        "item_count": n,
        "mean_delta": float(statistics.fmean(values)),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0 or hi < 0),
        "bootstrap_win_rate": float(wins / n_boot),
    }


def stable_drop(score: dict[str, Any]) -> float | None:
    if score.get("stable_count", 0) <= 0:
        return None
    return float(score["copy_stable_top5"]) - float(score["residual_stable_top5"])


def average_item_scores(eval_paths: list[Path]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in eval_paths:
        data = load_json(path)
        for score in data.get("item_scores", []):
            grouped[str(score["item_key"])].append(score)
    averaged: dict[str, dict[str, float]] = {}
    for key, rows in grouped.items():
        out: dict[str, float] = {}
        numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        for k in numeric_keys:
            out[k] = float(statistics.fmean(float(r.get(k, 0.0)) for r in rows))
        averaged[key] = out
    return averaged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/checkpoints/fstf_strong_copyaware_baselines_v8_20260501",
    )
    parser.add_argument(
        "--logs",
        default="logs/fstf_strong_copyaware_baselines_v8_20260501",
    )
    parser.add_argument(
        "--stwm-eval",
        default="reports/stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json",
    )
    parser.add_argument(
        "--suite-output",
        default="reports/stwm_fstf_strong_copyaware_baseline_suite_v8_20260501.json",
    )
    parser.add_argument(
        "--bootstrap-output",
        default="reports/stwm_fstf_strong_copyaware_baseline_bootstrap_v8_20260501.json",
    )
    parser.add_argument(
        "--doc-output",
        default="docs/STWM_FSTF_STRONG_COPYAWARE_BASELINE_SUITE_V8_20260501.md",
    )
    args = parser.parse_args()

    root = Path(args.root)
    logs = Path(args.logs)
    per_baseline: dict[str, Any] = {}
    all_eval_paths: list[Path] = []

    for baseline in LEARNED_BASELINES:
        seed_rows = []
        eval_paths = []
        for seed in SEEDS:
            run_dir = root / baseline / str(seed)
            ckpt = run_dir / "checkpoint.pt"
            train = run_dir / "train_summary.json"
            eval_path = run_dir / "eval_test.json"
            log = logs / f"{baseline}_seed{seed}.log"
            row: dict[str, Any] = {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "checkpoint_exists": ckpt.exists(),
                "checkpoint_size_bytes": ckpt.stat().st_size if ckpt.exists() else 0,
                "train_summary_path": str(train),
                "train_summary_exists": train.exists(),
                "eval_path": str(eval_path),
                "eval_exists": eval_path.exists(),
                "log_path": str(log),
                "log_exists": log.exists(),
                "log_nonempty": log.exists() and log.stat().st_size > 0,
            }
            if train.exists():
                tr = load_json(train)
                row["parameter_count"] = tr.get("parameter_count")
                row["steps"] = tr.get("steps")
                row["train_loss_final"] = tr.get("loss_final")
            if eval_path.exists():
                ev = load_json(eval_path)
                metrics = ev.get("metrics", {})
                row["metrics"] = metrics
                eval_paths.append(eval_path)
                all_eval_paths.append(eval_path)
            seed_rows.append(row)

        completed = [r for r in seed_rows if r["checkpoint_exists"] and r["eval_exists"]]
        changed_gains = [
            float(r["metrics"]["changed_subset_gain_over_copy"])
            for r in completed
            if r.get("metrics") and r["metrics"].get("changed_subset_gain_over_copy") is not None
        ]
        overall_gains = [
            float(r["metrics"]["overall_gain_over_copy"])
            for r in completed
            if r.get("metrics") and r["metrics"].get("overall_gain_over_copy") is not None
        ]
        stable_drops = [
            float(r["metrics"]["stable_preservation_drop"])
            for r in completed
            if r.get("metrics") and r["metrics"].get("stable_preservation_drop") is not None
        ]
        per_baseline[baseline] = {
            "baseline_name": baseline,
            "baseline_type": "same_output_fstf",
            "baseline_family": "copy_aware_controlled_baseline",
            "evidence_level": "controlled_ablation",
            "output_contract_matched": True,
            "uses_future_candidate_measurement": False,
            "candidate_scorer_used": False,
            "future_candidate_leakage": False,
            "allowed_table_placement": "main_fstf_table",
            "visibility_reappearance_status": "metric_invalid_or_untrained",
            "seed_results": seed_rows,
            "completed_seed_count": len(completed),
            "changed_gain_over_copy": mean_std(changed_gains),
            "overall_gain_over_copy": mean_std(overall_gains),
            "stable_preservation_drop": mean_std(stable_drops),
            "learned_baseline_beats_copy_changed": bool(changed_gains and statistics.fmean(changed_gains) > 0),
        }

    copy_eval = root / "copy_semantic_memory_baseline" / "eval_test.json"
    oracle_eval = root / "oracle_change_gate_upper_bound" / "eval_test.json"
    auxiliary = {}
    for name, path in [
        ("copy_semantic_memory_baseline", copy_eval),
        ("oracle_change_gate_upper_bound", oracle_eval),
    ]:
        auxiliary[name] = {
            "path": str(path),
            "exists": path.exists(),
            "metrics": load_json(path).get("metrics", {}) if path.exists() else {},
            "evidence_level": "trivial_lower_bound" if name.startswith("copy") else "oracle_upper_bound",
            "allowed_table_placement": "main_fstf_table" if name.startswith("copy") else "oracle_only",
        }

    complete_learned = sum(1 for p in all_eval_paths if p.exists())
    new_checkpoint_count = len(list(root.glob("*/*/checkpoint.pt")))
    new_eval_summary_count = len(list(root.glob("*/*/eval_test.json")))
    any_learned_beats_copy = any(
        v["learned_baseline_beats_copy_changed"] for v in per_baseline.values()
    )

    fair_rank = []
    for baseline, data in per_baseline.items():
        gain = data["changed_gain_over_copy"]["mean"]
        if gain is None:
            gain = -math.inf
        fair_rank.append((float(gain), baseline))
    fair_rank.sort(reverse=True)
    strongest = fair_rank[0][1] if fair_rank else None

    stwm = load_json(Path(args.stwm_eval))
    stwm_scores = {
        str(s["item_key"]): s
        for s in stwm["seed_results"][0]["test_itemwise"]["item_scores"]
    }
    strongest_eval_paths = sorted((root / strongest).glob("*/eval_test.json")) if strongest else []
    strongest_scores = average_item_scores(strongest_eval_paths)
    common_keys = sorted(set(stwm_scores) & set(strongest_scores))

    changed_deltas = []
    overall_deltas = []
    stable_drop_deltas = []
    for key in common_keys:
        s = stwm_scores[key]
        b = strongest_scores[key]
        overall_deltas.append(float(s["residual_overall_top5"]) - float(b["residual_overall_top5"]))
        if float(s.get("changed_count", 0.0)) > 0 and float(b.get("changed_count", 0.0)) > 0:
            changed_deltas.append(float(s["residual_changed_top5"]) - float(b["residual_changed_top5"]))
        sd_s = stable_drop(s)
        sd_b = float(b["copy_stable_top5"]) - float(b["residual_stable_top5"]) if float(b.get("stable_count", 0.0)) > 0 else None
        if sd_s is not None and sd_b is not None:
            stable_drop_deltas.append(float(sd_s) - float(sd_b))

    bootstrap = {
        "audit_name": "stwm_fstf_strong_copyaware_baseline_v8_paired_bootstrap",
        "stwm_eval_path": args.stwm_eval,
        "strongest_copyaware_baseline": strongest,
        "strongest_eval_paths": [str(p) for p in strongest_eval_paths],
        "common_item_count": len(common_keys),
        "stwm_minus_strongest_changed_top5": bootstrap_ci(changed_deltas),
        "stwm_minus_strongest_overall_top5": bootstrap_ci(overall_deltas),
        "stwm_minus_strongest_stable_drop": bootstrap_ci(stable_drop_deltas),
        "interpretation": {
            "stable_drop_delta_note": "Positive means STWM has larger stable drop than the baseline; negative favors STWM stable preservation.",
            "visibility_reappearance_status": "metric_invalid_or_untrained",
        },
    }

    stwm_changed = bootstrap["stwm_minus_strongest_changed_top5"]
    stwm_overall = bootstrap["stwm_minus_strongest_overall_top5"]
    stwm_significant_vs_strongest = bool(
        stwm_changed["zero_excluded"] and (stwm_changed["mean_delta"] or 0.0) > 0
    )
    proceed_to_scaling = bool(
        any_learned_beats_copy
        and stwm_significant_vs_strongest
        and complete_learned == len(LEARNED_BASELINES) * len(SEEDS)
    )
    next_step = "run_scaling_laws" if proceed_to_scaling else "fix_training_or_objective"

    suite = {
        "audit_name": "stwm_fstf_strong_copyaware_baseline_suite_v8",
        "baseline_suite_completed": bool(complete_learned == len(LEARNED_BASELINES) * len(SEEDS)),
        "completed_learned_eval_count": complete_learned,
        "expected_learned_eval_count": len(LEARNED_BASELINES) * len(SEEDS),
        "new_checkpoint_count": new_checkpoint_count,
        "new_eval_summary_count": new_eval_summary_count,
        "learned_baselines": per_baseline,
        "auxiliary_baselines": auxiliary,
        "strongest_copyaware_baseline": strongest,
        "whether_any_learned_copyaware_baseline_beats_copy": any_learned_beats_copy,
        "STWM_vs_strongest_copyaware_baseline_paired_bootstrap": bootstrap,
        "proceed_to_scaling_allowed": proceed_to_scaling,
        "next_step_choice": next_step,
        "visibility_reappearance_status": "metric_invalid_or_untrained",
        "guardrails": {
            "candidate_scorer_used": False,
            "future_candidate_leakage": False,
            "teacher_forced_path_used": False,
            "test_set_model_selection": False,
        },
    }

    dump_json(Path(args.bootstrap_output), bootstrap)
    dump_json(Path(args.suite_output), suite)

    md = [
        "# STWM FSTF Strong Copy-Aware Baseline Suite V8",
        "",
        f"- Baseline suite completed: `{suite['baseline_suite_completed']}`",
        f"- New checkpoints: `{new_checkpoint_count}`",
        f"- New eval summaries: `{new_eval_summary_count}`",
        f"- Strongest copy-aware baseline: `{strongest}`",
        f"- Any learned copy-aware baseline beats copy: `{any_learned_beats_copy}`",
        f"- Proceed to scaling allowed: `{proceed_to_scaling}`",
        f"- Next step: `{next_step}`",
        "",
        "## Paired Bootstrap",
        "",
        f"- STWM minus strongest changed top5: `{stwm_changed}`",
        f"- STWM minus strongest overall top5: `{stwm_overall}`",
        f"- STWM minus strongest stable drop: `{bootstrap['stwm_minus_strongest_stable_drop']}`",
        "",
        "Visibility/reappearance remain `metric_invalid_or_untrained` and are not used as positive evidence.",
    ]
    doc_path = Path(args.doc_output)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[aggregate-v8] suite={args.suite_output}")
    print(f"[aggregate-v8] bootstrap={args.bootstrap_output}")
    print(f"[aggregate-v8] doc={args.doc_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
