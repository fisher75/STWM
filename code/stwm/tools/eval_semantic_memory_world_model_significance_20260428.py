#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ci(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_delta": 0.0, "ci95": [0.0, 0.0], "zero_excluded": False}
    # With three seeds, this is a seed-level robustness interval, not a paper-grade item bootstrap.
    mean = float(arr.mean())
    if arr.size == 1:
        lo = hi = mean
    else:
        se = float(arr.std(ddof=1) / np.sqrt(arr.size))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return {"mean_delta": mean, "ci95": [float(lo), float(hi)], "zero_excluded": bool(lo > 0.0 or hi < 0.0)}


def _bootstrap_ci(values: list[float], *, seed: int = 20260428, samples: int = 2000) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_delta": 0.0, "ci95": [0.0, 0.0], "zero_excluded": False, "bootstrap_win_rate": 0.0, "item_count": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    lo, hi = np.percentile(np.asarray(means, dtype=np.float64), [2.5, 97.5])
    mean = float(arr.mean())
    return {
        "mean_delta": mean,
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0.0 or hi < 0.0),
        "bootstrap_win_rate": float((arr > 0.0).mean()),
        "item_count": int(arr.size),
    }


def _item_bootstrap(best_seed_payload: dict[str, Any]) -> dict[str, Any]:
    scores = best_seed_payload.get("test_itemwise", {}).get("item_scores", [])
    overall = [
        float(x.get("residual_overall_top5", 0.0) - x.get("copy_overall_top5", 0.0))
        for x in scores
        if int(x.get("overall_count", 0)) > 0
    ]
    changed = [
        float(x.get("residual_changed_top5", 0.0) - x.get("copy_changed_top5", 0.0))
        for x in scores
        if int(x.get("changed_count", 0)) > 0
    ]
    stable_drop = [
        float(x.get("copy_stable_top5", 0.0) - x.get("residual_stable_top5", 0.0))
        for x in scores
        if int(x.get("stable_count", 0)) > 0
    ]
    return {
        "bootstrap_unit": "item",
        "residual_vs_copy_overall_top5": _bootstrap_ci(overall),
        "residual_vs_copy_changed_top5": _bootstrap_ci(changed),
        "stable_preservation_drop": _bootstrap_ci(stable_drop),
    }


def _extract(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    copy = data["copy_baseline"]
    overall = [float(r["val_metrics"]["proto_top5"] - copy["proto_top5"]) for r in data["seed_results"]]
    changed = [
        float(r["val_metrics"]["changed_subset_top5"] - copy["changed_subset_top5"])
        for r in data["seed_results"]
    ]
    stable_drop = [
        float(copy["stable_subset_top5"] - r["val_metrics"]["stable_subset_top5"])
        for r in data["seed_results"]
    ]
    base = {
        "prototype_count": int(data["prototype_count"]),
        "residual_vs_copy_overall_top5": _ci(overall),
        "residual_vs_copy_changed_top5": _ci(changed),
        "stable_preservation_drop": _ci(stable_drop),
        "seed_count": int(len(data["seed_results"])),
        "seed_mean_std": data.get("seed_mean_std", {}),
    }
    best_seed = int(data.get("best_seed", data["seed_results"][0].get("seed", 0)))
    best_payload = next((r for r in data["seed_results"] if int(r.get("seed", -1)) == best_seed), data["seed_results"][0])
    if best_payload.get("test_itemwise", {}).get("item_scores"):
        base["item_level_bootstrap"] = _item_bootstrap(best_payload)
        base["bootstrap_unit"] = "item"
    else:
        base["bootstrap_unit"] = "seed"
    return base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-c32", default="reports/stwm_semantic_memory_world_model_v2_eval_c32_20260428.json")
    p.add_argument("--eval-c64", default="reports/stwm_semantic_memory_world_model_v2_eval_c64_20260428.json")
    p.add_argument("--output", default="reports/stwm_semantic_memory_world_model_v2_significance_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_WORLD_MODEL_V2_SIGNIFICANCE_20260428.md")
    p.add_argument("--audit-name", default="stwm_semantic_memory_world_model_v2_significance")
    p.add_argument("--title", default="STWM Semantic Memory World Model V2 Significance")
    args = p.parse_args()
    c32 = _extract(Path(args.eval_c32))
    c64 = _extract(Path(args.eval_c64))
    payload = {
        "audit_name": str(args.audit_name),
        "bootstrap_unit": "item" if c32.get("bootstrap_unit") == "item" or c64.get("bootstrap_unit") == "item" else "seed_level_due_current_eval_reports_not_saving_item_level_scores",
        "item_level_bootstrap_required_before_paper_claim": not (c32.get("bootstrap_unit") == "item" or c64.get("bootstrap_unit") == "item"),
        "c32": c32,
        "c64": c64,
        "v2_residual_vs_copy": c64 if c64["residual_vs_copy_overall_top5"]["mean_delta"] >= c32["residual_vs_copy_overall_top5"]["mean_delta"] else c32,
    }
    _write_json(Path(args.output), payload)
    _write_doc(Path(args.doc), str(args.title), payload)


if __name__ == "__main__":
    main()
