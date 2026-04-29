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
    return {
        "prototype_count": int(data["prototype_count"]),
        "residual_vs_copy_overall_top5": _ci(overall),
        "residual_vs_copy_changed_top5": _ci(changed),
        "stable_preservation_drop": _ci(stable_drop),
        "seed_count": int(len(data["seed_results"])),
        "seed_mean_std": data.get("seed_mean_std", {}),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-c32", default="reports/stwm_semantic_memory_world_model_v2_eval_c32_20260428.json")
    p.add_argument("--eval-c64", default="reports/stwm_semantic_memory_world_model_v2_eval_c64_20260428.json")
    p.add_argument("--output", default="reports/stwm_semantic_memory_world_model_v2_significance_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_WORLD_MODEL_V2_SIGNIFICANCE_20260428.md")
    args = p.parse_args()
    c32 = _extract(Path(args.eval_c32))
    c64 = _extract(Path(args.eval_c64))
    payload = {
        "audit_name": "stwm_semantic_memory_world_model_v2_significance",
        "bootstrap_unit": "seed_level_due_current_eval_reports_not_saving_item_level_scores",
        "item_level_bootstrap_required_before_paper_claim": True,
        "c32": c32,
        "c64": c64,
        "v2_residual_vs_copy": c64 if c64["residual_vs_copy_overall_top5"]["mean_delta"] >= c32["residual_vs_copy_overall_top5"]["mean_delta"] else c32,
    }
    _write_json(Path(args.output), payload)
    _write_doc(Path(args.doc), "STWM Semantic Memory World Model V2 Significance", payload)


if __name__ == "__main__":
    main()
