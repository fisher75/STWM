#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.eval_semantic_memory_world_model_significance_20260428 import main as generic_main


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


def _bootstrap(values: list[float], *, seed: int = 20260428, samples: int = 2000) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"item_count": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0], "zero_excluded": False, "bootstrap_win_rate": 0.0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    lo, hi = np.percentile(np.asarray(means, dtype=np.float64), [2.5, 97.5])
    return {
        "item_count": int(arr.size),
        "mean_delta": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0.0 or hi < 0.0),
        "bootstrap_win_rate": float((arr > 0.0).mean()),
    }


def _selected_test_main(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.test_eval).read_text(encoding="utf-8"))
    rows = payload["seed_results"][0]["test_itemwise"]["item_scores"]
    overall = [
        float(r.get("residual_overall_top5", 0.0) - r.get("copy_overall_top5", 0.0))
        for r in rows
        if int(r.get("overall_count", 0)) > 0
    ]
    changed = [
        float(r.get("residual_changed_top5", 0.0) - r.get("copy_changed_top5", 0.0))
        for r in rows
        if int(r.get("changed_count", 0)) > 0
    ]
    stable_drop = [
        float(r.get("copy_stable_top5", 0.0) - r.get("residual_stable_top5", 0.0))
        for r in rows
        if int(r.get("stable_count", 0)) > 0
    ]
    ce_delta = [
        float(r.get("copy_overall_ce", 0.0) - r.get("residual_overall_ce", 0.0))
        for r in rows
        if int(r.get("overall_count", 0)) > 0
    ]
    report = {
        "audit_name": "stwm_free_rollout_semantic_trace_field_v5_test_significance",
        "bootstrap_unit": "item",
        "prototype_count": int(payload.get("prototype_count", 0)),
        "selected_seed": int(payload.get("best_seed", -1)),
        "test_item_count": int(payload.get("heldout_item_count", len(rows))),
        "changed_item_count": int(sum(1 for r in rows if int(r.get("changed_count", 0)) > 0)),
        "residual_vs_copy_overall_top5": _bootstrap(overall),
        "residual_vs_copy_changed_top5": _bootstrap(changed),
        "stable_preservation_drop": _bootstrap(stable_drop),
        "residual_vs_copy_ce_improvement": _bootstrap(ce_delta),
        "low_sample_warning": bool(int(payload.get("heldout_item_count", len(rows))) < 100),
        "test_eval_path": str(args.test_eval),
    }
    _write_json(Path(args.output), report)
    _write_doc(Path(args.doc), "STWM Free-Rollout Semantic Trace Field V5 Test Significance", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test-eval", default="")
    parser.add_argument("--output", default="reports/stwm_free_rollout_semantic_trace_field_v5_test_significance_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_FREE_ROLLOUT_SEMANTIC_TRACE_FIELD_V5_TEST_SIGNIFICANCE_20260428.md")
    known, _ = parser.parse_known_args()
    if known.test_eval:
        _selected_test_main(known)
    else:
        generic_main()
