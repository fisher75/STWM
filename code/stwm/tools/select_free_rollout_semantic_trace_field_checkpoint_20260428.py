#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def _candidates(report_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    out = []
    for row in payload.get("seed_results", []):
        m = row.get("val_metrics", {})
        out.append(
            {
                "prototype_count": int(payload.get("prototype_count", 0)),
                "seed": int(row.get("seed", -1)),
                "checkpoint_path": str(row.get("checkpoint_path", "")),
                "changed_gain_over_copy": float(m.get("changed_subset_gain_over_copy", m.get("changed_subset_top5", 0.0) - m.get("copy_changed_subset_top5", 0.0))),
                "overall_gain_over_copy": float(m.get("overall_gain_over_copy", m.get("proto_top5", 0.0) - m.get("copy_proto_top5", 0.0))),
                "future_trace_coord_error": float(m.get("future_trace_coord_error", 0.0)),
                "proto_top5": float(m.get("proto_top5", 0.0)),
                "copy_proto_top5": float(m.get("copy_proto_top5", 0.0)),
                "stable_preservation_drop": float(m.get("stable_preservation_drop", 0.0)),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-eval-c32", default="reports/stwm_free_rollout_semantic_trace_field_v5_val_eval_c32_20260428.json")
    p.add_argument("--val-eval-c64", default="reports/stwm_free_rollout_semantic_trace_field_v5_val_eval_c64_20260428.json")
    p.add_argument("--output", default="reports/stwm_free_rollout_semantic_trace_field_v5_val_selection_20260428.json")
    p.add_argument("--doc", default="docs/STWM_FREE_ROLLOUT_SEMANTIC_TRACE_FIELD_V5_VAL_SELECTION_20260428.md")
    args = p.parse_args()
    candidates = _candidates(Path(args.val_eval_c32)) + _candidates(Path(args.val_eval_c64))
    best = max(candidates, key=lambda x: (x["changed_gain_over_copy"], x["overall_gain_over_copy"], -x["future_trace_coord_error"]))
    payload = {
        "audit_name": "stwm_free_rollout_semantic_trace_field_v5_val_selection",
        "selection_split": "val",
        "best_selected_on_val_only": True,
        "selection_rule": "primary changed_subset_top5 gain over copy; secondary overall top5 gain; tie lower trace coord error",
        "selected_prototype_count": int(best["prototype_count"]),
        "selected_seed": int(best["seed"]),
        "selected_checkpoint_path": str(best["checkpoint_path"]),
        "selected_changed_gain_over_copy": float(best["changed_gain_over_copy"]),
        "selected_overall_gain_over_copy": float(best["overall_gain_over_copy"]),
        "selected_future_trace_coord_error": float(best["future_trace_coord_error"]),
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
        "test_metrics_used_for_selection": False,
    }
    _write_json(Path(args.output), payload)
    _write_doc(Path(args.doc), "STWM Free-Rollout Semantic Trace Field V5 Val Selection", payload)


if __name__ == "__main__":
    main()
