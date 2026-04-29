#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-report", default="reports/stwm_semantic_memory_world_model_v2_splits_20260428.json")
    p.add_argument("--eval-report", default="reports/stwm_semantic_memory_world_model_v2_eval_c64_20260428.json")
    p.add_argument("--figure-dir", default="outputs/figures/stwm_semantic_trace_field_v2")
    p.add_argument("--output", default="reports/stwm_semantic_trace_field_v2_visualization_manifest_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_TRACE_FIELD_V2_VISUALIZATION_20260428.md")
    args = p.parse_args()
    splits = json.loads(Path(args.split_report).read_text(encoding="utf-8"))
    eval_payload = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    examples = [{"item_key": key, "planned_panels": ["observed_trace", "future_trace", "copy_proto", "residual_proto", "stable_changed_label"]} for key in splits["splits"]["test"][:16]]
    payload = {
        "audit_name": "stwm_semantic_trace_field_v2_visualization_manifest",
        "figure_dir": str(figure_dir),
        "example_count": len(examples),
        "examples": examples,
        "best_eval_report": str(args.eval_report),
        "best_prototype_count": int(eval_payload.get("prototype_count", 0)),
        "note": "Manifest only; no large videos generated in this controlled run.",
        "no_candidate_scorer": True,
        "future_candidate_leakage": False,
    }
    _write_json(Path(args.output), payload)
    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text(
        "# STWM Semantic Trace Field V2 Visualization\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in payload.items() if k != "examples")
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
