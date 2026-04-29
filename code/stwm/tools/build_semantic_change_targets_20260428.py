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


def _load_npz_from_report(report_path: Path, key: str = "target_cache_path") -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get(key) or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return payload, dict(np.load(cache_path, allow_pickle=True)), cache_path


def build_change_targets(
    *,
    observed_report: Path,
    future_reports: list[Path],
    output: Path,
    doc: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    observed_payload = json.loads(observed_report.read_text(encoding="utf-8"))
    observed_paths = observed_payload.get("target_cache_paths_by_prototype_count", {})
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    target_cache_paths: dict[str, str] = {}
    for future_report in future_reports:
        future_payload, future, _ = _load_npz_from_report(future_report)
        c = int(future_payload.get("prototype_count") or future["future_semantic_proto_distribution"].shape[-1])
        observed_path = Path(str(observed_paths.get(str(c), "")))
        if not observed_path.is_absolute():
            observed_path = observed_report.parent.parent / observed_path
        if not observed_path.exists():
            continue
        observed = dict(np.load(observed_path, allow_pickle=True))
        future_target = np.asarray(future["future_semantic_proto_target"], dtype=np.int64)
        future_mask = np.asarray(future["target_mask"], dtype=bool) & (future_target >= 0)
        observed_target = np.asarray(observed["observed_semantic_proto_target"], dtype=np.int64)
        observed_mask = np.asarray(observed["observed_semantic_proto_mask"], dtype=bool) & (observed_target >= 0)
        obs_h = observed_target[:, None, :]
        obs_mask_h = observed_mask[:, None, :]
        change_mask = future_mask & obs_mask_h
        change_target = change_mask & (future_target != obs_h)
        stable_target = change_mask & (~change_target)
        change_event_target = change_target.any(axis=1)
        change_event_mask = change_mask.any(axis=1)
        out_path = cache_dir / f"semantic_change_targets_c{c}.npz"
        np.savez_compressed(
            out_path,
            item_keys=future["item_keys"],
            semantic_change_target=change_target.astype(bool),
            semantic_change_mask=change_mask.astype(bool),
            semantic_change_event_target=change_event_target.astype(bool),
            semantic_change_event_mask=change_event_mask.astype(bool),
            semantic_stable_target=stable_target.astype(bool),
            prototype_count=np.asarray(c, dtype=np.int64),
            no_future_candidate_leakage=np.asarray(True),
        )
        valid_count = int(change_mask.sum())
        changed_count = int(change_target.sum())
        stable_count = int(stable_target.sum())
        event_valid_count = int(change_event_mask.sum())
        event_changed_count = int(change_event_target[change_event_mask].sum()) if event_valid_count else 0
        result = {
            "prototype_count": int(c),
            "target_cache_path": str(out_path),
            "change_positive_rate": float(changed_count / max(valid_count, 1)),
            "change_event_positive_rate": float(event_changed_count / max(event_valid_count, 1)),
            "changed_subset_count": changed_count,
            "stable_subset_count": stable_count,
            "change_mask_valid_count": valid_count,
            "change_event_valid_count": event_valid_count,
            "no_future_candidate_leakage": True,
        }
        results.append(result)
        target_cache_paths[str(c)] = str(out_path)
    selected = next((r for r in results if int(r["prototype_count"]) == 32), results[0] if results else {})
    payload = {
        "audit_name": "stwm_semantic_change_targets_v1",
        "observed_report": str(observed_report),
        "results_by_prototype_count": results,
        "target_cache_paths_by_prototype_count": target_cache_paths,
        "selected_prototype_count": int(selected.get("prototype_count", 0) or 0),
        "change_positive_rate": float(selected.get("change_positive_rate", 0.0) or 0.0),
        "change_event_positive_rate": float(selected.get("change_event_positive_rate", 0.0) or 0.0),
        "changed_subset_count": int(selected.get("changed_subset_count", 0) or 0),
        "stable_subset_count": int(selected.get("stable_subset_count", 0) or 0),
        "prototype_count": int(selected.get("prototype_count", 0) or 0),
        "no_future_candidate_leakage": True,
    }
    _write_json(output, payload)
    _write_doc(doc, "STWM Semantic Change Targets V1", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v2_20260428.json")
    parser.add_argument(
        "--future-reports",
        nargs="+",
        default=[
            "reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json",
            "reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json",
            "reports/stwm_future_semantic_trace_prototype_targets_v2_20260428.json",
        ],
    )
    parser.add_argument("--cache-dir", default="outputs/cache/stwm_semantic_change_targets_v1_20260428")
    parser.add_argument("--output", default="reports/stwm_semantic_change_targets_v1_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_SEMANTIC_CHANGE_TARGETS_V1_20260428.md")
    args = parser.parse_args()
    build_change_targets(
        observed_report=Path(args.observed_report),
        future_reports=[Path(x) for x in args.future_reports],
        output=Path(args.output),
        doc=Path(args.doc),
        cache_dir=Path(args.cache_dir),
    )


if __name__ == "__main__":
    main()
