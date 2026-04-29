#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.semantic_prototype_predictability_common_20260428 import frequency_scores, topk_metrics, write_doc, write_json


def _load_report_npz(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get("target_cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = path.parent.parent / cache_path
    return payload, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if scores.size == 0 or labels.size == 0:
        return {"top1": 0.0, "top5": 0.0, "ce": 0.0}
    return topk_metrics(scores, labels)


def _eval_one(observed_npz: Path, future_report: Path) -> dict[str, Any]:
    future_payload, future, _ = _load_report_npz(future_report)
    obs = dict(np.load(observed_npz, allow_pickle=True))
    c = int(future_payload.get("prototype_count") or future["prototypes"].shape[0])
    target = np.asarray(future["future_semantic_proto_target"], dtype=np.int64)
    mask = np.asarray(future["target_mask"], dtype=bool) & (target >= 0)
    obs_target = np.asarray(obs["observed_semantic_proto_target"], dtype=np.int64)
    obs_dist = np.asarray(obs["observed_semantic_proto_distribution"], dtype=np.float32)
    obs_mask = np.asarray(obs["observed_semantic_proto_mask"], dtype=bool)
    overlap = mask & obs_mask[:, None, :]
    labels = target[overlap]
    train_labels = target[mask]
    freq = _metrics(frequency_scores(train_labels, labels.shape[0], c), labels)
    one_hot_scores = np.eye(c, dtype=np.float32)[np.clip(obs_target, 0, c - 1)]
    copy_scores_all = np.repeat(one_hot_scores[:, None], target.shape[1], axis=1)
    dist_scores_all = np.repeat(obs_dist[:, None], target.shape[1], axis=1)
    copy = _metrics(copy_scores_all[overlap], labels)
    soft = _metrics(dist_scores_all[overlap], labels)
    stable = overlap & (target == obs_target[:, None, :])
    changed = overlap & (target != obs_target[:, None, :])
    return {
        "prototype_count": c,
        "future_report": str(future_report),
        "observed_cache_path": str(observed_npz),
        "eval_record_count": int(labels.shape[0]),
        "future_target_valid_count": int(mask.sum()),
        "observed_proto_valid_slot_count": int(obs_mask.sum()),
        "target_coverage_ratio": float(overlap.sum() / max(mask.sum(), 1)),
        "frequency": freq,
        "observed_last_copy": copy,
        "observed_soft_distribution_copy": soft,
        "observed_feature_nearest_prototype": copy,
        "stable_subset_count": int(stable.sum()),
        "changed_subset_count": int(changed.sum()),
        "changed_subset_ratio": float(changed.sum() / max(overlap.sum(), 1)),
        "copy_changed_subset": _metrics(copy_scores_all[changed], target[changed]) if changed.any() else {"top1": 0.0, "top5": 0.0, "ce": 0.0},
        "copy_stable_subset": _metrics(copy_scores_all[stable], target[stable]) if stable.any() else {"top1": 0.0, "top5": 0.0, "ce": 0.0},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v1_20260428.json")
    p.add_argument("--future-reports", nargs="+", default=[
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json",
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json",
        "reports/stwm_future_semantic_trace_prototype_targets_v2_20260428.json",
    ])
    p.add_argument("--output", default="reports/stwm_semantic_proto_persistence_baseline_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_PROTO_PERSISTENCE_BASELINE_V1_20260428.md")
    args = p.parse_args()
    observed_payload = json.loads(Path(args.observed_report).read_text(encoding="utf-8"))
    paths = observed_payload.get("target_cache_paths_by_prototype_count", {})
    results = []
    for future_path in args.future_reports:
        future_payload = json.loads(Path(future_path).read_text(encoding="utf-8"))
        c = str(int(future_payload.get("prototype_count")))
        observed_npz = Path(paths.get(c, ""))
        if not observed_npz.exists():
            continue
        results.append(_eval_one(observed_npz, Path(future_path)))
    selected = max(results, key=lambda x: (x["observed_last_copy"]["top5"], -x["prototype_count"])) if results else {}
    payload = {
        "audit_name": "stwm_semantic_proto_persistence_baseline_v1",
        "observed_report": str(args.observed_report),
        "results_by_prototype_count": results,
        "selected_prototype_count": int(selected.get("prototype_count", 0) or 0),
        "selection_reason": "highest observed-last copy top5 with low prototype count tie-break",
        "copy_baseline_strong": bool(selected.get("observed_last_copy", {}).get("top5", 0.0) > selected.get("frequency", {}).get("top5", 0.0)),
        "target_coverage_ratio": float(selected.get("target_coverage_ratio", 0.0) or 0.0),
        "copy_baseline_top1": float(selected.get("observed_last_copy", {}).get("top1", 0.0) or 0.0),
        "copy_baseline_top5": float(selected.get("observed_last_copy", {}).get("top5", 0.0) or 0.0),
        "observed_soft_distribution_copy_top1": float(selected.get("observed_soft_distribution_copy", {}).get("top1", 0.0) or 0.0),
        "observed_soft_distribution_copy_top5": float(selected.get("observed_soft_distribution_copy", {}).get("top5", 0.0) or 0.0),
        "frequency_top1": float(selected.get("frequency", {}).get("top1", 0.0) or 0.0),
        "frequency_top5": float(selected.get("frequency", {}).get("top5", 0.0) or 0.0),
        "changed_subset_top5": float(selected.get("copy_changed_subset", {}).get("top5", 0.0) or 0.0),
        "stable_subset_top5": float(selected.get("copy_stable_subset", {}).get("top5", 0.0) or 0.0),
        "changed_subset_count": int(selected.get("changed_subset_count", 0) or 0),
        "stable_subset_count": int(selected.get("stable_subset_count", 0) or 0),
        "copy_baseline_remains_strong": bool(selected.get("observed_last_copy", {}).get("top5", 0.0) > selected.get("frequency", {}).get("top5", 0.0)),
    }
    write_json(Path(args.output), payload)
    write_doc(
        Path(args.doc),
        "STWM Semantic Proto Persistence Baseline V1",
        payload,
        bullets=[
            "Observed semantic prototypes are copied across future horizon as a persistence baseline.",
            "This is a world-state memory diagnostic, not a candidate scorer.",
        ],
    )


if __name__ == "__main__":
    main()
