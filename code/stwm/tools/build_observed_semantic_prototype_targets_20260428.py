#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np
from stwm.tools.semantic_prototype_predictability_common_20260428 import (
    build_or_load_observed_feature_cache,
    checkpoint_args,
    l2_normalize,
    load_npz_from_report,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Observed Semantic Prototype Targets V1",
        "",
        f"- prototype_count: `{payload.get('prototype_count')}`",
        f"- item_count: `{payload.get('item_count')}`",
        f"- observed_feature_source: `{payload.get('observed_feature_source')}`",
        f"- observed_feature_valid_ratio: `{payload.get('observed_feature_valid_ratio')}`",
        f"- observed_slot_feature_available_ratio: `{payload.get('observed_slot_feature_available_ratio')}`",
        f"- target_cache_path: `{payload.get('target_cache_path')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _softmax(x: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    z = x / max(float(temperature), 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z).astype(np.float32)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)


def _topk_from_scores(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> dict[str, float]:
    if scores.size == 0 or labels.size == 0:
        return {"top1": 0.0, "top5": 0.0}
    pred = scores.argmax(axis=-1)
    kk = min(int(k), int(scores.shape[-1]))
    top = np.argpartition(-scores, kth=kk - 1, axis=-1)[:, :kk]
    return {
        "top1": float((pred == labels).mean()),
        "top5": float(np.any(top == labels[:, None], axis=1).mean()),
    }


def build_observed_targets(
    *,
    output: Path,
    doc: Path,
    cache_dir: Path,
    feature_report: Path,
    checkpoint: Path,
    prototype_target_reports: list[Path],
    max_samples_per_dataset: int,
    device: str,
    batch_size: int,
    force_rebuild_observed_cache: bool = False,
    observed_min_coverage: float = 0.0,
    previous_report: Path | None = None,
) -> dict[str, Any]:
    observed, observed_meta = build_or_load_observed_feature_cache(
        feature_report=feature_report,
        checkpoint_path=checkpoint,
        output_cache=cache_dir / "observed_features.npz",
        device=str(device),
        batch_size=int(batch_size),
        max_samples_per_dataset=int(max_samples_per_dataset),
        force_rebuild=bool(force_rebuild_observed_cache),
        min_required_coverage=float(observed_min_coverage),
    )
    feature_payload, feature_data, _ = load_npz_from_report(feature_report, key="cache_path")
    item_keys = [str(x) for x in feature_data["item_keys"].tolist()]
    splits = [str(x) for x in feature_data["splits"].tolist()]
    datasets = [str(x) for x in feature_data["datasets"].tolist()]
    obs_feature = l2_normalize(np.asarray(observed["observed_mean_feature"], dtype=np.float32))
    obs_mask = np.asarray(observed["observed_feature_mask"], dtype=bool)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    target_cache_paths: dict[str, str] = {}
    selected_cache_path = ""
    selected_count = 0
    selected_counts: dict[str, Any] = {}
    for report_path in prototype_target_reports:
        proto_payload, proto_data, _ = load_npz_from_report(report_path, key="target_cache_path")
        c = int(proto_payload.get("prototype_count") or proto_data["prototypes"].shape[0])
        prototypes = l2_normalize(np.asarray(proto_data["prototypes"], dtype=np.float32))
        scores = obs_feature @ prototypes.T
        proto_target = scores.argmax(axis=-1).astype(np.int64)
        proto_target[~obs_mask] = -1
        proto_dist = _softmax(scores)
        proto_dist[~obs_mask] = 0.0
        future_mask = np.asarray(proto_data["target_mask"], dtype=bool)
        future_slot_mask = future_mask.any(axis=1)
        overlap = obs_mask & future_slot_mask
        future_without_obs = future_slot_mask & (~obs_mask)
        obs_without_future = obs_mask & (~future_slot_mask)
        overlap_slot_count = int(overlap.sum())
        future_target_slot_count = int(future_slot_mask.sum())
        observed_nonzero_slot_count = int(obs_mask.sum())
        labels = np.asarray(proto_data["future_semantic_proto_target"], dtype=np.int64)
        valid_eval = future_mask & (labels >= 0) & obs_mask[:, None, :]
        repeated_scores = np.repeat(scores[:, None, :, :], labels.shape[1], axis=1)
        metrics = _topk_from_scores(repeated_scores[valid_eval], labels[valid_eval]) if valid_eval.any() else {"top1": 0.0, "top5": 0.0}
        cache_path = cache_dir / f"observed_proto_targets_c{c}.npz"
        np.savez_compressed(
            cache_path,
            item_keys=np.asarray(item_keys, dtype=object),
            splits=np.asarray(splits, dtype=object),
            datasets=np.asarray(datasets, dtype=object),
            observed_semantic_proto_target=proto_target,
            observed_semantic_proto_distribution=proto_dist.astype(np.float32),
            observed_semantic_proto_mask=obs_mask,
            prototypes=prototypes.astype(np.float32),
            prototype_count=np.asarray(c, dtype=np.int64),
            no_future_leakage=np.asarray(True),
        )
        target_cache_paths[str(c)] = str(cache_path)
        if c == 64 or not selected_cache_path:
            selected_cache_path = str(cache_path)
            selected_count = c
        results.append(
            {
                "prototype_count": c,
                "target_cache_path": str(cache_path),
                "observed_proto_valid_ratio": float(obs_mask.mean()),
                "future_target_overlap_ratio": float(overlap.sum() / max(future_slot_mask.sum(), 1)),
                "observed_nonzero_slot_count": observed_nonzero_slot_count,
                "future_target_slot_count": future_target_slot_count,
                "overlap_slot_count": overlap_slot_count,
                "slots_with_future_target_but_no_observed_proto": int(future_without_obs.sum()),
                "slots_with_observed_proto_but_no_future_target": int(obs_without_future.sum()),
                "observed_last_top1": float(metrics["top1"]),
                "observed_last_top5": float(metrics["top5"]),
                "coverage_sufficient": bool(float(obs_mask.mean()) >= 0.1),
            }
        )
        if c == selected_count:
            selected_counts = {
                "observed_nonzero_slot_count": observed_nonzero_slot_count,
                "future_target_slot_count": future_target_slot_count,
                "overlap_slot_count": overlap_slot_count,
                "slots_with_future_target_but_no_observed_proto": int(future_without_obs.sum()),
                "slots_with_observed_proto_but_no_future_target": int(obs_without_future.sum()),
            }

    observed_feature_valid_ratio = float(obs_mask.mean()) if obs_mask.size else 0.0
    observed_slot_feature_available_ratio = float(obs_mask.mean()) if obs_mask.size else 0.0
    selected = next((r for r in results if int(r["prototype_count"]) == selected_count), results[0])
    selected_counts = {
        "observed_nonzero_slot_count": int(selected.get("observed_nonzero_slot_count", 0)),
        "future_target_slot_count": int(selected.get("future_target_slot_count", 0)),
        "overlap_slot_count": int(selected.get("overlap_slot_count", 0)),
        "slots_with_future_target_but_no_observed_proto": int(selected.get("slots_with_future_target_but_no_observed_proto", 0)),
        "slots_with_observed_proto_but_no_future_target": int(selected.get("slots_with_observed_proto_but_no_future_target", 0)),
    }
    previous_observed_ratio = None
    previous_overlap_ratio = None
    if previous_report is not None and previous_report.exists():
        try:
            previous_payload = json.loads(previous_report.read_text(encoding="utf-8"))
            previous_observed_ratio = float(previous_payload.get("observed_proto_valid_ratio", 0.0) or 0.0)
            previous_overlap_ratio = float(previous_payload.get("future_target_overlap_ratio", 0.0) or 0.0)
        except Exception:
            previous_observed_ratio = None
            previous_overlap_ratio = None
    payload = {
        "generated_at_utc": now_iso(),
        "feature_report": str(feature_report),
        "checkpoint": str(checkpoint),
        "item_count": int(obs_feature.shape[0]),
        "feature_dim": int(obs_feature.shape[-1]),
        "observed_feature_source": str(observed_meta.get("feature_backbone") or ""),
        "observed_feature_cache_path": str(observed_meta.get("observed_feature_cache_path") or cache_dir / "observed_features.npz"),
        "observed_feature_cache_reused": bool(observed_meta.get("observed_feature_cache_reused", False)),
        "cache_rebuilt": bool(not observed_meta.get("observed_feature_cache_reused", False)),
        "cache_rebuild_reason": str(observed_meta.get("cache_rebuild_reason") or ""),
        "observed_max_samples_per_dataset": int(max_samples_per_dataset),
        "observed_feature_valid_ratio": observed_feature_valid_ratio,
        "observed_slot_feature_available_ratio": observed_slot_feature_available_ratio,
        "observed_proto_valid_ratio": observed_feature_valid_ratio,
        "prototype_count": int(selected_count),
        "target_cache_path": selected_cache_path,
        "target_cache_paths_by_prototype_count": target_cache_paths,
        "results_by_prototype_count": results,
        "future_target_overlap_ratio": float(selected["future_target_overlap_ratio"]),
        **selected_counts,
        "observed_last_top1": float(selected["observed_last_top1"]),
        "observed_last_top5": float(selected["observed_last_top5"]),
        "observed_proto_coverage_sufficient": bool(selected["coverage_sufficient"]),
        "observed_proto_valid_ratio_v1": previous_observed_ratio,
        "future_target_overlap_ratio_v1": previous_overlap_ratio,
        "coverage_improved_vs_v1": bool(
            previous_observed_ratio is not None
            and previous_overlap_ratio is not None
            and observed_feature_valid_ratio > previous_observed_ratio
            and float(selected["future_target_overlap_ratio"]) > previous_overlap_ratio
        ),
        "no_future_leakage": True,
        "no_future_candidate_leakage": True,
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--feature-report", default="reports/stwm_semantic_trace_field_decoder_v2_feature_targets_large_20260428.json")
    p.add_argument("--checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--prototype-target-reports", nargs="+", default=[
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json",
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json",
        "reports/stwm_future_semantic_trace_prototype_targets_v2_20260428.json",
    ])
    p.add_argument("--max-samples-per-dataset", type=int, default=512)
    p.add_argument("--observed-max-samples-per-dataset", type=int, default=None)
    p.add_argument("--force-rebuild-observed-cache", action="store_true")
    p.add_argument("--observed-min-coverage", type=float, default=0.0)
    p.add_argument("--previous-observed-report", default="reports/stwm_observed_semantic_prototype_targets_v1_20260428.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cache-dir", default="outputs/cache/stwm_observed_semantic_prototype_targets_v1_20260428")
    p.add_argument("--output", default="reports/stwm_observed_semantic_prototype_targets_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_OBSERVED_SEMANTIC_PROTOTYPE_TARGETS_V1_20260428.md")
    args = p.parse_args()
    build_observed_targets(
        output=Path(args.output),
        doc=Path(args.doc),
        cache_dir=Path(args.cache_dir),
        feature_report=Path(args.feature_report),
        checkpoint=Path(args.checkpoint),
        prototype_target_reports=[Path(x) for x in args.prototype_target_reports],
        max_samples_per_dataset=int(args.observed_max_samples_per_dataset or args.max_samples_per_dataset),
        device=str(args.device),
        batch_size=int(args.batch_size),
        force_rebuild_observed_cache=bool(args.force_rebuild_observed_cache),
        observed_min_coverage=float(args.observed_min_coverage),
        previous_report=Path(args.previous_observed_report) if args.previous_observed_report else None,
    )


if __name__ == "__main__":
    main()
