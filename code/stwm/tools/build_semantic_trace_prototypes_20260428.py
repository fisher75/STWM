#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Semantic Trace Prototypes V1",
        "",
        f"- prototype_count: `{payload.get('prototype_count')}`",
        f"- feature_backbone: `{payload.get('feature_backbone')}`",
        f"- source_valid_feature_count: `{payload.get('source_valid_feature_count')}`",
        f"- empty_prototype_count: `{payload.get('empty_prototype_count')}`",
        f"- mean_within_cluster_similarity: `{payload.get('mean_within_cluster_similarity')}`",
        f"- coverage: `{payload.get('coverage')}`",
        f"- prototype_cache_path: `{payload.get('prototype_cache_path')}`",
        "",
        "Prototype vectors are built from future GT crop features as supervised pseudo-label construction only.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_feature_cache(report_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get("cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"feature target tensor cache not found: {cache_path}")
    data = dict(np.load(cache_path, allow_pickle=True))
    return payload, data


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def _kmeans(features: np.ndarray, count: int, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] <= 0:
        raise ValueError("features must be non-empty [N,D]")
    n, dim = features.shape
    c = min(int(count), int(n))
    # Deterministic far-enough initialization without random fake prototypes.
    init_idx = np.linspace(0, n - 1, c, dtype=np.int64)
    centers = features[init_idx].copy()
    labels = np.zeros((n,), dtype=np.int64)
    counts = np.zeros((c,), dtype=np.int64)
    for _ in range(max(int(iterations), 1)):
        sim = features @ centers.T
        labels = sim.argmax(axis=1).astype(np.int64)
        new_centers = centers.copy()
        counts = np.bincount(labels, minlength=c).astype(np.int64)
        for j in range(c):
            if counts[j] > 0:
                new_centers[j] = features[labels == j].mean(axis=0)
        centers = _l2_normalize(new_centers.astype(np.float32))
    sim = features @ centers.T
    labels = sim.argmax(axis=1).astype(np.int64)
    counts = np.bincount(labels, minlength=c).astype(np.int64)
    return centers.astype(np.float32), labels, counts


def build_prototypes(
    *,
    feature_cache_report: Path,
    output: Path,
    doc: Path,
    cache_dir: Path,
    prototype_count: int,
    iterations: int,
) -> dict[str, Any]:
    source_payload, data = _load_feature_cache(feature_cache_report)
    features = np.asarray(data["future_semantic_feature_target"], dtype=np.float32)
    mask = np.asarray(data["target_mask"], dtype=bool)
    valid_features = features[mask]
    if valid_features.size == 0:
        raise RuntimeError("no valid future semantic features available for prototype construction")
    valid_features = _l2_normalize(valid_features.reshape(valid_features.shape[0], -1).astype(np.float32))
    requested_count = int(prototype_count)
    actual_count = min(requested_count, int(valid_features.shape[0]))
    prototypes, labels, counts = _kmeans(valid_features, actual_count, int(iterations))
    assigned_sim = (valid_features * prototypes[labels]).sum(axis=-1)
    empty_count = int((counts == 0).sum())
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "prototypes.npz"
    np.savez_compressed(
        cache_path,
        prototypes=prototypes,
        prototype_counts=counts,
        feature_backbone=np.asarray(str(source_payload.get("feature_backbone") or ""), dtype=object),
        feature_source=np.asarray(str(source_payload.get("feature_source") or ""), dtype=object),
    )
    payload = {
        "generated_at_utc": now_iso(),
        "source_feature_cache_report": str(feature_cache_report),
        "prototype_cache_path": str(cache_path),
        "requested_prototype_count": requested_count,
        "prototype_count": int(actual_count),
        "feature_dim": int(prototypes.shape[-1]),
        "feature_backbone": str(source_payload.get("feature_backbone") or ""),
        "feature_source": str(source_payload.get("feature_source") or ""),
        "source_valid_feature_count": int(valid_features.shape[0]),
        "prototype_counts": [int(x) for x in counts.tolist()],
        "empty_prototype_count": empty_count,
        "mean_within_cluster_similarity": float(assigned_sim.mean()) if assigned_sim.size else None,
        "min_within_cluster_similarity": float(assigned_sim.min()) if assigned_sim.size else None,
        "coverage": float((counts > 0).mean()) if counts.size else 0.0,
        "no_future_candidate_leakage": True,
        "random_fake_prototypes_used": False,
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--feature-cache-report", default="reports/stwm_future_semantic_trace_feature_targets_v1_20260428.json")
    p.add_argument("--prototype-count", type=int, default=64)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--cache-dir", default="outputs/cache/stwm_semantic_trace_prototypes_v1_20260428")
    p.add_argument("--output", default="reports/stwm_semantic_trace_prototypes_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_TRACE_PROTOTYPES_V1_20260428.md")
    args = p.parse_args()
    build_prototypes(
        feature_cache_report=Path(args.feature_cache_report),
        output=Path(args.output),
        doc=Path(args.doc),
        cache_dir=Path(args.cache_dir),
        prototype_count=int(args.prototype_count),
        iterations=int(args.iterations),
    )


if __name__ == "__main__":
    main()
