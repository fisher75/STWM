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
        "# STWM Future Semantic Trace Prototype Targets V1",
        "",
        f"- prototype_count: `{payload.get('prototype_count')}`",
        f"- item_count: `{payload.get('item_count')}`",
        f"- target_shape: `{payload.get('target_shape')}`",
        f"- target_valid_ratio: `{payload.get('target_valid_ratio')}`",
        f"- prototype_entropy: `{payload.get('prototype_entropy')}`",
        f"- long_tail_warning: `{payload.get('long_tail_warning')}`",
        f"- target_cache_path: `{payload.get('target_cache_path')}`",
        "",
        "Missing targets are masked, not filled as semantic class zero.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_report_npz(report_path: Path, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get(key) or payload.get("cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"npz cache not found: {cache_path}")
    return payload, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def build_targets(
    *,
    feature_cache_report: Path,
    prototype_report: Path,
    output: Path,
    doc: Path,
    cache_dir: Path,
    soft_temperature: float,
) -> dict[str, Any]:
    feature_payload, feature_data, _ = _load_report_npz(feature_cache_report, "cache_path")
    proto_payload, proto_data, proto_cache_path = _load_report_npz(prototype_report, "prototype_cache_path")
    features = np.asarray(feature_data["future_semantic_feature_target"], dtype=np.float32)
    mask = np.asarray(feature_data["target_mask"], dtype=bool)
    prototypes = np.asarray(proto_data["prototypes"], dtype=np.float32)
    prototypes = _l2_normalize(prototypes)
    feature_norm = _l2_normalize(features)
    n, h, k, _ = feature_norm.shape
    c = int(prototypes.shape[0])
    flat = feature_norm.reshape(-1, feature_norm.shape[-1])
    sims = flat @ prototypes.T
    logits = sims.reshape(n, h, k, c)
    proto_target = logits.argmax(axis=-1).astype(np.int64)
    proto_target[~mask] = -1
    temp = max(float(soft_temperature), 1e-6)
    shifted = logits / temp
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    proto_dist = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)
    proto_dist[~mask] = 0.0
    counts = np.bincount(proto_target[mask].reshape(-1), minlength=c).astype(np.int64) if bool(mask.any()) else np.zeros((c,), dtype=np.int64)
    entropy = _entropy_from_counts(counts)
    nonzero = counts[counts > 0]
    long_tail_warning = bool(nonzero.size > 0 and (counts == 0).sum() > 0 or (nonzero.size > 0 and nonzero.min() / max(nonzero.max(), 1) < 0.05))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "prototype_targets.npz"
    np.savez_compressed(
        cache_path,
        item_keys=feature_data["item_keys"],
        splits=feature_data["splits"],
        datasets=feature_data["datasets"],
        future_semantic_proto_target=proto_target,
        future_semantic_proto_distribution=proto_dist.astype(np.float32),
        target_mask=mask,
        future_visibility_target=np.asarray(feature_data["future_visibility_target"], dtype=bool),
        future_reappearance_target=np.asarray(feature_data["future_reappearance_target"], dtype=bool),
        identity_target=np.asarray(feature_data["identity_target"], dtype=np.int64),
        future_extent_box_target=np.asarray(feature_data["extent_box_target"], dtype=np.float32),
        prototypes=prototypes.astype(np.float32),
    )
    payload = {
        "generated_at_utc": now_iso(),
        "feature_cache_report": str(feature_cache_report),
        "prototype_report": str(prototype_report),
        "prototype_cache_path": str(proto_cache_path),
        "target_cache_path": str(cache_path),
        "item_count": int(n),
        "target_shape": [int(n), int(h), int(k)],
        "target_mask_shape": [int(n), int(h), int(k)],
        "prototype_count": int(c),
        "target_valid_ratio": float(mask.mean()) if mask.size else 0.0,
        "prototype_entropy": entropy,
        "prototype_class_counts": [int(x) for x in counts.tolist()],
        "empty_target_prototype_count": int((counts == 0).sum()),
        "long_tail_warning": long_tail_warning,
        "feature_backbone": str(feature_payload.get("feature_backbone") or proto_payload.get("feature_backbone") or ""),
        "future_visibility_target_available": "future_visibility_target" in feature_data,
        "future_reappearance_target_available": "future_reappearance_target" in feature_data,
        "future_extent_box_target_available": "extent_box_target" in feature_data,
        "identity_target_available": "identity_target" in feature_data,
        "no_future_candidate_leakage": True,
        "missing_target_filled_as_zero": False,
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--feature-cache-report", default="reports/stwm_future_semantic_trace_feature_targets_v1_20260428.json")
    p.add_argument("--prototype-report", default="reports/stwm_semantic_trace_prototypes_v1_20260428.json")
    p.add_argument("--soft-temperature", type=float, default=0.07)
    p.add_argument("--cache-dir", default="outputs/cache/stwm_future_semantic_trace_prototype_targets_v1_20260428")
    p.add_argument("--output", default="reports/stwm_future_semantic_trace_prototype_targets_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_FUTURE_SEMANTIC_TRACE_PROTOTYPE_TARGETS_V1_20260428.md")
    args = p.parse_args()
    build_targets(
        feature_cache_report=Path(args.feature_cache_report),
        prototype_report=Path(args.prototype_report),
        output=Path(args.output),
        doc=Path(args.doc),
        cache_dir=Path(args.cache_dir),
        soft_temperature=float(args.soft_temperature),
    )


if __name__ == "__main__":
    main()
