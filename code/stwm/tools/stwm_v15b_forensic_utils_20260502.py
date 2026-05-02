#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, title: str, payload: dict[str, Any], keys: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    keys = keys or list(payload.keys())
    for key in keys:
        if key in payload:
            lines.append(f"- {key}: `{to_jsonable(payload[key])}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cache_path(m: int) -> Path:
    return ROOT / f"outputs/cache/stwm_object_dense_trace_v15/M{m}/object_dense_trace_cache.npz"


def load_cache(m: int) -> np.lib.npyio.NpzFile:
    return np.load(cache_path(m), allow_pickle=True)


def scalar_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(arr.tolist())


def teacher_source_distribution(z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    source = scalar_str(z["teacher_source"]) if "teacher_source" in z.files else "missing"
    splits = [str(x) for x in z["splits"].tolist()]
    valid_objects = z["object_valid_mask"].astype(bool)
    per_split: dict[str, dict[str, int]] = {}
    for split in sorted(set(splits)):
        mask = np.asarray([s == split for s in splits], dtype=bool)
        per_split[split] = {
            source: int(valid_objects[mask].sum() * int(np.asarray(z["M"]).item())),
            "item_count": int(mask.sum()),
            "object_count": int(valid_objects[mask].sum()),
        }
    return {
        "source": source,
        "item_count": int(len(splits)),
        "object_count": int(valid_objects.sum()),
        "point_count": int(valid_objects.sum() * int(np.asarray(z["M"]).item())),
        "per_split": per_split,
    }


def is_fallback_source(source: str) -> bool:
    lower = source.lower()
    return any(token in lower for token in ["fallback", "bbox", "grid", "cv", "pseudo", "mask_bbox_relative"])


def norm_key_from_predecode(path: Path) -> str:
    stem = path.stem
    parts = stem.split("__", 2)
    if len(parts) == 3:
        ds, _split, clip = parts
        ds_norm = "VIPSEG" if ds.lower() == "vipseg" else ds.upper()
        return f"{ds_norm}::{clip}"
    return stem


def predecode_index() -> dict[str, Path]:
    root = ROOT / "data/processed/stage2_tusb_v3_predecode_cache_20260418"
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for path in sorted(root.glob("*/*.npz")):
        out.setdefault(norm_key_from_predecode(path), path)
    return out


def split_counts(z: np.lib.npyio.NpzFile) -> dict[str, int]:
    return dict(Counter(str(x) for x in z["splits"].tolist()))


def dataset_counts(z: np.lib.npyio.NpzFile) -> dict[str, int]:
    return dict(Counter(str(x) for x in z["datasets"].tolist()))


def valid_object_pairs(z: np.lib.npyio.NpzFile, limit: int | None = None) -> list[tuple[int, int]]:
    valid = z["object_valid_mask"].astype(bool)
    pairs = [(int(i), int(j)) for i, j in zip(*np.where(valid))]
    return pairs[:limit] if limit is not None else pairs


def predecode_for_item(item_key: str, index: dict[str, Path]) -> Path | None:
    return index.get(str(item_key))


def relative_point_unique_ratio(rel: np.ndarray) -> float:
    if rel.size == 0:
        return 0.0
    rounded = {(round(float(x), 4), round(float(y), 4)) for x, y in rel.reshape(-1, 2)}
    return float(len(rounded) / max(rel.reshape(-1, 2).shape[0], 1))


def trajectory_same_delta_ratio(points_t_m_2: np.ndarray, atol: float = 1e-3) -> float:
    if points_t_m_2.ndim != 3 or points_t_m_2.shape[1] <= 1:
        return 1.0
    p = points_t_m_2.astype(np.float32, copy=False)
    delta = p - p[:1]
    base = delta[:, :1]
    same = np.all(np.abs(delta - base) <= atol, axis=(0, 2))
    return float(same.mean())


def trajectory_variance(points_t_m_2: np.ndarray) -> float:
    if points_t_m_2.size == 0:
        return 0.0
    return float(np.mean(np.var(points_t_m_2.astype(np.float32, copy=False), axis=0)))


def summarize_sources_by_m(m_values: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total_points = 0
    fallback_points = 0
    for m in m_values:
        z = load_cache(m)
        dist = teacher_source_distribution(z)
        out[f"M{m}"] = dist
        total_points += int(dist["point_count"])
        if is_fallback_source(str(dist["source"])):
            fallback_points += int(dist["point_count"])
    out["fallback_point_ratio"] = float(fallback_points / max(total_points, 1))
    return out


def grouped_counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, "missing"))] += 1
    return dict(counts)
