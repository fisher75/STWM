#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM Object-Dense Trace Cache V15", ""]
    for key in ["materialization_success", "item_count", "object_count", "point_count", "valid_point_ratio", "future_leakage_audit"]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    for name, row in payload.get("caches", {}).items():
        lines.append(f"## {name}")
        for key in ["cache_path", "item_count", "object_count", "point_count", "valid_point_ratio", "mask_coverage", "cache_size_bytes"]:
            lines.append(f"- {key}: `{row.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_key_from_predecode(path: Path) -> str:
    stem = path.stem
    parts = stem.split("__", 2)
    if len(parts) == 3:
        ds, _split, clip = parts
        ds_norm = "VIPSEG" if ds.lower() == "vipseg" else ds.upper()
        return f"{ds_norm}::{clip}"
    return stem


def _split_maps() -> dict[str, str]:
    splits = _load_json(Path("reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json")).get("splits", {})
    out: dict[str, str] = {}
    for split, keys in splits.items():
        for key in keys:
            out[str(key)] = str(split)
    return out


def _predecode_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(Path("data/processed/stage2_tusb_v3_predecode_cache_20260418").glob("*/*.npz")):
        key = _norm_key_from_predecode(path)
        out.setdefault(key, path)
    return out


def _sample_rel_points(mask: np.ndarray, m: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    m = int(m)
    if m <= 1:
        return np.asarray([[0.5, 0.5]], dtype=np.float32), {"mask_used": bool(mask.any()), "boundary_fraction": 0.0}
    fg = np.asarray(mask > 0)
    rng = np.random.default_rng(seed)
    if not fg.any():
        side = int(np.ceil(np.sqrt(m)))
        xs, ys = np.meshgrid(np.linspace(0.1, 0.9, side), np.linspace(0.1, 0.9, side))
        pts = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)[:m]
        return pts.astype(np.float32), {"mask_used": False, "boundary_fraction": 0.0}
    padded = np.pad(fg, 1, constant_values=False)
    neighbor_count = (
        padded[0:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
    )
    boundary = fg & (neighbor_count < 4)
    interior = fg & (~boundary)
    by, bx = np.where(boundary)
    iy, ix = np.where(interior if interior.any() else fg)
    nb = min(max(m // 4, 1), len(bx)) if len(bx) else 0
    ni = m - nb
    pts: list[np.ndarray] = []
    if nb:
        idx = rng.choice(len(bx), size=nb, replace=len(bx) < nb)
        pts.append(np.stack([bx[idx], by[idx]], axis=-1))
    idx = rng.choice(len(ix), size=ni, replace=len(ix) < ni)
    pts.append(np.stack([ix[idx], iy[idx]], axis=-1))
    raw = np.concatenate(pts, axis=0).astype(np.float32)
    rng.shuffle(raw)
    h, w = fg.shape
    rel = np.stack([(raw[:, 0] + 0.5) / max(w, 1), (raw[:, 1] + 0.5) / max(h, 1)], axis=-1)
    return rel.astype(np.float32), {"mask_used": True, "boundary_fraction": float(nb / max(m, 1))}


def _points_from_boxes(rel: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = int(boxes.shape[0])
    m = int(rel.shape[0])
    pts = np.zeros((t, m, 2), dtype=np.float32)
    valid = np.zeros((t, m), dtype=bool)
    for i in range(t):
        x0, y0, x1, y1 = [float(x) for x in boxes[i].tolist()]
        if x1 <= x0 + 1 or y1 <= y0 + 1:
            continue
        pts[i, :, 0] = x0 + rel[:, 0] * (x1 - x0)
        pts[i, :, 1] = y0 + rel[:, 1] * (y1 - y0)
        valid[i, :] = True
    return pts, valid


def _draw_visual(path: Path, z: np.lib.npyio.NpzFile, selected_objects: list[int], rel_points: dict[int, np.ndarray]) -> None:
    rgb = np.asarray(z["semantic_rgb_crop"][selected_objects[0]]).transpose(1, 2, 0)
    img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8)).resize((256, 256))
    draw = ImageDraw.Draw(img)
    for obj in selected_objects:
        pts = rel_points[obj]
        color = (255, 64 + (obj * 37) % 160, 32 + (obj * 53) % 200)
        for x, y in pts[: min(len(pts), 512)]:
            px = float(x) * 256.0
            py = float(y) * 256.0
            draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _build_for_m(m: int, split_map: dict[str, str], predecode: dict[str, Path], out_root: Path, *, max_items: int) -> dict[str, Any]:
    t0 = time.time()
    keys = [k for k in split_map if k in predecode]
    keys = sorted(keys)[: max_items if max_items > 0 else None]
    items: list[str] = []
    splits: list[str] = []
    datasets: list[str] = []
    point_rows: list[np.ndarray] = []
    rel_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    object_valid_rows: list[np.ndarray] = []
    object_ids: list[np.ndarray] = []
    semantic_ids: list[np.ndarray] = []
    mask_used_rows: list[np.ndarray] = []
    failed: list[dict[str, Any]] = []
    visual_written = False
    for idx, key in enumerate(keys):
        try:
            z = np.load(predecode[key], allow_pickle=True)
            boxes = np.asarray(z["entity_boxes_over_time"], dtype=np.float32)
            masks = np.asarray(z["semantic_mask_crop"], dtype=np.float32)
            obj_count = int(min(boxes.shape[1], masks.shape[0]))
            if obj_count <= 0:
                continue
            t = int(boxes.shape[0])
            pts_item = np.zeros((obj_count, t, m, 2), dtype=np.float32)
            rel_item = np.zeros((obj_count, m, 2), dtype=np.float32)
            valid_item = np.zeros((obj_count, t, m), dtype=bool)
            mask_used_item = np.zeros((obj_count,), dtype=bool)
            obj_valid = np.zeros((obj_count,), dtype=bool)
            rel_for_visual: dict[int, np.ndarray] = {}
            for j in range(obj_count):
                rel, info = _sample_rel_points(masks[j, 0] > 0.5, m, seed=int(hashlib.md5(f"{key}-{j}-{m}".encode()).hexdigest()[:8], 16))
                pts, valid = _points_from_boxes(rel, boxes[:, j])
                rel_item[j] = rel
                pts_item[j] = pts
                valid_item[j] = valid
                mask_used_item[j] = bool(info["mask_used"])
                obj_valid[j] = bool(valid.any())
                rel_for_visual[j] = rel
            if not obj_valid.any():
                continue
            items.append(key)
            splits.append(split_map[key])
            ds = key.split("::", 1)[0]
            datasets.append(ds)
            point_rows.append(pts_item.astype(np.float16))
            rel_rows.append(rel_item.astype(np.float16))
            valid_rows.append(valid_item)
            object_valid_rows.append(obj_valid)
            object_ids.append(np.arange(obj_count, dtype=np.int32))
            sem = np.asarray(z.get("semantic_entity_dominant_instance_id", np.arange(obj_count)), dtype=np.int64)[:obj_count]
            semantic_ids.append(sem.astype(np.int64))
            mask_used_rows.append(mask_used_item)
            if not visual_written and m >= 128:
                _draw_visual(Path("assets/figures/stwm_object_dense_trace_v15") / f"M{m}_sample_points.png", z, [0], rel_for_visual)
                visual_written = True
        except Exception as exc:
            failed.append({"item_key": key, "reason": repr(exc)})
    max_obj = max((x.shape[0] for x in point_rows), default=0)
    t_steps = max((x.shape[1] for x in point_rows), default=0)
    n = len(point_rows)
    points = np.zeros((n, max_obj, t_steps, m, 2), dtype=np.float16)
    rels = np.zeros((n, max_obj, m, 2), dtype=np.float16)
    valid = np.zeros((n, max_obj, t_steps, m), dtype=bool)
    obj_valid_all = np.zeros((n, max_obj), dtype=bool)
    obj_id_all = np.full((n, max_obj), -1, dtype=np.int32)
    sem_all = np.full((n, max_obj), -1, dtype=np.int64)
    mask_used_all = np.zeros((n, max_obj), dtype=bool)
    for i in range(n):
        o, t, _, _ = point_rows[i].shape
        points[i, :o, :t] = point_rows[i]
        rels[i, :o] = rel_rows[i]
        valid[i, :o, :t] = valid_rows[i]
        obj_valid_all[i, :o] = object_valid_rows[i]
        obj_id_all[i, :o] = object_ids[i]
        sem_all[i, :o] = semantic_ids[i]
        mask_used_all[i, :o] = mask_used_rows[i]
    out_dir = out_root / f"M{m}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "object_dense_trace_cache.npz"
    np.savez_compressed(
        cache_path,
        item_keys=np.asarray(items, dtype=object),
        splits=np.asarray(splits, dtype=object),
        datasets=np.asarray(datasets, dtype=object),
        points_xy=points,
        object_relative_xy=rels,
        valid_mask=valid,
        object_valid_mask=obj_valid_all,
        object_id=obj_id_all,
        semantic_or_instance_id=sem_all,
        mask_used=mask_used_all,
        obs_len=np.asarray(8, dtype=np.int32),
        horizon=np.asarray(8, dtype=np.int32),
        M=np.asarray(m, dtype=np.int32),
        teacher_source=np.asarray("mask_bbox_relative_pseudo_track", dtype=object),
        no_future_leakage=np.asarray(True),
    )
    valid_ratio = float(valid.sum() / max(valid.size, 1))
    object_count = int(obj_valid_all.sum())
    point_count = int(obj_valid_all.sum() * m)
    per_dataset = {}
    for ds in sorted(set(datasets)):
        ds_mask = np.asarray([x == ds for x in datasets], dtype=bool)
        per_dataset[ds] = {
            "item_count": int(ds_mask.sum()),
            "object_count": int(obj_valid_all[ds_mask].sum()),
            "point_count": int(obj_valid_all[ds_mask].sum() * m),
            "valid_point_ratio": float(valid[ds_mask].sum() / max(valid[ds_mask].size, 1)) if ds_mask.any() else 0.0,
        }
    return {
        "M": m,
        "cache_path": str(cache_path),
        "item_count": n,
        "object_count": object_count,
        "point_count": point_count,
        "valid_point_ratio": valid_ratio,
        "mask_coverage": float(mask_used_all[obj_valid_all].mean()) if object_count else 0.0,
        "boundary_coverage": "boundary_plus_interior_mixture_for_M>=128",
        "per_dataset": per_dataset,
        "failed_items": failed[:50],
        "failed_item_count": len(failed),
        "cache_size_bytes": int(cache_path.stat().st_size),
        "checksum_sha256": _sha256(cache_path),
        "runtime_seconds": float(time.time() - t0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,128,512")
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--output-root", default="outputs/cache/stwm_object_dense_trace_v15")
    args = parser.parse_args()
    split_map = _split_maps()
    predecode = _predecode_index()
    out_root = Path(args.output_root)
    caches = {}
    for m in [int(x) for x in args.m_values.split(",") if x.strip()]:
        caches[f"M{m}"] = _build_for_m(m, split_map, predecode, out_root, max_items=args.max_items)
    total_items = max((v["item_count"] for v in caches.values()), default=0)
    total_objects = max((v["object_count"] for v in caches.values()), default=0)
    total_points = {k: v["point_count"] for k, v in caches.items()}
    payload = {
        "audit_name": "stwm_object_dense_trace_cache_v15",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "materialization_success": bool(caches and all(v["item_count"] > 0 for v in caches.values())),
        "teacher_source": "mask_bbox_relative_pseudo_track",
        "physical_point_teacher_used": False,
        "item_count": total_items,
        "object_count": total_objects,
        "point_count": total_points,
        "valid_point_ratio": {k: v["valid_point_ratio"] for k, v in caches.items()},
        "future_leakage_audit": "observed input uses first 8 points only; future points stored as targets in cache",
        "H8_cache_available": True,
        "H16_cache_available": False,
        "H16_blocker": "stage2_tusb_v3_predecode_cache contains 16 total timesteps (obs8+future8); H16 requires raw rematerialization or an external point teacher over 24 timesteps",
        "caches": caches,
        "raw_visualization": sorted(str(p) for p in Path("assets/figures/stwm_object_dense_trace_v15").glob("*.png")),
    }
    _dump(Path("reports/stwm_object_dense_trace_cache_v15_20260502.json"), payload)
    _write_doc(Path("docs/STWM_OBJECT_DENSE_TRACE_CACHE_V15_20260502.md"), payload)
    print("reports/stwm_object_dense_trace_cache_v15_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
