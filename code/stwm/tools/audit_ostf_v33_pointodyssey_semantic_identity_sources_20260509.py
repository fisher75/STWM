#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import (
    V30_POINTODYSSEY_CACHE,
    data_root,
    depth_path_for_frame,
    mask_path_for_frame,
)

REPORT = ROOT / "reports/stwm_ostf_v33_pointodyssey_semantic_identity_source_audit_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_POINTODYSSEY_SEMANTIC_IDENTITY_SOURCE_AUDIT_20260509.md"


def field(available: bool, path: str | None = None, count: int = 0, schema: Any = None, shape: Any = None, reliability: str = "unknown", blocker: str | None = None) -> dict[str, Any]:
    return {
        "available": bool(available),
        "path": path,
        "count": int(count),
        "example_schema": schema,
        "example_shape": shape,
        "reliability_status": reliability,
        "exact_blocker": blocker,
    }


def main() -> int:
    root = data_root()
    annos = sorted(root.glob("*/*/anno.npz")) if root else []
    caches = sorted(V30_POINTODYSSEY_CACHE.glob("M128_H32/*/*.npz"))
    anno_keys: list[str] = []
    anno_shapes: dict[str, Any] = {}
    if annos:
        z = np.load(annos[0], allow_pickle=True)
        anno_keys = list(z.files)
        anno_shapes = {k: list(np.asarray(z[k]).shape) for k in z.files}
    cache_keys: list[str] = []
    cache_shapes: dict[str, Any] = {}
    cache_scalar_values: dict[str, Any] = {}
    frame_paths = []
    if caches:
        zc = np.load(caches[0], allow_pickle=True)
        cache_keys = list(zc.files)
        cache_shapes = {k: list(np.asarray(zc[k]).shape) for k in zc.files}
        for key in ("object_id", "instance_id", "semantic_id"):
            if key in zc.files:
                cache_scalar_values[key] = np.asarray(zc[key]).item()
        frame_paths = [str(x) for x in np.asarray(zc["frame_paths"]).tolist()[:4]] if "frame_paths" in zc.files else []
    mask_dirs = sorted(root.glob("*/*/masks")) if root else []
    depth_dirs = sorted(root.glob("*/*/depths")) if root else []
    rgb_dirs = sorted(root.glob("*/*/rgbs")) if root else []
    scene_infos = sorted(root.glob("*/*/scene_info.json")) if root else []
    mask_example = None
    if frame_paths:
        mp = mask_path_for_frame(frame_paths[-1])
        mask_example = str(mp) if mp.exists() else None
    depth_example = None
    if frame_paths:
        dp = depth_path_for_frame(frame_paths[-1])
        depth_example = str(dp) if dp.exists() else None

    has_cache_point_id = "point_id" in cache_keys
    has_raw_point_tracks = {"trajs_2d", "valids", "visibs"}.issubset(set(anno_keys))
    instance_in_cache = any(cache_scalar_values.get(k, -1) != -1 for k in ("object_id", "instance_id", "semantic_id"))
    has_masks = bool(mask_dirs)
    fields = {
        "instance_segmentation": field(has_masks, str(mask_dirs[0]) if mask_dirs else None, len(mask_dirs), "per-frame indexed PNG masks", None, "usable_for_mask_lookup" if has_masks else "missing", None if has_masks else "no masks directory"),
        "object_or_instance_ids": field(instance_in_cache or has_masks, mask_example, len(mask_dirs), "mask id at point coordinate; V30 scalar ids are -1", cache_scalar_values, "mask_lookup_available_cache_scalar_missing" if has_masks else "unavailable"),
        "point_persistent_id": field(has_cache_point_id, str(caches[0]) if caches else None, len(caches), "V30 cache point_id [M]", cache_shapes.get("point_id"), "high" if has_cache_point_id else "missing"),
        "sampled_point_index_traceback": field(has_cache_point_id and has_raw_point_tracks, str(caches[0]) if caches else None, len(caches), "point_id indexes PointOdyssey anno trajs_* point dimension", {"cache_point_id": cache_shapes.get("point_id"), "anno_trajs_2d": anno_shapes.get("trajs_2d")}, "high" if has_cache_point_id and has_raw_point_tracks else "blocked"),
        "point_to_object_or_instance_mapping": field(has_masks, mask_example, len(mask_dirs), "query/future coordinate mask lookup", None, "derived_reliable_if_coordinate_inside_mask" if has_masks else "unavailable"),
        "per_frame_masks": field(has_masks, str(mask_dirs[0]) if mask_dirs else None, len(mask_dirs), "mask_XXXXX.png", None, "available" if has_masks else "missing"),
        "rgb_frame_paths": field(bool(rgb_dirs), str(rgb_dirs[0]) if rgb_dirs else None, len(rgb_dirs), "rgbs/rgb_XXXXX.jpg", None, "available" if rgb_dirs else "missing"),
        "depth": field(bool(depth_dirs), depth_example or (str(depth_dirs[0]) if depth_dirs else None), len(depth_dirs), "depths/depth_XXXXX.png", None, "available" if depth_dirs else "missing"),
        "intrinsics_extrinsics": field("intrinsics" in anno_keys and "extrinsics" in anno_keys, str(annos[0]) if annos else None, len(annos), "anno.npz intrinsics/extrinsics", {"intrinsics": anno_shapes.get("intrinsics"), "extrinsics": anno_shapes.get("extrinsics")}, "available"),
        "semantic_class_category_label": field(False, str(scene_infos[0]) if scene_infos else None, len(scene_infos), "scene_info assets names only, no per-point class id", None, "not_available_as_class_target", "PointOdyssey local anno/cache has no point-level class semantic ids"),
        "future_labels": field(True, str(caches[0]) if caches else None, len(caches), "future point validity and future mask ids are supervision only", {"fut_vis": cache_shapes.get("fut_vis")}, "available_supervision_only"),
        "v30_cache_metadata_sufficient": field(has_cache_point_id, str(caches[0]) if caches else None, len(caches), cache_keys, cache_shapes, "sufficient_for_point_identity_needs_sidecar_for_mask_instance" if has_cache_point_id else "insufficient"),
    }
    payload = {
        "generated_at_utc": utc_now(),
        "pointodyssey_root": str(root) if root else None,
        "anno_count": len(annos),
        "v30_m128_h32_cache_count": len(caches),
        "anno_keys": anno_keys,
        "cache_keys": cache_keys,
        "fields": fields,
        "identity_available": has_cache_point_id,
        "instance_identity_available": has_masks,
        "class_semantic_available": False,
        "visual_teacher_semantic_needed": True,
        "future_labels_supervision_only": True,
        "future_labels_enter_model_input": False,
        "current_v30_cache_enough_for_point_identity": has_cache_point_id,
        "current_v30_cache_enough_for_instance_mapping": False,
        "sidecar_needed": True,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 PointOdyssey Semantic Identity Source Audit", payload, [
        "identity_available",
        "instance_identity_available",
        "class_semantic_available",
        "visual_teacher_semantic_needed",
        "sidecar_needed",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
