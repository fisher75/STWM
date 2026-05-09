#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


V30_POINTODYSSEY_CACHE = ROOT / "outputs/cache/stwm_ostf_v30_external_gt/pointodyssey"
V33_IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"
DATA_ROOT_CANDIDATES = (
    Path("/raid/chen034/workspace/data/pointodyssey"),
    Path("/home/chen034/workspace/data/pointodyssey"),
    ROOT / "data/pointodyssey",
)

SCHEMA_DOC_PATH = ROOT / "docs/STWM_OSTF_V33_SEMANTIC_IDENTITY_TARGET_SCHEMA_20260509.md"


SEMANTIC_IDENTITY_SCHEMA: dict[str, Any] = {
    "sample_uid": "str",
    "dataset": "str",
    "split": "str",
    "source_npz": "str relative path to V30 OSTF cache",
    "frame_paths": "[Tobs + H] str",
    "level_1_point_persistence": {
        "point_id": "[M] int64, persistent source point id from PointOdyssey/V30 cache",
        "fut_same_point_valid": "[M,H] bool, future supervision mask for same sampled physical point",
    },
    "level_2_instance_identity": {
        "point_to_instance_id": "[M] int64, query-frame mask id if available else -1",
        "obs_instance_id": "[M,Tobs] int64 optional mask-id lookup",
        "fut_instance_id": "[M,H] int64 optional future mask-id lookup",
        "fut_same_instance_as_obs": "[M,H] bool, target only; never model input",
    },
    "level_3_semantic_class": {
        "semantic_class_id": "[M,H] int64 optional, -1 if unavailable",
        "class_available_mask": "[M,H] bool",
    },
    "level_4_visual_semantic_prototype": {
        "teacher_embedding": "[M,H,D] optional",
        "semantic_prototype_id": "[M,H] optional",
        "teacher_source": "DINOv2|CLIP|SigLIP|SAM2_crop|none",
    },
    "safety": {
        "leakage_safe": "bool",
        "input_uses_observed_only": "bool",
        "future_targets_supervision_only": "bool",
    },
}


def scalar(z: Any, key: str, default: Any = None) -> Any:
    if key not in z:
        return default
    arr = np.asarray(z[key])
    return arr.item() if arr.shape == () else arr


def cache_paths(m_values: tuple[int, ...] = (128, 512, 1024), horizons: tuple[int, ...] = (32, 64, 96)) -> list[Path]:
    out: list[Path] = []
    for m in m_values:
        for h in horizons:
            out.extend(sorted((V30_POINTODYSSEY_CACHE / f"M{m}_H{h}").glob("*/*.npz")))
    return out


def data_root() -> Path | None:
    for root in DATA_ROOT_CANDIDATES:
        if root.exists():
            return root
    return None


def mask_path_for_frame(frame_path: str | Path) -> Path:
    p = Path(str(frame_path))
    name = p.name
    match = re.search(r"(\d+)", name)
    frame_id = match.group(1) if match else p.stem.split("_")[-1]
    return p.parent.parent / "masks" / f"mask_{frame_id}.png"


def depth_path_for_frame(frame_path: str | Path) -> Path:
    p = Path(str(frame_path))
    match = re.search(r"(\d+)", p.name)
    frame_id = match.group(1) if match else p.stem.split("_")[-1]
    return p.parent.parent / "depths" / f"depth_{frame_id}.png"


@lru_cache(maxsize=192)
def load_mask(path_str: str) -> np.ndarray | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return np.asarray(Image.open(path))
    except Exception:
        return None


def assign_mask_ids(points_xy: np.ndarray, valid: np.ndarray, mask_path: str | Path) -> np.ndarray:
    mask = load_mask(str(mask_path))
    n = int(points_xy.shape[0])
    ids = np.full((n,), -1, dtype=np.int64)
    if mask is None or mask.ndim < 2:
        return ids
    h, w = mask.shape[:2]
    pts = np.nan_to_num(points_xy.astype(np.float32), nan=-1e9, posinf=-1e9, neginf=-1e9)
    xs = np.rint(pts[:, 0]).astype(np.int64)
    ys = np.rint(pts[:, 1]).astype(np.int64)
    good = valid.astype(bool) & (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    if good.any():
        ids[good] = mask[ys[good], xs[good]].astype(np.int64)
    return ids


def sidecar_path_for_cache(cache_path: Path) -> Path:
    z = np.load(cache_path, allow_pickle=True)
    split = str(scalar(z, "split", cache_path.parent.name))
    uid = str(scalar(z, "video_uid", cache_path.stem))
    return V33_IDENTITY_ROOT / split / f"{uid}.npz"


def write_schema_doc() -> None:
    payload = {
        "generated_at_utc": utc_now(),
        "schema": SEMANTIC_IDENTITY_SCHEMA,
        "rules": {
            "observed_identity_or_semantic_context_allowed": True,
            "future_identity_or_semantic_input_allowed": False,
            "future_targets_supervision_only": True,
            "class_semantic_absence_is_not_semantic_failure": True,
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v33_semantic_identity_schema_20260509.json", payload)
    write_doc(
        SCHEMA_DOC_PATH,
        "STWM OSTF V33 Semantic Identity Target Schema",
        payload,
        ["generated_at_utc", "schema", "rules"],
    )


def main() -> int:
    write_schema_doc()
    print(SCHEMA_DOC_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
