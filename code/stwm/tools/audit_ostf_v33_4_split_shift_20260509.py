#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"
VIS_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
REPORT = ROOT / "reports/stwm_ostf_v33_4_split_shift_audit_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_4_SPLIT_SHIFT_AUDIT_20260509.md"


def entropy(ids: np.ndarray, mask: np.ndarray) -> float | None:
    vals = ids[mask & (ids >= 0)]
    if vals.size == 0:
        return None
    _, counts = np.unique(vals, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / max(np.log(max(len(counts), 2)), 1e-6))


def split_stats(split: str, max_items: int = 128) -> dict[str, Any]:
    files = sorted((IDENTITY_ROOT / split).glob("*_M128_H32.npz"))[:max_items]
    pos = neg = avail = vis = occ = 0
    inst_counts = []
    confs = []
    proto_entropy = []
    teacher_cov = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        mask = np.asarray(z["fut_instance_available_mask"]).astype(bool)
        same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
        avail += int(mask.sum())
        pos += int((mask & same).sum())
        neg += int((mask & (~same)).sum())
        vis += int(np.asarray(z["fut_point_visible_target"]).astype(bool).sum())
        if "obs_instance_available_mask" in z.files:
            obs = np.asarray(z["obs_instance_available_mask"]).astype(bool)
            occ += int((obs[:, -1:] & (~mask)).sum())
        inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
        inst_counts.append(len(set(inst[inst >= 0].tolist())))
        confs.extend(np.asarray(z["point_assignment_confidence"], dtype=np.float32).tolist())
        uid = str(np.asarray(z["sample_uid"]).item())
        for root in [VIS_ROOT]:
            vp = root / split / f"{uid}.npz"
            if vp.exists():
                vz = np.load(vp, allow_pickle=True)
                teacher_cov.append(float(np.asarray(vz["fut_teacher_available_mask"]).astype(bool).mean()))
        pr = ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32" / split / f"{uid}.npz"
        if pr.exists():
            pz = np.load(pr, allow_pickle=True)
            proto_entropy.append(entropy(np.asarray(pz["semantic_prototype_id"], dtype=np.int64), np.asarray(pz["semantic_prototype_available_mask"]).astype(bool)))
    return {
        "sample_count": len(files),
        "instance_count_mean": float(np.mean(inst_counts)) if inst_counts else None,
        "identity_positive_ratio": float(pos / max(avail, 1)),
        "identity_negative_ratio": float(neg / max(avail, 1)),
        "future_visibility_ratio": float(vis / max(avail, 1)),
        "semantic_prototype_entropy": float(np.nanmean([x for x in proto_entropy if x is not None])) if proto_entropy else None,
        "teacher_embedding_coverage": float(np.mean(teacher_cov)) if teacher_cov else None,
        "crop_failure_ratio": float(1.0 - np.mean(teacher_cov)) if teacher_cov else None,
        "assignment_confidence_mean": float(np.mean(confs)) if confs else None,
        "occlusion_reappearance_count": int(occ),
    }


def main() -> int:
    stats = {split: split_stats(split) for split in ("train", "val", "test")}
    val = stats["val"]
    test = stats["test"]
    reasons = []
    if abs(float(test["identity_negative_ratio"]) - float(val["identity_negative_ratio"])) > 0.03:
        reasons.append("identity negative ratio differs between val and test")
    if (test.get("semantic_prototype_entropy") or 0.0) - (val.get("semantic_prototype_entropy") or 0.0) > 0.10:
        reasons.append("test semantic prototype entropy is higher than val")
    if abs(float(test.get("occlusion_reappearance_count") or 0.0) - float(val.get("occlusion_reappearance_count") or 0.0)) > 10000:
        reasons.append("occlusion/reappearance counts differ materially")
    split_shift = bool(reasons)
    which = "test" if (test["identity_negative_ratio"] > val["identity_negative_ratio"]) else "val"
    payload = {
        "generated_at_utc": utc_now(),
        "split_shift_suspected": split_shift,
        "split_shift_reason": reasons,
        "which_split_is_easier": which,
        "whether_test_hard_subset_is_easier_than_val": bool(test["identity_negative_ratio"] < val["identity_negative_ratio"]),
        "recommended_eval_protocol_fix": "Use separated identity/semantic masks, report val/test separately, and do not promote claims until validation and test agree.",
        "splits": stats,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.4 Split Shift Audit", payload, ["split_shift_suspected", "split_shift_reason", "which_split_is_easier", "whether_test_hard_subset_is_easier_than_val", "recommended_eval_protocol_fix", "splits"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
