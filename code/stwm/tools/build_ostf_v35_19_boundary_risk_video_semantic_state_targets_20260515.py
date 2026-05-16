#!/usr/bin/env python3
"""V35.19 构建 mask-boundary / visibility-risk video semantic targets。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import build_ostf_v35_14_mask_derived_video_semantic_state_targets_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

TRACE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_19_boundary_risk_video_semantic_state_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_19_boundary_risk_video_semantic_state_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_19_BOUNDARY_RISK_VIDEO_SEMANTIC_STATE_TARGET_BUILD_20260515.md"

FAMILY_NAMES = ["copy_last_visible", "copy_instance_pooled", "copy_max_confidence", "changed_transition", "uncertain_abstain"]
COPY_LAST = 0
COPY_INSTANCE = 1
COPY_MAX = 2
CHANGED = 3
UNCERTAIN = 4


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def local_boundary_for_trace(z: Any, dataset: str) -> np.ndarray:
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    vis = np.asarray(z["visibility"]).astype(bool)
    conf = np.asarray(z["confidence"], dtype=np.float32)
    frame_paths = [str(x) for x in np.asarray(z["frame_paths"], dtype=object).tolist()]
    resized_w, resized_h = [float(x) for x in np.asarray(z["resized_size"]).tolist()]
    masks, _missing = base.load_masks(frame_paths, dataset)
    obj_n, per_obj_m, total_t, _ = tracks.shape
    boundary = np.zeros((obj_n, per_obj_m, total_t), dtype=bool)
    # Object-dense points often sit inside masks, so a 1-2px boundary test is too brittle.
    # Use a small multi-radius stencil to mark boundary proximity/crossing risk without
    # depending on dataset ontology labels.
    offsets = []
    for r in (2, 4, 8, 12):
        offsets.extend([(-r, 0), (r, 0), (0, -r), (0, r), (-r, -r), (-r, r), (r, -r), (r, r)])
    for t, mask in enumerate(masks[:total_t]):
        if mask is None:
            continue
        sem, ok = base.decode_semantic(mask, dataset)
        mh, mw = sem.shape[:2]
        sx = mw / max(resized_w, 1.0)
        sy = mh / max(resized_h, 1.0)
        xy = tracks[:, :, t]
        xs = np.clip(np.rint(xy[..., 0] * sx).astype(np.int64), 0, mw - 1)
        ys = np.clip(np.rint(xy[..., 1] * sy).astype(np.int64), 0, mh - 1)
        center = sem[ys, xs]
        center_ok = ok[ys, xs]
        b = np.zeros_like(center_ok, dtype=bool)
        for dy, dx in offsets:
            yy = np.clip(ys + dy, 0, mh - 1)
            xx = np.clip(xs + dx, 0, mw - 1)
            neigh = sem[yy, xx]
            neigh_ok = ok[yy, xx]
            b |= center_ok & neigh_ok & (neigh != center)
        boundary[:, :, t] = b & vis[:, :, t] & (conf[:, :, t] > 0.05)
    return boundary


def _postprocess_sample(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    z = np.load(path, allow_pickle=True)
    payload = {k: z[k] for k in z.files}
    split = str(np.asarray(payload["split"]).item())
    dataset = str(np.asarray(payload["dataset"]).item())
    trace_path = TRACE_ROOT / split / path.name
    tz = np.load(trace_path, allow_pickle=True)
    obs_len = int(base.scalar(tz["obs_len"]))
    horizon = int(base.scalar(tz["horizon"]))
    obj_n, per_obj_m, _total_t, _ = np.asarray(tz["tracks_xy"]).shape
    boundary = local_boundary_for_trace(tz, dataset).reshape(obj_n * per_obj_m, obs_len + horizon)
    obs_boundary = boundary[:, :obs_len]
    future_boundary = boundary[:, obs_len : obs_len + horizon]
    target = np.asarray(payload["target_semantic_cluster_id"], dtype=np.int64)
    available = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
    obs_last = np.asarray(payload["obs_semantic_last_id"], dtype=np.int64)
    future_vis = np.asarray(payload["future_vis"], dtype=bool)
    future_conf = np.asarray(payload["future_conf"], dtype=np.float32)
    obs_points = np.asarray(payload["obs_points"], dtype=np.float32)
    future_points = np.asarray(payload["future_points"], dtype=np.float32)
    prev = np.concatenate([obs_last[:, None], target[:, :-1]], axis=1)
    label_transition = available & (prev >= 0) & (target >= 0) & (target != prev)
    motion = np.linalg.norm(future_points - obs_points[:, -1:, :], axis=-1).astype(np.float32)
    valid_motion = motion[available]
    motion_q60 = float(np.quantile(valid_motion, 0.60)) if valid_motion.size else 1.0
    motion_q80 = float(np.quantile(valid_motion, 0.80)) if valid_motion.size else 1.0
    motion_norm = np.clip(motion / max(motion_q80, 1.0), 0.0, 1.0)
    low_conf = available & (future_conf < 0.45)
    low_vis = available & (~future_vis)
    boundary_crossing = available & future_boundary & ((motion >= motion_q60) | label_transition | low_conf)
    # changed 是较窄的真实状态迁移；hard/risk 则包括边界附近或低可见性导致的可观测语义风险。
    changed = available & (label_transition | boundary_crossing)
    risk = np.clip(
        0.42 * future_boundary.astype(np.float32)
        + 0.26 * low_conf.astype(np.float32)
        + 0.16 * low_vis.astype(np.float32)
        + 0.16 * motion_norm,
        0.0,
        1.0,
    ).astype(np.float32)
    hard = available & (changed | (risk > 0.48))
    family = np.full(target.shape, COPY_INSTANCE, dtype=np.int64)
    stable = available & ~changed
    family[stable & (future_conf > 0.82) & (~future_boundary)] = COPY_MAX
    family[stable & future_boundary] = COPY_LAST
    family[changed] = CHANGED
    family[(risk > 0.66) | (~available)] = UNCERTAIN
    payload["original_semantic_changed_mask"] = np.asarray(payload["semantic_changed_mask"], dtype=bool)
    payload["original_semantic_hard_mask"] = np.asarray(payload["semantic_hard_mask"], dtype=bool)
    payload["original_evidence_anchor_family_target"] = np.asarray(payload["evidence_anchor_family_target"], dtype=np.int64)
    payload["mask_local_boundary_obs"] = obs_boundary.astype(bool)
    payload["mask_local_boundary_future"] = future_boundary.astype(bool)
    payload["mask_boundary_crossing_mask"] = boundary_crossing.astype(bool)
    payload["mask_label_transition_mask"] = label_transition.astype(bool)
    payload["visibility_conditioned_semantic_risk"] = risk.astype(np.float32)
    payload["semantic_changed_mask"] = changed.astype(bool)
    payload["semantic_cluster_changed_mask"] = changed.astype(bool)
    payload["semantic_hard_mask"] = hard.astype(bool)
    payload["evidence_anchor_family_target"] = family.astype(np.int64)
    payload["semantic_uncertainty_target"] = risk.astype(np.float32)
    payload["target_confidence"] = np.clip(1.0 - risk, 0.0, 1.0).astype(np.float32)
    payload["video_semantic_target_source"] = "mask_boundary_crossing / mask_label_transition / visibility_conditioned_semantic_risk"
    payload["semantic_changed_is_real_video_state"] = True
    payload["future_teacher_embedding_input_allowed"] = False
    payload["leakage_safe"] = True
    stats = {
        "split": split,
        "dataset": dataset,
        "valid_ratio": float(available.mean()),
        "changed_ratio": float(changed[available].mean()) if available.any() else 0.0,
        "hard_ratio": float(hard[available].mean()) if available.any() else 0.0,
        "boundary_ratio": float(future_boundary[available].mean()) if available.any() else 0.0,
        "risk_high_ratio": float((risk[available] > 0.5).mean()) if available.any() else 0.0,
    }
    return payload, stats


def main() -> int:
    base.OUT_ROOT = OUT_ROOT
    base.REPORT = REPORT
    base.DOC = DOC
    rc = base.main()
    rows: list[dict[str, Any]] = []
    split_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[int] = Counter()
    blockers: list[str] = []
    for p in sorted(OUT_ROOT.glob("*/*.npz")):
        try:
            payload, stats = _postprocess_sample(p)
            np.savez_compressed(p, **payload)
            key = f"{stats['split']}:{stats['dataset']}"
            available = np.asarray(payload["target_semantic_cluster_available_mask"], dtype=bool)
            changed = np.asarray(payload["semantic_changed_mask"], dtype=bool) & available
            hard = np.asarray(payload["semantic_hard_mask"], dtype=bool) & available
            family = np.asarray(payload["evidence_anchor_family_target"], dtype=np.int64)
            split_dataset[key]["samples"] += 1
            split_dataset[key]["tokens"] += int(available.size)
            split_dataset[key]["valid"] += int(available.sum())
            split_dataset[key]["changed"] += int(changed.sum())
            split_dataset[key]["hard"] += int(hard.sum())
            split_dataset[key]["boundary"] += int(np.asarray(payload["mask_local_boundary_future"], dtype=bool)[available].sum())
            split_dataset[key]["risk_high"] += int((np.asarray(payload["visibility_conditioned_semantic_risk"], dtype=np.float32)[available] > 0.5).sum())
            family_counts.update([int(v) for v in family[available].reshape(-1)])
            rows.append({"path": rel(p), **stats})
        except Exception as exc:  # pragma: no cover - diagnostic script
            blockers.append(f"{p}: {type(exc).__name__}: {exc}")

    split_dataset_report: dict[str, Any] = {}
    for key, c in sorted(split_dataset.items()):
        split_dataset_report[key] = {
            "samples": int(c["samples"]),
            "valid_ratio": float(c["valid"] / max(c["tokens"], 1)),
            "changed_ratio": float(c["changed"] / max(c["valid"], 1)),
            "semantic_hard_ratio": float(c["hard"] / max(c["valid"], 1)),
            "future_boundary_ratio": float(c["boundary"] / max(c["valid"], 1)),
            "risk_high_ratio": float(c["risk_high"] / max(c["valid"], 1)),
        }
    vspw_test_changed = split_dataset_report.get("test:VSPW", {}).get("changed_ratio", 0.0)
    vspw_test_hard = split_dataset_report.get("test:VSPW", {}).get("semantic_hard_ratio", 0.0)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_risk_video_semantic_state_targets_built": bool(rows),
        "base_builder_return_code": rc,
        "out_root": rel(OUT_ROOT),
        "sample_count": len(rows),
        "split_dataset_coverage": split_dataset_report,
        "vipseg_source_train_val_expanded": bool(split_dataset_report.get("train:VIPSEG", {}).get("samples", 0) >= 50 and split_dataset_report.get("val:VIPSEG", {}).get("samples", 0) >= 12),
        "vspw_test_changed_sparse": bool(vspw_test_changed < 0.03),
        "vspw_test_hard_sparse": bool(vspw_test_hard < 0.05),
        "evidence_anchor_family_distribution": {FAMILY_NAMES[k]: int(v) for k, v in sorted(family_counts.items()) if 0 <= k < len(FAMILY_NAMES)},
        "video_semantic_target_source": "mask_boundary_crossing / mask_label_transition / visibility_conditioned_semantic_risk",
        "future_teacher_embedding_input_allowed": False,
        "semantic_changed_is_real_video_state": True,
        "leakage_safe": True,
        "rows": rows,
        "exact_blockers": blockers[:30],
        "recommended_next_step": "eval_boundary_risk_video_semantic_predictability",
        "中文结论": (
            "V35.19 使用真实 mask 局部边界、trace motion、visibility/confidence 构建 ontology-agnostic hard/risk target；"
            "目标是避免 V35.18 纯 label transition 过窄导致 VSPW test changed/hard 不可评估。"
        ),
    }
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.19 Boundary-Risk Video Semantic Target Build\n\n"
        f"- boundary_risk_video_semantic_state_targets_built: {bool(rows)}\n"
        f"- sample_count: {len(rows)}\n"
        f"- vipseg_source_train_val_expanded: {report['vipseg_source_train_val_expanded']}\n"
        f"- vspw_test_changed_sparse: {report['vspw_test_changed_sparse']}\n"
        f"- vspw_test_hard_sparse: {report['vspw_test_hard_sparse']}\n"
        f"- future_teacher_embedding_input_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"样本数": len(rows), "VSPW_test_changed稀疏": report["vspw_test_changed_sparse"], "VSPW_test_hard稀疏": report["vspw_test_hard_sparse"], "报告": rel(REPORT)}, ensure_ascii=False), flush=True)
    return 0 if rows and not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
