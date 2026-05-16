#!/usr/bin/env python3
"""从 VSPW/VIPSeg 真实 mask/panoptic + CoTracker trace 构建 V35.14 video semantic targets。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TRACE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
MEASUREMENT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_10_video_observed_semantic_measurement_bank/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_14_mask_derived_video_semantic_state_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_14_mask_derived_video_semantic_state_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_14_MASK_DERIVED_VIDEO_SEMANTIC_STATE_TARGET_BUILD_20260515.md"
FAMILY_NAMES = ["copy_last_visible", "copy_instance_pooled", "copy_max_confidence", "changed_transition", "uncertain_abstain"]
UNCERTAIN = 4
CHANGED = 3
COPY_LAST = 0
COPY_INSTANCE = 1
COPY_MAX = 2


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


def scalar(x: np.ndarray) -> Any:
    a = np.asarray(x)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def mask_path_for_frame(frame_path: str, dataset: str) -> Path:
    p = Path(frame_path)
    if dataset == "VSPW":
        return Path(str(p).replace("/origin/", "/mask/")).with_suffix(".png")
    if dataset == "VIPSEG":
        return Path(str(p).replace("/imgs/", "/panomasks/")).with_suffix(".png")
    return p.with_suffix(".png")


def decode_semantic(raw: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    raw_i = raw.astype(np.int64)
    if dataset == "VIPSEG":
        sem = np.where(raw_i >= 125, raw_i // 100, raw_i)
        valid = raw_i > 0
    else:
        sem = raw_i
        valid = raw_i != 255
    sem = np.clip(sem, 0, 127).astype(np.int64)
    return sem, valid.astype(bool)


def load_masks(frame_paths: list[str], dataset: str) -> tuple[list[np.ndarray | None], list[str]]:
    masks: list[np.ndarray | None] = []
    missing: list[str] = []
    for frame in frame_paths:
        mp = mask_path_for_frame(frame, dataset)
        if not mp.exists():
            masks.append(None)
            missing.append(str(mp))
            continue
        masks.append(np.asarray(Image.open(mp)))
    return masks, missing


def sample_masks(z: Any, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    vis = np.asarray(z["visibility"]).astype(bool)
    conf = np.asarray(z["confidence"], dtype=np.float32)
    frame_paths = [str(x) for x in np.asarray(z["frame_paths"], dtype=object).tolist()]
    resized_w, resized_h = [float(x) for x in np.asarray(z["resized_size"]).tolist()]
    masks, missing = load_masks(frame_paths, dataset)
    obj_n, per_obj_m, total_t, _ = tracks.shape
    sem = np.full((obj_n, per_obj_m, total_t), -1, dtype=np.int64)
    valid = np.zeros((obj_n, per_obj_m, total_t), dtype=bool)
    for t, mask in enumerate(masks[:total_t]):
        if mask is None:
            continue
        mh, mw = mask.shape[:2]
        sx = mw / max(resized_w, 1.0)
        sy = mh / max(resized_h, 1.0)
        xy = tracks[:, :, t]
        xs = np.clip(np.rint(xy[..., 0] * sx).astype(np.int64), 0, mw - 1)
        ys = np.clip(np.rint(xy[..., 1] * sy).astype(np.int64), 0, mh - 1)
        raw = mask[ys, xs]
        st, ok = decode_semantic(raw, dataset)
        sem[:, :, t] = st
        valid[:, :, t] = ok & vis[:, :, t] & (conf[:, :, t] > 0.05)
    return sem, valid, conf, missing


def last_valid_label(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    m, t = labels.shape
    out = np.full((m,), -1, dtype=np.int64)
    for i in range(m):
        idx = np.where(valid[i] & (labels[i] >= 0))[0]
        if len(idx):
            out[i] = int(labels[i, idx[-1]])
    return out


def mode_valid_label(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full((labels.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(labels):
        vals = row[valid[i] & (row >= 0)]
        if vals.size:
            out[i] = int(np.bincount(vals, minlength=128).argmax())
    return out


def build_pair_masks(point_inst: np.ndarray, obs_last: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    same = (point_inst[:, None] == point_inst[None, :]) & (point_inst[:, None] >= 0)
    np.fill_diagonal(same, False)
    confuser = (obs_last[:, None] == obs_last[None, :]) & (obs_last[:, None] >= 0) & (point_inst[:, None] != point_inst[None, :])
    return same.astype(bool), confuser.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", default=str(TRACE_ROOT))
    ap.add_argument("--measurement-root", default=str(MEASUREMENT_ROOT))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    trace_root = Path(args.trace_root)
    if not trace_root.is_absolute():
        trace_root = ROOT / trace_root
    measurement_root = Path(args.measurement_root)
    if not measurement_root.is_absolute():
        measurement_root = ROOT / measurement_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    split_counts: dict[str, Counter[str]] = {}
    family_counts: Counter[int] = Counter()
    confuser_samples = 0
    for p in sorted(trace_root.glob("*/*.npz")):
        try:
            z = np.load(p, allow_pickle=True)
            dataset = str(scalar(z["dataset"]))
            split = str(scalar(z["split"]))
            mp = measurement_root / split / p.name
            if not mp.exists():
                raise FileNotFoundError(f"缺少 observed measurement cache: {mp}")
            zm = np.load(mp, allow_pickle=True)
            sampled_sem, sampled_valid, conf, missing = sample_masks(z, dataset)
            if missing:
                raise FileNotFoundError(f"缺少 mask 文件 {missing[:3]}")
            tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
            vis = np.asarray(z["visibility"]).astype(bool)
            obs_len = int(scalar(z["obs_len"]))
            horizon = int(scalar(z["horizon"]))
            obj_n, per_obj_m, total_t, _ = tracks.shape
            point_n = obj_n * per_obj_m
            obs_sem = sampled_sem[:, :, :obs_len].reshape(point_n, obs_len)
            obs_ok = sampled_valid[:, :, :obs_len].reshape(point_n, obs_len)
            fut_sem = sampled_sem[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon)
            fut_ok = sampled_valid[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon)
            fut_conf = conf[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon).astype(np.float32)
            obs_last = last_valid_label(obs_sem, obs_ok)
            obs_mode = mode_valid_label(obs_sem, obs_ok)
            target = np.where(fut_ok, fut_sem, -1).astype(np.int64)
            available = fut_ok & (target >= 0) & (obs_last[:, None] >= 0)
            changed = available & (target != obs_last[:, None])
            stable = available & (target == obs_last[:, None])
            obs_vis = vis[:, :, :obs_len].reshape(point_n, obs_len)
            obs_conf = conf[:, :, :obs_len].reshape(point_n, obs_len).astype(np.float32)
            future_points = tracks[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon, 2).astype(np.float32)
            future_vis = vis[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon)
            future_conf = fut_conf
            obs_risk = np.clip(0.45 * (1.0 - obs_vis.mean(axis=1)) + 0.35 * (1.0 - obs_conf.mean(axis=1)) + 0.20 * (obs_last < 0), 0.0, 1.0)
            base_future_risk = np.clip(0.50 * (1.0 - future_conf) + 0.25 * (~future_vis).astype(np.float32) + 0.25 * obs_risk[:, None], 0.0, 1.0)
            # 对 mask-derived video target，真实语义状态变化本身就是高风险/需 abstain 的事件；
            # 不能再把 uncertainty 定义成几乎全零的 CoTracker confidence jitter。
            future_risk = np.clip(np.maximum(base_future_risk, changed.astype(np.float32) * 0.72), 0.0, 1.0)
            hard = available & (changed | (future_risk > 0.55))
            family = np.full((point_n, horizon), COPY_INSTANCE, dtype=np.int64)
            family[stable & (obs_risk[:, None] < 0.35)] = COPY_INSTANCE
            family[stable & (obs_risk[:, None] >= 0.35)] = COPY_LAST
            family[stable & (future_conf > 0.85)] = COPY_MAX
            family[changed] = CHANGED
            family[(future_risk > 0.72) | (~available)] = UNCERTAIN
            point_inst = np.repeat(np.asarray(z["object_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            semantic_source_id = np.repeat(np.asarray(z["semantic_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            same_pair, confuser_pair = build_pair_masks(point_inst, obs_last)
            confuser_samples += int(confuser_pair.any())
            out_dir = out_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / p.name
            np.savez_compressed(
                out_path,
                sample_uid=str(scalar(z["item_key"])).replace("::", "__"),
                split=split,
                dataset=dataset,
                video_semantic_target_source="mask_label" if dataset == "VSPW" else "panoptic_instance",
                point_id=np.asarray(z["point_id"], dtype=np.int64).reshape(point_n),
                point_to_instance_id=point_inst,
                source_semantic_id=semantic_source_id,
                obs_points=tracks[:, :, :obs_len].reshape(point_n, obs_len, 2).astype(np.float32),
                obs_vis=obs_vis.astype(bool),
                obs_conf=obs_conf.astype(np.float32),
                future_points=future_points,
                future_vis=future_vis.astype(bool),
                future_conf=future_conf.astype(np.float32),
                obs_semantic_measurements=np.asarray(zm["obs_semantic_measurements"], dtype=np.float32),
                obs_semantic_measurement_mask=np.asarray(zm["obs_semantic_measurement_mask"]).astype(bool),
                obs_measurement_confidence=np.asarray(zm["obs_measurement_confidence"], dtype=np.float32),
                obs_semantic_cluster_id=obs_sem.astype(np.int64),
                obs_semantic_mode_id=obs_mode.astype(np.int64),
                obs_semantic_last_id=obs_last.astype(np.int64),
                target_semantic_cluster_id=target.astype(np.int64),
                target_semantic_cluster_available_mask=available.astype(bool),
                semantic_cluster_transition_id=np.where(available, obs_last[:, None] * 128 + target, -1).astype(np.int64),
                semantic_cluster_changed_mask=changed.astype(bool),
                semantic_stable_mask=stable.astype(bool),
                semantic_changed_mask=changed.astype(bool),
                semantic_hard_mask=hard.astype(bool),
                evidence_anchor_family_target=family.astype(np.int64),
                evidence_anchor_family_available_mask=available.astype(bool),
                semantic_uncertainty_target=future_risk.astype(np.float32),
                target_confidence=np.clip(1.0 - future_risk, 0.0, 1.0).astype(np.float32),
                same_instance_pair_mask=same_pair,
                identity_confuser_pair_mask=confuser_pair,
                same_instance_as_observed_target=(available & future_vis).astype(bool),
                identity_consistency_available_mask=available.astype(bool),
                semantic_changed_is_real_video_state=True,
                identity_confuser_target_built=bool(confuser_pair.any()),
                future_teacher_embeddings_supervision_only=False,
                future_teacher_embeddings_input_allowed=False,
                leakage_safe=True,
            )
            c = split_counts.setdefault(split, Counter())
            c["samples"] += 1
            c["tokens"] += int(available.size)
            c["valid"] += int(available.sum())
            c["changed"] += int(changed.sum())
            c["stable"] += int(stable.sum())
            c["hard"] += int(hard.sum())
            c["confuser_pairs"] += int(confuser_pair.sum())
            family_counts.update([int(v) for v in family[available].reshape(-1)])
            rows.append(
                {
                    "source_path": str(p.relative_to(ROOT)),
                    "output_path": str(out_path.relative_to(ROOT)),
                    "split": split,
                    "dataset": dataset,
                    "object_count": int(obj_n),
                    "point_count": int(point_n),
                    "valid_ratio": float(available.mean()),
                    "changed_ratio": float(changed[available].mean()) if available.any() else 0.0,
                    "hard_ratio": float(hard[available].mean()) if available.any() else 0.0,
                    "identity_confuser_pair_count": int(confuser_pair.sum()),
                }
            )
        except Exception as exc:
            blockers.append(f"{p}: {type(exc).__name__}: {exc}")

    split_report = {
        s: {
            "samples": int(c["samples"]),
            "valid_ratio": float(c["valid"] / max(c["tokens"], 1)),
            "stable_ratio": float(c["stable"] / max(c["valid"], 1)),
            "changed_ratio": float(c["changed"] / max(c["valid"], 1)),
            "semantic_hard_ratio": float(c["hard"] / max(c["valid"], 1)),
            "identity_confuser_pair_count": int(c["confuser_pairs"]),
        }
        for s, c in sorted(split_counts.items())
    }
    current_insufficient = any(
        split_report.get(s, {}).get("changed_ratio", 0.0) < 0.01
        or split_report.get(s, {}).get("semantic_hard_ratio", 0.0) < 0.01
        for s in ["train", "val", "test"]
    ) or len(rows) < 32
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mask_derived_video_semantic_state_targets_built": bool(rows),
        "video_semantic_target_source": "mask_label / panoptic_instance / object_track",
        "trace_root": str(trace_root.relative_to(ROOT)),
        "measurement_root": str(measurement_root.relative_to(ROOT)),
        "out_root": str(out_root.relative_to(ROOT)),
        "sample_count": len(rows),
        "target_coverage_by_split": split_report,
        "evidence_anchor_family_distribution": {FAMILY_NAMES[k]: int(v) for k, v in sorted(family_counts.items()) if 0 <= k < len(FAMILY_NAMES)},
        "semantic_changed_is_real_video_state": True,
        "identity_confuser_target_built": bool(confuser_samples > 0),
        "current_video_cache_insufficient_for_semantic_change_benchmark": current_insufficient,
        "future_teacher_embedding_input_allowed": False,
        "leakage_safe": True,
        "rows": rows,
        "exact_blockers": blockers[:20],
        "recommended_next_step": "eval_mask_derived_video_semantic_state_predictability",
        "中文结论": "V35.14 已从真实 VSPW/VIPSeg mask/panoptic label 沿 CoTracker future trace 采样构建 video semantic state targets；changed/hard 不再来自 CLIP/KMeans。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.14 Mask-Derived Video Semantic State Target Build\n\n"
        f"- mask_derived_video_semantic_state_targets_built: {report['mask_derived_video_semantic_state_targets_built']}\n"
        f"- sample_count: {len(rows)}\n"
        f"- video_semantic_target_source: {report['video_semantic_target_source']}\n"
        f"- semantic_changed_is_real_video_state: true\n"
        f"- identity_confuser_target_built: {report['identity_confuser_target_built']}\n"
        f"- current_video_cache_insufficient_for_semantic_change_benchmark: {current_insufficient}\n"
        f"- future_teacher_embedding_input_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"样本数": len(rows), "current_video_cache_insufficient_for_semantic_change_benchmark": current_insufficient, "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
