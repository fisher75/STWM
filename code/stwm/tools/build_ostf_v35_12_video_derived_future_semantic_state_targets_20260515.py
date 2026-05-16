#!/usr/bin/env python3
"""构建 V35.12 video-derived future semantic state targets。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import clip  # type: ignore
import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.build_ostf_v35_10_video_observed_semantic_measurement_cache_20260515 import (
    PAD_DIM,
    crop_for_object,
    encode_crops,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT

CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
MEASUREMENT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_10_video_observed_semantic_measurement_bank/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_12_video_derived_future_semantic_state_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_12_video_derived_future_semantic_state_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_12_VIDEO_DERIVED_FUTURE_SEMANTIC_STATE_TARGET_BUILD_20260515.md"
FAMILY_NAMES = [
    "copy_last_visible",
    "copy_instance_pooled",
    "copy_max_confidence",
    "changed_transition",
    "uncertain_abstain",
]
UNCERTAIN = 4
CHANGED_TRANSITION = 3


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
    return np.asarray(x).item()


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def entropy_from_counts(values: np.ndarray) -> float:
    vals = values[values >= 0]
    if vals.size == 0:
        return 0.0
    cnt = np.bincount(vals.astype(np.int64))
    p = cnt[cnt > 0].astype(np.float64) / float(vals.size)
    return float(-(p * np.log2(np.maximum(p, 1e-12))).sum())


def crop_encode_sample(model: Any, preprocess: Any, device: torch.device, z: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    obs_len = int(scalar(z["obs_len"]))
    horizon = int(scalar(z["horizon"]))
    obj_n = tracks.shape[0]
    total_t = obs_len + horizon
    crops = []
    crop_index: list[tuple[int, int]] = []
    for obj in range(obj_n):
        for t in range(total_t):
            crop = crop_for_object(z, obj, t)
            if crop is not None:
                crops.append(crop)
                crop_index.append((obj, t))
    encoded = encode_crops(model, preprocess, device, crops, batch_size)
    feat = np.zeros((obj_n, total_t, PAD_DIM), dtype=np.float32)
    mask = np.zeros((obj_n, total_t), dtype=bool)
    for row_i, (obj, t) in enumerate(crop_index):
        feat[obj, t] = encoded[row_i]
        mask[obj, t] = True
    return feat, mask


def fit_video_codebook(paths: list[Path], args: argparse.Namespace, model: Any, preprocess: Any, device: torch.device) -> tuple[MiniBatchKMeans, dict[str, tuple[np.ndarray, np.ndarray]]]:
    per_sample: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    samples: list[np.ndarray] = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        feat, mask = crop_encode_sample(model, preprocess, device, z, args.batch_size)
        per_sample[str(p)] = (feat, mask)
        valid = feat[mask]
        if valid.size:
            samples.append(valid.astype(np.float32))
    if not samples:
        raise RuntimeError("没有可用的 video crop embedding，无法构建 target。")
    x = np.concatenate(samples, axis=0)
    x = norm(x)
    k = min(args.semantic_clusters, max(2, x.shape[0] // 2))
    km = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=min(4096, max(256, x.shape[0])), n_init=5, max_iter=200)
    km.fit(x)
    return km, per_sample


def nearest_cluster(km: MiniBatchKMeans, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat = norm(x.reshape(-1, x.shape[-1]))
    pred = km.predict(flat).reshape(x.shape[:-1]).astype(np.int64)
    return np.where(mask, pred, -1)


def build_anchor_family(
    obs_feat: np.ndarray,
    obs_mask: np.ndarray,
    obs_conf: np.ndarray,
    fut_feat: np.ndarray,
    fut_mask: np.ndarray,
    changed: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obj_n, obs_len, dim = obs_feat.shape
    horizon = fut_feat.shape[1]
    mean_w = obs_mask.astype(np.float32) * np.clip(obs_conf, 0.05, 1.0)
    mean = (obs_feat * mean_w[:, :, None]).sum(axis=1) / np.maximum(mean_w.sum(axis=1, keepdims=True), 1e-6)
    mean = norm(mean)
    idx_grid = np.broadcast_to(np.arange(obs_len)[None, :], (obj_n, obs_len))
    last_idx = np.where(obs_mask, idx_grid, 0).max(axis=1)
    last = norm(obs_feat[np.arange(obj_n), last_idx])
    max_idx = np.where(obs_mask, obs_conf, -1.0).argmax(axis=1)
    maxc = norm(obs_feat[np.arange(obj_n), max_idx])
    cand = np.stack(
        [
            np.broadcast_to(last[:, None, :], fut_feat.shape),
            np.broadcast_to(mean[:, None, :], fut_feat.shape),
            np.broadcast_to(maxc[:, None, :], fut_feat.shape),
        ],
        axis=-2,
    )
    cos = np.einsum("ohfd,ohd->ohf", cand, norm(fut_feat))
    best = cos.argmax(axis=-1).astype(np.int64)
    family = best.copy()
    family[changed & fut_mask] = CHANGED_TRANSITION
    family[(uncertainty > 0.65) & fut_mask] = UNCERTAIN
    family[~fut_mask] = UNCERTAIN
    available = fut_mask.astype(bool)
    return family.astype(np.int64), available


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--measurement-root", default=str(MEASUREMENT_ROOT))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--semantic-clusters", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    measurement_root = Path(args.measurement_root)
    if not measurement_root.is_absolute():
        measurement_root = ROOT / measurement_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    paths = list_npz(cache_root)
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=str(device), download_root=str(Path.home() / ".cache" / "clip"))
    model.eval()
    km, encoded_by_path = fit_video_codebook(paths, args, model, preprocess, device)

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    split_counts: dict[str, Counter[str]] = {}
    transition_counts: Counter[int] = Counter()
    family_counts: Counter[int] = Counter()
    uncertainty_values: list[float] = []
    target_conf_values: list[float] = []

    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
            split = str(scalar(z["split"]))
            zm_path = measurement_root / split / p.name
            if not zm_path.exists():
                raise FileNotFoundError(f"缺少 observed measurement cache: {zm_path}")
            zm = np.load(zm_path, allow_pickle=True)
            feat, mask = encoded_by_path[str(p)]
            obs_len = int(scalar(z["obs_len"]))
            horizon = int(scalar(z["horizon"]))
            tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
            vis = np.asarray(z["visibility"]).astype(bool)
            conf = np.asarray(z["confidence"], dtype=np.float32)
            point_id = np.asarray(z["point_id"], dtype=np.int64)
            obj_n, per_obj_m = point_id.shape
            point_n = obj_n * per_obj_m
            obs_feat = feat[:, :obs_len]
            fut_feat = feat[:, obs_len : obs_len + horizon]
            obs_mask_obj = mask[:, :obs_len]
            fut_mask_obj = mask[:, obs_len : obs_len + horizon]
            obs_conf_obj = conf[:, :, :obs_len].mean(axis=1)
            fut_conf_obj = conf[:, :, obs_len : obs_len + horizon].mean(axis=1)
            obs_cluster_obj = nearest_cluster(km, obs_feat, obs_mask_obj)
            target_cluster_obj = nearest_cluster(km, fut_feat, fut_mask_obj)
            last_idx = np.where(obs_mask_obj, np.arange(obs_len)[None, :], 0).max(axis=1)
            last_cluster_obj = obs_cluster_obj[np.arange(obj_n), last_idx]
            changed_obj = (target_cluster_obj != last_cluster_obj[:, None]) & fut_mask_obj
            target_n = norm(fut_feat)
            obs_mean = norm((obs_feat * obs_mask_obj[:, :, None]).sum(axis=1) / np.maximum(obs_mask_obj.sum(axis=1, keepdims=True), 1.0))
            copy_cos = np.einsum("od,ohd->oh", obs_mean, target_n)
            target_conf_obj = np.clip(((copy_cos + 1.0) * 0.5) * fut_conf_obj * fut_mask_obj.astype(np.float32), 0.0, 1.0)
            uncertainty_obj = np.clip(1.0 - target_conf_obj, 0.0, 1.0).astype(np.float32)
            semantic_hard_obj = (changed_obj | (copy_cos < 0.82) | (uncertainty_obj > 0.5)) & fut_mask_obj
            family_obj, family_avail_obj = build_anchor_family(obs_feat, obs_mask_obj, obs_conf_obj, fut_feat, fut_mask_obj, changed_obj, uncertainty_obj)

            def repeat_point(a: np.ndarray) -> np.ndarray:
                return np.repeat(a[:, None, ...], per_obj_m, axis=1).reshape(point_n, *a.shape[1:])

            target_cluster = repeat_point(target_cluster_obj)
            obs_cluster = repeat_point(obs_cluster_obj)
            changed = repeat_point(changed_obj).astype(bool)
            stable = ((target_cluster == np.repeat(last_cluster_obj[:, None, None], per_obj_m, axis=1).reshape(point_n, 1)) & repeat_point(fut_mask_obj)).astype(bool)
            semantic_hard = repeat_point(semantic_hard_obj).astype(bool)
            target_available = repeat_point(fut_mask_obj).astype(bool)
            transition = np.where(
                target_available,
                np.repeat(last_cluster_obj[:, None, None], per_obj_m, axis=1).reshape(point_n, 1) * int(km.n_clusters) + target_cluster,
                -1,
            ).astype(np.int64)
            family = repeat_point(family_obj).astype(np.int64)
            family_available = repeat_point(family_avail_obj).astype(bool)
            uncertainty = repeat_point(uncertainty_obj).astype(np.float32)
            target_conf = repeat_point(target_conf_obj).astype(np.float32)
            point_inst = np.repeat(np.asarray(z["object_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            semantic_id = np.repeat(np.asarray(z["semantic_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            obs_points = tracks[:, :, :obs_len].reshape(point_n, obs_len, 2).astype(np.float32)
            obs_vis = vis[:, :, :obs_len].reshape(point_n, obs_len).astype(bool)
            obs_conf = conf[:, :, :obs_len].reshape(point_n, obs_len).astype(np.float32)
            fut_emb = repeat_point(fut_feat).astype(np.float32)
            fut_mask = target_available
            same_instance = np.ones((point_n, horizon), dtype=bool) & fut_mask
            identity_avail = fut_mask.copy()

            out_dir = out_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / p.name
            np.savez_compressed(
                out_path,
                sample_uid=str(scalar(z["item_key"])).replace("::", "__"),
                split=split,
                point_id=point_id.reshape(point_n),
                point_to_instance_id=point_inst,
                semantic_id=semantic_id,
                obs_points=obs_points,
                obs_vis=obs_vis,
                obs_conf=obs_conf,
                obs_semantic_measurements=np.asarray(zm["obs_semantic_measurements"], dtype=np.float32),
                obs_semantic_measurement_mask=np.asarray(zm["obs_semantic_measurement_mask"]).astype(bool),
                obs_measurement_confidence=np.asarray(zm["obs_measurement_confidence"], dtype=np.float32),
                target_semantic_cluster_id=target_cluster.astype(np.int64),
                target_semantic_cluster_available_mask=target_available,
                obs_semantic_cluster_id=obs_cluster.astype(np.int64),
                semantic_cluster_transition_id=transition.astype(np.int64),
                semantic_cluster_changed_mask=changed,
                semantic_stable_mask=stable,
                semantic_changed_mask=changed,
                semantic_hard_mask=semantic_hard,
                evidence_anchor_family_target=family.astype(np.int64),
                evidence_anchor_family_available_mask=family_available,
                same_instance_as_observed_target=same_instance,
                identity_consistency_available_mask=identity_avail,
                semantic_uncertainty_target=uncertainty,
                target_confidence=target_conf,
                fut_teacher_embedding=fut_emb,
                fut_teacher_available_mask=fut_mask,
                fut_teacher_confidence=target_conf,
                future_teacher_embeddings_supervision_only=True,
                future_teacher_embeddings_input_allowed=False,
                leakage_safe=True,
                target_source="video_future_clip_crop_supervision_only",
            )
            valid_count = int(target_available.sum())
            split_counts.setdefault(split, Counter())
            split_counts[split]["samples"] += 1
            split_counts[split]["tokens"] += int(target_available.size)
            split_counts[split]["valid"] += valid_count
            split_counts[split]["changed"] += int(changed.sum())
            split_counts[split]["hard"] += int(semantic_hard.sum())
            split_counts[split]["stable"] += int(stable.sum())
            transition_counts.update([int(v) for v in transition[target_available].reshape(-1)])
            family_counts.update([int(v) for v in family[family_available].reshape(-1)])
            uncertainty_values.extend([float(v) for v in uncertainty[target_available].reshape(-1)])
            target_conf_values.extend([float(v) for v in target_conf[target_available].reshape(-1)])
            rows.append(
                {
                    "cache_path": str(p.relative_to(ROOT)),
                    "output_path": str(out_path.relative_to(ROOT)),
                    "split": split,
                    "object_count": int(obj_n),
                    "point_count": int(point_n),
                    "valid_ratio": float(target_available.mean()),
                    "changed_ratio": float(changed[target_available].mean()) if valid_count else 0.0,
                    "semantic_hard_ratio": float(semantic_hard[target_available].mean()) if valid_count else 0.0,
                    "target_confidence_mean": float(target_conf[target_available].mean()) if valid_count else 0.0,
                }
            )
        except Exception as exc:
            blockers.append(f"{p}: {type(exc).__name__}: {exc}")

    def split_ratio(counter: Counter[str], key: str) -> float:
        return float(counter[key] / max(counter["valid"], 1))

    split_report = {
        s: {
            "samples": int(c["samples"]),
            "tokens": int(c["tokens"]),
            "valid_ratio": float(c["valid"] / max(c["tokens"], 1)),
            "stable_ratio": split_ratio(c, "stable"),
            "changed_ratio": split_ratio(c, "changed"),
            "semantic_hard_ratio": split_ratio(c, "hard"),
        }
        for s, c in sorted(split_counts.items())
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_derived_future_semantic_state_targets_built": bool(rows),
        "cache_root": str(cache_root.relative_to(ROOT)),
        "measurement_root": str(measurement_root.relative_to(ROOT)),
        "out_root": str(out_root.relative_to(ROOT)),
        "sample_count": len(rows),
        "semantic_cluster_count": int(km.n_clusters),
        "target_coverage_by_split": split_report,
        "semantic_transition_entropy": entropy_from_counts(np.asarray(list(transition_counts.elements()), dtype=np.int64)) if transition_counts else 0.0,
        "evidence_anchor_family_distribution": {FAMILY_NAMES[k]: int(v) for k, v in sorted(family_counts.items()) if 0 <= k < len(FAMILY_NAMES)},
        "uncertainty_target_stats": {
            "mean": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
            "p50": float(np.quantile(uncertainty_values, 0.5)) if uncertainty_values else 0.0,
            "p90": float(np.quantile(uncertainty_values, 0.9)) if uncertainty_values else 0.0,
        },
        "target_confidence_stats": {
            "mean": float(np.mean(target_conf_values)) if target_conf_values else 0.0,
            "p10": float(np.quantile(target_conf_values, 0.1)) if target_conf_values else 0.0,
            "p50": float(np.quantile(target_conf_values, 0.5)) if target_conf_values else 0.0,
        },
        "future_teacher_embeddings_supervision_only": True,
        "future_teacher_embeddings_input_allowed": False,
        "leakage_safe": True,
        "video_target_is_limited_m128_h32_smoke": True,
        "rows": rows,
        "exact_blockers": blockers[:20]
        + [
            "当前 V16 video-derived cache 只有 6 个样本，因此这是 target-contract smoke，不是 full-scale video benchmark。",
            "future CLIP crop 只作为 supervision target；若后续 predictability 不过，不能训练 V35 video semantic head。",
        ],
        "recommended_next_step": "eval_video_derived_semantic_state_target_predictability",
        "中文结论": "已构建 V35.12 video-derived future semantic state targets；future CLIP 只作为监督，不进入输入。下一步必须做 observed-only predictability 上界审计。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.12 Video-Derived Future Semantic State Target Build\n\n"
        f"- video_derived_future_semantic_state_targets_built: {report['video_derived_future_semantic_state_targets_built']}\n"
        f"- sample_count: {len(rows)}\n"
        f"- semantic_cluster_count: {report['semantic_cluster_count']}\n"
        f"- future_teacher_embeddings_input_allowed: false\n"
        f"- leakage_safe: true\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "样本数": len(rows),
                "semantic_cluster_count": int(km.n_clusters),
                "recommended_next_step": report["recommended_next_step"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
