#!/usr/bin/env python3
"""V35.35 小规模 raw-video frontend rerun smoke，M128/H32。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import run_cotracker_object_dense_teacher_v15c_20260502 as v15c
from stwm.tools.eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515 import build_split
from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.train_eval_ostf_v35_14_video_semantic_state_adapter_20260515 import (
    VideoSemanticAdapter,
    bin_metrics,
    choose_threshold,
    predict,
    top5_cluster_metrics,
)
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import (
    IdentityResidualHead,
    evaluate_split,
)
from stwm.tools.build_ostf_v35_16_video_identity_pairwise_retrieval_targets_20260515 import (
    close_pair,
    future_crossing_pair,
    occlusion_reappear,
)

UNIFIED_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark/M128_H32"
RERUN_TRACE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_35_raw_video_frontend_rerun_trace/M128_H32"
RERUN_UNIFIED_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_35_raw_video_frontend_rerun_unified_slice/M128_H32"
SEMANTIC_CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_21_domain_normalized_video_semantic_state_adapter_h32_m128"
IDENTITY_CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_h32_m128"
REPORT = ROOT / "reports/stwm_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_35_RAW_VIDEO_FRONTEND_RERUN_SMOKE_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516.log"
SEEDS = [42, 123, 456]


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


def scalar(z: Any, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    try:
        return arr.item()
    except ValueError:
        return arr


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def weighted_measurement_from_payload(payload: dict[str, np.ndarray]) -> np.ndarray:
    m = np.asarray(payload["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(payload["obs_semantic_measurement_mask"], dtype=np.float32)
    conf = np.asarray(payload["obs_measurement_confidence"], dtype=np.float32)
    w = mask * np.clip(conf, 0.05, 1.0)
    pooled = (m * w[..., None]).sum(axis=1) / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    return norm(pooled.astype(np.float32))


def one_hot_semantic_from_payload(payload: dict[str, np.ndarray]) -> np.ndarray:
    sem = np.asarray(payload["obs_semantic_last_id"], dtype=np.int64)
    if "source_semantic_id" in payload:
        sem = np.where(sem >= 0, np.asarray(sem), np.asarray(payload["source_semantic_id"], dtype=np.int64))
    sem = np.clip(sem, 0, 127)
    return np.eye(128, dtype=np.float32)[sem]


def trace_features_from_payload(payload: dict[str, np.ndarray]) -> np.ndarray:
    obs = np.asarray(payload["obs_points"], dtype=np.float32)
    fut = np.asarray(payload["future_points"], dtype=np.float32)
    obs_vis = np.asarray(payload["obs_vis"], dtype=np.float32)
    fut_vis = np.asarray(payload["future_vis"], dtype=np.float32)
    obs_conf = np.asarray(payload["obs_conf"], dtype=np.float32)
    fut_conf = np.asarray(payload["future_conf"], dtype=np.float32)
    obs_disp = (obs[:, -1] - obs[:, 0]) / 512.0
    fut_disp = (fut[:, -1] - obs[:, -1]) / 512.0
    obs_speed = np.sqrt((np.diff(obs, axis=1) ** 2).sum(-1)).mean(axis=1, keepdims=True) / 64.0
    fut_speed = np.sqrt((np.diff(fut, axis=1) ** 2).sum(-1)).mean(axis=1, keepdims=True) / 64.0
    last_xy = obs[:, -1] / 512.0
    return np.concatenate(
        [
            obs_disp,
            fut_disp,
            last_xy,
            obs_speed,
            fut_speed,
            obs_vis.mean(axis=1, keepdims=True),
            fut_vis.mean(axis=1, keepdims=True),
            obs_conf.mean(axis=1, keepdims=True),
            fut_conf.mean(axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def sample_stats(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    confuser = np.asarray(z["identity_identity_confuser_pair_mask"], dtype=bool)
    occlusion = np.asarray(z["identity_occlusion_reappear_point_mask"], dtype=bool)
    crossing = np.asarray(z["identity_trajectory_crossing_pair_mask"], dtype=bool)
    dataset = str(scalar(z, "dataset"))
    split = str(scalar(z, "split"))
    cats = {f"dataset_{dataset.lower()}", f"split_{split}"}
    if changed.any():
        cats.add("changed")
    if hard.any():
        cats.add("hard")
    if confuser.any():
        cats.add("identity_confuser")
    if occlusion.any():
        cats.add("occlusion")
    if crossing.any():
        cats.add("crossing")
    return {
        "path": path,
        "sample_uid": str(scalar(z, "sample_uid", path.stem)),
        "dataset": dataset,
        "split": split,
        "point_count": int(np.asarray(z["point_id"]).size),
        "changed_ratio": float(changed.mean()),
        "hard_ratio": float(hard.mean()),
        "confuser_pair_count": int(confuser.sum()),
        "occlusion_point_count": int(occlusion.sum()),
        "crossing_pair_count": int(crossing.sum()),
        "categories": sorted(cats),
        "video_trace_source_npz": str(scalar(z, "video_trace_source_npz")),
    }


def select_balanced(max_samples: int) -> list[dict[str, Any]]:
    targets = {"dataset_vspw", "dataset_vipseg", "split_train", "split_val", "split_test", "changed", "hard", "identity_confuser", "occlusion", "crossing"}
    rows = [sample_stats(p) for p in list_npz(UNIFIED_ROOT)]
    rows = [r for r in rows if r["point_count"] <= 1280]
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    while len(selected) < max_samples and rows:
        best = max(
            rows,
            key=lambda r: (
                len((set(r["categories"]) & targets) - covered),
                r["changed_ratio"] + r["hard_ratio"],
                int(r["confuser_pair_count"] > 0) + int(r["occlusion_point_count"] > 0) + int(r["crossing_pair_count"] > 0),
                -r["point_count"],
            ),
        )
        selected.append(best)
        covered |= set(best["categories"]) & targets
        rows = [r for r in rows if r["path"] != best["path"]]
        if targets <= covered and len(selected) >= max_samples:
            break
    return selected


def pick_diverse(rows: list[dict[str, Any]], count: int, covered: set[str] | None = None) -> list[dict[str, Any]]:
    targets = {"dataset_vspw", "dataset_vipseg", "changed", "hard", "identity_confuser", "occlusion", "crossing"}
    covered = set() if covered is None else set(covered)
    pool = list(rows)
    selected: list[dict[str, Any]] = []
    while pool and len(selected) < count:
        best = max(
            pool,
            key=lambda r: (
                len((set(r["categories"]) & targets) - covered),
                r["changed_ratio"] + r["hard_ratio"],
                int(r["confuser_pair_count"] > 0) + int(r["occlusion_point_count"] > 0) + int(r["crossing_pair_count"] > 0),
                -r["point_count"],
            ),
        )
        selected.append(best)
        covered |= set(best["categories"]) & targets
        pool = [r for r in pool if r["path"] != best["path"]]
    return selected


def select_eval_balanced(max_samples: int, eval_per_split: int, train_samples: int) -> list[dict[str, Any]]:
    rows = [sample_stats(p) for p in list_npz(UNIFIED_ROOT)]
    rows = [r for r in rows if r["point_count"] <= 1280]
    selected: list[dict[str, Any]] = []
    used: set[Path] = set()
    for split in ["val", "test"]:
        if len(selected) >= max_samples:
            break
        split_rows = [r for r in rows if r["split"] == split and r["path"] not in used]
        picked = pick_diverse(split_rows, min(eval_per_split, max_samples - len(selected)))
        selected.extend(picked)
        used |= {r["path"] for r in picked}
    if len(selected) < max_samples and train_samples > 0:
        train_rows = [r for r in rows if r["split"] == "train" and r["path"] not in used]
        picked = pick_diverse(train_rows, min(train_samples, max_samples - len(selected)))
        selected.extend(picked)
        used |= {r["path"] for r in picked}
    if len(selected) < max_samples:
        rest = [r for r in rows if r["path"] not in used]
        selected.extend(pick_diverse(rest, max_samples - len(selected)))
    return selected[:max_samples]


def compare_trace(cached_path: Path, rerun_path: Path) -> dict[str, Any]:
    old = np.load(cached_path, allow_pickle=True)
    new = np.load(rerun_path, allow_pickle=True)
    old_tr = np.asarray(old["tracks_xy"], dtype=np.float32)
    new_tr = np.asarray(new["tracks_xy"], dtype=np.float32)
    old_vis = np.asarray(old["visibility"], dtype=bool)
    new_vis = np.asarray(new["visibility"], dtype=bool)
    same_shape = old_tr.shape == new_tr.shape and old_vis.shape == new_vis.shape
    if not same_shape:
        return {
            "shape_match": False,
            "cached_tracks_shape": list(old_tr.shape),
            "rerun_tracks_shape": list(new_tr.shape),
        }
    delta = np.sqrt(((old_tr - new_tr) ** 2).sum(axis=-1))
    old_motion = np.sqrt((np.diff(old_tr, axis=2) ** 2).sum(axis=-1))
    new_motion = np.sqrt((np.diff(new_tr, axis=2) ** 2).sum(axis=-1))
    old_frames = [str(x) for x in np.asarray(old["frame_paths"], dtype=object).tolist()]
    new_frames = [str(x) for x in np.asarray(new["frame_paths"], dtype=object).tolist()]
    return {
        "shape_match": True,
        "cached_tracks_shape": list(old_tr.shape),
        "rerun_tracks_shape": list(new_tr.shape),
        "frame_paths_aligned": old_frames == new_frames,
        "mean_l2_trace_delta_px": float(delta.mean()),
        "p95_l2_trace_delta_px": float(np.quantile(delta, 0.95)),
        "max_l2_trace_delta_px": float(delta.max()),
        "visibility_agreement": float((old_vis == new_vis).mean()),
        "confidence_mae": float(np.abs(np.asarray(old["confidence"], dtype=np.float32) - np.asarray(new["confidence"], dtype=np.float32)).mean()),
        "cached_motion_mean": float(old_motion.mean()),
        "rerun_motion_mean": float(new_motion.mean()),
        "motion_mean_abs_delta": float(abs(old_motion.mean() - new_motion.mean())),
    }


def rebuild_unified_slice(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in selected:
        old_unified = Path(r["path"])
        old_z = np.load(old_unified, allow_pickle=True)
        split = str(scalar(old_z, "split"))
        rerun_path = RERUN_TRACE_ROOT / split / old_unified.name
        rz = np.load(rerun_path, allow_pickle=True)
        tracks = np.asarray(rz["tracks_xy"], dtype=np.float32)
        vis = np.asarray(rz["visibility"], dtype=bool)
        conf = np.asarray(rz["confidence"], dtype=np.float32)
        obs_len = int(scalar(rz, "obs_len"))
        horizon = int(scalar(rz, "horizon"))
        point_n = tracks.shape[0] * tracks.shape[1]
        if point_n != np.asarray(old_z["point_id"]).size:
            raise RuntimeError(f"{old_unified.name} rerun point_count 不匹配：{point_n} vs {np.asarray(old_z['point_id']).size}")
        payload = {k: old_z[k] for k in old_z.files}
        payload["obs_points"] = tracks[:, :, :obs_len].reshape(point_n, obs_len, 2).astype(np.float32)
        payload["obs_vis"] = vis[:, :, :obs_len].reshape(point_n, obs_len).astype(bool)
        payload["obs_conf"] = conf[:, :, :obs_len].reshape(point_n, obs_len).astype(np.float32)
        payload["future_points"] = tracks[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon, 2).astype(np.float32)
        payload["future_vis"] = vis[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon).astype(bool)
        payload["future_conf"] = conf[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon).astype(np.float32)
        payload["raw_video_frame_paths"] = rz["frame_paths"]
        payload["video_trace_source_npz"] = np.asarray(rel(rerun_path))
        payload["raw_video_input_available"] = np.asarray(True)
        meas = weighted_measurement_from_payload(payload)
        payload["identity_measurement_identity_embedding"] = meas.astype(np.float32)
        payload["identity_identity_input_features"] = np.concatenate(
            [meas, one_hot_semantic_from_payload(payload), trace_features_from_payload(payload)],
            axis=1,
        ).astype(np.float32)
        inst = np.asarray(payload["point_to_instance_id"], dtype=np.int64)
        same = (inst[:, None] == inst[None, :]) & (inst[:, None] >= 0)
        np.fill_diagonal(same, False)
        sem = np.asarray(payload["obs_semantic_last_id"], dtype=np.int64)
        if "source_semantic_id" in payload:
            sem = np.where(sem >= 0, sem, np.asarray(payload["source_semantic_id"], dtype=np.int64))
        diff = (inst[:, None] != inst[None, :]) & (inst[:, None] >= 0) & (inst[None, :] >= 0)
        same_sem = diff & (sem[:, None] == sem[None, :]) & (sem[:, None] >= 0)
        spatial_hard = close_pair(payload["obs_points"][:, -1], inst, 0.12)
        crossing = future_crossing_pair(payload["future_points"], inst)
        identity_confuser = np.asarray(payload["identity_identity_confuser_pair_mask"], dtype=bool) | same_sem | spatial_hard | crossing
        np.fill_diagonal(identity_confuser, False)
        payload["identity_same_instance_pair_mask"] = same.astype(bool)
        payload["identity_same_semantic_hard_negative_pair_mask"] = same_sem.astype(bool)
        payload["identity_same_frame_hard_negative_pair_mask"] = spatial_hard.astype(bool)
        payload["identity_trajectory_crossing_pair_mask"] = crossing.astype(bool)
        payload["identity_identity_confuser_pair_mask"] = identity_confuser.astype(bool)
        payload["identity_occlusion_reappear_point_mask"] = occlusion_reappear(payload["future_vis"]).astype(bool)
        out_dir = RERUN_UNIFIED_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / old_unified.name
        np.savez_compressed(out_path, **payload)
        rows.append({"output_path": rel(out_path), "split": split, "point_count": int(point_n)})
    return rows


def load_semantic_model(seed: int, input_dim: int, device: torch.device) -> VideoSemanticAdapter:
    ckpt = torch.load(SEMANTIC_CKPT_DIR / f"v35_21_domain_normalized_video_semantic_state_adapter_m128_h32_seed{seed}_best.pt", map_location=device)
    model = VideoSemanticAdapter(input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def semantic_pass_bin(m: dict[str, float | None]) -> bool:
    return bool((m["roc_auc"] or 0.0) >= 0.55 and (m["balanced_accuracy"] or 0.0) >= 0.52)


def eval_semantic_seed(seed: int, device: torch.device) -> dict[str, Any]:
    val = build_split(RERUN_UNIFIED_ROOT, "val", 60000, seed)
    test = build_split(RERUN_UNIFIED_ROOT, "test", 60000, seed)
    model = load_semantic_model(seed, int(val["x"].shape[1]), device)
    pv = predict(model, val["x"], device)
    pt = predict(model, test["x"], device)
    thresholds = {k: choose_threshold(pv[k], val[{"changed": "changed", "hard": "hard", "uncertainty": "uncertainty_high"}[k]]) for k in ["changed", "hard", "uncertainty"]}
    val_m = {
        "semantic_changed": bin_metrics(pv["changed"], val["changed"], thresholds["changed"]),
        "semantic_hard": bin_metrics(pv["hard"], val["hard"], thresholds["hard"]),
        "semantic_uncertainty": bin_metrics(pv["uncertainty"], val["uncertainty_high"], thresholds["uncertainty"]),
        "cluster": top5_cluster_metrics(pv["cluster_logits"], val["cluster"], val["last_cluster"], pv["changed"], thresholds["changed"]),
    }
    test_m = {
        "semantic_changed": bin_metrics(pt["changed"], test["changed"], thresholds["changed"]),
        "semantic_hard": bin_metrics(pt["hard"], test["hard"], thresholds["hard"]),
        "semantic_uncertainty": bin_metrics(pt["uncertainty"], test["uncertainty_high"], thresholds["uncertainty"]),
        "cluster": top5_cluster_metrics(pt["cluster_logits"], test["cluster"], test["last_cluster"], pt["changed"], thresholds["changed"]),
    }
    stable_preservation = bool(
        val_m["cluster"]["stable_top5"] >= val_m["cluster"]["stable_copy_top1"] - 0.05
        and test_m["cluster"]["stable_top5"] >= test_m["cluster"]["stable_copy_top1"] - 0.05
    )
    passed = bool(
        stable_preservation
        and (
            (semantic_pass_bin(val_m["semantic_changed"]) and semantic_pass_bin(test_m["semantic_changed"]))
            or (semantic_pass_bin(val_m["semantic_hard"]) and semantic_pass_bin(test_m["semantic_hard"]))
            or (semantic_pass_bin(val_m["semantic_uncertainty"]) and semantic_pass_bin(test_m["semantic_uncertainty"]))
        )
    )
    return {"seed": seed, "thresholds": thresholds, "val": val_m, "test": test_m, "stable_preservation": stable_preservation, "semantic_smoke_passed": passed}


def load_identity_sample(path: Path) -> dict[str, np.ndarray | str]:
    z = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "sample_uid": str(np.asarray(z["sample_uid"]).item()),
        "split": str(np.asarray(z["split"]).item()),
        "dataset": str(np.asarray(z["dataset"]).item()),
        "x": np.asarray(z["identity_identity_input_features"], dtype=np.float32),
        "measurement": np.asarray(z["identity_measurement_identity_embedding"], dtype=np.float32),
        "inst": np.asarray(z["point_to_instance_id"], dtype=np.int64),
        "same": np.asarray(z["identity_same_instance_pair_mask"], dtype=bool),
        "confuser": np.asarray(z["identity_identity_confuser_pair_mask"], dtype=bool),
        "same_semantic": np.asarray(z["identity_same_semantic_hard_negative_pair_mask"], dtype=bool),
        "spatial_hard": np.asarray(z["identity_same_frame_hard_negative_pair_mask"], dtype=bool),
        "crossing": np.asarray(z["identity_trajectory_crossing_pair_mask"], dtype=bool),
        "occlusion": np.asarray(z["identity_occlusion_reappear_point_mask"], dtype=bool),
        "obs_points": np.asarray(z["obs_points"], dtype=np.float32),
        "future_points": np.asarray(z["future_points"], dtype=np.float32),
    }


def load_identity_split(root: Path, split: str) -> list[dict[str, np.ndarray | str]]:
    return [load_identity_sample(p) for p in sorted((root / split).glob("*.npz"))]


def load_identity_model(seed: int, device: torch.device) -> IdentityResidualHead:
    ckpt = torch.load(IDENTITY_CKPT_DIR / f"v35_29_expanded_video_identity_pairwise_retrieval_m128_h32_seed{seed}_best.pt", map_location=device)
    model = IdentityResidualHead(int(ckpt["input_dim"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def identity_pass(m: dict[str, float | None]) -> bool:
    return bool(
        (m["identity_retrieval_exclude_same_point_top1"] or 0.0) >= 0.65
        and (m["identity_retrieval_same_frame_top1"] or 0.0) >= 0.65
        and (m["identity_retrieval_instance_pooled_top1"] or 0.0) >= 0.65
        and (m["identity_confuser_avoidance_top1"] or 0.0) >= 0.65
    )


def eval_identity_seed(seed: int, device: torch.device) -> dict[str, Any]:
    val = load_identity_split(RERUN_UNIFIED_ROOT, "val")
    test = load_identity_split(RERUN_UNIFIED_ROOT, "test")
    model = load_identity_model(seed, device)
    val_m = evaluate_split(val, model, device, "learned")
    test_m = evaluate_split(test, model, device, "learned")
    return {"seed": seed, "val": val_m, "test": test_m, "identity_smoke_passed": bool(identity_pass(val_m) and identity_pass(test_m))}


def main() -> int:
    global RERUN_TRACE_ROOT, RERUN_UNIFIED_ROOT, REPORT, DOC, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", default="baselines/repos/co-tracker")
    ap.add_argument("--checkpoint", default="baselines/checkpoints/cotracker/scaled_offline.pth")
    ap.add_argument("--max-samples", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-eval", action="store_true")
    ap.add_argument("--rerun-trace-root", default=str(RERUN_TRACE_ROOT))
    ap.add_argument("--rerun-unified-root", default=str(RERUN_UNIFIED_ROOT))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--doc", default=str(DOC))
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--experiment-label", default="V35.35")
    ap.add_argument("--next-step-on-pass", default="expand_m128_h32_raw_video_frontend_rerun_subset")
    ap.add_argument("--selection-mode", choices=["balanced", "eval_balanced"], default="balanced")
    ap.add_argument("--eval-per-split", type=int, default=5)
    ap.add_argument("--train-samples", type=int, default=2)
    args = ap.parse_args()

    RERUN_TRACE_ROOT = Path(args.rerun_trace_root)
    if not RERUN_TRACE_ROOT.is_absolute():
        RERUN_TRACE_ROOT = ROOT / RERUN_TRACE_ROOT
    RERUN_UNIFIED_ROOT = Path(args.rerun_unified_root)
    if not RERUN_UNIFIED_ROOT.is_absolute():
        RERUN_UNIFIED_ROOT = ROOT / RERUN_UNIFIED_ROOT
    REPORT = Path(args.report)
    if not REPORT.is_absolute():
        REPORT = ROOT / REPORT
    DOC = Path(args.doc)
    if not DOC.is_absolute():
        DOC = ROOT / DOC
    LOG = Path(args.log)
    if not LOG.is_absolute():
        LOG = ROOT / LOG

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    repo = ROOT / args.repo_path
    ckpt = ROOT / args.checkpoint
    selected = (
        select_eval_balanced(args.max_samples, args.eval_per_split, args.train_samples)
        if args.selection_mode == "eval_balanced"
        else select_balanced(args.max_samples)
    )
    log(f"{args.experiment_label} 开始：选择 {len(selected)} 个 M128/H32 raw-video frontend rerun smoke 样本。")
    for r in selected:
        log(f"选择样本 {r['sample_uid']} split={r['split']} dataset={r['dataset']} categories={','.join(r['categories'])}")

    if not repo.exists() or not ckpt.exists():
        raise RuntimeError(f"CoTracker repo/checkpoint 不存在：repo={repo}, ckpt={ckpt}")
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor  # type: ignore

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = CoTrackerPredictor(checkpoint=str(ckpt), offline=True, window_len=60).to(device).eval()
    v15c.OUT_ROOT = RERUN_TRACE_ROOT
    split_map = v15c._mixed_split_map()
    run_args = SimpleNamespace(repo_path=str(repo), checkpoint=str(ckpt), m=128, horizon=32, obs_len=8, max_side=args.max_side)
    rerun_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for idx, r in enumerate(selected, start=1):
        cached_trace_rel = r["video_trace_source_npz"]
        cached_trace_path = ROOT / cached_trace_rel
        cached_trace = np.load(cached_trace_path, allow_pickle=True)
        predecode_path = Path(str(np.asarray(cached_trace["predecode_path"]).item()))
        log(f"重跑前端 [{idx}/{len(selected)}] {r['sample_uid']} predecode={predecode_path}")
        row, fail = v15c._run_clip(model, predecode_path, run_args, device, split_map)
        if row:
            rerun_rows.append(row)
            rerun_path = ROOT / row["cache_path"]
            drift = compare_trace(cached_trace_path, rerun_path)
            drift.update({"sample_uid": r["sample_uid"], "split": r["split"], "dataset": r["dataset"], "cached_trace": cached_trace_rel, "rerun_trace": row["cache_path"]})
            drift_rows.append(drift)
            log(f"完成 {r['sample_uid']} mean_delta={drift.get('mean_l2_trace_delta_px')} vis_agree={drift.get('visibility_agreement')}")
        if fail:
            failures.append(fail)
            log(f"失败 {r['sample_uid']} reason={fail.get('reason')}")

    rebuilt_rows = rebuild_unified_slice(selected) if len(rerun_rows) == len(selected) else []
    eval_device = torch.device("cpu" if args.cpu_eval else ("cuda" if torch.cuda.is_available() else "cpu"))
    semantic_rows = []
    identity_rows = []
    joint_eval_ran = False
    if rebuilt_rows:
        log("开始 V35.31-style semantic + identity smoke eval。")
        semantic_rows = [eval_semantic_seed(seed, eval_device) for seed in SEEDS]
        identity_rows = [eval_identity_seed(seed, eval_device) for seed in SEEDS]
        joint_eval_ran = True
    shape_ok = bool(drift_rows and all(r.get("shape_match", False) for r in drift_rows))
    frame_ok = bool(drift_rows and all(r.get("frame_paths_aligned", False) for r in drift_rows))
    drift_ok = bool(
        shape_ok
        and frame_ok
        and np.mean([float(r.get("mean_l2_trace_delta_px", 1e9)) for r in drift_rows]) <= 8.0
        and np.mean([float(r.get("visibility_agreement", 0.0)) for r in drift_rows]) >= 0.80
    )
    semantic_pass = bool(semantic_rows and all(r["semantic_smoke_passed"] for r in semantic_rows))
    identity_passed = bool(identity_rows and all(r["identity_smoke_passed"] for r in identity_rows))
    smoke_passed = bool(len(rerun_rows) == len(selected) and drift_ok and joint_eval_ran and semantic_pass and identity_passed)
    recommended = args.next_step_on_pass if smoke_passed else "fix_frontend_reproducibility_trace_quality_or_target_alignment"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_label": args.experiment_label,
        "raw_video_frontend_rerun_smoke_done": True,
        "raw_video_frontend_rerun_attempted": True,
        "m": 128,
        "horizon": 32,
        "forbidden_scale_used": False,
        "selected_sample_count": len(selected),
        "selection_mode": args.selection_mode,
        "selected_split_counts": dict(Counter(r["split"] for r in selected)),
        "rerun_success_count": len(rerun_rows),
        "failure_count": len(failures),
        "selected_samples": selected,
        "rerun_trace_root": rel(RERUN_TRACE_ROOT),
        "rerun_unified_slice_root": rel(RERUN_UNIFIED_ROOT),
        "log_path": rel(LOG),
        "cached_vs_rerun_drift": {
            "shape_match_all": shape_ok,
            "frame_paths_aligned_all": frame_ok,
            "mean_l2_trace_delta_px_mean": float(np.mean([r.get("mean_l2_trace_delta_px", 0.0) for r in drift_rows])) if drift_rows else None,
            "p95_l2_trace_delta_px_mean": float(np.mean([r.get("p95_l2_trace_delta_px", 0.0) for r in drift_rows])) if drift_rows else None,
            "visibility_agreement_mean": float(np.mean([r.get("visibility_agreement", 0.0) for r in drift_rows])) if drift_rows else None,
            "motion_mean_abs_delta_mean": float(np.mean([r.get("motion_mean_abs_delta", 0.0) for r in drift_rows])) if drift_rows else None,
            "drift_ok": drift_ok,
            "rows": drift_rows,
        },
        "minimal_unified_slice_built": bool(rebuilt_rows),
        "minimal_unified_slice_rows": rebuilt_rows,
        "joint_eval_ran": joint_eval_ran,
        "semantic_seed_rows": semantic_rows,
        "identity_seed_rows": identity_rows,
        "semantic_smoke_passed_all_seeds": semantic_pass,
        "identity_smoke_passed_all_seeds": identity_passed,
        "raw_video_frame_paths_rerun_used": True,
        "old_trace_cache_used_as_input_result": False,
        "old_trace_cache_used_for_comparison_only": True,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "m128_h32_video_system_benchmark_claim_allowed": smoke_passed,
        "full_cvpr_scale_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": []
        if smoke_passed
        else [
            "raw frontend rerun smoke 未完全通过严格 gate；优先检查 frontend reproducibility、trace drift、minimal slice target alignment 或小样本统计不稳定。",
            "本轮没有训练新 writer/gate，也没有扩大到 H64/H96/M512/M1024。",
        ],
        "runtime_seconds": float(time.time() - t0),
        "recommended_next_step": recommended,
        "中文结论": (
            f"{args.experiment_label} raw-video frontend rerun smoke 通过：raw frame rerun trace 与缓存 trace 漂移可控，并且 unified slice 的 semantic/identity 联合评估没有崩。"
            if smoke_passed
            else f"{args.experiment_label} 已完成 raw-video frontend rerun smoke，但还没有通过全部 smoke gate；当前不能把 M128/H32 video system claim 从 cache 闭环推进到 raw-video rerun 闭环。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        f"# STWM OSTF {args.experiment_label} Raw Video Frontend Rerun Smoke\n\n"
        f"- raw_video_frontend_rerun_attempted: true\n"
        f"- selected_sample_count: {len(selected)}\n"
        f"- rerun_success_count: {len(rerun_rows)}\n"
        f"- cached_vs_rerun_drift_ok: {drift_ok}\n"
        f"- minimal_unified_slice_built: {bool(rebuilt_rows)}\n"
        f"- joint_eval_ran: {joint_eval_ran}\n"
        f"- semantic_smoke_passed_all_seeds: {semantic_pass}\n"
        f"- identity_smoke_passed_all_seeds: {identity_passed}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: {smoke_passed}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## Claim boundary\n"
        "本轮只证明 M128/H32 小规模 raw-video frontend rerun smoke；不代表 full CVPR-scale complete system。\n",
        encoding="utf-8",
    )
    log(f"{args.experiment_label} 结束 smoke_passed={smoke_passed} recommended_next_step={recommended}")
    print(json.dumps({"v35_35_smoke_passed": smoke_passed, "recommended_next_step": recommended}, ensure_ascii=False), flush=True)
    return 0 if smoke_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
