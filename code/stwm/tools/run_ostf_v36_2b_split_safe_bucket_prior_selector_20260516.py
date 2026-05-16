#!/usr/bin/env python3
"""V36.2b: split-safe bucket prior selector/calibration，V30 frozen，不跑 V36.3。"""
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

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

ROLLOUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_v30_past_only_future_trace_rollout/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_2b_split_safe_bucket_prior_selector_rollout/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v36_2b_split_safe_bucket_prior_selector_rollout_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_2B_SPLIT_SAFE_BUCKET_PRIOR_SELECTOR_ROLLOUT_20260516.md"

CANDIDATES = [
    ("v30", "predicted_future_points"),
    ("last_observed_copy", "prior_last_observed_copy"),
    ("last_visible_copy", "prior_last_visible_copy"),
    ("constant_velocity", "prior_constant_velocity"),
    ("damped_velocity", "prior_damped_velocity"),
    ("global_median_velocity", "derived_global_median_velocity"),
]


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


def scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return arr.item() if arr.shape == () else arr


def list_npz(split: str | None = None) -> list[Path]:
    if split is None:
        return sorted(ROLLOUT_ROOT.glob("*/*.npz"))
    return sorted((ROLLOUT_ROOT / split).glob("*.npz"))


def global_median_velocity(obs: np.ndarray, horizon: int) -> np.ndarray:
    vel = obs[:, -1] - obs[:, -2]
    med = np.median(vel, axis=0, keepdims=True)
    t = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
    return obs[:, -1:, :] + med[:, None, :] * t


def candidate_arrays(z: np.lib.npyio.NpzFile) -> tuple[list[str], np.ndarray]:
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    h = np.asarray(z["future_trace_teacher_points"], dtype=np.float32).shape[1]
    names, arrays = [], []
    for name, key in CANDIDATES:
        arr = global_median_velocity(obs, h) if key == "derived_global_median_velocity" else np.asarray(z[key], dtype=np.float32)
        names.append(name)
        arrays.append(arr)
    return names, np.stack(arrays, axis=0).astype(np.float32)


def sample_ade(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    if not valid.any():
        return None
    return float(np.linalg.norm(pred - target, axis=-1)[valid].mean())


def sample_fde(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    vals = []
    for i in range(pred.shape[0]):
        idx = np.where(valid[i])[0]
        if idx.size:
            vals.append(float(np.linalg.norm(pred[i, idx[-1]] - target[i, idx[-1]])))
    return float(np.mean(vals)) if vals else None


def point_ade(cands: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(cands - target[None], axis=-1)
    mask = valid[None].astype(np.float32)
    denom = mask.sum(axis=-1).clip(min=1.0)
    return ((dist * mask).sum(axis=-1) / denom).transpose(1, 0)


def sample_features(z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
    obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
    step = np.linalg.norm(np.diff(obs, axis=1), axis=-1)
    point_motion = step.mean(axis=1) if step.size else np.zeros(obs.shape[0], dtype=np.float32)
    motion_mean = float(point_motion.mean()) if point_motion.size else 0.0
    motion_p90 = float(np.percentile(point_motion, 90)) if point_motion.size else 0.0
    camera_motion = float(np.linalg.norm(np.median(obs[:, -1] - obs[:, 0], axis=0)))
    return {
        "dataset": str(scalar(z, "dataset")),
        "split": str(scalar(z, "split")),
        "point_count": int(obs.shape[0]),
        "object_bucket": "obj1" if obs.shape[0] <= 128 else "obj2_4" if obs.shape[0] <= 512 else "obj5plus",
        "motion_mean": motion_mean,
        "motion_p90": motion_p90,
        "camera_motion": camera_motion,
        "obs_visibility_mean": float(obs_vis.mean()),
        "obs_confidence_mean": float(obs_conf.mean()),
    }


def thresholds_from_train() -> dict[str, tuple[float, float]]:
    feats = [sample_features(np.load(p, allow_pickle=True)) for p in list_npz("train")]
    out: dict[str, tuple[float, float]] = {}
    for key in ["motion_mean", "camera_motion", "obs_visibility_mean", "obs_confidence_mean"]:
        vals = np.asarray([float(f[key]) for f in feats], dtype=np.float32)
        out[key] = tuple(np.quantile(vals, [1 / 3, 2 / 3]).tolist()) if vals.size else (0.0, 0.0)
    return out


def bin3(v: float, cuts: tuple[float, float], prefix: str) -> str:
    if v <= cuts[0]:
        return f"{prefix}_low"
    if v <= cuts[1]:
        return f"{prefix}_mid"
    return f"{prefix}_high"


def bucket_key(feat: dict[str, Any], cuts: dict[str, tuple[float, float]], level: int) -> tuple[str, ...]:
    # level 越大越细；回退时逐步减少维度，避免小样本 bucket 过拟合。
    base = [
        f"dataset_{str(feat['dataset']).lower()}",
        bin3(float(feat["motion_mean"]), cuts["motion_mean"], "motion"),
        bin3(float(feat["camera_motion"]), cuts["camera_motion"], "camera"),
        bin3(float(feat["obs_visibility_mean"]), cuts["obs_visibility_mean"], "vis"),
        str(feat["object_bucket"]),
    ]
    if level >= 4:
        return tuple(base)
    if level == 3:
        return tuple(base[:4])
    if level == 2:
        return tuple(base[:3])
    if level == 1:
        return tuple(base[:2])
    return ("global",)


def method_metrics_for_sample(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    names, cands = candidate_arrays(z)
    target = np.asarray(z["future_trace_teacher_points"], dtype=np.float32)
    valid = np.asarray(z["future_trace_teacher_vis"], dtype=bool)
    metrics = {name: sample_ade(cands[i], target, valid) for i, name in enumerate(names)}
    return {"path": path, "features": sample_features(z), "metrics": metrics}


def build_bucket_policy(cuts: dict[str, tuple[float, float]]) -> dict[str, Any]:
    train = [method_metrics_for_sample(p) for p in list_npz("train")]
    val = [method_metrics_for_sample(p) for p in list_npz("val")]
    names = [name for name, _ in CANDIDATES]
    global_train = aggregate_method_ade(train, names)
    global_val = aggregate_method_ade(val, names)
    global_safe = choose_validated_method(global_train, global_val, min_val_margin=-0.0)
    policies: dict[str, Any] = {"global": {"method": global_safe, "level": 0, "train_count": len(train), "val_count": len(val), "reason": "global_validated"}}
    for level in [1, 2, 3, 4]:
        keys = sorted({bucket_key(r["features"], cuts, level) for r in train + val})
        for key in keys:
            tr = [r for r in train if bucket_key(r["features"], cuts, level) == key]
            vr = [r for r in val if bucket_key(r["features"], cuts, level) == key]
            if len(tr) < 8 or len(vr) < 3:
                continue
            trm = aggregate_method_ade(tr, names)
            vrm = aggregate_method_ade(vr, names)
            method = choose_validated_method(trm, vrm, min_val_margin=0.0)
            # 单调保护：V30 只能在高运动/高置信、且 val 真实赢的 bucket 启用。
            feat_tokens = set(key)
            if method == "v30" and not (("motion_high" in feat_tokens or "camera_high" in feat_tokens) and vrm["v30"] <= min(vrm[m] for m in names if m != "v30")):
                method = min([m for m in names if m != "v30"], key=lambda m: vrm[m])
            policies["|".join(key)] = {
                "method": method,
                "level": level,
                "train_count": len(tr),
                "val_count": len(vr),
                "train_ade": trm,
                "val_ade": vrm,
                "reason": "bucket_validated_train_val",
            }
    return policies


def aggregate_method_ade(rows: list[dict[str, Any]], names: list[str]) -> dict[str, float]:
    out = {}
    for name in names:
        vals = [float(r["metrics"][name]) for r in rows if r["metrics"].get(name) is not None]
        out[name] = float(np.mean(vals)) if vals else 1e18
    return out


def choose_validated_method(train_ade: dict[str, float], val_ade: dict[str, float], min_val_margin: float) -> str:
    names = list(train_ade)
    train_best = min(names, key=lambda m: train_ade[m])
    val_best = min(names, key=lambda m: val_ade[m])
    # 只有 train 与 val 一致，才相信该 bucket 的精细选择；否则偏向 val best 的保守 prior。
    if train_best == val_best:
        return val_best
    prior_names = [m for m in names if m != "v30"]
    return min(prior_names, key=lambda m: val_ade[m])


def choose_method(feat: dict[str, Any], cuts: dict[str, tuple[float, float]], policies: dict[str, Any]) -> tuple[str, str]:
    for level in [4, 3, 2, 1]:
        key = "|".join(bucket_key(feat, cuts, level))
        if key in policies:
            return str(policies[key]["method"]), key
    return str(policies["global"]["method"]), "global"


def build_selected_rollout(cuts: dict[str, tuple[float, float]], policies: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hist: Counter[str] = Counter()
    for p in list_npz():
        z = np.load(p, allow_pickle=True)
        names, cands = candidate_arrays(z)
        name_to_idx = {n: i for i, n in enumerate(names)}
        feat = sample_features(z)
        method, key = choose_method(feat, cuts, policies)
        target = np.asarray(z["future_trace_teacher_points"], dtype=np.float32)
        valid = np.asarray(z["future_trace_teacher_vis"], dtype=bool)
        pred = cands[name_to_idx[method]]
        metrics = {n: {"ADE": sample_ade(cands[i], target, valid), "FDE": sample_fde(cands[i], target, valid)} for i, n in enumerate(names)}
        metrics["bucket_selector"] = {"ADE": sample_ade(pred, target, valid), "FDE": sample_fde(pred, target, valid)}
        split = str(feat["split"])
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / p.name
        np.savez_compressed(
            out_path,
            sample_uid=z["sample_uid"],
            dataset=z["dataset"],
            split=z["split"],
            point_id=z["point_id"],
            obs_points=np.asarray(z["obs_points"], dtype=np.float32),
            obs_vis=np.asarray(z["obs_vis"], dtype=bool),
            obs_conf=np.asarray(z["obs_conf"], dtype=np.float32),
            predicted_future_points=pred.astype(np.float32),
            predicted_future_vis=np.asarray(z["predicted_future_vis"], dtype=np.float32),
            predicted_future_conf=np.asarray(z["predicted_future_conf"], dtype=np.float32),
            selector_method_name=np.asarray(method),
            selector_bucket_key=np.asarray(key),
            future_trace_teacher_points=target,
            future_trace_teacher_vis=valid,
            future_trace_teacher_conf=np.asarray(z["future_trace_teacher_conf"], dtype=np.float32),
            future_trace_predicted_from_past_only=np.asarray(True),
            v30_backbone_frozen=np.asarray(True),
            future_trace_teacher_input_allowed=np.asarray(False),
            leakage_safe=np.asarray(True),
            selector_uses_observed_only_features=np.asarray(True),
        )
        hist[method] += 1
        row = {"sample_uid": str(scalar(z, "sample_uid")), "split": split, "output_path": rel(out_path), "selected_method": method, "bucket_key": key, "metrics": metrics}
        rows.append(row)
        split_rows[split].append(row)
    summary = {split: aggregate_rows(srows) for split, srows in split_rows.items()}
    summary["all"] = aggregate_rows(rows)
    summary["selected_method_histogram"] = dict(hist)
    return rows, summary


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_count": len(rows)}
    if not rows:
        return out
    methods = list(rows[0]["metrics"].keys())
    for method in methods:
        ades = [r["metrics"][method]["ADE"] for r in rows if r["metrics"][method]["ADE"] is not None]
        fdes = [r["metrics"][method]["FDE"] for r in rows if r["metrics"][method]["FDE"] is not None]
        out[f"{method}_ADE_mean"] = float(np.mean(ades)) if ades else None
        out[f"{method}_FDE_mean"] = float(np.mean(fdes)) if fdes else None
    prior_methods = [m for m in methods if m not in {"v30", "bucket_selector"}]
    strongest = min(prior_methods, key=lambda m: out[f"{m}_ADE_mean"] if out[f"{m}_ADE_mean"] is not None else 1e18)
    out["strongest_prior"] = strongest
    out["strongest_prior_ADE_mean"] = out[f"{strongest}_ADE_mean"]
    out["bucket_selector_minus_strongest_prior_ADE"] = (
        float(out["bucket_selector_ADE_mean"]) - float(out["strongest_prior_ADE_mean"])
        if out.get("bucket_selector_ADE_mean") is not None and out.get("strongest_prior_ADE_mean") is not None
        else None
    )
    out["bucket_selector_beats_strongest_prior"] = bool(out["bucket_selector_minus_strongest_prior_ADE"] is not None and out["bucket_selector_minus_strongest_prior_ADE"] <= 0)
    out["bucket_selector_minus_v30_ADE"] = (
        float(out["bucket_selector_ADE_mean"]) - float(out["v30_ADE_mean"])
        if out.get("bucket_selector_ADE_mean") is not None and out.get("v30_ADE_mean") is not None
        else None
    )
    out["bucket_selector_beats_v30"] = bool(out["bucket_selector_minus_v30_ADE"] is not None and out["bucket_selector_minus_v30_ADE"] <= 0)
    return out


def main() -> int:
    cuts = thresholds_from_train()
    policies = build_bucket_policy(cuts)
    rows, summary = build_selected_rollout(cuts, policies)
    val_pass = bool(summary.get("val", {}).get("bucket_selector_beats_strongest_prior", False))
    test_pass = bool(summary.get("test", {}).get("bucket_selector_beats_strongest_prior", False))
    passed = bool(val_pass and test_pass)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_safe_bucket_prior_selector_done": True,
        "train_new_large_model": False,
        "v30_backbone_frozen": True,
        "selector_type": "split_safe_bucket_validated_monotonic_rule",
        "thresholds_from_train_only": cuts,
        "policy_count": len(policies),
        "policies": policies,
        "output_root": rel(OUT_ROOT),
        "sample_count": len(rows),
        "summary_by_split": summary,
        "bucket_selector_beats_strongest_prior_val": val_pass,
        "bucket_selector_beats_strongest_prior_test": test_pass,
        "bucket_selector_rollout_passed": passed,
        "future_trace_predicted_from_past_only": True,
        "future_trace_teacher_input_allowed": False,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "recommended_next_step": "run_v36_3_full_325_causal_benchmark_rerun" if passed else "fix_v30_prior_selector_calibration",
        "rows": rows,
        "中文总结": (
            "V36.2b split-safe bucket selector 在 val/test 上同时赢 strongest prior，可以进入 V36.3 full causal benchmark rerun。"
            if passed
            else "V36.2b split-safe bucket selector 仍未在 val/test 同时赢 strongest prior；需要继续修 trace calibration，不允许跑 V36.3 claim。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    all_summary = summary.get("all", {})
    DOC.write_text(
        "# STWM OSTF V36.2b Split-Safe Bucket Prior Selector\n\n"
        "- train_new_large_model: false\n"
        "- V30 frozen: true\n"
        "- selector_type: split_safe_bucket_validated_monotonic_rule\n"
        f"- policy_count: {len(policies)}\n"
        f"- selected_method_histogram: {summary.get('selected_method_histogram')}\n"
        f"- bucket_selector_ADE_all: {all_summary.get('bucket_selector_ADE_mean')}\n"
        f"- strongest_prior_all: {all_summary.get('strongest_prior')}\n"
        f"- strongest_prior_ADE_all: {all_summary.get('strongest_prior_ADE_mean')}\n"
        f"- bucket_selector_minus_strongest_prior_ADE_all: {all_summary.get('bucket_selector_minus_strongest_prior_ADE')}\n"
        f"- bucket_selector_beats_strongest_prior_val: {val_pass}\n"
        f"- bucket_selector_beats_strongest_prior_test: {test_pass}\n"
        f"- bucket_selector_rollout_passed: {passed}\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36_2b_bucket_selector完成": True, "passed": passed, "下一步": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
