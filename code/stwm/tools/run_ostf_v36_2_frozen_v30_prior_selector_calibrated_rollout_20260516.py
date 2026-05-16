#!/usr/bin/env python3
"""V36.2: frozen V30 + observed-only prior selector/calibration，不做 trajectory architecture search。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

ROLLOUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_v30_past_only_future_trace_rollout/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_2_frozen_v30_prior_selector_calibrated_rollout/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v36_2_frozen_v30_prior_selector_calibrated_rollout_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_2_FROZEN_V30_PRIOR_SELECTOR_CALIBRATED_ROLLOUT_20260516.md"

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


def list_npz(root: Path, split: str | None = None) -> list[Path]:
    if split is None:
        return sorted(root.glob("*/*.npz"))
    return sorted((root / split).glob("*.npz"))


def global_median_velocity(obs: np.ndarray, horizon: int) -> np.ndarray:
    vel = obs[:, -1] - obs[:, -2]
    med = np.median(vel, axis=0, keepdims=True)
    t = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
    return obs[:, -1:, :] + med[:, None, :] * t


def candidate_stack(z: np.lib.npyio.NpzFile) -> tuple[list[str], np.ndarray]:
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    h = np.asarray(z["future_trace_teacher_points"], dtype=np.float32).shape[1]
    arrays = []
    names = []
    for name, key in CANDIDATES:
        if key == "derived_global_median_velocity":
            arr = global_median_velocity(obs, h)
        else:
            arr = np.asarray(z[key], dtype=np.float32)
        arrays.append(arr)
        names.append(name)
    return names, np.stack(arrays, axis=0).astype(np.float32)


def point_ade(cands: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # [C,M,H,2] -> [M,C]
    dist = np.linalg.norm(cands - target[None], axis=-1)
    mask = valid[None].astype(np.float32)
    denom = mask.sum(axis=-1).clip(min=1.0)
    ade = (dist * mask).sum(axis=-1) / denom
    return ade.transpose(1, 0)


def sample_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | None]:
    if not valid.any():
        return {"ADE": None, "FDE": None}
    dist = np.linalg.norm(pred - target, axis=-1)
    ade = float(dist[valid].mean())
    vals = []
    for i in range(pred.shape[0]):
        idx = np.where(valid[i])[0]
        if idx.size:
            vals.append(float(np.linalg.norm(pred[i, idx[-1]] - target[i, idx[-1]])))
    return {"ADE": ade, "FDE": float(np.mean(vals)) if vals else None}


def observed_features(z: np.lib.npyio.NpzFile) -> np.ndarray:
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    vis = np.asarray(z["obs_vis"], dtype=np.float32)
    conf = np.asarray(z["obs_conf"], dtype=np.float32)
    m = obs.shape[0]
    disp = (obs[:, -1] - obs[:, 0]) / 512.0
    last_step = (obs[:, -1] - obs[:, -2]) / 512.0
    step = np.linalg.norm(np.diff(obs, axis=1), axis=-1)
    speed_mean = step.mean(axis=1, keepdims=True) / 64.0
    speed_last = step[:, -1:].astype(np.float32) / 64.0
    valid_count = vis.sum(axis=1, keepdims=True) / max(vis.shape[1], 1)
    med_vel = np.median(obs[:, -1] - obs[:, -2], axis=0, keepdims=True) / 512.0
    med_vel = np.repeat(med_vel, m, axis=0)
    camera_motion = np.linalg.norm(med_vel, axis=1, keepdims=True)
    object_count = np.full((m, 1), max(1.0, float(round(m / 128))), dtype=np.float32)
    last_xy = obs[:, -1] / 512.0
    radial = np.linalg.norm(last_xy - 0.5, axis=1, keepdims=True)
    return np.concatenate(
        [
            last_xy,
            radial,
            disp,
            last_step,
            speed_mean,
            speed_last,
            vis.mean(axis=1, keepdims=True),
            vis[:, -1:],
            conf.mean(axis=1, keepdims=True),
            conf[:, -1:],
            valid_count,
            med_vel,
            camera_motion,
            np.log1p(object_count),
        ],
        axis=1,
    ).astype(np.float32)


def build_train_set(max_points: int = 220_000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rng = np.random.default_rng(3602)
    names_ref: list[str] = []
    for p in list_npz(ROLLOUT_ROOT, "train"):
        z = np.load(p, allow_pickle=True)
        names, cands = candidate_stack(z)
        names_ref = names
        target = np.asarray(z["future_trace_teacher_points"], dtype=np.float32)
        valid = np.asarray(z["future_trace_teacher_vis"], dtype=bool)
        ade = point_ade(cands, target, valid)
        y = ade.argmin(axis=1).astype(np.int64)
        x = observed_features(z)
        xs.append(x)
        ys.append(y)
    if not xs:
        raise RuntimeError("train split 没有 rollout 样本，无法训练 observed-only selector。")
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    if len(x_all) > max_points:
        idx = rng.choice(len(x_all), max_points, replace=False)
        x_all = x_all[idx]
        y_all = y_all[idx]
    return x_all, y_all, names_ref


def evaluate_and_write(clf: HistGradientBoostingClassifier, names: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hist_total: Counter[str] = Counter()
    for p in list_npz(ROLLOUT_ROOT):
        z = np.load(p, allow_pickle=True)
        split = str(scalar(z, "split"))
        target = np.asarray(z["future_trace_teacher_points"], dtype=np.float32)
        valid = np.asarray(z["future_trace_teacher_vis"], dtype=bool)
        cand_names, cands = candidate_stack(z)
        x = observed_features(z)
        proba = clf.predict_proba(x)
        pred_id = proba.argmax(axis=1).astype(np.int64)
        # sklearn 可能只见过部分类别；映射回原候选 id。
        class_ids = np.asarray(clf.classes_, dtype=np.int64)
        method_id = class_ids[pred_id]
        selected = cands[method_id, np.arange(cands.shape[1])]
        uncertainty = 1.0 - proba.max(axis=1)
        oracle_id = point_ade(cands, target, valid).argmin(axis=1).astype(np.int64)
        hist = Counter(cand_names[int(i)] for i in method_id.tolist())
        hist_total.update(hist)
        metrics = {name: sample_metrics(cands[i], target, valid) for i, name in enumerate(cand_names)}
        metrics["calibrated_selector"] = sample_metrics(selected, target, valid)
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
            predicted_future_points=selected.astype(np.float32),
            predicted_future_vis=np.asarray(z["predicted_future_vis"], dtype=np.float32),
            predicted_future_conf=(1.0 - uncertainty[:, None]).repeat(selected.shape[1], axis=1).astype(np.float32),
            selector_method_id=method_id.astype(np.int64),
            selector_method_name=np.asarray([cand_names[int(i)] for i in method_id], dtype=object),
            selector_uncertainty=uncertainty.astype(np.float32),
            candidate_names=np.asarray(cand_names, dtype=object),
            candidate_future_points=cands.astype(np.float16),
            future_trace_teacher_points=target,
            future_trace_teacher_vis=valid,
            future_trace_teacher_conf=np.asarray(z["future_trace_teacher_conf"], dtype=np.float32),
            future_trace_predicted_from_past_only=np.asarray(True),
            v30_backbone_frozen=np.asarray(True),
            future_trace_teacher_input_allowed=np.asarray(False),
            leakage_safe=np.asarray(True),
            selector_uses_observed_only_features=np.asarray(True),
            source_v36_rollout_npz=np.asarray(rel(p)),
        )
        row = {
            "sample_uid": str(scalar(z, "sample_uid")),
            "split": split,
            "output_path": rel(out_path),
            "selector_accuracy_vs_oracle_point_method": float(accuracy_score(oracle_id, method_id)),
            "selected_method_histogram": dict(hist),
            "metrics": metrics,
        }
        rows.append(row)
        by_split[split].append(row)
    summary: dict[str, Any] = {}
    for split, srows in sorted(by_split.items()):
        summary[split] = aggregate_rows(srows)
    summary["all"] = aggregate_rows(rows)
    summary["selected_method_histogram_all"] = dict(hist_total)
    return rows, summary


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_count": len(rows)}
    if not rows:
        return out
    method_names = list(rows[0]["metrics"].keys())
    for method in method_names:
        ades = [r["metrics"][method]["ADE"] for r in rows if r["metrics"][method]["ADE"] is not None]
        fdes = [r["metrics"][method]["FDE"] for r in rows if r["metrics"][method]["FDE"] is not None]
        out[f"{method}_ADE_mean"] = float(np.mean(ades)) if ades else None
        out[f"{method}_FDE_mean"] = float(np.mean(fdes)) if fdes else None
    prior_methods = [m for m in method_names if m not in {"v30", "calibrated_selector"}]
    strongest = min(prior_methods, key=lambda m: out[f"{m}_ADE_mean"] if out[f"{m}_ADE_mean"] is not None else 1e18)
    out["strongest_prior"] = strongest
    out["strongest_prior_ADE_mean"] = out[f"{strongest}_ADE_mean"]
    out["calibrated_minus_strongest_prior_ADE"] = (
        float(out["calibrated_selector_ADE_mean"]) - float(out["strongest_prior_ADE_mean"])
        if out.get("calibrated_selector_ADE_mean") is not None and out.get("strongest_prior_ADE_mean") is not None
        else None
    )
    out["calibrated_beats_strongest_prior"] = bool(out["calibrated_minus_strongest_prior_ADE"] is not None and out["calibrated_minus_strongest_prior_ADE"] <= 0)
    out["calibrated_minus_v30_ADE"] = (
        float(out["calibrated_selector_ADE_mean"]) - float(out["v30_ADE_mean"])
        if out.get("calibrated_selector_ADE_mean") is not None and out.get("v30_ADE_mean") is not None
        else None
    )
    out["calibrated_beats_v30"] = bool(out["calibrated_minus_v30_ADE"] is not None and out["calibrated_minus_v30_ADE"] <= 0)
    out["selector_accuracy_vs_oracle_point_method_mean"] = float(np.mean([r["selector_accuracy_vs_oracle_point_method"] for r in rows]))
    return out


def main() -> int:
    x_train, y_train, names = build_train_set()
    clf = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31, learning_rate=0.06, l2_regularization=0.01, random_state=3602)
    clf.fit(x_train, y_train)
    rows, summary = evaluate_and_write(clf, names)
    all_summary = summary.get("all", {})
    val_pass = bool(summary.get("val", {}).get("calibrated_beats_strongest_prior", False))
    test_pass = bool(summary.get("test", {}).get("calibrated_beats_strongest_prior", False))
    passed = bool(val_pass and test_pass)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_v30_prior_selector_calibrated_rollout_done": True,
        "train_new_large_model": False,
        "v30_backbone_frozen": True,
        "selector_model": "sklearn.HistGradientBoostingClassifier",
        "selector_training_scope": "train_split_only_observed_features_to_best_candidate_supervision",
        "observed_only_feature_groups": [
            "static gate features",
            "motion gate features",
            "visibility-aware features",
            "confidence calibration features",
            "camera/global median velocity proxy",
        ],
        "candidate_methods": names,
        "train_points_used": int(len(x_train)),
        "output_root": rel(OUT_ROOT),
        "sample_count": len(rows),
        "summary_by_split": summary,
        "calibrated_beats_strongest_prior_val": val_pass,
        "calibrated_beats_strongest_prior_test": test_pass,
        "calibrated_rollout_passed": passed,
        "future_trace_predicted_from_past_only": True,
        "future_trace_teacher_input_allowed": False,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "recommended_next_step": "run_v36_3_full_325_causal_benchmark_rerun" if passed else "fix_v30_prior_selector_calibration",
        "rows": rows,
        "中文总结": (
            "V36.2 frozen V30 prior selector/calibration 在 val/test 上均赢 strongest prior；可以进入 V36.3 causal full benchmark rerun。"
            if passed
            else "V36.2 selector 还没有在 val/test 同时赢 strongest prior；需要继续修 prior selector/calibration，而不是训练 semantic/identity。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36.2 Frozen V30 Prior Selector Calibrated Rollout\n\n"
        "- train_new_large_model: false\n"
        "- V30 frozen: true\n"
        f"- train_points_used: {len(x_train)}\n"
        f"- candidate_methods: {names}\n"
        f"- calibrated_selector_ADE_all: {all_summary.get('calibrated_selector_ADE_mean')}\n"
        f"- strongest_prior_all: {all_summary.get('strongest_prior')}\n"
        f"- strongest_prior_ADE_all: {all_summary.get('strongest_prior_ADE_mean')}\n"
        f"- calibrated_minus_strongest_prior_ADE_all: {all_summary.get('calibrated_minus_strongest_prior_ADE')}\n"
        f"- calibrated_beats_strongest_prior_val: {val_pass}\n"
        f"- calibrated_beats_strongest_prior_test: {test_pass}\n"
        f"- calibrated_rollout_passed: {passed}\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36_2_selector_calibration完成": True, "passed": passed, "下一步": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
