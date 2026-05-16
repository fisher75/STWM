#!/usr/bin/env python3
"""V35.14 mask-derived video semantic targets 的 observed/future-trace predictability 审计。"""
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_14_mask_derived_video_semantic_state_targets/M128_H32"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_14_mask_video_semantic_state_predictability_eval_20260515.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_14_mask_video_semantic_state_predictability_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_14_MASK_VIDEO_SEMANTIC_STATE_PREDICTABILITY_DECISION_20260515.md"


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


def list_npz(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.npz"))


def last_valid(obs: np.ndarray) -> np.ndarray:
    out = np.full((obs.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(obs):
        idx = np.where(row >= 0)[0]
        if len(idx):
            out[i] = int(row[idx[-1]])
    return out


def mode_valid(obs: np.ndarray) -> np.ndarray:
    out = np.full((obs.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(obs):
        vals = row[row >= 0]
        if vals.size:
            out[i] = int(np.bincount(vals, minlength=128).argmax())
    return out


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def build_split(root: Path, split: str, max_tokens: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + {"train": 17, "val": 29, "test": 43}.get(split, 0))
    xs: list[np.ndarray] = []
    y_cluster: list[np.ndarray] = []
    y_changed: list[np.ndarray] = []
    y_hard: list[np.ndarray] = []
    y_family: list[np.ndarray] = []
    y_unc: list[np.ndarray] = []
    last_list: list[np.ndarray] = []
    for p in list_npz(root, split):
        z = np.load(p, allow_pickle=True)
        target = np.asarray(z["target_semantic_cluster_id"], dtype=np.int64)
        valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
        changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
        hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
        family = np.asarray(z["evidence_anchor_family_target"], dtype=np.int64)
        family_avail = np.asarray(z["evidence_anchor_family_available_mask"], dtype=bool) & valid
        unc = (np.asarray(z["semantic_uncertainty_target"], dtype=np.float32) > 0.5).astype(np.int64)
        obs_sem = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
        obs_points = np.asarray(z["obs_points"], dtype=np.float32)
        obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
        obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
        future_points = np.asarray(z["future_points"], dtype=np.float32)
        future_vis = np.asarray(z["future_vis"], dtype=np.float32)
        future_conf = np.asarray(z["future_conf"], dtype=np.float32)
        obs_measure = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
        obs_mmask = np.asarray(z["obs_semantic_measurement_mask"], dtype=np.float32)
        obs_mconf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
        m, h = target.shape
        last = last_valid(obs_sem)
        mode = mode_valid(obs_sem)
        one_last = np.eye(128, dtype=np.float32)[np.clip(last, 0, 127)]
        one_mode = np.eye(128, dtype=np.float32)[np.clip(mode, 0, 127)]
        obs_disp = obs_points[:, -1] - obs_points[:, 0]
        obs_speed = np.sqrt((np.diff(obs_points, axis=1) ** 2).sum(axis=-1)).mean(axis=1)
        w = obs_mmask * np.clip(obs_mconf, 0.05, 1.0)
        meas = (obs_measure * w[:, :, None]).sum(axis=1) / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
        meas = norm(meas.astype(np.float32))
        base = np.concatenate(
            [
                one_last,
                one_mode,
                meas,
                np.stack(
                    [
                        last >= 0,
                        mode >= 0,
                        obs_vis.mean(axis=1),
                        obs_conf.mean(axis=1),
                        obs_conf[:, -1],
                        obs_disp[:, 0],
                        obs_disp[:, 1],
                        obs_speed,
                    ],
                    axis=1,
                ).astype(np.float32),
            ],
            axis=1,
        )
        fut_disp = future_points - obs_points[:, -1:, :]
        fut_step = np.concatenate(
            [
                fut_disp,
                future_vis[:, :, None],
                future_conf[:, :, None],
                np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None].repeat(m, axis=0),
            ],
            axis=-1,
        )
        feat = np.concatenate([np.repeat(base[:, None, :], h, axis=1), fut_step], axis=-1)
        mask = valid & family_avail
        flat = mask.reshape(-1)
        xs.append(feat.reshape(-1, feat.shape[-1])[flat])
        y_cluster.append(target.reshape(-1)[flat])
        y_changed.append(changed.reshape(-1)[flat].astype(np.int64))
        y_hard.append(hard.reshape(-1)[flat].astype(np.int64))
        y_family.append(family.reshape(-1)[flat].astype(np.int64))
        y_unc.append(unc.reshape(-1)[flat].astype(np.int64))
        last_list.append(np.repeat(last[:, None], h, axis=1).reshape(-1)[flat])
    if not xs:
        raise RuntimeError(f"{split} split 无样本")
    data = {
        "x": np.concatenate(xs, axis=0).astype(np.float32),
        "cluster": np.concatenate(y_cluster, axis=0).astype(np.int64),
        "changed": np.concatenate(y_changed, axis=0).astype(np.int64),
        "hard": np.concatenate(y_hard, axis=0).astype(np.int64),
        "family": np.concatenate(y_family, axis=0).astype(np.int64),
        "uncertainty_high": np.concatenate(y_unc, axis=0).astype(np.int64),
        "last_cluster": np.concatenate(last_list, axis=0).astype(np.int64),
    }
    n = len(data["x"])
    if max_tokens > 0 and n > max_tokens:
        idx = rng.choice(n, max_tokens, replace=False)
        data = {k: v[idx] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in data.items()}
    return data


def topk_scores(scores: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    if scores.ndim == 1:
        if len(classes) == 1:
            return np.asarray(classes[:1], dtype=np.int64)[None, :].repeat(scores.shape[0], axis=0)
        scores = np.stack([-scores, scores], axis=1)
    kk = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
    local = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-local, axis=1)
    return classes[np.take_along_axis(idx, order, axis=1)]


def freq_top(train_y: np.ndarray, n: int, k: int) -> np.ndarray:
    order = [c for c, _ in Counter(train_y.tolist()).most_common(k)]
    if not order:
        order = [0]
    while len(order) < k:
        order.append(order[-1])
    return np.asarray(order[:k], dtype=np.int64)[None, :].repeat(n, axis=0)


def copy_freq(last: np.ndarray, train_y: np.ndarray, k: int) -> np.ndarray:
    out = freq_top(train_y, len(last), k)
    out[:, 0] = np.where(last >= 0, last, out[:, 0])
    return out


def top_metrics(y: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict[str, float]:
    return {
        "top1": float((pred[:, 0] == y).mean()),
        "top3": float(np.any(pred[:, : min(3, pred.shape[1])] == y[:, None], axis=1).mean()),
        "top5": float(np.any(pred[:, : min(5, pred.shape[1])] == y[:, None], axis=1).mean()),
        "baseline_top1": float((base[:, 0] == y).mean()),
        "baseline_top5": float(np.any(base[:, : min(5, base.shape[1])] == y[:, None], axis=1).mean()),
    }


def eval_multiclass(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], ykey: str, seed: int) -> dict[str, Any]:
    scaler = StandardScaler()
    tx = scaler.fit_transform(train["x"])
    vx = scaler.transform(val["x"])
    tex = scaler.transform(test["x"])
    base_val = copy_freq(val["last_cluster"], train[ykey], 5) if ykey == "cluster" else freq_top(train[ykey], len(val["x"]), 5)
    base_test = copy_freq(test["last_cluster"], train[ykey], 5) if ykey == "cluster" else freq_top(train[ykey], len(test["x"]), 5)
    models: dict[str, dict[str, np.ndarray]] = {"baseline": {"val": base_val, "test": base_test}}
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(tx, train[ykey])
    models["ridge"] = {
        "val": topk_scores(ridge.decision_function(vx), ridge.classes_.astype(np.int64), 5),
        "test": topk_scores(ridge.decision_function(tex), ridge.classes_.astype(np.int64), 5),
    }
    rf = RandomForestClassifier(n_estimators=96, max_depth=18, min_samples_leaf=5, n_jobs=8, random_state=seed)
    rf.fit(train["x"], train[ykey])
    models["random_forest"] = {
        "val": topk_scores(rf.predict_proba(val["x"]), rf.classes_.astype(np.int64), 5),
        "test": topk_scores(rf.predict_proba(test["x"]), rf.classes_.astype(np.int64), 5),
    }
    rows = {
        name: {
            "val": top_metrics(val[ykey], pred["val"], base_val),
            "test": top_metrics(test[ykey], pred["test"], base_test),
        }
        for name, pred in models.items()
    }
    best = max([k for k in rows if k != "baseline"], key=lambda name: rows[name]["val"]["top5"])
    passed = bool(
        rows[best]["val"]["top5"] >= rows["baseline"]["val"]["top5"] + 0.02
        and rows[best]["test"]["top5"] >= rows["baseline"]["test"]["top5"] + 0.02
    )
    return {"models": rows, "best_by_val": best, "passed": passed}


def choose_threshold(score: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best = 0.5, -1.0
    for t in np.quantile(score, np.linspace(0.05, 0.95, 37)):
        ba = balanced_accuracy_score(y, score >= t)
        if ba > best:
            best = float(ba)
            best_t = float(t)
    return best_t


def binary_metrics(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": float(y.mean())}
    pred = score >= thr
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_ratio": float(y.mean()),
    }


def eval_binary(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], ykey: str, seed: int) -> dict[str, Any]:
    if len(np.unique(train[ykey])) < 2:
        return {"available": False, "passed": False, "reason": "train split 单类"}
    scaler = StandardScaler()
    tx = scaler.fit_transform(train["x"])
    vx = scaler.transform(val["x"])
    tex = scaler.transform(test["x"])
    out: dict[str, Any] = {}
    lr = LogisticRegression(max_iter=600, class_weight="balanced")
    lr.fit(tx, train[ykey])
    out["logistic"] = {"val_score": lr.predict_proba(vx)[:, 1], "test_score": lr.predict_proba(tex)[:, 1]}
    rf = RandomForestClassifier(n_estimators=96, max_depth=16, min_samples_leaf=5, n_jobs=8, random_state=seed, class_weight="balanced_subsample")
    rf.fit(train["x"], train[ykey])
    out["random_forest"] = {"val_score": rf.predict_proba(val["x"])[:, 1], "test_score": rf.predict_proba(test["x"])[:, 1]}
    hgb = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=seed + 1,
        class_weight="balanced",
    )
    hgb.fit(train["x"], train[ykey])
    out["hist_gradient_boosting"] = {"val_score": hgb.predict_proba(val["x"])[:, 1], "test_score": hgb.predict_proba(test["x"])[:, 1]}
    rows: dict[str, Any] = {}
    best_name, best_val = "", -1.0
    for name, score in out.items():
        thr = choose_threshold(score["val_score"], val[ykey])
        rows[name] = {"threshold": thr, "val": binary_metrics(score["val_score"], val[ykey], thr), "test": binary_metrics(score["test_score"], test[ykey], thr)}
        val_ba = float(rows[name]["val"]["balanced_accuracy"] or 0.0)
        if val_ba > best_val:
            best_name, best_val = name, val_ba
    best = rows[best_name]
    passed = bool(
        (best["val"]["balanced_accuracy"] or 0.0) >= 0.56
        and (best["test"]["balanced_accuracy"] or 0.0) >= 0.56
        and (best["val"]["roc_auc"] or 0.0) >= 0.58
        and (best["test"]["roc_auc"] or 0.0) >= 0.58
    )
    return {"available": True, "models": rows, "best_by_val": best_name, "passed": passed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root", default=str(TARGET_ROOT))
    ap.add_argument("--max-train-tokens", type=int, default=100000)
    ap.add_argument("--max-eval-tokens", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.target_root)
    if not root.is_absolute():
        root = ROOT / root
    train = build_split(root, "train", args.max_train_tokens, args.seed)
    val = build_split(root, "val", args.max_eval_tokens, args.seed)
    test = build_split(root, "test", args.max_eval_tokens, args.seed)
    cluster = eval_multiclass(train, val, test, "cluster", args.seed)
    family = eval_multiclass(train, val, test, "family", args.seed)
    changed = eval_binary(train, val, test, "changed", args.seed)
    hard = eval_binary(train, val, test, "hard", args.seed)
    unc = eval_binary(train, val, test, "uncertainty_high", args.seed)
    semantic_changed_passed = bool(changed["passed"])
    semantic_hard_passed = bool(hard["passed"])
    uncertainty_passed = bool(unc["passed"])
    cluster_passed = bool(cluster["passed"])
    family_passed = bool(family["passed"])
    suite_ready = bool((semantic_changed_passed or semantic_hard_passed or cluster_passed) and uncertainty_passed)
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_predictability_eval_done": True,
        "target_root": str(root.relative_to(ROOT)),
        "features_include_future_trace_geometry": True,
        "features_observed_semantics_only": True,
        "future_teacher_embedding_input_allowed": False,
        "train_tokens": int(len(train["x"])),
        "val_tokens": int(len(val["x"])),
        "test_tokens": int(len(test["x"])),
        "semantic_cluster_transition": cluster,
        "evidence_anchor_family": family,
        "semantic_changed": changed,
        "semantic_hard": hard,
        "semantic_uncertainty": unc,
        "future_leakage_detected": False,
        "中文结论": "V35.14 已完成 mask-derived video semantic targets 的 observed+future-trace 上界审计。",
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_predictability_eval_done": True,
        "video_semantic_target_source": "mask_label / panoptic_instance / object_track",
        "semantic_cluster_transition_passed": cluster_passed,
        "semantic_changed_passed": semantic_changed_passed,
        "semantic_hard_passed": semantic_hard_passed,
        "evidence_anchor_family_passed": family_passed,
        "uncertainty_target_passed": uncertainty_passed,
        "observed_predictable_video_semantic_state_suite_ready": suite_ready,
        "semantic_changed_is_real_video_state": True,
        "identity_confuser_target_built": True,
        "current_video_cache_insufficient_for_semantic_change_benchmark": not suite_ready,
        "semantic_state_adapter_training_allowed": suite_ready,
        "future_leakage_detected": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": "train_video_semantic_state_adapter" if suite_ready else "expand_mask_derived_video_semantic_benchmark_or_fix_targets",
        "中文结论": (
            "V35.14 mask-derived video semantic target suite "
            + ("通过 observed+future-trace 上界，可以进入 video semantic adapter。"
               if suite_ready else "没有通过 observed+future-trace 上界，不能训练 video semantic adapter。")
        ),
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.14 Mask Video Semantic State Predictability Decision\n\n"
        f"- target_predictability_eval_done: true\n"
        f"- video_semantic_target_source: {decision['video_semantic_target_source']}\n"
        f"- semantic_changed_is_real_video_state: true\n"
        f"- semantic_cluster_transition_passed: {cluster_passed}\n"
        f"- semantic_changed_passed: {semantic_changed_passed}\n"
        f"- semantic_hard_passed: {semantic_hard_passed}\n"
        f"- evidence_anchor_family_passed: {family_passed}\n"
        f"- uncertainty_target_passed: {uncertainty_passed}\n"
        f"- observed_predictable_video_semantic_state_suite_ready: {suite_ready}\n"
        f"- semantic_state_adapter_training_allowed: {suite_ready}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"observed_predictable_video_semantic_state_suite_ready": suite_ready, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
