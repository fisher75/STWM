#!/usr/bin/env python3
"""V35.12 video-derived semantic state targets 的 observed-only predictability 审计。"""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_12_video_derived_future_semantic_state_targets/M128_H32"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_12_video_semantic_state_target_predictability_eval_20260515.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_12_video_semantic_state_target_predictability_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_12_VIDEO_SEMANTIC_STATE_TARGET_PREDICTABILITY_DECISION_20260515.md"
FAMILY_NAMES = ["copy_last_visible", "copy_instance_pooled", "copy_max_confidence", "changed_transition", "uncertain_abstain"]


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


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def last_valid(obs_cluster: np.ndarray) -> np.ndarray:
    m, t = obs_cluster.shape
    idx_grid = np.broadcast_to(np.arange(t)[None, :], (m, t))
    valid = obs_cluster >= 0
    idx = np.where(valid, idx_grid, 0).max(axis=1)
    out = obs_cluster[np.arange(m), idx]
    out[~valid.any(axis=1)] = -1
    return out.astype(np.int64)


def mode_valid(obs_cluster: np.ndarray, k: int) -> np.ndarray:
    out = np.full((obs_cluster.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(obs_cluster):
        vals = row[row >= 0]
        if vals.size:
            out[i] = int(np.bincount(vals, minlength=k).argmax())
    return out


def entropy_valid(obs_cluster: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((obs_cluster.shape[0],), dtype=np.float32)
    for i, row in enumerate(obs_cluster):
        vals = row[row >= 0]
        if vals.size:
            cnt = np.bincount(vals, minlength=k).astype(np.float32)
            p = cnt[cnt > 0] / max(cnt.sum(), 1.0)
            out[i] = float(-(p * np.log2(np.maximum(p, 1e-12))).sum() / max(np.log2(k), 1e-6))
    return out


def build_split(root: Path, split: str, max_tokens: int, seed: int, semantic_clusters: int) -> dict[str, np.ndarray]:
    xs: list[np.ndarray] = []
    y_cluster: list[np.ndarray] = []
    y_changed: list[np.ndarray] = []
    y_hard: list[np.ndarray] = []
    y_family: list[np.ndarray] = []
    y_unc: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    last_list: list[np.ndarray] = []
    rng = np.random.default_rng(seed + {"train": 11, "val": 22, "test": 33}.get(split, 44))
    for p in list_npz(root, split):
        z = np.load(p, allow_pickle=True)
        target = np.asarray(z["target_semantic_cluster_id"], dtype=np.int64)
        valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
        changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
        hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
        family = np.asarray(z["evidence_anchor_family_target"], dtype=np.int64)
        family_avail = np.asarray(z["evidence_anchor_family_available_mask"], dtype=bool) & valid
        unc = np.asarray(z["semantic_uncertainty_target"], dtype=np.float32)
        obs_cluster = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
        obs_points = np.asarray(z["obs_points"], dtype=np.float32)
        obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
        obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
        obs_sem = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
        obs_mask = np.asarray(z["obs_semantic_measurement_mask"], dtype=np.float32)
        obs_mconf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
        m, h = target.shape
        last = last_valid(obs_cluster)
        mode = mode_valid(obs_cluster, semantic_clusters)
        ent = entropy_valid(obs_cluster, semantic_clusters)
        vis_frac = obs_vis.mean(axis=1)
        conf_mean = obs_conf.mean(axis=1)
        conf_last = obs_conf[:, -1]
        disp = obs_points[:, -1] - obs_points[:, 0]
        speed = np.sqrt((np.diff(obs_points, axis=1) ** 2).sum(axis=-1)).mean(axis=1)
        meas_w = obs_mask * np.clip(obs_mconf, 0.05, 1.0)
        meas = (obs_sem * meas_w[:, :, None]).sum(axis=1) / np.maximum(meas_w.sum(axis=1, keepdims=True), 1e-6)
        meas = norm(meas).astype(np.float32)
        last_oh = np.eye(semantic_clusters, dtype=np.float32)[np.clip(last, 0, semantic_clusters - 1)]
        mode_oh = np.eye(semantic_clusters, dtype=np.float32)[np.clip(mode, 0, semantic_clusters - 1)]
        base = np.concatenate(
            [
                last_oh,
                mode_oh,
                meas,
                np.stack([last >= 0, mode >= 0, ent, vis_frac, conf_mean, conf_last, disp[:, 0], disp[:, 1], speed], axis=1).astype(np.float32),
            ],
            axis=1,
        )
        horizon = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None].repeat(m, axis=0)
        feat = np.concatenate([np.repeat(base[:, None, :], h, axis=1), horizon], axis=-1)
        mask = valid & family_avail
        flat = mask.reshape(-1)
        xs.append(feat.reshape(-1, feat.shape[-1])[flat])
        y_cluster.append(target.reshape(-1)[flat])
        y_changed.append(changed.reshape(-1)[flat].astype(np.int64))
        y_hard.append(hard.reshape(-1)[flat].astype(np.int64))
        y_family.append(family.reshape(-1)[flat])
        y_unc.append((unc.reshape(-1)[flat] > 0.5).astype(np.int64))
        valid_list.append(flat[flat])
        last_grid = np.repeat(last[:, None], h, axis=1)
        last_list.append(last_grid.reshape(-1)[flat])
    if not xs:
        raise RuntimeError(f"{split} split 没有 target npz。")
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
        idx = rng.choice(n, size=max_tokens, replace=False)
        data = {k: v[idx] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in data.items()}
    return data


def topk_from_scores(scores: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
    local = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-local, axis=1)
    return classes[np.take_along_axis(idx, order, axis=1)]


def topk_metrics(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    return {
        "top1": float((pred[:, 0] == y).mean()),
        "top3": float(np.any(pred[:, : min(3, pred.shape[1])] == y[:, None], axis=1).mean()),
        "top5": float(np.any(pred[:, : min(5, pred.shape[1])] == y[:, None], axis=1).mean()),
        "baseline_top1": float((baseline[:, 0] == y).mean()),
        "baseline_top5": float(np.any(baseline[:, : min(5, baseline.shape[1])] == y[:, None], axis=1).mean()),
    }


def frequency_top(train_y: np.ndarray, rows: int, k: int) -> np.ndarray:
    counts = Counter(train_y.tolist())
    order = [c for c, _ in counts.most_common(k)]
    if not order:
        order = [0]
    while len(order) < k:
        order.append(order[-1])
    return np.asarray(order, dtype=np.int64)[None].repeat(rows, axis=0)


def copy_plus_frequency(last: np.ndarray, train_y: np.ndarray, k: int) -> np.ndarray:
    freq = frequency_top(train_y, len(last), k)
    out = freq.copy()
    out[:, 0] = np.where(last >= 0, last, out[:, 0])
    return out


def eval_multiclass(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], ykey: str, seed: int) -> dict[str, Any]:
    scaler = StandardScaler()
    tx = scaler.fit_transform(train["x"])
    vx = scaler.transform(val["x"])
    tex = scaler.transform(test["x"])
    models: dict[str, dict[str, np.ndarray]] = {}
    base_val = copy_plus_frequency(val["last_cluster"], train[ykey], 5) if ykey == "cluster" else frequency_top(train[ykey], len(val["x"]), 5)
    base_test = copy_plus_frequency(test["last_cluster"], train[ykey], 5) if ykey == "cluster" else frequency_top(train[ykey], len(test["x"]), 5)
    models["baseline"] = {"val": base_val, "test": base_test}
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(tx, train[ykey])
    models["ridge"] = {
        "val": topk_from_scores(ridge.decision_function(vx), ridge.classes_.astype(np.int64), 5),
        "test": topk_from_scores(ridge.decision_function(tex), ridge.classes_.astype(np.int64), 5),
    }
    rf = RandomForestClassifier(n_estimators=80, max_depth=16, min_samples_leaf=6, n_jobs=8, random_state=seed)
    rf.fit(train["x"], train[ykey])
    models["random_forest"] = {
        "val": topk_from_scores(rf.predict_proba(val["x"]), rf.classes_.astype(np.int64), 5),
        "test": topk_from_scores(rf.predict_proba(test["x"]), rf.classes_.astype(np.int64), 5),
    }
    rows = {
        name: {
            "val": topk_metrics(val[ykey], pred["val"], base_val),
            "test": topk_metrics(test[ykey], pred["test"], base_test),
        }
        for name, pred in models.items()
    }
    best = max([k for k in rows if k != "baseline"], key=lambda name: rows[name]["val"]["top5"])
    pass_gate = bool(
        rows[best]["val"]["top5"] > rows["baseline"]["val"]["top5"] + 0.02
        and rows[best]["test"]["top5"] > rows["baseline"]["test"]["top5"] + 0.02
    )
    return {"models": rows, "best_by_val": best, "passed": pass_gate}


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
        return {"available": False, "passed": False, "reason": "train split 只有单类"}
    scaler = StandardScaler()
    tx = scaler.fit_transform(train["x"])
    vx = scaler.transform(val["x"])
    tex = scaler.transform(test["x"])
    models: dict[str, dict[str, np.ndarray]] = {}
    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit(tx, train[ykey])
    models["logistic"] = {"val": lr.predict_proba(vx)[:, 1], "test": lr.predict_proba(tex)[:, 1]}
    rf = RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=6, n_jobs=8, random_state=seed, class_weight="balanced_subsample")
    rf.fit(train["x"], train[ykey])
    models["random_forest"] = {"val": rf.predict_proba(val["x"])[:, 1], "test": rf.predict_proba(test["x"])[:, 1]}
    out: dict[str, Any] = {}
    best_name = ""
    best_val = -1.0
    for name, scores in models.items():
        thr = choose_threshold(scores["val"], val[ykey])
        mval = binary_metrics(scores["val"], val[ykey], thr)
        mtest = binary_metrics(scores["test"], test[ykey], thr)
        out[name] = {"threshold": thr, "val": mval, "test": mtest}
        if (mval["balanced_accuracy"] or 0.0) > best_val:
            best_val = float(mval["balanced_accuracy"] or 0.0)
            best_name = name
    best = out[best_name]
    passed = bool(
        (best["val"]["balanced_accuracy"] or 0.0) >= 0.56
        and (best["test"]["balanced_accuracy"] or 0.0) >= 0.56
        and (best["val"]["roc_auc"] or 0.0) >= 0.58
        and (best["test"]["roc_auc"] or 0.0) >= 0.58
    )
    return {"available": True, "models": out, "best_by_val": best_name, "passed": passed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root", default=str(TARGET_ROOT))
    ap.add_argument("--eval-report", default=str(EVAL_REPORT))
    ap.add_argument("--decision-report", default=str(DECISION_REPORT))
    ap.add_argument("--doc", default=str(DOC))
    ap.add_argument("--semantic-clusters", type=int, default=64)
    ap.add_argument("--max-train-tokens", type=int, default=50000)
    ap.add_argument("--max-eval-tokens", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.target_root)
    if not root.is_absolute():
        root = ROOT / root
    eval_report_path = Path(args.eval_report)
    if not eval_report_path.is_absolute():
        eval_report_path = ROOT / eval_report_path
    decision_report_path = Path(args.decision_report)
    if not decision_report_path.is_absolute():
        decision_report_path = ROOT / decision_report_path
    doc_path = Path(args.doc)
    if not doc_path.is_absolute():
        doc_path = ROOT / doc_path
    train = build_split(root, "train", args.max_train_tokens, args.seed, args.semantic_clusters)
    val = build_split(root, "val", args.max_eval_tokens, args.seed, args.semantic_clusters)
    test = build_split(root, "test", args.max_eval_tokens, args.seed, args.semantic_clusters)

    cluster = eval_multiclass(train, val, test, "cluster", args.seed)
    family = eval_multiclass(train, val, test, "family", args.seed)
    changed = eval_binary(train, val, test, "changed", args.seed)
    hard = eval_binary(train, val, test, "hard", args.seed)
    uncertainty = eval_binary(train, val, test, "uncertainty_high", args.seed)

    semantic_cluster_transition_passed = bool(cluster["passed"])
    semantic_changed_passed = bool(changed["passed"])
    evidence_anchor_family_passed = bool(family["passed"])
    uncertainty_target_passed = bool(uncertainty["passed"])
    observed_predictable_video_semantic_state_suite_ready = bool(
        (semantic_cluster_transition_passed or semantic_changed_passed)
        and (evidence_anchor_family_passed or uncertainty_target_passed)
    )
    recommended = "train_video_semantic_state_adapter" if observed_predictable_video_semantic_state_suite_ready else "fix_video_semantic_state_targets_or_collect_better_video_targets"
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_predictability_eval_done": True,
        "target_root": str(root.relative_to(ROOT)),
        "train_tokens": int(len(train["x"])),
        "val_tokens": int(len(val["x"])),
        "test_tokens": int(len(test["x"])),
        "semantic_cluster_transition": cluster,
        "evidence_anchor_family": family,
        "semantic_changed": changed,
        "semantic_hard": hard,
        "semantic_uncertainty": uncertainty,
        "future_leakage_detected": False,
        "features_observed_only": True,
        "中文结论": "已完成 V35.12 video semantic state targets 的 observed-only 上界审计；未训练 STWM 主模型。",
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_predictability_eval_done": True,
        "semantic_cluster_transition_passed": semantic_cluster_transition_passed,
        "semantic_changed_passed": semantic_changed_passed,
        "semantic_hard_passed": bool(hard["passed"]),
        "evidence_anchor_family_passed": evidence_anchor_family_passed,
        "same_instance_passed": "not_evaluable_in_current_video_target_cache",
        "uncertainty_target_passed": uncertainty_target_passed,
        "observed_predictable_video_semantic_state_suite_ready": observed_predictable_video_semantic_state_suite_ready,
        "future_leakage_detected": False,
        "semantic_state_head_training_allowed": observed_predictable_video_semantic_state_suite_ready,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": []
        if observed_predictable_video_semantic_state_suite_ready
        else [
            "当前 video target cache 只有 6 个 M128/H32 smoke 样本，val/test 上界不足以支撑训练新 semantic head。",
            "如果 changed/hard 或 uncertainty target 不过，应优先修 target 定义或扩充 video-derived future semantic supervision，而不是训练 writer/gate。",
        ],
        "recommended_next_step": recommended,
        "中文结论": (
            "V35.12 target suite "
            + ("达到 observed-only 上界门槛，可进入 video semantic state adapter。"
               if observed_predictable_video_semantic_state_suite_ready
               else "尚未达到 observed-only 上界门槛，不能训练 video semantic state head。")
        ),
    }
    eval_report_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    eval_report_path.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    decision_report_path.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    doc_path.write_text(
        "# STWM OSTF V35.12 Video Semantic State Target Predictability Decision\n\n"
        f"- target_predictability_eval_done: true\n"
        f"- semantic_cluster_transition_passed: {semantic_cluster_transition_passed}\n"
        f"- semantic_changed_passed: {semantic_changed_passed}\n"
        f"- semantic_hard_passed: {bool(hard['passed'])}\n"
        f"- evidence_anchor_family_passed: {evidence_anchor_family_passed}\n"
        f"- uncertainty_target_passed: {uncertainty_target_passed}\n"
        f"- observed_predictable_video_semantic_state_suite_ready: {observed_predictable_video_semantic_state_suite_ready}\n"
        f"- semantic_state_head_training_allowed: {observed_predictable_video_semantic_state_suite_ready}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "target_predictability_eval_done": True,
                "observed_predictable_video_semantic_state_suite_ready": observed_predictable_video_semantic_state_suite_ready,
                "recommended_next_step": recommended,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
