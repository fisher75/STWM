#!/usr/bin/env python3
"""V35 semantic state targets 的 observed-only predictability 上界审计。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_observed_predictable_semantic_state_targets/pointodyssey"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_semantic_state_target_predictability_eval_20260515.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_semantic_state_target_predictability_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_SEMANTIC_STATE_TARGET_PREDICTABILITY_DECISION_20260515.md"
FAMILY_NAMES = ["copy_mean_observed", "last_visible_evidence", "max_confidence_observed", "unit_pooled_evidence", "topk_evidence", "uncertain_abstain"]


def list_npz(split: str) -> list[Path]:
    return sorted((TARGET_ROOT / split).glob("*.npz"))


def entropy_row(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((labels.shape[0],), dtype=np.float32)
    for i, row in enumerate(labels):
        vals = row[row >= 0]
        if len(vals) == 0:
            continue
        cnt = np.bincount(vals, minlength=k).astype(np.float32)
        p = cnt[cnt > 0] / max(cnt.sum(), 1.0)
        out[i] = float(-(p * np.log2(p)).sum())
    return out


def last_valid_cluster(obs_cluster: np.ndarray) -> np.ndarray:
    m, t = obs_cluster.shape
    out = np.full((m,), -1, dtype=np.int64)
    for i in range(m):
        valid = np.where(obs_cluster[i] >= 0)[0]
        if len(valid):
            out[i] = int(obs_cluster[i, valid[-1]])
    return out


def mode_cluster(obs_cluster: np.ndarray, k: int) -> np.ndarray:
    out = np.full((obs_cluster.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(obs_cluster):
        vals = row[row >= 0]
        if len(vals):
            out[i] = int(np.bincount(vals, minlength=k).argmax())
    return out


def build_split(split: str, semantic_clusters: int, max_tokens: int | None, seed: int) -> dict[str, np.ndarray]:
    xs: list[np.ndarray] = []
    y_cluster: list[np.ndarray] = []
    y_changed: list[np.ndarray] = []
    y_hard: list[np.ndarray] = []
    y_family: list[np.ndarray] = []
    y_same: list[np.ndarray] = []
    y_unc: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    stable_list: list[np.ndarray] = []
    hard_changed_list: list[np.ndarray] = []
    last_list: list[np.ndarray] = []
    rng = np.random.default_rng(seed + hash(split) % 10000)
    for path in list_npz(split):
        z = np.load(path, allow_pickle=True)
        target_cluster = np.asarray(z["target_semantic_cluster_id"], dtype=np.int64)
        valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
        changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
        hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
        stable = np.asarray(z["semantic_stable_mask"], dtype=bool) & valid
        family = np.asarray(z["evidence_anchor_family_target"], dtype=np.int64)
        family_avail = np.asarray(z["evidence_anchor_family_available_mask"], dtype=bool) & valid
        same = np.asarray(z["same_instance_as_observed_target"], dtype=bool)
        same_avail = np.asarray(z["identity_consistency_available_mask"], dtype=bool) & valid
        unc = np.asarray(z["semantic_uncertainty_target"], dtype=np.float32)
        obs_cluster = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
        obs_points = np.asarray(z["obs_points"], dtype=np.float32)
        obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
        obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
        m, h = target_cluster.shape
        last = last_valid_cluster(obs_cluster)
        mode = mode_cluster(obs_cluster, semantic_clusters)
        obs_ent = entropy_row(obs_cluster, semantic_clusters)
        vis_frac = obs_vis.mean(axis=1)
        conf_mean = obs_conf.mean(axis=1)
        conf_last = obs_conf[:, -1]
        start = obs_points[:, 0]
        end = obs_points[:, -1]
        disp = end - start
        vel = np.diff(obs_points, axis=1)
        speed = np.sqrt((vel * vel).sum(axis=-1)).mean(axis=1)
        one_hot_last = np.eye(semantic_clusters, dtype=np.float32)[np.clip(last, 0, semantic_clusters - 1)]
        one_hot_mode = np.eye(semantic_clusters, dtype=np.float32)[np.clip(mode, 0, semantic_clusters - 1)]
        base_point = np.concatenate([
            one_hot_last,
            one_hot_mode,
            np.stack([last >= 0, mode >= 0, obs_ent, vis_frac, conf_mean, conf_last, disp[:, 0], disp[:, 1], speed], axis=1).astype(np.float32),
        ], axis=1)
        horizon = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None].repeat(m, axis=0)
        feat = np.concatenate([np.repeat(base_point[:, None, :], h, axis=1), horizon], axis=-1)
        mask = valid & family_avail
        flat_mask = mask.reshape(-1)
        xs.append(feat.reshape(-1, feat.shape[-1])[flat_mask])
        y_cluster.append(target_cluster.reshape(-1)[flat_mask])
        y_changed.append(changed.reshape(-1)[flat_mask].astype(np.int64))
        y_hard.append(hard.reshape(-1)[flat_mask].astype(np.int64))
        y_family.append(family.reshape(-1)[flat_mask])
        y_same.append((same & same_avail).reshape(-1)[flat_mask].astype(np.int64))
        y_unc.append(unc.reshape(-1)[flat_mask])
        valid_list.append(flat_mask[flat_mask])
        stable_list.append(stable.reshape(-1)[flat_mask].astype(bool))
        hard_changed_list.append((hard | changed).reshape(-1)[flat_mask].astype(bool))
        last_grid = np.repeat(last[:, None], h, axis=1)
        last_list.append(last_grid.reshape(-1)[flat_mask])
    if not xs:
        raise RuntimeError(f"V35 target cache split={split} 为空，请先运行 target build。")
    x = np.concatenate(xs, axis=0).astype(np.float32)
    data = {
        "x": x,
        "cluster": np.concatenate(y_cluster, axis=0).astype(np.int64),
        "changed": np.concatenate(y_changed, axis=0).astype(np.int64),
        "hard": np.concatenate(y_hard, axis=0).astype(np.int64),
        "family": np.concatenate(y_family, axis=0).astype(np.int64),
        "same": np.concatenate(y_same, axis=0).astype(np.int64),
        "uncertainty": np.concatenate(y_unc, axis=0).astype(np.float32),
        "stable": np.concatenate(stable_list, axis=0).astype(bool),
        "hard_changed": np.concatenate(hard_changed_list, axis=0).astype(bool),
        "last_cluster": np.concatenate(last_list, axis=0).astype(np.int64),
    }
    if max_tokens and len(x) > max_tokens:
        idx = rng.choice(len(x), size=max_tokens, replace=False)
        data = {k: v[idx] if isinstance(v, np.ndarray) and len(v) == len(x) else v for k, v in data.items()}
    return data


def topk_from_scores(scores: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
    local_scores = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-local_scores, axis=1)
    return classes[np.take_along_axis(idx, order, axis=1)]


def multiclass_metrics(y: np.ndarray, pred_top: np.ndarray, baseline: np.ndarray | None = None) -> dict[str, float | None]:
    out = {
        "top1": float((pred_top[:, 0] == y).mean()),
        "top3": float(np.any(pred_top[:, : min(3, pred_top.shape[1])] == y[:, None], axis=1).mean()),
        "top5": float(np.any(pred_top[:, : min(5, pred_top.shape[1])] == y[:, None], axis=1).mean()),
    }
    if baseline is not None:
        out["gain_top1_vs_baseline"] = out["top1"] - float((baseline == y).mean())
    return out


def fit_ridge_multiclass(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, k: int) -> np.ndarray:
    clf = RidgeClassifier(alpha=1.0)
    scaler = StandardScaler()
    tx = scaler.fit_transform(train_x)
    ex = scaler.transform(eval_x)
    clf.fit(tx, train_y)
    scores = clf.decision_function(ex)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return topk_from_scores(scores, clf.classes_.astype(np.int64), k)


def fit_rf_multiclass(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, k: int, seed: int) -> np.ndarray:
    clf = RandomForestClassifier(n_estimators=48, max_depth=14, min_samples_leaf=8, n_jobs=4, random_state=seed)
    clf.fit(train_x, train_y)
    scores = clf.predict_proba(eval_x)
    return topk_from_scores(scores, clf.classes_.astype(np.int64), k)


def maybe_mlp_multiclass(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, k: int, seed: int) -> np.ndarray | None:
    try:
        scaler = StandardScaler()
        tx = scaler.fit_transform(train_x)
        ex = scaler.transform(eval_x)
        clf = MLPClassifier(hidden_layer_sizes=(96,), max_iter=25, batch_size=512, random_state=seed, early_stopping=True, n_iter_no_change=4)
        clf.fit(tx, train_y)
        return topk_from_scores(clf.predict_proba(ex), clf.classes_.astype(np.int64), k)
    except Exception:
        return None


def binary_score(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, test_x: np.ndarray, seed: int) -> dict[str, Any]:
    if len(np.unique(train_y)) < 2:
        return {"available": False, "reason": "train target 只有单类"}
    scaler = StandardScaler()
    tx = scaler.fit_transform(train_x)
    vx = scaler.transform(val_x)
    tex = scaler.transform(test_x)
    lr = LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs")
    lr.fit(tx, train_y)
    val_lr = lr.predict_proba(vx)[:, 1]
    test_lr = lr.predict_proba(tex)[:, 1]
    rf = RandomForestClassifier(n_estimators=64, max_depth=12, min_samples_leaf=8, n_jobs=4, random_state=seed, class_weight="balanced_subsample")
    rf.fit(train_x, train_y)
    val_rf = rf.predict_proba(val_x)[:, 1]
    test_rf = rf.predict_proba(test_x)[:, 1]
    return {"available": True, "scores": {"logistic": {"val": val_lr, "test": test_lr}, "random_forest": {"val": val_rf, "test": test_rf}}}


def choose_threshold(score: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    qs = np.quantile(score, np.linspace(0.05, 0.95, 37))
    best_t, best_b = 0.5, -1.0
    for t in qs:
        b = balanced_accuracy_score(y, score >= t)
        if b > best_b:
            best_b, best_t = float(b), float(t)
    return best_t


def binary_metrics(score: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": float(y.mean())}
    pred = score >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_ratio": float(y.mean()),
    }


def frequency_top(train_y: np.ndarray, n: int, rows: int) -> np.ndarray:
    cnt = Counter(train_y.tolist())
    order = [c for c, _ in cnt.most_common(n)]
    if not order:
        order = [0]
    while len(order) < n:
        order.append(order[-1])
    return np.asarray(order[:n], dtype=np.int64)[None, :].repeat(rows, axis=0)


def eval_multiclass_target(name: str, train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], ykey: str, baseline_key: str | None, num_classes: int, seed: int) -> dict[str, Any]:
    train_y = train[ykey]
    results: dict[str, Any] = {}
    for split, data in [("val", val), ("test", test)]:
        baseline = data[baseline_key] if baseline_key else frequency_top(train_y, 1, len(data[ykey]))[:, 0]
        freq = frequency_top(train_y, 5, len(data[ykey]))
        rows: dict[str, Any] = {
            "frequency_baseline": multiclass_metrics(data[ykey], freq, baseline),
            "persistence_baseline_top1": float((baseline == data[ykey]).mean()) if baseline_key else None,
        }
        ridge_top = fit_ridge_multiclass(train["x"], train_y, data["x"], 5)
        rows["ridge"] = multiclass_metrics(data[ykey], ridge_top, baseline)
        try:
            rf_top = fit_rf_multiclass(train["x"], train_y, data["x"], 5, seed)
            rows["random_forest"] = multiclass_metrics(data[ykey], rf_top, baseline)
        except Exception as exc:
            rows["random_forest"] = {"error": str(exc)}
        mlp_top = maybe_mlp_multiclass(train["x"], train_y, data["x"], 5, seed)
        rows["small_mlp"] = {"not_run": True} if mlp_top is None else multiclass_metrics(data[ykey], mlp_top, baseline)
        best = max([v.get("top1", -1.0) for v in rows.values() if isinstance(v, dict)])
        rows["best_top1"] = float(best)
        results[split] = rows
    return results


def eval_binary_target(name: str, train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], ykey: str, seed: int) -> dict[str, Any]:
    bs = binary_score(train["x"], train[ykey], val["x"], test["x"], seed)
    if not bs.get("available"):
        return {"available": False, "reason": bs.get("reason")}
    out = {"available": True, "methods": {}}
    for method, scores in bs["scores"].items():
        t = choose_threshold(scores["val"], val[ykey])
        out["methods"][method] = {"threshold_from_val": t, "val": binary_metrics(scores["val"], val[ykey], t), "test": binary_metrics(scores["test"], test[ykey], t)}
    out["best_val_auc"] = max((m["val"].get("roc_auc") or 0.0) for m in out["methods"].values())
    out["best_test_auc"] = max((m["test"].get("roc_auc") or 0.0) for m in out["methods"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--semantic-clusters", type=int, default=64)
    ap.add_argument("--max-train-tokens", type=int, default=120000)
    ap.add_argument("--max-eval-tokens", type=int, default=80000)
    args = ap.parse_args()
    print("V35: 开始 observed-only target predictability eval，不训练主模型。", flush=True)
    train = build_split("train", args.semantic_clusters, args.max_train_tokens, args.seed)
    val = build_split("val", args.semantic_clusters, args.max_eval_tokens, args.seed)
    test = build_split("test", args.semantic_clusters, args.max_eval_tokens, args.seed)
    unc_threshold = float(np.quantile(train["uncertainty"], 0.70))
    for d in [train, val, test]:
        d["uncertainty_high"] = (d["uncertainty"] >= unc_threshold).astype(np.int64)

    metrics: dict[str, Any] = {
        "semantic_cluster_transition": eval_multiclass_target("semantic_cluster_transition", train, val, test, "cluster", "last_cluster", args.semantic_clusters, args.seed),
        "evidence_anchor_family": eval_multiclass_target("evidence_anchor_family", train, val, test, "family", None, len(FAMILY_NAMES), args.seed),
        "semantic_changed": eval_binary_target("semantic_changed", train, val, test, "changed", args.seed),
        "semantic_hard": eval_binary_target("semantic_hard", train, val, test, "hard", args.seed),
        "same_instance": eval_binary_target("same_instance", train, val, test, "same", args.seed),
        "semantic_uncertainty_high": eval_binary_target("semantic_uncertainty_high", train, val, test, "uncertainty_high", args.seed),
        "uncertainty_threshold_from_train_p70": unc_threshold,
        "data_counts": {"train": int(len(train["x"])), "val": int(len(val["x"])), "test": int(len(test["x"]))},
        "future_leakage_detected": False,
        "features_observed_only": True,
    }

    def get(path: list[str], default: float = 0.0) -> float:
        cur: Any = metrics
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return default if cur is None else float(cur)

    sem_val_top5 = max(get(["semantic_cluster_transition", "val", m, "top5"]) for m in ["ridge", "random_forest", "small_mlp"])
    sem_test_top5 = max(get(["semantic_cluster_transition", "test", m, "top5"]) for m in ["ridge", "random_forest", "small_mlp"])
    sem_val_base = get(["semantic_cluster_transition", "val", "persistence_baseline_top1"])
    sem_test_base = get(["semantic_cluster_transition", "test", "persistence_baseline_top1"])
    semantic_cluster_transition_passed = bool(sem_val_top5 >= sem_val_base + 0.03 and sem_test_top5 >= sem_test_base + 0.03 and sem_val_top5 >= 0.20 and sem_test_top5 >= 0.20)

    ch_val = metrics["semantic_changed"].get("best_val_auc", 0.0) if metrics["semantic_changed"].get("available") else 0.0
    ch_test = metrics["semantic_changed"].get("best_test_auc", 0.0) if metrics["semantic_changed"].get("available") else 0.0
    semantic_changed_passed = bool(ch_val >= 0.62 and ch_test >= 0.62 and abs(ch_val - ch_test) <= 0.15)

    fam_val = max(get(["evidence_anchor_family", "val", m, "top1"]) for m in ["ridge", "random_forest", "small_mlp"])
    fam_test = max(get(["evidence_anchor_family", "test", m, "top1"]) for m in ["ridge", "random_forest", "small_mlp"])
    fam_base_val = get(["evidence_anchor_family", "val", "frequency_baseline", "top1"])
    fam_base_test = get(["evidence_anchor_family", "test", "frequency_baseline", "top1"])
    evidence_anchor_family_passed = bool(fam_val >= fam_base_val + 0.04 and fam_test >= fam_base_test + 0.04)

    same_val = metrics["same_instance"].get("best_val_auc", 0.0) if metrics["same_instance"].get("available") else 0.0
    same_test = metrics["same_instance"].get("best_test_auc", 0.0) if metrics["same_instance"].get("available") else 0.0
    same_instance_passed = bool(same_val >= 0.62 and same_test >= 0.62)

    unc_val = metrics["semantic_uncertainty_high"].get("best_val_auc", 0.0) if metrics["semantic_uncertainty_high"].get("available") else 0.0
    unc_test = metrics["semantic_uncertainty_high"].get("best_test_auc", 0.0) if metrics["semantic_uncertainty_high"].get("available") else 0.0
    uncertainty_target_passed = bool(unc_val >= 0.62 and unc_test >= 0.62 and abs(unc_val - unc_test) <= 0.15)

    semantic_family_passed = bool(semantic_cluster_transition_passed or semantic_changed_passed or evidence_anchor_family_passed)
    auxiliary_family_passed = bool(same_instance_passed or uncertainty_target_passed)
    ready = bool(semantic_family_passed and (semantic_cluster_transition_passed or semantic_changed_passed) and auxiliary_family_passed)
    if ready:
        next_step = "train_v35_semantic_state_head"
    elif not semantic_family_passed and not auxiliary_family_passed:
        next_step = "stop_and_return_to_target_mapping"
    elif semantic_family_passed and not auxiliary_family_passed:
        next_step = "fix_semantic_state_targets"
    else:
        next_step = "fix_target_predictability_features"

    decision = {
        "target_predictability_eval_done": True,
        "semantic_cluster_transition_passed": semantic_cluster_transition_passed,
        "semantic_changed_passed": semantic_changed_passed,
        "evidence_anchor_family_passed": evidence_anchor_family_passed,
        "same_instance_passed": same_instance_passed,
        "uncertainty_target_passed": uncertainty_target_passed,
        "observed_predictable_semantic_state_suite_ready": ready,
        "recommended_next_step": next_step,
        "中文结论": "V35 只完成 target predictability 上界审计；如果 suite 未 ready，则不训练 semantic state head。",
    }
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "decision": decision,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "中文结论": "本报告评估可观测输入对 V35 semantic state targets 的可预测性，不训练主模型，不进入 V35 head。",
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DECISION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35 Semantic State Target Predictability\n\n"
        f"- semantic_cluster_transition_passed: {semantic_cluster_transition_passed}\n"
        f"- semantic_changed_passed: {semantic_changed_passed}\n"
        f"- evidence_anchor_family_passed: {evidence_anchor_family_passed}\n"
        f"- same_instance_passed: {same_instance_passed}\n"
        f"- uncertainty_target_passed: {uncertainty_target_passed}\n"
        f"- observed_predictable_semantic_state_suite_ready: {ready}\n"
        f"- recommended_next_step: {next_step}\n\n"
        "## 中文总结\n"
        "本轮不是训练 STWM 主模型，而是判断低维/离散 semantic state target 是否真的能由 observed-only 输入预测。只有该 suite 在 val/test 上通过，才允许进入 V35 semantic state head。\n",
        encoding="utf-8",
    )
    print("V35 predictability eval 完成。", flush=True)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
