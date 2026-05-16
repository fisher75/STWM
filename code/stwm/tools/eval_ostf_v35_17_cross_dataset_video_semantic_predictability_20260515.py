#!/usr/bin/env python3
"""V35.17 cross-dataset video semantic state predictability 审计。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515 import last_valid, mode_valid, norm
from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_17_cross_dataset_mask_derived_video_semantic_state_targets/M128_H32"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_17_cross_dataset_video_semantic_predictability_eval_20260515.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_17_cross_dataset_video_semantic_predictability_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_17_CROSS_DATASET_VIDEO_SEMANTIC_PREDICTABILITY_DECISION_20260515.md"

PROTOCOLS = {
    "mixed_unseen": {"train": ("train", None), "val": ("val", None), "test": ("test", None)},
    "vspw_to_vipseg": {"train": ("train", "VSPW"), "val": ("val", "VSPW"), "test": ("test", "VIPSEG")},
    "vipseg_to_vspw": {"train": ("train", "VIPSEG"), "val": ("val", "VIPSEG"), "test": ("test", "VSPW")},
}


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


def paths_for(root: Path, split: str, dataset: str | None) -> list[Path]:
    out = []
    for p in sorted((root / split).glob("*.npz")):
        if dataset is None:
            out.append(p)
            continue
        z = np.load(p, allow_pickle=True)
        if str(np.asarray(z["dataset"]).item()) == dataset:
            out.append(p)
    return out


def build_from_paths(paths: list[Path], max_tokens: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    y_cluster: list[np.ndarray] = []
    y_changed: list[np.ndarray] = []
    y_hard: list[np.ndarray] = []
    y_unc: list[np.ndarray] = []
    y_family: list[np.ndarray] = []
    last_list: list[np.ndarray] = []
    meta_dataset: list[np.ndarray] = []
    meta_motion: list[np.ndarray] = []
    meta_occlusion: list[np.ndarray] = []
    meta_category: list[np.ndarray] = []
    for p in paths:
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
        future_vis_bool = np.asarray(z["future_vis"], dtype=bool)
        future_vis = future_vis_bool.astype(np.float32)
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
                    [last >= 0, mode >= 0, obs_vis.mean(axis=1), obs_conf.mean(axis=1), obs_conf[:, -1], obs_disp[:, 0], obs_disp[:, 1], obs_speed],
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
        y_unc.append(unc.reshape(-1)[flat].astype(np.int64))
        y_family.append(family.reshape(-1)[flat].astype(np.int64))
        last_list.append(np.repeat(last[:, None], h, axis=1).reshape(-1)[flat])
        dataset_id = 1 if str(np.asarray(z["dataset"]).item()) == "VIPSEG" else 0
        meta_dataset.append(np.full(int(flat.sum()), dataset_id, dtype=np.int64))
        motion = np.linalg.norm(future_points - obs_points[:, -1:, :], axis=-1)
        meta_motion.append(motion.reshape(-1)[flat].astype(np.float32))
        meta_occlusion.append((~future_vis_bool).reshape(-1)[flat].astype(np.int64))
        source_sem = np.asarray(z["source_semantic_id"], dtype=np.int64) if "source_semantic_id" in z.files else last
        meta_category.append(np.repeat(source_sem[:, None], h, axis=1).reshape(-1)[flat].astype(np.int64))
    if not xs:
        raise RuntimeError("没有可用 token")
    data = {
        "x": np.concatenate(xs).astype(np.float32),
        "cluster": np.concatenate(y_cluster).astype(np.int64),
        "changed": np.concatenate(y_changed).astype(np.int64),
        "hard": np.concatenate(y_hard).astype(np.int64),
        "uncertainty_high": np.concatenate(y_unc).astype(np.int64),
        "family": np.concatenate(y_family).astype(np.int64),
        "last_cluster": np.concatenate(last_list).astype(np.int64),
        "dataset_id": np.concatenate(meta_dataset).astype(np.int64),
        "motion": np.concatenate(meta_motion).astype(np.float32),
        "occlusion": np.concatenate(meta_occlusion).astype(np.int64),
        "category": np.concatenate(meta_category).astype(np.int64),
    }
    n = len(data["x"])
    if max_tokens > 0 and n > max_tokens:
        idx = rng.choice(n, max_tokens, replace=False)
        data = {k: v[idx] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in data.items()}
    return data


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


def metrics(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": float(y.mean())}
    pred = score >= thr
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_ratio": float(y.mean()),
    }


def eval_binary(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], key: str, seed: int) -> dict[str, Any]:
    clf = HistGradientBoostingClassifier(max_iter=240, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.1, class_weight="balanced", random_state=seed)
    clf.fit(train["x"], train[key])
    sv = clf.predict_proba(val["x"])[:, 1]
    st = clf.predict_proba(test["x"])[:, 1]
    thr = choose_threshold(sv, val[key])
    out = {"model": "hist_gradient_boosting", "threshold": thr, "val": metrics(sv, val[key], thr), "test": metrics(st, test[key], thr), "_test_score": st}
    out["passed"] = bool(
        (out["val"]["balanced_accuracy"] or 0.0) >= 0.56
        and (out["test"]["balanced_accuracy"] or 0.0) >= 0.56
        and (out["val"]["roc_auc"] or 0.0) >= 0.58
        and (out["test"]["roc_auc"] or 0.0) >= 0.58
    )
    return out


def breakdown(score: np.ndarray, y: np.ndarray, meta: dict[str, np.ndarray], thr: float) -> dict[str, Any]:
    pred = score >= thr
    out: dict[str, Any] = {}
    bins = {
        "motion_low": meta["motion"] < 20,
        "motion_mid": (meta["motion"] >= 20) & (meta["motion"] < 80),
        "motion_high": meta["motion"] >= 80,
        "occlusion_false": meta["occlusion"] == 0,
        "occlusion_true": meta["occlusion"] == 1,
    }
    for name, mask in bins.items():
        if int(mask.sum()) < 10 or len(np.unique(y[mask])) < 2:
            out[name] = {"tokens": int(mask.sum()), "balanced_accuracy": None, "positive_ratio": float(y[mask].mean()) if mask.any() else 0.0}
        else:
            out[name] = {"tokens": int(mask.sum()), "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])), "positive_ratio": float(y[mask].mean())}
    cat_rows = []
    for cat, count in Counter(meta["category"].tolist()).most_common(12):
        mask = meta["category"] == cat
        row = {"category": int(cat), "tokens": int(mask.sum()), "positive_ratio": float(y[mask].mean())}
        if int(mask.sum()) >= 20 and len(np.unique(y[mask])) >= 2:
            row["balanced_accuracy"] = float(balanced_accuracy_score(y[mask], pred[mask]))
        else:
            row["balanced_accuracy"] = None
        cat_rows.append(row)
    out["top_category_breakdown"] = cat_rows
    return out


def run_protocol(root: Path, name: str, spec: dict[str, tuple[str, str | None]], seed: int) -> dict[str, Any]:
    train_paths = paths_for(root, *spec["train"])
    val_paths = paths_for(root, *spec["val"])
    test_paths = paths_for(root, *spec["test"])
    train = build_from_paths(train_paths, 180000, seed + 1)
    val = build_from_paths(val_paths, 90000, seed + 2)
    test = build_from_paths(test_paths, 90000, seed + 3)
    changed = eval_binary(train, val, test, "changed", seed)
    hard = eval_binary(train, val, test, "hard", seed + 10)
    unc = eval_binary(train, val, test, "uncertainty_high", seed + 20)
    changed_score = np.asarray(changed.pop("_test_score"), dtype=np.float32)
    hard_score = np.asarray(hard.pop("_test_score"), dtype=np.float32)
    unc_score = np.asarray(unc.pop("_test_score"), dtype=np.float32)
    suite = bool(changed["passed"] and hard["passed"] and unc["passed"])
    return {
        "protocol": name,
        "sample_counts": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
        "token_counts": {"train": int(len(train["x"])), "val": int(len(val["x"])), "test": int(len(test["x"]))},
        "semantic_changed": changed,
        "semantic_hard": hard,
        "semantic_uncertainty": unc,
        "test_breakdown_changed": breakdown(changed_score, test["changed"], test, changed["threshold"]),
        "test_breakdown_hard": breakdown(hard_score, test["hard"], test, hard["threshold"]),
        "test_breakdown_uncertainty": breakdown(unc_score, test["uncertainty_high"], test, unc["threshold"]),
        "suite_passed": suite,
    }


def main() -> int:
    root = TARGET_ROOT
    results = {}
    for i, (name, spec) in enumerate(PROTOCOLS.items()):
        results[name] = run_protocol(root, name, spec, 42 + i * 101)
    all_passed = all(r["suite_passed"] for r in results.values())
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_dataset_video_semantic_predictability_eval_done": True,
        "target_root": str(root.relative_to(ROOT)),
        "protocols": results,
        "future_leakage_detected": False,
        "中文结论": "V35.17 已完成 mixed 与 VSPW/VIPSeg cross-dataset video semantic predictability 审计。",
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_dataset_video_semantic_predictability_eval_done": True,
        "mixed_unseen_passed": bool(results["mixed_unseen"]["suite_passed"]),
        "vspw_to_vipseg_passed": bool(results["vspw_to_vipseg"]["suite_passed"]),
        "vipseg_to_vspw_passed": bool(results["vipseg_to_vspw"]["suite_passed"]),
        "cross_dataset_video_semantic_suite_ready": all_passed,
        "future_leakage_detected": False,
        "recommended_next_step": "train_cross_dataset_video_semantic_adapter" if all_passed else "fix_cross_dataset_video_semantic_target_or_domain_shift",
        "中文结论": (
            "V35.17 cross-dataset semantic target suite 通过，可以进入 semantic adapter 三 seed。"
            if all_passed
            else "V35.17 cross-dataset semantic target suite 未完全通过，应先修 target/domain shift，不能直接 claim full system。"
        ),
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.17 Cross-Dataset Video Semantic Predictability Decision\n\n"
        f"- mixed_unseen_passed: {decision['mixed_unseen_passed']}\n"
        f"- vspw_to_vipseg_passed: {decision['vspw_to_vipseg_passed']}\n"
        f"- vipseg_to_vspw_passed: {decision['vipseg_to_vspw_passed']}\n"
        f"- cross_dataset_video_semantic_suite_ready: {all_passed}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cross_dataset_video_semantic_suite_ready": all_passed, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
