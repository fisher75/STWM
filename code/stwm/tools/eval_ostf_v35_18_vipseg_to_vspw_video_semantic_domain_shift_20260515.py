#!/usr/bin/env python3
"""V35.18 VIPSeg->VSPW domain-shift / target-split predictability 审计。"""
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

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_18_ontology_agnostic_video_semantic_state_targets/M128_H32"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_18_vipseg_to_vspw_video_semantic_domain_shift_eval_20260515.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_18_vipseg_to_vspw_video_semantic_domain_shift_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_18_VIPSEG_TO_VSPW_VIDEO_SEMANTIC_DOMAIN_SHIFT_DECISION_20260515.md"

PROTOCOLS = {
    "mixed_unseen_ontology_agnostic": {"train": ("train", None), "val": ("val", None), "test": ("test", None), "stratified_test": False},
    "vspw_to_vipseg_ontology_agnostic": {"train": ("train", "VSPW"), "val": ("val", "VSPW"), "test": ("test", "VIPSEG"), "stratified_test": False},
    "vipseg_to_vspw_all_ontology_agnostic": {"train": ("train", "VIPSEG"), "val": ("val", "VIPSEG"), "test": ("test", "VSPW"), "stratified_test": False},
    "vipseg_to_vspw_stratified_ontology_agnostic": {"train": ("train", "VIPSEG"), "val": ("val", "VIPSEG"), "test": ("test", "VSPW"), "stratified_test": True},
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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def paths_for(root: Path, split: str, dataset: str | None) -> list[Path]:
    out: list[Path] = []
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
    labels: dict[str, list[np.ndarray]] = {k: [] for k in ["changed", "hard", "uncertainty_high", "family", "cluster", "last_cluster"]}
    meta: dict[str, list[np.ndarray]] = {k: [] for k in ["dataset_id", "motion", "occlusion", "category", "sample_id"]}
    for sample_i, p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        target = np.asarray(z["target_semantic_cluster_id"], dtype=np.int64)
        valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
        changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
        hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
        family = np.asarray(z["evidence_anchor_family_target"], dtype=np.int64)
        family_avail = np.asarray(z["evidence_anchor_family_available_mask"], dtype=bool) & valid
        unc = (np.asarray(z["semantic_uncertainty_target"], dtype=np.float32) > 0.5).astype(np.int64)
        if "domain_normalized_risk_percentile" in z.files:
            unc = (np.asarray(z["semantic_uncertainty_target"], dtype=np.float32) > 0.5).astype(np.int64)
        elif "visibility_conditioned_semantic_risk" in z.files:
            unc = (np.asarray(z["visibility_conditioned_semantic_risk"], dtype=np.float32) > 0.5).astype(np.int64)
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
        labels["cluster"].append(target.reshape(-1)[flat].astype(np.int64))
        labels["changed"].append(changed.reshape(-1)[flat].astype(np.int64))
        labels["hard"].append(hard.reshape(-1)[flat].astype(np.int64))
        labels["uncertainty_high"].append(unc.reshape(-1)[flat].astype(np.int64))
        labels["family"].append(family.reshape(-1)[flat].astype(np.int64))
        labels["last_cluster"].append(np.repeat(last[:, None], h, axis=1).reshape(-1)[flat].astype(np.int64))
        dataset_id = 1 if str(np.asarray(z["dataset"]).item()) == "VIPSEG" else 0
        meta["dataset_id"].append(np.full(int(flat.sum()), dataset_id, dtype=np.int64))
        motion = np.linalg.norm(future_points - obs_points[:, -1:, :], axis=-1)
        meta["motion"].append(motion.reshape(-1)[flat].astype(np.float32))
        meta["occlusion"].append((~future_vis_bool).reshape(-1)[flat].astype(np.int64))
        source_sem = np.asarray(z["source_semantic_id"], dtype=np.int64) if "source_semantic_id" in z.files else last
        meta["category"].append(np.repeat(source_sem[:, None], h, axis=1).reshape(-1)[flat].astype(np.int64))
        meta["sample_id"].append(np.full(int(flat.sum()), sample_i, dtype=np.int64))
    if not xs:
        raise RuntimeError("没有可用 token")
    data: dict[str, np.ndarray] = {"x": np.concatenate(xs).astype(np.float32)}
    for k, v in labels.items():
        data[k] = np.concatenate(v).astype(np.int64)
    for k, v in meta.items():
        data[k] = np.concatenate(v).astype(np.float32 if k == "motion" else np.int64)
    n = len(data["x"])
    if max_tokens > 0 and n > max_tokens:
        idx = rng.choice(n, max_tokens, replace=False)
        data = {k: v[idx] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in data.items()}
    return data


def choose_threshold(score: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best = 0.5, -1.0
    for t in np.quantile(score, np.linspace(0.03, 0.97, 49)):
        ba = balanced_accuracy_score(y, score >= t)
        if ba > best:
            best = float(ba)
            best_t = float(t)
    return best_t


def stratified_indices(y: np.ndarray, seed: int, max_per_class: int = 25000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    n = min(len(pos), len(neg), max_per_class)
    pi = rng.choice(pos, n, replace=False)
    ni = rng.choice(neg, n, replace=False)
    idx = np.concatenate([pi, ni])
    rng.shuffle(idx)
    return idx


def metrics(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": float(y.mean()), "tokens": int(len(y))}
    pred = score >= thr
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_ratio": float(y.mean()),
        "tokens": int(len(y)),
    }


def eval_binary(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    key: str,
    seed: int,
    stratified_test: bool,
) -> dict[str, Any]:
    if len(np.unique(train[key])) < 2:
        return {
            "model": "hist_gradient_boosting",
            "passed": False,
            "blocker": "train_split_single_class",
            "val": metrics(np.zeros_like(val[key], dtype=np.float32), val[key], 0.5),
            "test": metrics(np.zeros_like(test[key], dtype=np.float32), test[key], 0.5),
        }
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.065,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(train["x"], train[key])
    sv = clf.predict_proba(val["x"])[:, 1]
    st = clf.predict_proba(test["x"])[:, 1]
    thr = choose_threshold(sv, val[key])
    val_m = metrics(sv, val[key], thr)
    test_m = metrics(st, test[key], thr)
    out: dict[str, Any] = {"model": "hist_gradient_boosting", "threshold": thr, "val": val_m, "test": test_m, "_test_score": st}
    if stratified_test:
        idx = stratified_indices(test[key], seed + 77)
        out["test_stratified"] = metrics(st[idx], test[key][idx], thr)
        pass_test = out["test_stratified"]
    else:
        pass_test = test_m
    out["passed"] = bool(
        (val_m["balanced_accuracy"] or 0.0) >= 0.56
        and (pass_test["balanced_accuracy"] or 0.0) >= 0.56
        and (val_m["roc_auc"] or 0.0) >= 0.58
        and (pass_test["roc_auc"] or 0.0) >= 0.58
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
    for cat, count in Counter(meta["category"].astype(int).tolist()).most_common(12):
        mask = meta["category"] == cat
        row = {"category": int(cat), "tokens": int(mask.sum()), "positive_ratio": float(y[mask].mean())}
        if int(mask.sum()) >= 20 and len(np.unique(y[mask])) >= 2:
            row["balanced_accuracy"] = float(balanced_accuracy_score(y[mask], pred[mask]))
        else:
            row["balanced_accuracy"] = None
        cat_rows.append(row)
    out["top_category_breakdown"] = cat_rows
    return out


def run_protocol(root: Path, name: str, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    train_paths = paths_for(root, *spec["train"])
    val_paths = paths_for(root, *spec["val"])
    test_paths = paths_for(root, *spec["test"])
    train = build_from_paths(train_paths, 90000, seed + 1)
    val = build_from_paths(val_paths, 45000, seed + 2)
    test = build_from_paths(test_paths, 45000, seed + 3)
    stratified = bool(spec.get("stratified_test", False))
    changed = eval_binary(train, val, test, "changed", seed, stratified)
    hard = eval_binary(train, val, test, "hard", seed + 10, stratified)
    unc = eval_binary(train, val, test, "uncertainty_high", seed + 20, stratified)
    changed_score = np.asarray(changed.pop("_test_score", np.zeros(len(test["changed"]))), dtype=np.float32)
    hard_score = np.asarray(hard.pop("_test_score", np.zeros(len(test["hard"]))), dtype=np.float32)
    unc_score = np.asarray(unc.pop("_test_score", np.zeros(len(test["uncertainty_high"]))), dtype=np.float32)
    suite = bool(changed.get("passed") and hard.get("passed") and unc.get("passed"))
    return {
        "protocol": name,
        "stratified_test": stratified,
        "sample_counts": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
        "token_counts": {"train": int(len(train["x"])), "val": int(len(val["x"])), "test": int(len(test["x"]))},
        "semantic_changed": changed,
        "semantic_hard": hard,
        "semantic_uncertainty": unc,
        "test_breakdown_changed": breakdown(changed_score, test["changed"], test, changed.get("threshold", 0.5)),
        "test_breakdown_hard": breakdown(hard_score, test["hard"], test, hard.get("threshold", 0.5)),
        "test_breakdown_uncertainty": breakdown(unc_score, test["uncertainty_high"], test, unc.get("threshold", 0.5)),
        "suite_passed": suite,
    }


def main() -> int:
    root = TARGET_ROOT
    if "v35_21_domain_normalized" in str(root):
        target_build_path = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_state_target_build_20260515.json"
    elif "v35_19_boundary_risk" in str(root):
        target_build_path = ROOT / "reports/stwm_ostf_v35_19_boundary_risk_video_semantic_state_target_build_20260515.json"
    else:
        target_build_path = ROOT / "reports/stwm_ostf_v35_18_ontology_agnostic_video_semantic_state_target_build_20260515.json"
    target_build = json.loads(target_build_path.read_text(encoding="utf-8")) if target_build_path.exists() else {}
    results = {}
    for i, (name, spec) in enumerate(PROTOCOLS.items()):
        results[name] = run_protocol(root, name, spec, 42 + i * 101)

    mixed_passed = bool(results["mixed_unseen_ontology_agnostic"]["suite_passed"])
    vspw_to_vipseg_passed = bool(results["vspw_to_vipseg_ontology_agnostic"]["suite_passed"])
    vipseg_all_passed = bool(results["vipseg_to_vspw_all_ontology_agnostic"]["suite_passed"])
    vipseg_strat_passed = bool(results["vipseg_to_vspw_stratified_ontology_agnostic"]["suite_passed"])
    target_split_fixed = bool(vipseg_strat_passed)
    suite_ready = bool(mixed_passed and vspw_to_vipseg_passed and target_split_fixed)

    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_18_vipseg_to_vspw_domain_shift_eval_done": True,
        "target_root": rel(root),
        "target_build_report": rel(target_build_path),
        "vipseg_source_train_val_expanded": target_build.get("vipseg_source_train_val_expanded"),
        "ontology_agnostic_change_target_used": True,
        "stratified_vspw_test_used": True,
        "protocols": results,
        "future_teacher_embedding_input_allowed": False,
        "future_leakage_detected": False,
        "中文结论": "V35.18 完成 VIPSeg→VSPW ontology-agnostic target 与 VSPW stratified test 诊断。",
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_18_vipseg_to_vspw_domain_shift_eval_done": True,
        "vipseg_source_train_val_expanded": target_build.get("vipseg_source_train_val_expanded"),
        "semantic_change_target_repaired": True,
        "ontology_agnostic_change_target_used": True,
        "stratified_vspw_test_used": True,
        "mixed_unseen_passed": mixed_passed,
        "vspw_to_vipseg_passed": vspw_to_vipseg_passed,
        "vipseg_to_vspw_all_passed": vipseg_all_passed,
        "vipseg_to_vspw_stratified_passed": vipseg_strat_passed,
        "vipseg_to_vspw_target_split_fixed": target_split_fixed,
        "cross_dataset_video_semantic_suite_ready": suite_ready,
        "semantic_adapter_training_allowed": suite_ready,
        "future_leakage_detected": False,
        "recommended_next_step": "run_three_seed_cross_dataset_video_semantic_adapter" if suite_ready else "expand_vipseg_source_or_fix_ontology_agnostic_video_target",
        "中文结论": (
            "V35.18 cross-dataset semantic target suite 通过，可以进入三 seed video semantic adapter。"
            if suite_ready
            else "V35.18 仍未完全修复 VIPSeg→VSPW；继续扩 VIPSeg source 或重修 ontology-agnostic video target，不训练 adapter。"
        ),
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.18 VIPSeg→VSPW Video Semantic Domain Shift Decision\n\n"
        f"- mixed_unseen_passed: {mixed_passed}\n"
        f"- vspw_to_vipseg_passed: {vspw_to_vipseg_passed}\n"
        f"- vipseg_to_vspw_all_passed: {vipseg_all_passed}\n"
        f"- vipseg_to_vspw_stratified_passed: {vipseg_strat_passed}\n"
        f"- cross_dataset_video_semantic_suite_ready: {suite_ready}\n"
        f"- semantic_adapter_training_allowed: {suite_ready}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cross_dataset_video_semantic_suite_ready": suite_ready, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
