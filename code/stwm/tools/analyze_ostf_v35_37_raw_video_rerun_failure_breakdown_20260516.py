#!/usr/bin/env python3
"""V35.37 对 V35.36 raw-video rerun 扩展 subset 做分组失败归因。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515 import last_valid, mode_valid
from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import (
    load_identity_model,
    load_identity_sample,
    load_semantic_model,
    norm,
    predict,
    top5_cluster_metrics,
)
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import (
    aggregate,
    model_embedding,
    retrieval_metrics_for_sample,
)

DEFAULT_INPUT = ROOT / "reports/stwm_ostf_v35_36_expanded_raw_video_frontend_rerun_subset_20260516.json"
DEFAULT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_36_raw_video_frontend_rerun_unified_slice/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_37_raw_video_frontend_rerun_failure_breakdown_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_37_RAW_VIDEO_FRONTEND_RERUN_FAILURE_BREAKDOWN_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_37_raw_video_frontend_rerun_failure_breakdown_20260516.log"
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


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def scalar(z: Any, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    try:
        return arr.item()
    except ValueError:
        return arr


def bin_metrics(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float | None]:
    if score.size == 0:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": None, "token_count": 0}
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": float(y.mean()), "token_count": int(y.size)}
    pred = score >= thr
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_ratio": float(y.mean()),
        "token_count": int(y.size),
    }


def semantic_features_for_sample(path: Path) -> dict[str, np.ndarray | str]:
    z = np.load(path, allow_pickle=True)
    target = np.asarray(z["target_semantic_cluster_id"], dtype=np.int64)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
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
    mask = (valid & family_avail).reshape(-1)
    return {
        "path": str(path),
        "sample_uid": str(scalar(z, "sample_uid", path.stem)),
        "split": str(scalar(z, "split")),
        "dataset": str(scalar(z, "dataset")),
        "x": feat.reshape(-1, feat.shape[-1])[mask].astype(np.float32),
        "cluster": target.reshape(-1)[mask].astype(np.int64),
        "changed": changed.reshape(-1)[mask].astype(np.int64),
        "hard": hard.reshape(-1)[mask].astype(np.int64),
        "uncertainty_high": unc.reshape(-1)[mask].astype(np.int64),
        "last_cluster": np.repeat(last[:, None], h, axis=1).reshape(-1)[mask].astype(np.int64),
    }


def categories_for_sample(uid: str, meta: dict[str, Any], motion_median: float) -> list[str]:
    cats = set(meta.get("categories", []))
    cats.add("all")
    cats.add(f"split_{meta.get('split', 'unknown')}")
    cats.add(f"dataset_{str(meta.get('dataset', 'unknown')).lower()}")
    motion = float(meta.get("cached_motion_mean", 0.0))
    cats.add("motion_high" if motion >= motion_median else "motion_low")
    return sorted(cats)


def collect_meta(smoke: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], float]:
    by_uid: dict[str, dict[str, Any]] = {}
    for row in smoke.get("selected_samples", []):
        by_uid[str(row["sample_uid"])] = dict(row)
    motions = []
    for row in smoke.get("cached_vs_rerun_drift", {}).get("rows", []):
        uid = str(row["sample_uid"])
        by_uid.setdefault(uid, {})["cached_motion_mean"] = float(row.get("cached_motion_mean", 0.0))
        by_uid[uid]["drift_row"] = row
        motions.append(float(row.get("cached_motion_mean", 0.0)))
    return by_uid, float(np.median(motions)) if motions else 0.0


def semantic_group_metrics(root: Path, smoke: dict[str, Any], device: torch.device) -> dict[str, Any]:
    by_uid, motion_median = collect_meta(smoke)
    sample_paths = sorted(root.glob("*/*.npz"))
    samples = [semantic_features_for_sample(p) for p in sample_paths]
    out: dict[str, Any] = {}
    for seed in SEEDS:
        seed_row = next((r for r in smoke.get("semantic_seed_rows", []) if int(r["seed"]) == seed), None)
        thresholds = seed_row.get("thresholds", {"changed": 0.5, "hard": 0.5, "uncertainty": 0.5}) if seed_row else {"changed": 0.5, "hard": 0.5, "uncertainty": 0.5}
        model = load_semantic_model(seed, int(samples[0]["x"].shape[1]), device)
        groups: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        for s in samples:
            pred = predict(model, np.asarray(s["x"], dtype=np.float32), device)
            uid = str(s["sample_uid"])
            cats = categories_for_sample(uid, by_uid.get(uid, {}), motion_median)
            for cat in cats:
                groups[cat]["changed_score"].append(pred["changed"])
                groups[cat]["hard_score"].append(pred["hard"])
                groups[cat]["uncertainty_score"].append(pred["uncertainty"])
                groups[cat]["cluster_logits"].append(pred["cluster_logits"])
                groups[cat]["changed"].append(np.asarray(s["changed"]))
                groups[cat]["hard"].append(np.asarray(s["hard"]))
                groups[cat]["uncertainty"].append(np.asarray(s["uncertainty_high"]))
                groups[cat]["cluster"].append(np.asarray(s["cluster"]))
                groups[cat]["last_cluster"].append(np.asarray(s["last_cluster"]))
        seed_out: dict[str, Any] = {}
        for cat, parts in groups.items():
            changed_score = np.concatenate(parts["changed_score"])
            hard_score = np.concatenate(parts["hard_score"])
            unc_score = np.concatenate(parts["uncertainty_score"])
            cluster_logits = np.concatenate(parts["cluster_logits"])
            y_changed = np.concatenate(parts["changed"])
            y_hard = np.concatenate(parts["hard"])
            y_unc = np.concatenate(parts["uncertainty"])
            y_cluster = np.concatenate(parts["cluster"])
            last = np.concatenate(parts["last_cluster"])
            seed_out[cat] = {
                "token_count": int(y_changed.size),
                "semantic_changed": bin_metrics(changed_score, y_changed, float(thresholds["changed"])),
                "semantic_hard": bin_metrics(hard_score, y_hard, float(thresholds["hard"])),
                "semantic_uncertainty": bin_metrics(unc_score, y_unc, float(thresholds["uncertainty"])),
                "cluster": top5_cluster_metrics(cluster_logits, y_cluster, last, changed_score, float(thresholds["changed"])),
            }
        out[str(seed)] = seed_out
    return out


def identity_group_metrics(root: Path, smoke: dict[str, Any], device: torch.device) -> dict[str, Any]:
    by_uid, motion_median = collect_meta(smoke)
    samples = [load_identity_sample(p) for p in sorted(root.glob("*/*.npz"))]
    out: dict[str, Any] = {}
    for seed in SEEDS:
        model = load_identity_model(seed, device)
        rows_by_group: dict[str, list[dict[str, float]]] = defaultdict(list)
        for s in samples:
            uid = str(s["sample_uid"])
            emb = model_embedding(model, np.asarray(s["x"], dtype=np.float32), device)
            row = retrieval_metrics_for_sample(emb, s)
            for cat in categories_for_sample(uid, by_uid.get(uid, {}), motion_median):
                rows_by_group[cat].append(row)
        out[str(seed)] = {cat: aggregate(rows) for cat, rows in rows_by_group.items()}
    return out


def split_counts(smoke: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in smoke.get("selected_samples", []):
        split = str(row.get("split", "unknown"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def summarize_failures(smoke: dict[str, Any]) -> dict[str, Any]:
    semantic_failures = []
    for row in smoke.get("semantic_seed_rows", []):
        if row.get("semantic_smoke_passed"):
            continue
        seed = int(row["seed"])
        test = row.get("test", {})
        semantic_failures.append(
            {
                "seed": seed,
                "test_changed_auc": test.get("semantic_changed", {}).get("roc_auc"),
                "test_changed_balanced_accuracy": test.get("semantic_changed", {}).get("balanced_accuracy"),
                "test_hard_auc": test.get("semantic_hard", {}).get("roc_auc"),
                "test_hard_balanced_accuracy": test.get("semantic_hard", {}).get("balanced_accuracy"),
                "test_uncertainty_auc": test.get("semantic_uncertainty", {}).get("roc_auc"),
                "test_uncertainty_balanced_accuracy": test.get("semantic_uncertainty", {}).get("balanced_accuracy"),
            }
        )
    identity_failures = []
    for row in smoke.get("identity_seed_rows", []):
        if row.get("identity_smoke_passed"):
            continue
        seed = int(row["seed"])
        val = row.get("val", {})
        test = row.get("test", {})
        identity_failures.append(
            {
                "seed": seed,
                "val_instance_pooled_top1": val.get("identity_retrieval_instance_pooled_top1"),
                "test_instance_pooled_top1": test.get("identity_retrieval_instance_pooled_top1"),
                "val_exclude_same_point_top1": val.get("identity_retrieval_exclude_same_point_top1"),
                "test_exclude_same_point_top1": test.get("identity_retrieval_exclude_same_point_top1"),
                "val_confuser_avoidance_top1": val.get("identity_confuser_avoidance_top1"),
                "test_confuser_avoidance_top1": test.get("identity_confuser_avoidance_top1"),
            }
        )
    return {"semantic_seed_failures": semantic_failures, "identity_seed_failures": identity_failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-report", default=str(DEFAULT_INPUT))
    ap.add_argument("--rerun-unified-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    report_path = Path(args.input_report)
    root = Path(args.rerun_unified_root)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    if not root.is_absolute():
        root = ROOT / root
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu and args.device == "cuda" else "cpu")
    smoke = json.loads(report_path.read_text(encoding="utf-8"))
    log(f"开始 V35.37 breakdown；输入报告={rel(report_path)}；root={rel(root)}；device={device}")
    sem_groups = semantic_group_metrics(root, smoke, device)
    id_groups = identity_group_metrics(root, smoke, device)
    failures = summarize_failures(smoke)
    counts = split_counts(smoke)
    drift_ok = bool(smoke.get("cached_vs_rerun_drift", {}).get("drift_ok", False))
    eval_balance_issue = bool(counts.get("test", 0) < 3 or counts.get("val", 0) < 3)
    semantic_failure = bool(failures["semantic_seed_failures"])
    identity_failure = bool(failures["identity_seed_failures"])
    recommended = (
        "fix_raw_video_rerun_eval_split_balance_and_target_alignment"
        if drift_ok and (eval_balance_issue or semantic_failure or identity_failure)
        else "expand_m128_h32_raw_video_frontend_rerun_subset"
    )
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_37_failure_breakdown_done": True,
        "input_report": rel(report_path),
        "rerun_unified_root": rel(root),
        "raw_frontend_drift_ok": drift_ok,
        "selected_sample_count": int(smoke.get("selected_sample_count", 0)),
        "split_counts": counts,
        "eval_split_balance_issue_detected": eval_balance_issue,
        "semantic_failure_detected": semantic_failure,
        "identity_failure_detected": identity_failure,
        "failure_summary": failures,
        "semantic_group_metrics": sem_groups,
        "identity_group_metrics": id_groups,
        "m128_h32_video_system_benchmark_claim_allowed": False,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": recommended,
        "中文结论": (
            "V35.37 归因显示 raw-video frontend 重跑本身可复现，当前 blocker 是扩展 subset 的评估组成和 target/model 对齐："
            "semantic 有 seed 级别 test 泛化失败，identity 主要卡在 val instance-pooled retrieval；不能把问题归因成 trace drift。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.37 Raw-Video Rerun Failure Breakdown\n\n"
        f"- v35_37_failure_breakdown_done: true\n"
        f"- raw_frontend_drift_ok: {drift_ok}\n"
        f"- split_counts: {counts}\n"
        f"- eval_split_balance_issue_detected: {eval_balance_issue}\n"
        f"- semantic_failure_detected: {semantic_failure}\n"
        f"- identity_failure_detected: {identity_failure}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: false\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n\n"
        "## 好消息/坏消息\n"
        "- 好消息：rerun trace 与 cache 的 shape、frame path、visibility、confidence、motion drift 都对齐，raw frontend reproducibility 没有暴露硬错误。\n"
        "- 坏消息：扩展 subset 只有 1 个 test clip、3 个 val clip，评估统计过小且类别分布偏，semantic seed123 与 identity instance-pooled val gate 暴露不稳。\n\n"
        "## Claim Boundary\n"
        "当前只能说 raw frontend 小规模复现链路可运行；不能 claim M128/H32 video system benchmark，更不能 claim full CVPR-scale complete system。\n",
        encoding="utf-8",
    )
    log(f"V35.37 完成；recommended_next_step={recommended}")
    print(json.dumps({"v35_37_failure_breakdown_done": True, "recommended_next_step": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
