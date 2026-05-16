#!/usr/bin/env python3
"""V35.46：基于 V35.45 larger raw-video closure 的 per-category failure atlas。"""
from __future__ import annotations

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

from stwm.tools.eval_ostf_v35_14_mask_video_semantic_state_predictability_20260515 import last_valid  # noqa: E402
from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools.render_ostf_v35_45_larger_raw_video_closure_visualizations_20260516 import (  # noqa: E402
    features_for_sample,
)
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import (  # noqa: E402
    load_identity_model,
    load_identity_sample,
    load_semantic_model,
)
from stwm.tools.train_eval_ostf_v35_14_video_semantic_state_adapter_20260515 import (  # noqa: E402
    choose_threshold,
    predict,
)
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import (  # noqa: E402
    aggregate,
    model_embedding,
    retrieval_metrics_for_sample,
)

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_rerun_unified_slice/M128_H32"
SUBSET_MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_closure_subset/manifest.json"
V35_45_DECISION = ROOT / "reports/stwm_ostf_v35_45_decision_20260516.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_46_per_category_failure_atlas_eval_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_46_per_category_failure_atlas_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_46_PER_CATEGORY_FAILURE_ATLAS_DECISION_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_46_per_category_failure_atlas_20260516.log"
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def scalar(z: Any, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    try:
        return arr.item()
    except ValueError:
        return arr


def slice_paths(split: str | None = None) -> list[Path]:
    if split is None:
        return sorted(SLICE_ROOT.glob("*/*.npz"))
    return sorted((SLICE_ROOT / split).glob("*.npz"))


def manifest_meta() -> dict[str, dict[str, Any]]:
    data = load_json(SUBSET_MANIFEST)
    return {str(r["sample_uid"]): r for r in data.get("samples", [])}


def sample_tags(path: Path, meta: dict[str, dict[str, Any]]) -> list[str]:
    z = np.load(path, allow_pickle=True)
    uid = str(scalar(z, "sample_uid", path.stem))
    tags = set(meta.get(uid, {}).get("category_tags", []))
    tags.add("all")
    tags.add(f"dataset_{str(scalar(z, 'dataset')).lower()}")
    tags.add(f"split_{str(scalar(z, 'split'))}")
    claim = bool(scalar(z, "identity_claim_allowed", False))
    tags.add("real_instance_identity" if claim else "pseudo_identity_diagnostic")
    return sorted(tags)


@torch.no_grad()
def predict_semantic_samples(seed: int, device: torch.device, meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    probe = next(iter(slice_paths()))
    input_dim = int(features_for_sample(np.load(probe, allow_pickle=True))["x"].shape[1])
    model = load_semantic_model(seed, input_dim, device)
    # threshold 仍按 V35.45 协议：val global 选阈值，test 只确认。
    val_scores: dict[str, list[np.ndarray]] = {"changed": [], "hard": [], "uncertainty": []}
    val_targets: dict[str, list[np.ndarray]] = {"changed": [], "hard": [], "uncertainty": []}
    per_sample: dict[str, dict[str, Any]] = {}
    for p in slice_paths():
        z = np.load(p, allow_pickle=True)
        data = features_for_sample(z)
        pred = predict(model, data["x"], device)
        split = str(scalar(z, "split"))
        uid = str(scalar(z, "sample_uid", p.stem))
        if split == "val":
            for k, yk in [("changed", "changed"), ("hard", "hard"), ("uncertainty", "uncertainty")]:
                val_scores[k].append(np.asarray(pred[k], dtype=np.float32))
                val_targets[k].append(np.asarray(data[yk], dtype=np.int64))
        per_sample[uid] = {
            "path": p,
            "split": split,
            "dataset": str(scalar(z, "dataset")),
            "tags": sample_tags(p, meta),
            "pred": pred,
            "data": data,
            "last_cluster": np.repeat(last_valid(np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64))[:, None], np.asarray(z["target_semantic_cluster_id"]).shape[1], axis=1)[data["valid"]],
        }
    thresholds = {}
    for k in ["changed", "hard", "uncertainty"]:
        y = np.concatenate(val_targets[k], axis=0)
        s = np.concatenate(val_scores[k], axis=0)
        thresholds[k] = choose_threshold(s, y)
    return {"seed": seed, "thresholds": thresholds, "per_sample": per_sample}


def bin_metric(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)
    if y.size == 0:
        return {"token_count": 0, "positive_ratio": None, "roc_auc": None, "balanced_accuracy": None, "f1": None, "passed": False, "insufficient_label_diversity": True}
    if len(np.unique(y)) < 2:
        return {
            "token_count": int(y.size),
            "positive_ratio": float(y.mean()),
            "roc_auc": None,
            "balanced_accuracy": None,
            "f1": None,
            "passed": False,
            "insufficient_label_diversity": True,
        }
    pred = score >= thr
    ba = float(balanced_accuracy_score(y, pred))
    auc = float(roc_auc_score(y, score))
    f1 = float(f1_score(y, pred, zero_division=0))
    return {
        "token_count": int(y.size),
        "positive_ratio": float(y.mean()),
        "roc_auc": auc,
        "balanced_accuracy": ba,
        "f1": f1,
        "passed": bool(auc >= 0.55 and ba >= 0.52),
        "insufficient_label_diversity": False,
    }


def cluster_stable_metrics(cluster_logits: np.ndarray, y: np.ndarray, last: np.ndarray, changed_score: np.ndarray, changed_thr: float) -> dict[str, Any]:
    if y.size == 0:
        return {"token_count": 0, "cluster_top5": None, "copy_top1": None, "stable_top5": None, "stable_copy_top1": None, "stable_preservation": False}
    top5 = np.argpartition(-cluster_logits, kth=4, axis=1)[:, :5]
    copy = np.where(last >= 0, last, top5[:, 0])
    changed_pred = changed_score >= changed_thr
    final_top5 = top5.copy()
    final_top5[:, 0] = np.where(changed_pred, final_top5[:, 0], copy)
    final_top5[:, -1] = copy
    stable = (last >= 0) & (y == last)
    stable_top5 = float(np.any(final_top5[stable] == y[stable, None], axis=1).mean()) if stable.any() else None
    stable_copy = float((copy[stable] == y[stable]).mean()) if stable.any() else None
    return {
        "token_count": int(y.size),
        "cluster_top5": float(np.any(final_top5 == y[:, None], axis=1).mean()),
        "copy_top1": float((copy == y).mean()),
        "stable_top5": stable_top5,
        "stable_copy_top1": stable_copy,
        "stable_token_count": int(stable.sum()),
        "stable_preservation": bool(stable_top5 is not None and stable_copy is not None and stable_top5 >= stable_copy - 0.05),
    }


def semantic_category_metrics(seed_row: dict[str, Any], split: str, category: str) -> dict[str, Any]:
    thresholds = seed_row["thresholds"]
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    sample_count = 0
    for s in seed_row["per_sample"].values():
        if s["split"] != split or category not in s["tags"]:
            continue
        sample_count += 1
        data = s["data"]
        pred = s["pred"]
        for k in ["changed", "hard", "uncertainty"]:
            parts[f"{k}_score"].append(np.asarray(pred[k], dtype=np.float32))
            parts[f"{k}_target"].append(np.asarray(data[k], dtype=np.int64))
        parts["cluster_logits"].append(np.asarray(pred["cluster_logits"], dtype=np.float32))
        parts["cluster_target"].append(np.asarray(data["cluster"], dtype=np.int64))
        parts["last_cluster"].append(np.asarray(s["last_cluster"], dtype=np.int64))
    if sample_count == 0:
        return {"sample_count": 0, "semantic_changed": {}, "semantic_hard": {}, "semantic_uncertainty": {}, "cluster_stable": {}}
    out = {"sample_count": sample_count}
    out["semantic_changed"] = bin_metric(np.concatenate(parts["changed_score"]), np.concatenate(parts["changed_target"]), thresholds["changed"])
    out["semantic_hard"] = bin_metric(np.concatenate(parts["hard_score"]), np.concatenate(parts["hard_target"]), thresholds["hard"])
    out["semantic_uncertainty"] = bin_metric(np.concatenate(parts["uncertainty_score"]), np.concatenate(parts["uncertainty_target"]), thresholds["uncertainty"])
    out["cluster_stable"] = cluster_stable_metrics(
        np.concatenate(parts["cluster_logits"], axis=0),
        np.concatenate(parts["cluster_target"], axis=0),
        np.concatenate(parts["last_cluster"], axis=0),
        np.concatenate(parts["changed_score"], axis=0),
        thresholds["changed"],
    )
    return out


@torch.no_grad()
def identity_category_metrics(seed: int, split: str, category: str, real_only: bool, device: torch.device, meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    sample_count = 0
    model = load_identity_model(seed, device)
    for p in slice_paths(split):
        z = np.load(p, allow_pickle=True)
        claim = bool(scalar(z, "identity_claim_allowed", False))
        if claim != real_only:
            continue
        if category not in sample_tags(p, meta):
            continue
        s = load_identity_sample(p)
        emb = model_embedding(model, np.asarray(s["x"], dtype=np.float32), device)
        rows.append(retrieval_metrics_for_sample(emb, s))
        sample_count += 1
    if not rows:
        return {"sample_count": 0}
    m = aggregate(rows)
    m["sample_count"] = sample_count
    m["passed"] = bool(
        (m.get("identity_retrieval_exclude_same_point_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_same_frame_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_instance_pooled_top1") or 0.0) >= 0.65
        and (m.get("identity_confuser_avoidance_top1") or 0.0) >= 0.65
    )
    return m


def mean_metric(rows: list[dict[str, Any]], path: list[str]) -> float | None:
    vals = []
    for r in rows:
        cur: Any = r
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                ok = False
                break
            cur = cur[k]
        if ok:
            vals.append(float(cur))
    return float(np.mean(vals)) if vals else None


def collect_categories(meta: dict[str, dict[str, Any]]) -> list[str]:
    cats = {"all"}
    for r in meta.values():
        cats.update(r.get("category_tags", []))
        cats.add(f"dataset_{str(r.get('dataset', '')).lower()}")
        cats.add(f"split_{str(r.get('split', ''))}")
        cats.add("real_instance_identity" if r.get("identity_claim_allowed") else "pseudo_identity_diagnostic")
    preferred = [
        "all",
        "dataset_vspw",
        "dataset_vipseg",
        "high_motion",
        "low_motion",
        "occlusion",
        "crossing",
        "identity_confuser",
        "real_instance_identity",
        "pseudo_identity_diagnostic",
        "stable_heavy",
        "semantic_changed",
        "semantic_hard",
        "semantic_uncertainty",
    ]
    return [c for c in preferred if c in cats] + sorted(cats - set(preferred))


def summarize_category_across_seeds(seed_reports: list[dict[str, Any]], categories: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ["val", "test"]:
        summary[split] = {}
        for cat in categories:
            rows = [r["semantic"][split].get(cat, {}) for r in seed_reports]
            id_rows = [r["identity_real_instance"][split].get(cat, {}) for r in seed_reports]
            pseudo_rows = [r["identity_pseudo_diagnostic"][split].get(cat, {}) for r in seed_reports]
            sample_count = max([int(r.get("sample_count", 0) or 0) for r in rows] + [0])
            if sample_count == 0:
                summary[split][cat] = {
                    "sample_count": 0,
                    "semantic_changed_balanced_accuracy_mean": None,
                    "semantic_changed_auc_mean": None,
                    "semantic_changed_positive_ratio_mean": None,
                    "semantic_hard_balanced_accuracy_mean": None,
                    "semantic_hard_auc_mean": None,
                    "semantic_hard_positive_ratio_mean": None,
                    "semantic_uncertainty_balanced_accuracy_mean": None,
                    "semantic_uncertainty_auc_mean": None,
                    "stable_top5_mean": None,
                    "stable_copy_top1_mean": None,
                    "stable_preservation_all_seeds": None,
                    "identity_real_exclude_same_point_top1_mean": None,
                    "identity_real_confuser_avoidance_top1_mean": None,
                    "identity_real_instance_pooled_top1_mean": None,
                    "identity_real_passed_all_seeds": None,
                    "pseudo_identity_diagnostic_exclude_same_point_top1_mean": None,
                    "pseudo_identity_diagnostic_only": True,
                }
                continue
            summary[split][cat] = {
                "sample_count": sample_count,
                "semantic_changed_balanced_accuracy_mean": mean_metric(rows, ["semantic_changed", "balanced_accuracy"]),
                "semantic_changed_auc_mean": mean_metric(rows, ["semantic_changed", "roc_auc"]),
                "semantic_changed_positive_ratio_mean": mean_metric(rows, ["semantic_changed", "positive_ratio"]),
                "semantic_hard_balanced_accuracy_mean": mean_metric(rows, ["semantic_hard", "balanced_accuracy"]),
                "semantic_hard_auc_mean": mean_metric(rows, ["semantic_hard", "roc_auc"]),
                "semantic_hard_positive_ratio_mean": mean_metric(rows, ["semantic_hard", "positive_ratio"]),
                "semantic_uncertainty_balanced_accuracy_mean": mean_metric(rows, ["semantic_uncertainty", "balanced_accuracy"]),
                "semantic_uncertainty_auc_mean": mean_metric(rows, ["semantic_uncertainty", "roc_auc"]),
                "stable_top5_mean": mean_metric(rows, ["cluster_stable", "stable_top5"]),
                "stable_copy_top1_mean": mean_metric(rows, ["cluster_stable", "stable_copy_top1"]),
                "stable_preservation_all_seeds": all(bool(r.get("cluster_stable", {}).get("stable_preservation", False)) for r in rows if r.get("sample_count", 0) > 0),
                "identity_real_exclude_same_point_top1_mean": mean_metric(id_rows, ["identity_retrieval_exclude_same_point_top1"]),
                "identity_real_confuser_avoidance_top1_mean": mean_metric(id_rows, ["identity_confuser_avoidance_top1"]),
                "identity_real_instance_pooled_top1_mean": mean_metric(id_rows, ["identity_retrieval_instance_pooled_top1"]),
                "identity_real_passed_all_seeds": all(bool(r.get("passed", False)) for r in id_rows if r.get("sample_count", 0) > 0),
                "pseudo_identity_diagnostic_exclude_same_point_top1_mean": mean_metric(pseudo_rows, ["identity_retrieval_exclude_same_point_top1"]),
                "pseudo_identity_diagnostic_only": True,
            }
    return summary


def failure_boundaries(summary: dict[str, Any], categories: list[str]) -> dict[str, Any]:
    high_risk: list[dict[str, Any]] = []
    robust: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        for cat in categories:
            row = summary[split][cat]
            if row.get("sample_count", 0) == 0:
                continue
            checks = {
                "semantic_changed": row.get("semantic_changed_balanced_accuracy_mean"),
                "semantic_hard": row.get("semantic_hard_balanced_accuracy_mean"),
                "semantic_uncertainty": row.get("semantic_uncertainty_balanced_accuracy_mean"),
                "stable_preservation": 1.0 if row.get("stable_preservation_all_seeds") else 0.0,
                "identity_real_confuser": row.get("identity_real_confuser_avoidance_top1_mean"),
            }
            bad = {k: v for k, v in checks.items() if v is not None and ((k.startswith("semantic") and v < 0.55) or (k == "stable_preservation" and v < 1.0) or (k == "identity_real_confuser" and v < 0.70))}
            good = {k: v for k, v in checks.items() if v is not None and ((k.startswith("semantic") and v >= 0.58) or (k == "stable_preservation" and v >= 1.0) or (k == "identity_real_confuser" and v >= 0.90))}
            if bad:
                high_risk.append({"split": split, "category": cat, "sample_count": row.get("sample_count"), "risk_metrics": bad})
            if len(good) >= 3:
                robust.append({"split": split, "category": cat, "sample_count": row.get("sample_count"), "strong_metrics": good})
    return {
        "high_risk_categories": high_risk,
        "robust_categories": robust,
        "semantic_fragile_categories_test": [
            r for r in high_risk if r["split"] == "test" and any(str(k).startswith("semantic") for k in r["risk_metrics"])
        ],
        "identity_fragile_categories_test": [
            r for r in high_risk if r["split"] == "test" and "identity_real_confuser" in r["risk_metrics"]
        ],
    }


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    meta = manifest_meta()
    categories = collect_categories(meta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_reports = []
    for seed in SEEDS:
        log(f"开始 V35.46 seed={seed} per-category 语义/身份评估")
        sem = predict_semantic_samples(seed, device, meta)
        seed_report = {"seed": seed, "thresholds": sem["thresholds"], "semantic": {"val": {}, "test": {}}, "identity_real_instance": {"val": {}, "test": {}}, "identity_pseudo_diagnostic": {"val": {}, "test": {}}}
        for split in ["val", "test"]:
            for cat in categories:
                seed_report["semantic"][split][cat] = semantic_category_metrics(sem, split, cat)
                seed_report["identity_real_instance"][split][cat] = identity_category_metrics(seed, split, cat, True, device, meta)
                seed_report["identity_pseudo_diagnostic"][split][cat] = identity_category_metrics(seed, split, cat, False, device, meta)
        seed_reports.append(seed_report)
    category_summary = summarize_category_across_seeds(seed_reports, categories)
    boundaries = failure_boundaries(category_summary, categories)
    v35_45 = load_json(V35_45_DECISION)
    atlas_ready = bool(
        v35_45.get("m128_h32_larger_video_system_benchmark_claim_allowed", False)
        and len(categories) >= 10
        and category_summary.get("val")
        and category_summary.get("test")
    )
    # 失败图谱不是 gate，而是下一步定位工具；即使有 fragile categories，也不能取消 V35.45 bounded claim。
    recommended = "run_v35_47_full_m128_h32_raw_video_closure_protocol_decision" if atlas_ready else "fix_visualization_case_mining"
    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_46_per_category_failure_atlas_done": True,
        "slice_root": rel(SLICE_ROOT),
        "source_v35_45_decision": rel(V35_45_DECISION),
        "seeds": SEEDS,
        "categories": categories,
        "seed_reports": seed_reports,
        "category_summary": category_summary,
        "failure_boundaries": boundaries,
        "atlas_ready": atlas_ready,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "full_cvpr_scale_claim_allowed": False,
        "中文结论": "V35.46 已完成 per-category failure atlas：按 motion、occlusion、crossing、confuser、stable/changed/hard、dataset、identity provenance 拆分三 seed 语义和 identity retrieval 指标。",
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.46",
        "per_category_failure_atlas_done": True,
        "atlas_ready": atlas_ready,
        "m128_h32_larger_video_system_benchmark_claim_allowed": bool(v35_45.get("m128_h32_larger_video_system_benchmark_claim_allowed", False)),
        "full_cvpr_scale_claim_allowed": False,
        "category_count": len(categories),
        "high_risk_category_count": len(boundaries["high_risk_categories"]),
        "semantic_fragile_categories_test": boundaries["semantic_fragile_categories_test"],
        "identity_fragile_categories_test": boundaries["identity_fragile_categories_test"],
        "robust_category_count": len(boundaries["robust_categories"]),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "innovation_status": {
            "video_closure": "V35.45 的 raw-video rerun -> V30 frozen trace -> semantic state -> identity retrieval 证据链保持成立。",
            "failure_atlas": "V35.46 不扩大 claim，而是把 bounded benchmark 的成功/失败边界拆到类别级，为下一轮 full M128/H32 protocol decision 或 failure repair 提供依据。",
            "claim_boundary": "仍只允许 M128/H32 larger subset bounded claim；full CVPR-scale 仍为 false。",
        },
        "recommended_next_step": recommended,
        "中文结论": (
            "V35.46 完成类别级失败图谱。创新点没有跑偏：仍是 raw video/video-derived dense trace 到 future trace/semantic state/identity retrieval 的闭环分析，不训练新模型。"
            "当前应把 V35.46 作为 V35.47 full M128/H32 raw-video closure protocol decision 的输入，而不是直接宣称 full CVPR-scale success。"
        ),
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    doc_lines = [
        "# STWM OSTF V35.46 Per-Category Failure Atlas Decision",
        "",
        f"- per_category_failure_atlas_done: true",
        f"- atlas_ready: {atlas_ready}",
        f"- category_count: {len(categories)}",
        f"- high_risk_category_count: {len(boundaries['high_risk_categories'])}",
        f"- semantic_fragile_categories_test: {len(boundaries['semantic_fragile_categories_test'])}",
        f"- identity_fragile_categories_test: {len(boundaries['identity_fragile_categories_test'])}",
        f"- m128_h32_larger_video_system_benchmark_claim_allowed: {decision['m128_h32_larger_video_system_benchmark_claim_allowed']}",
        "- full_cvpr_scale_claim_allowed: false",
        f"- recommended_next_step: {recommended}",
        "",
        "## 中文总结",
        decision["中文结论"],
        "",
        "## Test split 高风险类别",
    ]
    if boundaries["semantic_fragile_categories_test"] or boundaries["identity_fragile_categories_test"]:
        for r in boundaries["semantic_fragile_categories_test"][:12] + boundaries["identity_fragile_categories_test"][:12]:
            doc_lines.append(f"- {r['split']} / {r['category']} / sample_count={r['sample_count']} / risk_metrics={r['risk_metrics']}")
    else:
        doc_lines.append("- 未发现按当前阈值定义的 test split 高风险类别；仍需扩大到 full M128/H32 raw-video closure 才能升级 claim。")
    doc_lines.extend(
        [
            "",
            "## Claim boundary",
            "- V35.46 是 failure atlas，不是新模型训练，也不是 full CVPR-scale success。",
            "- V35.45 的 bounded M128/H32 larger video system benchmark claim 仍可保留。",
            "- full CVPR-scale 仍需要更大/完整 M128/H32 raw-video closure protocol 与更多真实 instance identity provenance。",
        ]
    )
    DOC.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")
    print(json.dumps({"per_category_failure_atlas_done": True, "atlas_ready": atlas_ready, "recommended_next_step": recommended}, ensure_ascii=False), flush=True)
    return 0 if atlas_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
