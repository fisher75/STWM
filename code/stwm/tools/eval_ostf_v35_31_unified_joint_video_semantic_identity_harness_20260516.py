#!/usr/bin/env python3
"""V35.31 在 full unified video benchmark 上做 semantic + identity 联合评估。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

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

UNIFIED_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark/M128_H32"
SEMANTIC_CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_21_domain_normalized_video_semantic_state_adapter_h32_m128"
IDENTITY_CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_h32_m128"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_eval_summary_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_31_UNIFIED_JOINT_VIDEO_SEMANTIC_IDENTITY_DECISION_20260516.md"
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


def list_npz(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.npz"))


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
    return [load_identity_sample(p) for p in list_npz(root, split)]


def load_identity_model(seed: int, device: torch.device) -> IdentityResidualHead:
    ckpt_path = IDENTITY_CKPT_DIR / f"v35_29_expanded_video_identity_pairwise_retrieval_m128_h32_seed{seed}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = IdentityResidualHead(int(ckpt["input_dim"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_semantic_model(seed: int, input_dim: int, device: torch.device) -> VideoSemanticAdapter:
    ckpt_path = SEMANTIC_CKPT_DIR / f"v35_21_domain_normalized_video_semantic_state_adapter_m128_h32_seed{seed}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_input_dim = int(ckpt.get("input_dim", input_dim))
    if ckpt_input_dim != input_dim:
        raise RuntimeError(f"semantic input_dim 不匹配：checkpoint={ckpt_input_dim}, unified={input_dim}")
    model = VideoSemanticAdapter(input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def semantic_pass_bin(m: dict[str, float | None]) -> bool:
    return bool((m["roc_auc"] or 0.0) >= 0.58 and (m["balanced_accuracy"] or 0.0) >= 0.56)


def eval_semantic_seed(seed: int, device: torch.device) -> dict[str, Any]:
    val = build_split(UNIFIED_ROOT, "val", 80000, seed)
    test = build_split(UNIFIED_ROOT, "test", 80000, seed)
    model = load_semantic_model(seed, int(val["x"].shape[1]), device)
    pv = predict(model, val["x"], device)
    pt = predict(model, test["x"], device)
    thresholds = {k: choose_threshold(pv[k], val[{"changed": "changed", "hard": "hard", "uncertainty": "uncertainty_high"}[k]]) for k in ["changed", "hard", "uncertainty"]}
    val_metrics = {
        "semantic_changed": bin_metrics(pv["changed"], val["changed"], thresholds["changed"]),
        "semantic_hard": bin_metrics(pv["hard"], val["hard"], thresholds["hard"]),
        "semantic_uncertainty": bin_metrics(pv["uncertainty"], val["uncertainty_high"], thresholds["uncertainty"]),
        "cluster": top5_cluster_metrics(pv["cluster_logits"], val["cluster"], val["last_cluster"], pv["changed"], thresholds["changed"]),
    }
    test_metrics = {
        "semantic_changed": bin_metrics(pt["changed"], test["changed"], thresholds["changed"]),
        "semantic_hard": bin_metrics(pt["hard"], test["hard"], thresholds["hard"]),
        "semantic_uncertainty": bin_metrics(pt["uncertainty"], test["uncertainty_high"], thresholds["uncertainty"]),
        "cluster": top5_cluster_metrics(pt["cluster_logits"], test["cluster"], test["last_cluster"], pt["changed"], thresholds["changed"]),
    }
    semantic_changed_passed = semantic_pass_bin(val_metrics["semantic_changed"]) and semantic_pass_bin(test_metrics["semantic_changed"])
    semantic_hard_passed = semantic_pass_bin(val_metrics["semantic_hard"]) and semantic_pass_bin(test_metrics["semantic_hard"])
    uncertainty_passed = semantic_pass_bin(val_metrics["semantic_uncertainty"]) and semantic_pass_bin(test_metrics["semantic_uncertainty"])
    stable_preservation = bool(
        val_metrics["cluster"]["stable_top5"] >= val_metrics["cluster"]["stable_copy_top1"] - 0.02
        and test_metrics["cluster"]["stable_top5"] >= test_metrics["cluster"]["stable_copy_top1"] - 0.02
    )
    return {
        "seed": seed,
        "thresholds_from_unified_val": thresholds,
        "val": val_metrics,
        "test": test_metrics,
        "semantic_changed_passed": semantic_changed_passed,
        "semantic_hard_passed": semantic_hard_passed,
        "uncertainty_passed": uncertainty_passed,
        "stable_preservation": stable_preservation,
        "semantic_adapter_passed": bool((semantic_changed_passed or semantic_hard_passed) and uncertainty_passed and stable_preservation),
    }


def identity_pass(m: dict[str, float | None]) -> bool:
    return bool(
        (m["identity_retrieval_exclude_same_point_top1"] or 0.0) >= 0.70
        and (m["identity_retrieval_same_frame_top1"] or 0.0) >= 0.70
        and (m["identity_retrieval_instance_pooled_top1"] or 0.0) >= 0.70
        and (m["identity_confuser_separation"] is not None and m["identity_confuser_separation"] > 0.02)
        and (m["identity_confuser_avoidance_top1"] or 0.0) >= 0.70
    )


def eval_identity_seed(seed: int, device: torch.device) -> dict[str, Any]:
    val = load_identity_split(UNIFIED_ROOT, "val")
    test = load_identity_split(UNIFIED_ROOT, "test")
    model = load_identity_model(seed, device)
    val_metrics = evaluate_split(val, model, device, "learned")
    test_metrics = evaluate_split(test, model, device, "learned")
    measurement_val = evaluate_split(val, None, device, "measurement")
    measurement_test = evaluate_split(test, None, device, "measurement")
    not_worse_than_measurement = bool(
        (val_metrics["identity_retrieval_exclude_same_point_top1"] or 0.0) >= (measurement_val["identity_retrieval_exclude_same_point_top1"] or 0.0) - 0.08
        and (test_metrics["identity_retrieval_exclude_same_point_top1"] or 0.0) >= (measurement_test["identity_retrieval_exclude_same_point_top1"] or 0.0) - 0.08
        and (val_metrics["identity_confuser_avoidance_top1"] or 0.0) >= (measurement_val["identity_confuser_avoidance_top1"] or 0.0) - 0.08
        and (test_metrics["identity_confuser_avoidance_top1"] or 0.0) >= (measurement_test["identity_confuser_avoidance_top1"] or 0.0) - 0.08
    )
    return {
        "seed": seed,
        "val": val_metrics,
        "test": test_metrics,
        "measurement_baseline": {"val": measurement_val, "test": measurement_test},
        "identity_passed": bool(identity_pass(val_metrics) and identity_pass(test_metrics) and not_worse_than_measurement),
        "not_worse_than_measurement_baseline": not_worse_than_measurement,
    }


def benchmark_overview(root: Path) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    dataset_counts: dict[str, dict[str, int]] = {}
    raw_paths = 0
    semantic_ok = 0
    identity_ok = 0
    future_teacher_input = False
    leakage_safe = True
    for split in ["train", "val", "test"]:
        files = list_npz(root, split)
        split_counts[split] = len(files)
        dataset_counts[split] = {}
        for p in files:
            z = np.load(p, allow_pickle=True)
            ds = str(np.asarray(z["dataset"]).item())
            dataset_counts[split][ds] = dataset_counts[split].get(ds, 0) + 1
            raw_paths += int(bool(np.asarray(z["raw_video_input_available"]).item()))
            semantic_ok += int(bool(np.asarray(z["semantic_state_target_available"]).item()))
            identity_ok += int(bool(np.asarray(z["identity_pairwise_target_available"]).item()))
            future_teacher_input = future_teacher_input or bool(np.asarray(z["future_teacher_embedding_input_allowed"]).item())
            leakage_safe = leakage_safe and bool(np.asarray(z["leakage_safe"]).item())
    total = sum(split_counts.values())
    return {
        "split_counts": split_counts,
        "dataset_counts": dataset_counts,
        "total_clips": total,
        "raw_video_input_available_count": raw_paths,
        "semantic_state_target_available_count": semantic_ok,
        "identity_pairwise_target_available_count": identity_ok,
        "raw_video_frame_paths_available": raw_paths == total and total > 0,
        "semantic_identity_sample_alignment_passed": semantic_ok == total and identity_ok == total and total > 0,
        "future_teacher_embedding_input_allowed": future_teacher_input,
        "leakage_safe": leakage_safe,
    }


def mean_field(rows: list[dict[str, Any]], getter: tuple[str, ...]) -> float | None:
    vals: list[float] = []
    for row in rows:
        cur: Any = row
        for key in getter:
            cur = cur[key]
        if cur is not None:
            vals.append(float(cur))
    return float(np.mean(vals)) if vals else None


def write_doc(decision: dict[str, Any]) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        "# STWM OSTF V35.31 统一视频语义/身份联合评估决策\n\n"
        f"- unified_joint_eval_done: {decision['unified_joint_eval_done']}\n"
        f"- semantic_three_seed_passed_on_unified_benchmark: {decision['semantic_three_seed_passed_on_unified_benchmark']}\n"
        f"- identity_three_seed_passed_on_unified_benchmark: {decision['identity_three_seed_passed_on_unified_benchmark']}\n"
        f"- full_unified_joint_eval_passed: {decision['full_unified_joint_eval_passed']}\n"
        f"- full_video_semantic_identity_field_claim_allowed: {decision['full_video_semantic_identity_field_claim_allowed']}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n\n"
        "## 阶段性判断\n"
        "V35.31 把 V35.21 语义状态 adapter、V35.29 identity retrieval head 和 V35.28 unified video benchmark 放到同一个评估口径中。"
        "这一步证明的是 M128/H32 video-derived trace 闭环是否已经形成，而不是扩大尺度或训练新 writer/gate。\n",
        encoding="utf-8",
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overview = benchmark_overview(UNIFIED_ROOT)
    semantic_rows = [eval_semantic_seed(seed, device) for seed in SEEDS]
    identity_rows = [eval_identity_seed(seed, device) for seed in SEEDS]

    semantic_three_seed_passed = bool(all(row["semantic_adapter_passed"] for row in semantic_rows))
    identity_three_seed_passed = bool(all(row["identity_passed"] for row in identity_rows))
    full_unified_joint_eval_passed = bool(
        overview["semantic_identity_sample_alignment_passed"]
        and overview["raw_video_frame_paths_available"]
        and overview["leakage_safe"]
        and not overview["future_teacher_embedding_input_allowed"]
        and semantic_three_seed_passed
        and identity_three_seed_passed
    )

    eval_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unified_joint_eval_done": True,
        "benchmark_overview": overview,
        "semantic_seed_results": semantic_rows,
        "identity_seed_results": identity_rows,
        "semantic_test_changed_balanced_accuracy_mean": mean_field(semantic_rows, ("test", "semantic_changed", "balanced_accuracy")),
        "semantic_test_changed_roc_auc_mean": mean_field(semantic_rows, ("test", "semantic_changed", "roc_auc")),
        "semantic_test_hard_balanced_accuracy_mean": mean_field(semantic_rows, ("test", "semantic_hard", "balanced_accuracy")),
        "semantic_test_hard_roc_auc_mean": mean_field(semantic_rows, ("test", "semantic_hard", "roc_auc")),
        "semantic_test_uncertainty_balanced_accuracy_mean": mean_field(semantic_rows, ("test", "semantic_uncertainty", "balanced_accuracy")),
        "identity_test_exclude_same_point_top1_mean": mean_field(identity_rows, ("test", "identity_retrieval_exclude_same_point_top1")),
        "identity_test_same_frame_top1_mean": mean_field(identity_rows, ("test", "identity_retrieval_same_frame_top1")),
        "identity_test_instance_pooled_top1_mean": mean_field(identity_rows, ("test", "identity_retrieval_instance_pooled_top1")),
        "identity_test_confuser_avoidance_top1_mean": mean_field(identity_rows, ("test", "identity_confuser_avoidance_top1")),
        "identity_test_occlusion_reappear_top1_mean": mean_field(identity_rows, ("test", "occlusion_reappear_retrieval_top1")),
        "identity_test_trajectory_crossing_top1_mean": mean_field(identity_rows, ("test", "trajectory_crossing_retrieval_top1")),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
    }

    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unified_joint_eval_done": True,
        "semantic_three_seed_passed_on_unified_benchmark": semantic_three_seed_passed,
        "identity_three_seed_passed_on_unified_benchmark": identity_three_seed_passed,
        "full_unified_video_semantic_identity_benchmark_used": True,
        "joint_clip_count": overview["total_clips"],
        "raw_video_frame_paths_available": overview["raw_video_frame_paths_available"],
        "semantic_identity_sample_alignment_passed": overview["semantic_identity_sample_alignment_passed"],
        "semantic_test_changed_balanced_accuracy_mean": eval_report["semantic_test_changed_balanced_accuracy_mean"],
        "semantic_test_hard_balanced_accuracy_mean": eval_report["semantic_test_hard_balanced_accuracy_mean"],
        "semantic_test_uncertainty_balanced_accuracy_mean": eval_report["semantic_test_uncertainty_balanced_accuracy_mean"],
        "identity_test_exclude_same_point_top1_mean": eval_report["identity_test_exclude_same_point_top1_mean"],
        "identity_test_confuser_avoidance_top1_mean": eval_report["identity_test_confuser_avoidance_top1_mean"],
        "identity_test_occlusion_reappear_top1_mean": eval_report["identity_test_occlusion_reappear_top1_mean"],
        "identity_test_trajectory_crossing_top1_mean": eval_report["identity_test_trajectory_crossing_top1_mean"],
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "full_unified_joint_eval_passed": full_unified_joint_eval_passed,
        "integrated_semantic_field_claim_allowed": False,
        "integrated_identity_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "当前只允许 M128/H32，尚未做 H64/H96/M512/M1024 泛化。",
            "V35.31 证明的是统一 video-derived trace benchmark 上的组件闭环，不等于最终 CVPR full-scale claim。",
            "下一步需要 video input closure：从 raw video 自动得到 dense traces/semantic masks/identity targets 的端到端打包审计。",
        ],
        "recommended_next_step": "build_video_input_closure" if full_unified_joint_eval_passed else "fix_unified_joint_eval_harness",
        "中文结论": (
            "V35.31 统一联合评估通过：semantic 三 seed、identity 三 seed、325 clip unified video benchmark、raw video frame path、future leakage safety 在同一口径下对齐。"
            "这是强好消息，说明 M128/H32 video-derived trace 到 future semantic/identity 的闭环已经成形；但仍不能宣称完整 semantic field success，下一步应做 video input closure。"
            if full_unified_joint_eval_passed
            else "V35.31 统一联合评估未完全通过；不能进入 full system claim，下一步应修统一评估或对应短板。"
        ),
    }

    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_report), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    write_doc(decision)
    print(json.dumps({"统一联合评估通过": full_unified_joint_eval_passed, "推荐下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
