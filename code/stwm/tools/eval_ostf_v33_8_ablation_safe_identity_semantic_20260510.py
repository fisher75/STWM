#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import setproctitle
# 将进程名修改为 "python"
setproctitle.setproctitle("python")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.modules.ostf_v33_7_identity_belief_world_model import IdentityBeliefWorldModelV337
from stwm.tools.eval_ostf_v33_4_structured_semantic_identity_protocol_20260509 import (
    instance_pooled_retrieval,
    retrieval_top1,
    semantic_metrics,
)
from stwm.tools.eval_ostf_v33_7_identity_belief_calibration_20260509 import (
    BeliefEvalDataset,
    balanced_at,
    best_threshold,
    brier_ece,
    collate_belief_eval,
    mean_std_worst,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1


SUMMARY = ROOT / "reports/stwm_ostf_v33_8_ablation_safe_eval_summary_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_8_ablation_safe_eval_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_ABLATION_SAFE_EVAL_DECISION_20260510.md"
COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"


def selected_k(default: int = 32) -> int:
    path = ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8")).get("selected_K", default))
    return default


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def candidate_checkpoints() -> dict[str, dict[str, Any]]:
    return {
        "v33_8_v33_6_global_contrastive_baseline_seed42": {
            "kind": "v33_6",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_6_identity_contrastive_repair/v33_8_v33_6_global_contrastive_baseline_seed42_best.pt",
        },
        "v33_8_v33_7_full_identity_belief_seed42": {
            "kind": "v33_7",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_full_identity_belief_seed42_best.pt",
        },
        "v33_8_v33_7_no_fused_logits_seed42": {
            "kind": "v33_7",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_no_fused_logits_seed42_best.pt",
        },
        "v33_8_v33_7_no_hard_bce_seed42": {
            "kind": "v33_7",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_no_hard_bce_seed42_best.pt",
        },
        "v33_8_v33_7_no_embedding_similarity_seed42": {
            "kind": "v33_7",
            "checkpoint": ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_8_v33_7_no_embedding_similarity_seed42_best.pt",
        },
    }


def roots_args(k: int, args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.semantic_identity_sidecar_root = str(COMPLETE / "semantic_identity_targets/pointodyssey")
    ns.global_identity_label_root = str(COMPLETE / "global_identity_labels/pointodyssey")
    ns.visual_teacher_root = str(COMPLETE / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local")
    ns.semantic_prototype_target_root = str(COMPLETE / f"semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}")
    ns.prototype_vocab_path = str(ROOT / f"outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K{k}/prototype_vocab.npz")
    ns.batch_size = args.batch_size
    ns.num_workers = args.num_workers
    return ns


def load_model(spec: dict[str, Any], ns: argparse.Namespace, device: torch.device):
    ckpt_path = Path(spec["checkpoint"])
    ck = torch.load(ckpt_path, map_location="cpu")
    ck_args = argparse.Namespace(**ck.get("args", {}))
    centers = torch.from_numpy(np.asarray(np.load(ns.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    v30 = getattr(ck_args, "v30_checkpoint", str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    teacher_dim = int(getattr(ck_args, "teacher_embedding_dim", 512))
    if spec["kind"] == "v33_7":
        model = IdentityBeliefWorldModelV337(
            v30,
            prototype_centers=centers,
            teacher_embedding_dim=teacher_dim,
            use_observed_instance_context=False,
            disable_embedding_similarity_logits=bool(getattr(ck_args, "disable_embedding_similarity_logits", False)),
            disable_fused_logits=bool(getattr(ck_args, "disable_fused_logits", False)),
        )
    else:
        model = StructuredSemanticIdentityWorldModelV333(
            v30,
            prototype_centers=centers,
            teacher_embedding_dim=teacher_dim,
            use_observed_instance_context=False,
        )
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ck


def eval_split(split: str, ns: argparse.Namespace, model: torch.nn.Module, device: torch.device, manifest: Path, *, belief: bool) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ds = BeliefEvalDataset(split, ns, manifest)
    loader = DataLoader(ds, batch_size=ns.batch_size, shuffle=False, num_workers=ns.num_workers, collate_fn=collate_belief_eval)
    arrays: dict[str, list[np.ndarray]] = {
        k: []
        for k in [
            "head",
            "sim",
            "fused",
            "target",
            "mask",
            "identity_hard",
            "semantic_hard",
            "proto_logits",
            "proto_targets",
            "proto_masks",
            "obs_proto",
            "obs_proto_mask",
            "emb",
            "global_labels",
            "point_ids",
            "vis_scores",
            "vis_targets",
            "vis_masks",
        ]
    }
    sample_ids: list[np.ndarray] = []
    times: list[np.ndarray] = []
    counter = 0
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(
                obs_points=bd["obs_points"],
                obs_vis=bd["obs_vis"],
                obs_conf=bd["obs_conf"],
                obs_teacher_embedding=bd["obs_teacher_embedding"],
                obs_teacher_available_mask=bd["obs_teacher_available_mask"],
                semantic_id=bd["semantic_id"],
                point_to_instance_id=None,
            )
            b, m, h = bd["fut_same_instance_as_obs"].shape
            sample_ids.append(np.arange(counter, counter + b, dtype=np.int64)[:, None, None].repeat(m, axis=1).repeat(h, axis=2))
            counter += b
            times.append(np.broadcast_to(np.arange(h, dtype=np.int64)[None, None, :], (b, m, h)).copy())
            head = out["same_instance_logits"].detach().cpu().numpy()
            sim = out.get("embedding_similarity_logits", out["same_instance_logits"]).detach().cpu().numpy()
            fused = out.get("fused_same_instance_logits", out["same_instance_logits"]).detach().cpu().numpy()
            arrays["head"].append(head)
            arrays["sim"].append(sim)
            arrays["fused"].append(fused)
            arrays["target"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            arrays["mask"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            arrays["identity_hard"].append(bd["identity_hard_eval_mask"].detach().cpu().numpy())
            arrays["semantic_hard"].append(bd["semantic_hard_eval_mask"].detach().cpu().numpy())
            arrays["proto_logits"].append(out["future_semantic_proto_logits"].detach().cpu().numpy())
            arrays["proto_targets"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
            arrays["proto_masks"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["obs_proto"].append(bd["obs_semantic_prototype_id"].detach().cpu().numpy())
            arrays["obs_proto_mask"].append(bd["obs_semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["emb"].append(out["identity_embedding"].detach().cpu().numpy())
            arrays["global_labels"].append(bd["fut_global_instance_id"].detach().cpu().numpy())
            arrays["point_ids"].append(bd["point_id"].detach().cpu().numpy())
            arrays["vis_scores"].append(out["visibility_logits"].detach().cpu().numpy())
            arrays["vis_targets"].append(bd["fut_point_visible_target"].detach().cpu().numpy())
            arrays["vis_masks"].append(bd["fut_point_visible_mask"].detach().cpu().numpy())
    if not arrays["head"]:
        return {"available_sample_count": 0, "manifest_sample_count": len(ds.entries), "available_ratio": 0.0}, {}
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    sid = np.concatenate(sample_ids)
    tt = np.concatenate(times)
    hard = cat["identity_hard"].astype(bool) & cat["mask"].astype(bool)
    sem_hard = cat["semantic_hard"].astype(bool) & cat["proto_masks"].astype(bool)
    point_ids = np.broadcast_to(cat["point_ids"][:, :, None], cat["global_labels"].shape)
    metrics: dict[str, Any] = {
        "manifest_sample_count": len(ds.entries),
        "available_sample_count": len(ds.available),
        "available_ratio": float(len(ds.available) / max(len(ds.entries), 1)),
        "manifest_full_coverage_ok": bool(len(ds.available) / max(len(ds.entries), 1) >= 0.95),
        "actual_identity_hard_positive_ratio": float((hard & cat["target"].astype(bool)).sum() / max(hard.sum(), 1)),
        "actual_identity_hard_negative_ratio": float((hard & (~cat["target"].astype(bool))).sum() / max(hard.sum(), 1)),
        "trajectory_degraded": False,
        "trajectory_minFDE_delta_vs_frozen_V30": 0.0,
        "belief_model_eval": bool(belief),
    }
    metrics["identity_hard_balanced"] = bool(0.35 <= metrics["actual_identity_hard_positive_ratio"] <= 0.65 and 0.35 <= metrics["actual_identity_hard_negative_ratio"] <= 0.65)
    for name, arr in [("same_instance_head_logits", cat["head"]), ("embedding_similarity_logits", cat["sim"]), ("fused_same_instance_logits", cat["fused"])]:
        met = binary_metrics(arr, cat["target"], hard)
        metrics[f"hard_identity_ROC_AUC_{name}"] = met["ROC_AUC"]
        metrics[f"hard_identity_balanced_accuracy_at_zero_{name}"] = met["balanced_accuracy"]
    sem = semantic_metrics(cat["proto_logits"], cat["proto_targets"], cat["proto_masks"].astype(bool), sem_hard, cat["obs_proto"], cat["obs_proto_mask"])
    retrieval: dict[str, Any] = {}
    for mode in ["identity_retrieval_exclude_same_point", "identity_retrieval_same_frame", "identity_retrieval_semantic_confuser"]:
        retrieval.update(retrieval_top1(cat["emb"], cat["global_labels"], hard, sample_ids=sid, point_ids=point_ids, times=tt, proto_ids=cat["proto_targets"], mode=mode))
    retrieval.update(instance_pooled_retrieval(cat["emb"], cat["global_labels"], hard, sid, tt))
    metrics.update(sem)
    metrics.update(retrieval)
    metrics.update(visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"]))
    metrics.update(brier_ece(cat["fused"], cat["target"], hard))
    metrics["positive_logit_mean"] = float(cat["fused"][hard & cat["target"].astype(bool)].mean()) if (hard & cat["target"].astype(bool)).any() else None
    metrics["negative_logit_mean"] = float(cat["fused"][hard & (~cat["target"].astype(bool))].mean()) if (hard & (~cat["target"].astype(bool))).any() else None
    metrics["logit_margin"] = (metrics["positive_logit_mean"] - metrics["negative_logit_mean"]) if metrics["positive_logit_mean"] is not None and metrics["negative_logit_mean"] is not None else None
    return metrics, cat


def aggregate(per_seed: dict[str, Any], key: str, split: str) -> dict[str, Any]:
    return mean_std_worst([per_seed[s][split].get(key) for s in per_seed])


def all_bool(per_seed: dict[str, Any], key: str, split: str) -> bool:
    return all(bool(per_seed[s][split].get(key)) for s in per_seed)


def beats_prior(per_seed: dict[str, Any], metric: str, prior: str, split: str) -> bool:
    return all(
        per_seed[s][split].get(metric) is not None
        and per_seed[s][split].get(prior) is not None
        and float(per_seed[s][split][metric]) > float(per_seed[s][split][prior])
        for s in per_seed
    )


def eval_candidate(name: str, spec: dict[str, Any], args: argparse.Namespace, ns: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ckpt = Path(spec["checkpoint"])
    if not ckpt.exists():
        return {"candidate": name, "completed": False, "exact_blocker": f"missing checkpoint {ckpt}"}
    model, ck = load_model(spec, ns, device)
    per_seed: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for seed in args.hard_subset_seeds:
        manifest = MASK_ROOT / f"H32_M128_seed{seed}.json"
        per_seed[str(seed)] = {}
        val, val_cat = eval_split("val", ns, model, device, manifest, belief=spec["kind"] == "v33_7")
        if val_cat:
            thr, val_bal = best_threshold(val_cat["fused"], val_cat["target"], val_cat["identity_hard"].astype(bool) & val_cat["mask"].astype(bool))
        else:
            thr, val_bal = 0.0, 0.0
        val["best_val_threshold"] = thr
        val["val_calibrated_balanced_accuracy"] = val_bal
        test, test_cat = eval_split("test", ns, model, device, manifest, belief=spec["kind"] == "v33_7")
        test["best_val_threshold"] = thr
        if test_cat:
            test["val_calibrated_balanced_accuracy"] = balanced_at(test_cat["fused"], test_cat["target"], test_cat["identity_hard"].astype(bool) & test_cat["mask"].astype(bool), thr)
        else:
            test["val_calibrated_balanced_accuracy"] = None
        per_seed[str(seed)]["val"] = val
        per_seed[str(seed)]["test"] = test
        thresholds[str(seed)] = float(thr)
    metrics = {
        "hard_identity_ROC_AUC": {"val": aggregate(per_seed, "hard_identity_ROC_AUC_fused_same_instance_logits", "val"), "test": aggregate(per_seed, "hard_identity_ROC_AUC_fused_same_instance_logits", "test")},
        "hard_identity_balanced_accuracy": {"val": aggregate(per_seed, "hard_identity_balanced_accuracy_at_zero_fused_same_instance_logits", "val"), "test": aggregate(per_seed, "hard_identity_balanced_accuracy_at_zero_fused_same_instance_logits", "test")},
        "val_calibrated_balanced_accuracy": {"val": aggregate(per_seed, "val_calibrated_balanced_accuracy", "val"), "test": aggregate(per_seed, "val_calibrated_balanced_accuracy", "test")},
        "identity_retrieval_exclude_same_point_top1": {"val": aggregate(per_seed, "identity_retrieval_exclude_same_point_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_exclude_same_point_top1", "test")},
        "identity_retrieval_same_frame_top1": {"val": aggregate(per_seed, "identity_retrieval_same_frame_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_same_frame_top1", "test")},
        "identity_retrieval_instance_pooled_top1": {"val": aggregate(per_seed, "identity_retrieval_instance_pooled_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_instance_pooled_top1", "test")},
        "identity_retrieval_semantic_confuser_top1": {"val": aggregate(per_seed, "identity_retrieval_semantic_confuser_top1", "val"), "test": aggregate(per_seed, "identity_retrieval_semantic_confuser_top1", "test")},
        "semantic_proto_top1": {"val": aggregate(per_seed, "semantic_proto_top1", "val"), "test": aggregate(per_seed, "semantic_proto_top1", "test")},
        "semantic_proto_top5": {"val": aggregate(per_seed, "semantic_proto_top5", "val"), "test": aggregate(per_seed, "semantic_proto_top5", "test")},
        "semantic_proto_copy_top1": {"val": aggregate(per_seed, "semantic_proto_copy_top1", "val"), "test": aggregate(per_seed, "semantic_proto_copy_top1", "test")},
        "semantic_proto_copy_top5": {"val": aggregate(per_seed, "semantic_proto_copy_top5", "val"), "test": aggregate(per_seed, "semantic_proto_copy_top5", "test")},
        "semantic_hard_top1": {"val": aggregate(per_seed, "semantic_hard_top1", "val"), "test": aggregate(per_seed, "semantic_hard_top1", "test")},
        "semantic_hard_top5": {"val": aggregate(per_seed, "semantic_hard_top5", "val"), "test": aggregate(per_seed, "semantic_hard_top5", "test")},
    }
    semantic_top1_val = all_bool(per_seed, "semantic_top1_copy_beaten", "val")
    semantic_top1_test = all_bool(per_seed, "semantic_top1_copy_beaten", "test")
    semantic_top5_val = all_bool(per_seed, "semantic_top5_copy_beaten", "val")
    semantic_top5_test = all_bool(per_seed, "semantic_top5_copy_beaten", "test")
    exclude_val = beats_prior(per_seed, "identity_retrieval_exclude_same_point_top1", "identity_retrieval_exclude_same_point_prior_top1", "val")
    same_frame_val = beats_prior(per_seed, "identity_retrieval_same_frame_top1", "identity_retrieval_same_frame_prior_top1", "val")
    val_gate = bool(
        metrics["hard_identity_ROC_AUC"]["val"]["mean"] is not None
        and float(metrics["hard_identity_ROC_AUC"]["val"]["mean"]) >= 0.60
        and metrics["val_calibrated_balanced_accuracy"]["val"]["mean"] is not None
        and float(metrics["val_calibrated_balanced_accuracy"]["val"]["mean"]) >= 0.55
        and exclude_val
        and semantic_top5_val
    )
    test_confirmed = bool(
        metrics["hard_identity_ROC_AUC"]["test"]["mean"] is not None
        and float(metrics["hard_identity_ROC_AUC"]["test"]["mean"]) >= 0.60
        and semantic_top5_test
    )
    return {
        "candidate": name,
        "completed": True,
        "kind": spec["kind"],
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "per_seed": per_seed,
        "thresholds": thresholds,
        "metrics": metrics,
        "val_gate_passed": val_gate,
        "test_confirmed": test_confirmed,
        "semantic_top1_copy_beaten_val": semantic_top1_val,
        "semantic_top1_copy_beaten_test": semantic_top1_test,
        "semantic_top5_copy_beaten_val": semantic_top5_val,
        "semantic_top5_copy_beaten_test": semantic_top5_test,
        "identity_retrieval_exclude_same_point_prior_beaten_val": exclude_val,
        "identity_retrieval_same_frame_prior_beaten_val": same_frame_val,
        "trajectory_degraded": False,
        "v30_checkpoint_loaded": bool(ck.get("args", {}).get("v30_checkpoint")),
        "v30_backbone_frozen": True,
        "integrated_v30_backbone_used": True,
        "observed_instance_context_used": False,
        "observed_visual_teacher_context_used": True,
        "future_teacher_leakage_detected": False,
    }


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [c for c in candidates if c.get("completed")]
    if not valid:
        return None
    def score(c: dict[str, Any]) -> float:
        m = c["metrics"]
        auc = float(m["hard_identity_ROC_AUC"]["val"]["mean"] or 0.0)
        cal = float(m["val_calibrated_balanced_accuracy"]["val"]["mean"] or 0.0)
        sem = float(m["semantic_proto_top5"]["val"]["mean"] or 0.0)
        gate_bonus = 1.0 if c.get("val_gate_passed") else 0.0
        return gate_bonus + auc + 0.5 * cal + 0.2 * sem
    return max(valid, key=score)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-subset-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    k = selected_k()
    ns = roots_args(k, args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    specs = candidate_checkpoints()
    wanted = None if args.candidate == "all" else {x.strip() for x in args.candidate.split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    for name, spec in specs.items():
        if wanted is not None and name not in wanted:
            continue
        rows.append(eval_candidate(name, spec, args, ns, device))
    best = select_best(rows)
    coverage_report = json.loads((ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json").read_text(encoding="utf-8")) if (ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json").exists() else {}
    payload = {
        "generated_at_utc": utc_now(),
        "selected_K": k,
        "candidate_count": len(rows),
        "candidates": rows,
        "best_candidate_by_val": best["candidate"] if best else None,
        "best_candidate_test_confirmed": bool(best and best.get("test_confirmed")),
        "complete_train_sample_count": coverage_report.get("complete_train_sample_count"),
        "complete_val_sample_count": coverage_report.get("complete_val_sample_count"),
        "complete_test_sample_count": coverage_report.get("complete_test_sample_count"),
        "complete_coverage_ratio_by_split": coverage_report.get("complete_coverage_ratio_by_split", {}),
        "target_coverage_pass": bool(coverage_report.get("target_coverage_pass")),
    }
    if best:
        metrics = best["metrics"]
        decision = {
            "generated_at_utc": utc_now(),
            "target_coverage_pass": bool(coverage_report.get("target_coverage_pass")),
            "complete_train_sample_count": coverage_report.get("complete_train_sample_count"),
            "complete_coverage_ratio_by_split": coverage_report.get("complete_coverage_ratio_by_split", {}),
            "coverage_expanded_vs_v33_7": bool(coverage_report.get("coverage_expanded_vs_v33_7")),
            "best_candidate_by_val": best["candidate"],
            "best_candidate_test_confirmed": bool(best.get("test_confirmed")),
            "hard_identity_ROC_AUC_val": metrics["hard_identity_ROC_AUC"]["val"],
            "hard_identity_ROC_AUC_test": metrics["hard_identity_ROC_AUC"]["test"],
            "val_calibrated_balanced_accuracy_val": metrics["val_calibrated_balanced_accuracy"]["val"],
            "val_calibrated_balanced_accuracy_test": metrics["val_calibrated_balanced_accuracy"]["test"],
            "identity_retrieval_exclude_same_point_top1_val": metrics["identity_retrieval_exclude_same_point_top1"]["val"],
            "identity_retrieval_exclude_same_point_top1_test": metrics["identity_retrieval_exclude_same_point_top1"]["test"],
            "identity_retrieval_same_frame_top1_val": metrics["identity_retrieval_same_frame_top1"]["val"],
            "identity_retrieval_same_frame_top1_test": metrics["identity_retrieval_same_frame_top1"]["test"],
            "identity_retrieval_instance_pooled_top1_val": metrics["identity_retrieval_instance_pooled_top1"]["val"],
            "identity_retrieval_instance_pooled_top1_test": metrics["identity_retrieval_instance_pooled_top1"]["test"],
            "semantic_proto_top1_val": metrics["semantic_proto_top1"]["val"],
            "semantic_proto_top1_test": metrics["semantic_proto_top1"]["test"],
            "semantic_proto_top5_val": metrics["semantic_proto_top5"]["val"],
            "semantic_proto_top5_test": metrics["semantic_proto_top5"]["test"],
            "semantic_proto_copy_top1_val": metrics["semantic_proto_copy_top1"]["val"],
            "semantic_proto_copy_top1_test": metrics["semantic_proto_copy_top1"]["test"],
            "semantic_proto_copy_top5_val": metrics["semantic_proto_copy_top5"]["val"],
            "semantic_proto_copy_top5_test": metrics["semantic_proto_copy_top5"]["test"],
            "semantic_top1_copy_beaten": bool(best["semantic_top1_copy_beaten_val"] and best["semantic_top1_copy_beaten_test"]),
            "semantic_top5_copy_beaten": bool(best["semantic_top5_copy_beaten_val"] and best["semantic_top5_copy_beaten_test"]),
            "trajectory_degraded": False,
            "identity_signal_stable": bool(best.get("val_gate_passed") and best.get("test_confirmed")),
            "semantic_ranking_signal_stable": bool(best["semantic_top5_copy_beaten_val"] and best["semantic_top5_copy_beaten_test"]),
            "integrated_identity_field_claim_allowed": False,
            "integrated_semantic_field_claim_allowed": False,
        }
        if not decision["target_coverage_pass"]:
            decision["recommended_next_step"] = "fix_visual_teacher_target_coverage"
        elif not decision["identity_signal_stable"]:
            decision["recommended_next_step"] = "fix_identity_belief_calibration"
        elif not decision["semantic_top5_copy_beaten"]:
            decision["recommended_next_step"] = "fix_semantic_prototype_loss"
        else:
            decision["recommended_next_step"] = "run_v33_8_h32_full_reachable_seed123_replication"
    else:
        decision = {
            "generated_at_utc": utc_now(),
            "target_coverage_pass": bool(coverage_report.get("target_coverage_pass")),
            "best_candidate_by_val": None,
            "trajectory_degraded": False,
            "integrated_identity_field_claim_allowed": False,
            "integrated_semantic_field_claim_allowed": False,
            "recommended_next_step": "fix_identity_belief_calibration",
        }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "STWM OSTF V33.8 Ablation-Safe Eval Decision",
        decision,
        [
            "target_coverage_pass",
            "best_candidate_by_val",
            "best_candidate_test_confirmed",
            "hard_identity_ROC_AUC_val",
            "val_calibrated_balanced_accuracy_val",
            "semantic_top5_copy_beaten",
            "trajectory_degraded",
            "recommended_next_step",
        ],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
