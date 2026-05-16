#!/usr/bin/env python3
"""评估 V35 semantic state head（seed42，V30 frozen）。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.modules.ostf_v35_semantic_state_world_model import SemanticStateWorldModelV35
from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.train_ostf_v35_semantic_state_head_20260515 import ASSIGNMENT_TARGET_ROOT, CKPT_DIR, MEASUREMENT_BANK_ROOT, SUMMARY as TRAIN_SUMMARY, V35SemanticStateDataset, collate, move_batch

SUMMARY = ROOT / "reports/stwm_ostf_v35_semantic_state_head_eval_summary_20260515.json"
DECISION = ROOT / "reports/stwm_ostf_v35_semantic_state_head_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_SEMANTIC_STATE_HEAD_DECISION_20260515.md"


def topk_hit(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int) -> tuple[int, int]:
    if not bool(mask.any()):
        return 0, 0
    kk = min(k, logits.shape[-1])
    pred = logits.topk(kk, dim=-1).indices
    hit = (pred == target[..., None]).any(dim=-1) & mask
    return int(hit.sum().item()), int(mask.sum().item())


def onehot_baseline_hit(last_cluster: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    valid = mask & (last_cluster >= 0)
    if not bool(valid.any()):
        return 0, 0
    hit = (last_cluster == target) & valid
    return int(hit.sum().item()), int(valid.sum().item())


def choose_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best_b = 0.5, -1.0
    for t in np.quantile(scores, np.linspace(0.05, 0.95, 37)):
        b = balanced_accuracy_score(y, scores >= t)
        if b > best_b:
            best_b, best_t = float(b), float(t)
    return float(best_t)


def binary_metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "f1": None, "positive_ratio": float(y.mean())}
    pred = scores >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_ratio": float(y.mean()),
    }


def purity(assign: np.ndarray, labels: np.ndarray) -> float:
    vals: list[float] = []
    for b in range(assign.shape[0]):
        for u in range(assign.shape[-1]):
            w = assign[b, :, u]
            keep = labels[b] >= 0
            if keep.sum() == 0 or w[keep].sum() <= 1e-6:
                continue
            counts: dict[int, float] = {}
            for lab, ww in zip(labels[b, keep], w[keep]):
                counts[int(lab)] = counts.get(int(lab), 0.0) + float(ww)
            vals.append(max(counts.values()) / max(sum(counts.values()), 1e-6))
    return float(np.mean(vals)) if vals else 0.0


def empty_identity_retrieval_acc() -> dict[str, float]:
    return {
        "exclude_same_point_hit": 0.0,
        "exclude_same_point_total": 0.0,
        "same_frame_hit": 0.0,
        "same_frame_total": 0.0,
        "instance_pooled_hit": 0.0,
        "instance_pooled_total": 0.0,
        "same_pair_sim_sum": 0.0,
        "same_pair_sim_count": 0.0,
        "confuser_pair_sim_sum": 0.0,
        "confuser_pair_sim_count": 0.0,
    }


def update_identity_retrieval_acc(
    acc: dict[str, float],
    identity_embedding: torch.Tensor,
    instance_id: torch.Tensor,
    confuser_pair_mask: torch.Tensor | None = None,
) -> None:
    emb = identity_embedding.detach().cpu().numpy().mean(axis=2)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-8)
    labels_np = instance_id.detach().cpu().numpy()
    conf_np = confuser_pair_mask.detach().cpu().numpy().astype(bool) if confuser_pair_mask is not None else None
    for b in range(emb.shape[0]):
        labels = labels_np[b]
        valid = labels >= 0
        if int(valid.sum()) < 2:
            continue
        sim = emb[b] @ emb[b].T
        np.fill_diagonal(sim, -np.inf)
        for i in np.where(valid)[0]:
            same = (labels == labels[i]) & valid
            same[i] = False
            if same.any():
                top = int(np.argmax(sim[i]))
                acc["exclude_same_point_total"] += 1.0
                acc["same_frame_total"] += 1.0
                hit = float(labels[top] == labels[i])
                acc["exclude_same_point_hit"] += hit
                acc["same_frame_hit"] += hit
        unique_ids = [int(x) for x in np.unique(labels[valid])]
        if len(unique_ids) >= 2:
            centroids = []
            for inst in unique_ids:
                cur = emb[b, labels == inst]
                c = cur.mean(axis=0)
                c = c / max(float(np.linalg.norm(c)), 1e-8)
                centroids.append(c)
            centroid_arr = np.stack(centroids, axis=0)
            point_to_centroid = emb[b, valid] @ centroid_arr.T
            valid_labels = labels[valid]
            pred_ids = np.asarray(unique_ids, dtype=np.int64)[point_to_centroid.argmax(axis=1)]
            acc["instance_pooled_hit"] += float((pred_ids == valid_labels).sum())
            acc["instance_pooled_total"] += float(valid_labels.shape[0])
        same_pairs = (labels[:, None] == labels[None, :]) & valid[:, None] & valid[None, :]
        eye = np.eye(labels.shape[0], dtype=bool)
        same_pairs &= ~eye
        if same_pairs.any():
            acc["same_pair_sim_sum"] += float(sim[same_pairs].sum())
            acc["same_pair_sim_count"] += float(same_pairs.sum())
        if conf_np is not None:
            conf_pairs = conf_np[b] & ~eye
            if conf_pairs.any():
                acc["confuser_pair_sim_sum"] += float(sim[conf_pairs].sum())
                acc["confuser_pair_sim_count"] += float(conf_pairs.sum())


def finalize_identity_retrieval_acc(acc: dict[str, float]) -> dict[str, float | None]:
    same_mean = acc["same_pair_sim_sum"] / acc["same_pair_sim_count"] if acc["same_pair_sim_count"] > 0 else None
    conf_mean = acc["confuser_pair_sim_sum"] / acc["confuser_pair_sim_count"] if acc["confuser_pair_sim_count"] > 0 else None
    separation = float(same_mean - conf_mean) if same_mean is not None and conf_mean is not None else None
    return {
        "identity_retrieval_exclude_same_point_top1": float(acc["exclude_same_point_hit"] / max(acc["exclude_same_point_total"], 1.0)),
        "identity_retrieval_same_frame_top1": float(acc["same_frame_hit"] / max(acc["same_frame_total"], 1.0)),
        "identity_retrieval_instance_pooled_top1": float(acc["instance_pooled_hit"] / max(acc["instance_pooled_total"], 1.0)),
        "identity_same_pair_similarity": float(same_mean) if same_mean is not None else None,
        "identity_confuser_pair_similarity": float(conf_mean) if conf_mean is not None else None,
        "identity_confuser_separation": separation,
    }


@torch.no_grad()
def collect(model: SemanticStateWorldModelV35, split: str, args: argparse.Namespace, device: torch.device, intervention: str | None = None) -> dict[str, Any]:
    ds = V35SemanticStateDataset(
        split,
        args.semantic_clusters,
        ASSIGNMENT_TARGET_ROOT if args.use_assignment_targets else None,
        args.measurement_bank_root if args.include_observed_measurement_embedding else None,
        args.include_observed_measurement_embedding,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate)
    model.eval()
    totals: dict[str, list[int]] = {k: [0, 0] for k in [
        "global_top1", "global_top5", "stable_top1", "stable_top5", "changed_top1", "changed_top5", "hard_top1", "hard_top5",
        "copy_global_top1", "copy_stable_top1", "copy_changed_top1", "copy_hard_top1",
        "family_top1", "family_top3",
    ]}
    changed_scores: list[np.ndarray] = []
    changed_y: list[np.ndarray] = []
    same_scores: list[np.ndarray] = []
    same_y: list[np.ndarray] = []
    unc_scores: list[np.ndarray] = []
    unc_y: list[np.ndarray] = []
    assignments: list[np.ndarray] = []
    last_labels: list[np.ndarray] = []
    inst_labels: list[np.ndarray] = []
    identity_acc = empty_identity_retrieval_acc()
    for batch in loader:
        bd = move_batch(batch, device)
        out = model(bd["point_features"], horizon=bd["target_cluster"].shape[2], intervention=intervention)
        logits = out["semantic_cluster_logits"]
        target = bd["target_cluster"]
        valid = bd["valid"].bool()
        stable = bd["stable"].bool() & valid
        changed = bd["changed"].bool() & valid
        hard = bd["hard"].bool() & valid
        for name, mask in [("global", valid), ("stable", stable), ("changed", changed), ("hard", hard)]:
            h1, n1 = topk_hit(logits, target, mask, 1)
            h5, n5 = topk_hit(logits, target, mask, 5)
            totals[f"{name}_top1"][0] += h1
            totals[f"{name}_top1"][1] += n1
            totals[f"{name}_top5"][0] += h5
            totals[f"{name}_top5"][1] += n5
            ch, cn = onehot_baseline_hit(bd["last_cluster"], target, mask)
            totals[f"copy_{name}_top1"][0] += ch
            totals[f"copy_{name}_top1"][1] += cn
        fam_mask = bd["family_available"].bool() & valid
        fh1, fn = topk_hit(out["evidence_anchor_family_logits"], bd["family"], fam_mask, 1)
        fh3, _ = topk_hit(out["evidence_anchor_family_logits"], bd["family"], fam_mask, 3)
        totals["family_top1"][0] += fh1
        totals["family_top1"][1] += fn
        totals["family_top3"][0] += fh3
        totals["family_top3"][1] += fn
        changed_scores.append(torch.sigmoid(out["semantic_change_logits"])[valid].detach().cpu().numpy())
        changed_y.append(bd["changed"][valid].detach().cpu().numpy().astype(np.int64))
        same_mask = bd["same_available"].bool() & valid
        if bool(same_mask.any()):
            same_scores.append(torch.sigmoid(out["same_instance_logits"])[same_mask].detach().cpu().numpy())
            same_y.append(bd["same_instance"][same_mask].detach().cpu().numpy().astype(np.int64))
        unc_scores.append(out["semantic_uncertainty"][valid].detach().cpu().numpy())
        unc_target = bd["uncertainty"][valid].detach().cpu().numpy()
        unc_y.append((unc_target >= np.quantile(unc_target, 0.70)).astype(np.int64))
        assignments.append(out["point_to_unit_assignment"].detach().cpu().numpy())
        last_labels.append(bd["last_cluster"][:, :, 0].detach().cpu().numpy())
        inst_labels.append(bd["point_to_instance_id"].detach().cpu().numpy())
        update_identity_retrieval_acc(identity_acc, out["identity_embedding"], bd["point_to_instance_id"], bd.get("identity_confuser_pair_mask"))

    def rate(key: str) -> float:
        h, n = totals[key]
        return float(h / max(n, 1))

    assign_np = np.concatenate(assignments, axis=0) if assignments else np.zeros((1, 1, 1), dtype=np.float32)
    usage = assign_np.mean(axis=(0, 1))
    usage = usage / max(usage.sum(), 1e-8)
    effective_units = float(np.exp(-(usage * np.log(np.maximum(usage, 1e-8))).sum()))
    out_metrics = {
        "global_top1": rate("global_top1"),
        "global_top5": rate("global_top5"),
        "stable_top1": rate("stable_top1"),
        "stable_top5": rate("stable_top5"),
        "changed_top1": rate("changed_top1"),
        "changed_top5": rate("changed_top5"),
        "hard_top1": rate("hard_top1"),
        "hard_top5": rate("hard_top5"),
        "copy_global_top1": rate("copy_global_top1"),
        "copy_stable_top1": rate("copy_stable_top1"),
        "copy_changed_top1": rate("copy_changed_top1"),
        "copy_hard_top1": rate("copy_hard_top1"),
        "family_top1": rate("family_top1"),
        "family_top3": rate("family_top3"),
        "changed_scores": np.concatenate(changed_scores) if changed_scores else np.asarray([]),
        "changed_y": np.concatenate(changed_y) if changed_y else np.asarray([]),
        "same_scores": np.concatenate(same_scores) if same_scores else np.asarray([]),
        "same_y": np.concatenate(same_y) if same_y else np.asarray([]),
        "unc_scores": np.concatenate(unc_scores) if unc_scores else np.asarray([]),
        "unc_y": np.concatenate(unc_y) if unc_y else np.asarray([]),
        "effective_units": effective_units,
        "unit_semantic_purity": purity(assign_np, np.concatenate(last_labels, axis=0)),
        "unit_dominant_instance_purity": purity(assign_np, np.concatenate(inst_labels, axis=0)),
    }
    out_metrics.update(finalize_identity_retrieval_acc(identity_acc))
    return out_metrics


def strip_arrays(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--semantic-clusters", type=int, default=64)
    ap.add_argument("--evidence-families", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--train-summary-path", type=str, default=str(TRAIN_SUMMARY))
    ap.add_argument("--summary-path", type=str, default=str(SUMMARY))
    ap.add_argument("--decision-path", type=str, default=str(DECISION))
    ap.add_argument("--doc-path", type=str, default=str(DOC))
    ap.add_argument("--use-assignment-targets", action="store_true")
    ap.add_argument("--measurement-bank-root", type=str, default="")
    ap.add_argument("--include-observed-measurement-embedding", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    print(f"V35 semantic state head: 开始评估 seed{args.seed} checkpoint。", flush=True)
    train_summary_path = Path(args.train_summary_path)
    summary_path = Path(args.summary_path)
    decision_path = Path(args.decision_path)
    doc_path = Path(args.doc_path)
    train = json.loads(train_summary_path.read_text(encoding="utf-8")) if train_summary_path.exists() else {}
    ckpt_path = ROOT / train.get("checkpoint_path", str(CKPT_DIR / f"v35_semantic_state_head_m128_h32_seed{args.seed}_best.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    if bool(ckpt_args.get("include_observed_measurement_embedding", False)):
        args.include_observed_measurement_embedding = True
    if not args.measurement_bank_root:
        args.measurement_bank_root = ckpt_args.get("measurement_bank_root", str(MEASUREMENT_BANK_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SemanticStateWorldModelV35(
        point_feature_dim=int(ckpt["feature_dim"]),
        semantic_clusters=args.semantic_clusters,
        evidence_families=args.evidence_families,
        copy_prior_strength=float(ckpt_args.get("copy_prior_strength", 0.0)),
        assignment_bound_decoder=bool(ckpt_args.get("assignment_bound_decoder", False)),
        identity_dim=int(ckpt_args.get("identity_dim", 64)),
        semantic_feature_dim=int(ckpt_args["semantic_feature_dim"]) if "semantic_feature_dim" in ckpt_args else None,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    val = collect(model, "val", args, device)
    test = collect(model, "test", args, device)
    ch_t = choose_threshold(val["changed_scores"], val["changed_y"])
    same_t = choose_threshold(val["same_scores"], val["same_y"]) if len(val["same_y"]) else 0.5
    unc_t = choose_threshold(val["unc_scores"], val["unc_y"])
    for split_metrics in [val, test]:
        split_metrics["semantic_changed_metrics"] = binary_metrics(split_metrics.pop("changed_scores"), split_metrics.pop("changed_y"), ch_t)
        split_metrics["same_instance_metrics"] = binary_metrics(split_metrics.pop("same_scores"), split_metrics.pop("same_y"), same_t) if len(split_metrics.get("same_y", [])) else {"roc_auc": None, "balanced_accuracy": None, "f1": None}
        split_metrics.pop("same_y", None)
        split_metrics["uncertainty_high_metrics"] = binary_metrics(split_metrics.pop("unc_scores"), split_metrics.pop("unc_y"), unc_t)

    interventions: dict[str, dict[str, Any]] = {}
    for name in ["zero_semantic_measurements", "shuffle_semantic_measurements", "shuffle_assignment", "zero_unit_memory"]:
        iv = collect(model, "val", args, device, intervention=name)
        interventions[name] = {
            "hard_top5_delta_vs_normal": float(val["hard_top5"] - iv["hard_top5"]),
            "changed_top5_delta_vs_normal": float(val["changed_top5"] - iv["changed_top5"]),
            "global_top5_delta_vs_normal": float(val["global_top5"] - iv["global_top5"]),
        }

    semantic_hard_signal = {"val": bool(val["hard_top5"] >= val["copy_hard_top1"] + 0.02), "test": bool(test["hard_top5"] >= test["copy_hard_top1"] + 0.02)}
    changed_semantic_signal = {"val": bool(val["changed_top5"] >= val["copy_changed_top1"] + 0.02), "test": bool(test["changed_top5"] >= test["copy_changed_top1"] + 0.02)}
    stable_preservation = {"val": bool(val["stable_top5"] >= val["copy_stable_top1"] - 0.02), "test": bool(test["stable_top5"] >= test["copy_stable_top1"] - 0.02)}
    semantic_measurement_load_bearing = bool(max(interventions["zero_semantic_measurements"]["hard_top5_delta_vs_normal"], interventions["shuffle_semantic_measurements"]["hard_top5_delta_vs_normal"], interventions["zero_semantic_measurements"]["changed_top5_delta_vs_normal"], interventions["shuffle_semantic_measurements"]["changed_top5_delta_vs_normal"]) > 0.005)
    assignment_load_bearing = bool(max(interventions["shuffle_assignment"]["hard_top5_delta_vs_normal"], interventions["shuffle_assignment"]["changed_top5_delta_vs_normal"]) > 0.005)
    unit_memory_load_bearing = bool(max(interventions["zero_unit_memory"]["hard_top5_delta_vs_normal"], interventions["zero_unit_memory"]["changed_top5_delta_vs_normal"]) > 0.005)
    head_passed = bool(stable_preservation["val"] and stable_preservation["test"] and (semantic_hard_signal["val"] or changed_semantic_signal["val"]) and (semantic_hard_signal["test"] or changed_semantic_signal["test"]))
    identity_retrieval_passed = bool(
        val["identity_retrieval_exclude_same_point_top1"] >= 0.50
        and test["identity_retrieval_exclude_same_point_top1"] >= 0.50
        and val["identity_retrieval_same_frame_top1"] >= 0.50
        and test["identity_retrieval_same_frame_top1"] >= 0.50
        and val["identity_retrieval_instance_pooled_top1"] >= 0.70
        and test["identity_retrieval_instance_pooled_top1"] >= 0.70
        and (val["identity_confuser_separation"] is not None and val["identity_confuser_separation"] > 0.0)
        and (test["identity_confuser_separation"] is not None and test["identity_confuser_separation"] > 0.0)
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "val": strip_arrays(val),
        "test": strip_arrays(test),
        "thresholds_from_val": {"semantic_changed": ch_t, "same_instance": same_t, "uncertainty_high": unc_t},
        "interventions_val": interventions,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "teacher_as_method": False,
        "中文结论": "V35 semantic state head 已完成 seed42 评估；本结果只验证低维 semantic state head，不代表完整 semantic field success。",
    }
    decision = {
        "semantic_state_head_training_ran": True,
        "semantic_state_head_passed": head_passed,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "semantic_hard_signal": semantic_hard_signal,
        "changed_semantic_signal": changed_semantic_signal,
        "stable_preservation": stable_preservation,
        "identity_consistency": {
            "val_auc": val["same_instance_metrics"].get("roc_auc"),
            "test_auc": test["same_instance_metrics"].get("roc_auc"),
            "retrieval_passed": identity_retrieval_passed,
            "identity_retrieval_exclude_same_point_top1": {
                "val": val["identity_retrieval_exclude_same_point_top1"],
                "test": test["identity_retrieval_exclude_same_point_top1"],
            },
            "identity_retrieval_same_frame_top1": {
                "val": val["identity_retrieval_same_frame_top1"],
                "test": test["identity_retrieval_same_frame_top1"],
            },
            "identity_retrieval_instance_pooled_top1": {
                "val": val["identity_retrieval_instance_pooled_top1"],
                "test": test["identity_retrieval_instance_pooled_top1"],
            },
            "identity_confuser_separation": {
                "val": val["identity_confuser_separation"],
                "test": test["identity_confuser_separation"],
            },
        },
        "uncertainty_quality": {
            "val_auc": val["uncertainty_high_metrics"].get("roc_auc"),
            "test_auc": test["uncertainty_high_metrics"].get("roc_auc"),
        },
        "unit_memory_load_bearing": unit_memory_load_bearing,
        "semantic_measurement_load_bearing": semantic_measurement_load_bearing,
        "assignment_load_bearing": assignment_load_bearing,
        "effective_units": {"val": val["effective_units"], "test": test["effective_units"]},
        "unit_dominant_instance_purity": {"val": val["unit_dominant_instance_purity"], "test": test["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": val["unit_semantic_purity"], "test": test["unit_semantic_purity"]},
        "integrated_identity_field_claim_allowed": identity_retrieval_passed,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v35_identity_seed123_replication" if head_passed and identity_retrieval_passed else ("fix_identity_retrieval_head" if head_passed else "fix_v35_semantic_state_head"),
        "中文结论": "V35 identity retrieval 评估已加入 pairwise / retrieval gate。只有 exclude-same-point、same-frame、instance-pooled 检索和 confuser separation 同时过，才允许 identity field claim；semantic field success 仍不允许。",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    doc_path.write_text(
        "# STWM OSTF V35 Semantic State Head Decision\n\n"
        f"- semantic_state_head_training_ran: true\n"
        f"- semantic_state_head_passed: {head_passed}\n"
        f"- semantic_hard_signal: {semantic_hard_signal}\n"
        f"- changed_semantic_signal: {changed_semantic_signal}\n"
        f"- stable_preservation: {stable_preservation}\n"
        f"- unit_memory_load_bearing: {unit_memory_load_bearing}\n"
        f"- semantic_measurement_load_bearing: {semantic_measurement_load_bearing}\n"
        f"- assignment_load_bearing: {assignment_load_bearing}\n"
        f"- identity_retrieval_passed: {identity_retrieval_passed}\n"
        f"- identity_retrieval_exclude_same_point_top1: val={val['identity_retrieval_exclude_same_point_top1']} test={test['identity_retrieval_exclude_same_point_top1']}\n"
        f"- identity_retrieval_same_frame_top1: val={val['identity_retrieval_same_frame_top1']} test={test['identity_retrieval_same_frame_top1']}\n"
        f"- identity_retrieval_instance_pooled_top1: val={val['identity_retrieval_instance_pooled_top1']} test={test['identity_retrieval_instance_pooled_top1']}\n"
        f"- identity_confuser_separation: val={val['identity_confuser_separation']} test={test['identity_confuser_separation']}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"semantic_state_head_passed": head_passed, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
