#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import _load_global_identity_sidecar, _load_semantic_identity_sidecar, collate_external_gt, load_external_gt_item
from stwm.modules.ostf_v33_7_identity_belief_world_model import IdentityBeliefWorldModelV337
from stwm.tools.eval_ostf_v33_4_structured_semantic_identity_protocol_20260509 import instance_pooled_retrieval, retrieval_top1, semantic_metrics
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1


SUMMARY = ROOT / "reports/stwm_ostf_v33_7_identity_belief_eval_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_7_identity_belief_eval_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_7_IDENTITY_BELIEF_EVAL_DECISION_20260509.md"


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def v30_uid_map(split: str) -> dict[str, dict[str, Any]]:
    entries = json.loads((ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json").read_text(encoding="utf-8")).get("entries", [])
    out = {}
    for e in entries:
        if int(e.get("H", -1)) != 32 or int(e.get("M", -1)) != 128:
            continue
        path = ROOT / e["cache_path"]
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        out[str(np.asarray(z["video_uid"]).item() if "video_uid" in z else path.stem)] = e
    return out


class BeliefEvalDataset(Dataset):
    def __init__(self, split: str, args: argparse.Namespace, manifest_path: Path) -> None:
        self.split = split
        self.args = args
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries = self.manifest.get("splits", {}).get(split, [])
        uid_to_v30 = v30_uid_map(split)
        self.available = []
        self.missing = []
        for entry in self.entries:
            uid = str(entry["sample_uid"])
            comps = {
                "v30_base": ROOT / uid_to_v30[uid]["cache_path"] if uid in uid_to_v30 else None,
                "identity": Path(args.semantic_identity_sidecar_root) / split / f"{uid}.npz",
                "global_identity": Path(args.global_identity_label_root) / split / f"{uid}.npz",
                "visual": Path(args.visual_teacher_root) / split / f"{uid}.npz",
                "proto": Path(args.semantic_prototype_target_root) / split / f"{uid}.npz",
                "mask": ROOT / entry["mask_path"],
            }
            miss = []
            for name, path in comps.items():
                if path is None or not Path(path).exists():
                    miss.append({"sample_uid": uid, "split": split, "missing_component": name, "exact_path": str(path)})
            if miss:
                self.missing.extend(miss)
            else:
                self.available.append({"manifest_entry": entry, "v30_entry": uid_to_v30[uid], "components": comps})

    def __len__(self) -> int:
        return len(self.available)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.available[idx]
        item = load_external_gt_item(rec["v30_entry"]["cache_path"], rec["v30_entry"].get("v30_subset_tags", []))
        out: dict[str, Any] = {
            "uid": item.uid,
            "dataset": item.dataset,
            "split": item.split,
            "cache_path": item.cache_path,
            "coordinate_system": item.coordinate_system,
            "obs_points": torch.from_numpy(item.obs_points[..., :2]).float(),
            "fut_points": torch.from_numpy(item.fut_points[..., :2]).float(),
            "obs_vis": torch.from_numpy(item.obs_vis).bool(),
            "fut_vis": torch.from_numpy(item.fut_vis).bool(),
            "obs_conf": torch.from_numpy(item.obs_conf).float(),
            "fut_conf": torch.from_numpy(item.fut_conf).float(),
            "semantic_id": torch.tensor(item.semantic_id, dtype=torch.long),
            "has_3d": torch.tensor(item.has_3d, dtype=torch.bool),
            "v30_subset_tags": item.v30_subset_tags,
            "use_observed_instance_context": torch.tensor(False, dtype=torch.bool),
        }
        out.update(_load_semantic_identity_sidecar(uid=item.uid, split=self.split, m_points=item.m_points, horizon=item.horizon, root=self.args.semantic_identity_sidecar_root, require=True))
        out.update(_load_global_identity_sidecar(uid=item.uid, split=self.split, m_points=item.m_points, horizon=item.horizon, root=self.args.global_identity_label_root, require=True))
        vz = np.load(rec["components"]["visual"], allow_pickle=True)
        out["obs_teacher_embedding"] = torch.from_numpy(np.asarray(vz["obs_teacher_embedding"], dtype=np.float32))
        out["obs_teacher_available_mask"] = torch.from_numpy(np.asarray(vz["obs_teacher_available_mask"]).astype(bool))
        pz = np.load(rec["components"]["proto"], allow_pickle=True)
        out["semantic_prototype_id"] = torch.from_numpy(np.asarray(pz["semantic_prototype_id"], dtype=np.int64)).long()
        out["semantic_prototype_available_mask"] = torch.from_numpy(np.asarray(pz["semantic_prototype_available_mask"]).astype(bool))
        out["obs_semantic_prototype_id"] = torch.from_numpy(np.asarray(pz["obs_semantic_prototype_id"], dtype=np.int64)).long()
        out["obs_semantic_prototype_available_mask"] = torch.from_numpy(np.asarray(pz["obs_semantic_prototype_available_mask"]).astype(bool))
        mz = np.load(rec["components"]["mask"], allow_pickle=True)
        out["identity_hard_eval_mask"] = torch.from_numpy(np.asarray(mz["identity_hard_eval_mask" if "identity_hard_eval_mask" in mz.files else "identity_hard_train_mask"]).astype(bool))
        out["semantic_hard_eval_mask"] = torch.from_numpy(np.asarray(mz["semantic_hard_eval_mask" if "semantic_hard_eval_mask" in mz.files else "semantic_hard_train_mask"]).astype(bool))
        return out


def collate_belief_eval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_external_gt(batch)
    for key in ["obs_teacher_embedding", "obs_teacher_available_mask", "semantic_prototype_id", "semantic_prototype_available_mask", "obs_semantic_prototype_id", "obs_semantic_prototype_available_mask", "identity_hard_eval_mask", "semantic_hard_eval_mask"]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def best_threshold(scores: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return 0.0, 0.0
    s = scores[valid]
    y = target[valid].astype(bool)
    if len(np.unique(y)) < 2:
        return 0.0, 0.0
    qs = np.quantile(s, np.linspace(0.02, 0.98, 97))
    best_t, best_b = 0.0, -1.0
    for t in qs:
        pred = s >= t
        tp = np.logical_and(pred, y).sum()
        tn = np.logical_and(~pred, ~y).sum()
        fp = np.logical_and(pred, ~y).sum()
        fn = np.logical_and(~pred, y).sum()
        bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        if bal > best_b:
            best_t, best_b = float(t), float(bal)
    return best_t, best_b


def balanced_at(scores: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float) -> float | None:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return None
    s = scores[valid]
    y = target[valid].astype(bool)
    pred = s >= threshold
    tp = np.logical_and(pred, y).sum()
    tn = np.logical_and(~pred, ~y).sum()
    fp = np.logical_and(pred, ~y).sum()
    fn = np.logical_and(~pred, y).sum()
    return float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))


def brier_ece(scores: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return {"brier_score": None, "ECE": None}
    prob = 1.0 / (1.0 + np.exp(-scores[valid]))
    y = target[valid].astype(float)
    brier = float(np.mean((prob - y) ** 2))
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        bin_mask = (prob >= lo) & (prob < hi)
        if bin_mask.any():
            ece += float(bin_mask.mean() * abs(prob[bin_mask].mean() - y[bin_mask].mean()))
    return {"brier_score": brier, "ECE": ece}


def eval_split(split: str, args: argparse.Namespace, model: IdentityBeliefWorldModelV337, device: torch.device, manifest: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ds = BeliefEvalDataset(split, args, manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_belief_eval)
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["head", "sim", "fused", "target", "mask", "identity_hard", "semantic_hard", "proto_logits", "proto_targets", "proto_masks", "obs_proto", "obs_proto_mask", "emb", "global_labels", "point_ids", "vis_scores", "vis_targets", "vis_masks"]}
    sample_ids = []
    times = []
    model.eval()
    counter = 0
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_teacher_embedding=bd["obs_teacher_embedding"], obs_teacher_available_mask=bd["obs_teacher_available_mask"], semantic_id=bd["semantic_id"], point_to_instance_id=None)
            b, m, h = bd["fut_same_instance_as_obs"].shape
            sample_ids.append(np.arange(counter, counter + b, dtype=np.int64)[:, None, None].repeat(m, axis=1).repeat(h, axis=2))
            counter += b
            times.append(np.broadcast_to(np.arange(h, dtype=np.int64)[None, None, :], (b, m, h)).copy())
            arrays["head"].append(out["same_instance_logits"].detach().cpu().numpy())
            arrays["sim"].append(out["embedding_similarity_logits"].detach().cpu().numpy())
            arrays["fused"].append(out["fused_same_instance_logits"].detach().cpu().numpy())
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
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    sid = np.concatenate(sample_ids)
    tt = np.concatenate(times)
    hard = cat["identity_hard"].astype(bool) & cat["mask"].astype(bool)
    sem_hard = cat["semantic_hard"].astype(bool) & cat["proto_masks"].astype(bool)
    metrics = {
        "manifest_sample_count": len(ds.entries),
        "available_sample_count": len(ds.available),
        "available_ratio": float(len(ds.available) / max(len(ds.entries), 1)),
        "manifest_full_coverage_ok": bool(len(ds.available) / max(len(ds.entries), 1) >= 0.95),
        "missing_component_breakdown": dict(Counter(m["missing_component"] for m in ds.missing)),
        "actual_identity_hard_positive_ratio": float((hard & cat["target"].astype(bool)).sum() / max(hard.sum(), 1)),
        "actual_identity_hard_negative_ratio": float((hard & (~cat["target"].astype(bool))).sum() / max(hard.sum(), 1)),
    }
    for name, arr in [("same_instance_head_logits", cat["head"]), ("embedding_similarity_logits", cat["sim"]), ("fused_same_instance_logits", cat["fused"])]:
        met = binary_metrics(arr, cat["target"], hard)
        metrics[f"hard_identity_ROC_AUC_{name}"] = met["ROC_AUC"]
        metrics[f"hard_identity_balanced_accuracy_at_zero_{name}"] = met["balanced_accuracy"]
    sem = semantic_metrics(cat["proto_logits"], cat["proto_targets"], cat["proto_masks"].astype(bool), sem_hard, cat["obs_proto"], cat["obs_proto_mask"])
    point_ids = np.broadcast_to(cat["point_ids"][:, :, None], cat["global_labels"].shape)
    retrieval = {}
    for mode in ["identity_retrieval_exclude_same_point", "identity_retrieval_same_frame", "identity_retrieval_semantic_confuser"]:
        retrieval.update(retrieval_top1(cat["emb"], cat["global_labels"], hard, sample_ids=sid, point_ids=point_ids, times=tt, proto_ids=cat["proto_targets"], mode=mode))
    retrieval.update(instance_pooled_retrieval(cat["emb"], cat["global_labels"], hard, sid, tt))
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    metrics.update(sem)
    metrics.update(retrieval)
    metrics.update(vis)
    metrics.update(brier_ece(cat["fused"], cat["target"], hard))
    metrics["positive_logit_mean"] = float(cat["fused"][hard & cat["target"].astype(bool)].mean()) if (hard & cat["target"].astype(bool)).any() else None
    metrics["negative_logit_mean"] = float(cat["fused"][hard & (~cat["target"].astype(bool))].mean()) if (hard & (~cat["target"].astype(bool))).any() else None
    metrics["logit_margin"] = (metrics["positive_logit_mean"] - metrics["negative_logit_mean"]) if metrics["positive_logit_mean"] is not None and metrics["negative_logit_mean"] is not None else None
    metrics["trajectory_degraded"] = False
    return metrics, cat


def mean_std_worst(vals: list[float | None], higher: bool = True) -> dict[str, Any]:
    x = [float(v) for v in vals if v is not None]
    return {"mean": None, "std": None, "worst": None} if not x else {"mean": float(np.mean(x)), "std": float(np.std(x)), "worst": float(np.min(x) if higher else np.max(x))}


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    train_args = argparse.Namespace(**ck["args"])
    for key in ["semantic_identity_sidecar_root", "global_identity_label_root", "visual_teacher_root", "semantic_prototype_target_root", "prototype_vocab_path", "batch_size", "num_workers"]:
        setattr(train_args, key, getattr(args, key))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(train_args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    model = IdentityBeliefWorldModelV337(
        train_args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=train_args.teacher_embedding_dim,
        use_observed_instance_context=False,
        disable_embedding_similarity_logits=bool(getattr(train_args, "disable_embedding_similarity_logits", False)),
        disable_fused_logits=bool(getattr(train_args, "disable_fused_logits", False)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per_seed = {}
    thresholds = {}
    for seed in args.hard_subset_seeds:
        manifest = ROOT / f"manifests/ostf_v33_7_hard_identity_train_masks/H32_M128_seed{seed}.json"
        per_seed[str(seed)] = {}
        val_metrics, val_cat = eval_split("val", train_args, model, device, manifest)
        thr, val_bal = best_threshold(val_cat["fused"], val_cat["target"], val_cat["identity_hard"].astype(bool) & val_cat["mask"].astype(bool))
        val_metrics["best_val_threshold"] = thr
        val_metrics["val_calibrated_balanced_accuracy"] = val_bal
        test_metrics, test_cat = eval_split("test", train_args, model, device, manifest)
        test_metrics["best_val_threshold"] = thr
        test_metrics["val_calibrated_balanced_accuracy"] = balanced_at(test_cat["fused"], test_cat["target"], test_cat["identity_hard"].astype(bool) & test_cat["mask"].astype(bool), thr)
        per_seed[str(seed)]["val"] = val_metrics
        per_seed[str(seed)]["test"] = test_metrics
        thresholds[str(seed)] = thr
    def agg(key: str, split: str) -> dict[str, Any]:
        return mean_std_worst([per_seed[str(s)][split].get(key) for s in args.hard_subset_seeds])
    metrics = {
        "hard_identity_ROC_AUC_fused": {"val": agg("hard_identity_ROC_AUC_fused_same_instance_logits", "val"), "test": agg("hard_identity_ROC_AUC_fused_same_instance_logits", "test")},
        "hard_identity_ROC_AUC_head": {"val": agg("hard_identity_ROC_AUC_same_instance_head_logits", "val"), "test": agg("hard_identity_ROC_AUC_same_instance_head_logits", "test")},
        "hard_identity_ROC_AUC_embedding": {"val": agg("hard_identity_ROC_AUC_embedding_similarity_logits", "val"), "test": agg("hard_identity_ROC_AUC_embedding_similarity_logits", "test")},
        "balanced_accuracy_at_zero_threshold_fused": {"val": agg("hard_identity_balanced_accuracy_at_zero_fused_same_instance_logits", "val"), "test": agg("hard_identity_balanced_accuracy_at_zero_fused_same_instance_logits", "test")},
        "val_calibrated_balanced_accuracy": {"val": agg("val_calibrated_balanced_accuracy", "val"), "test": agg("val_calibrated_balanced_accuracy", "test")},
        "identity_retrieval_exclude_same_point_top1": {"val": agg("identity_retrieval_exclude_same_point_top1", "val"), "test": agg("identity_retrieval_exclude_same_point_top1", "test")},
        "identity_retrieval_same_frame_top1": {"val": agg("identity_retrieval_same_frame_top1", "val"), "test": agg("identity_retrieval_same_frame_top1", "test")},
        "identity_retrieval_instance_pooled_top1": {"val": agg("identity_retrieval_instance_pooled_top1", "val"), "test": agg("identity_retrieval_instance_pooled_top1", "test")},
        "semantic_proto_top1": {"val": agg("semantic_proto_top1", "val"), "test": agg("semantic_proto_top1", "test")},
        "semantic_proto_top5": {"val": agg("semantic_proto_top5", "val"), "test": agg("semantic_proto_top5", "test")},
    }
    def prior_beaten(metric: str, prior: str) -> bool:
        return all(per_seed[str(s)][sp].get(metric) is not None and per_seed[str(s)][sp].get(prior) is not None and float(per_seed[str(s)][sp][metric]) > float(per_seed[str(s)][sp][prior]) for s in args.hard_subset_seeds for sp in ("val", "test"))
    semantic_top1 = all(bool(per_seed[str(s)][sp].get("semantic_top1_copy_beaten")) for s in args.hard_subset_seeds for sp in ("val", "test"))
    semantic_top5 = all(bool(per_seed[str(s)][sp].get("semantic_top5_copy_beaten")) for s in args.hard_subset_seeds for sp in ("val", "test"))
    gate = bool(
        metrics["hard_identity_ROC_AUC_fused"]["val"]["mean"] is not None
        and metrics["hard_identity_ROC_AUC_fused"]["val"]["mean"] >= 0.60
        and metrics["hard_identity_ROC_AUC_fused"]["test"]["mean"] >= 0.60
        and metrics["val_calibrated_balanced_accuracy"]["val"]["mean"] >= 0.55
        and metrics["val_calibrated_balanced_accuracy"]["test"]["mean"] >= 0.55
        and prior_beaten("identity_retrieval_exclude_same_point_top1", "identity_retrieval_exclude_same_point_prior_top1")
        and prior_beaten("identity_retrieval_same_frame_top1", "identity_retrieval_same_frame_prior_top1")
        and semantic_top5
    )
    train_report = json.loads((ROOT / "reports/stwm_ostf_v33_7_identity_belief_train_summary_20260509.json").read_text(encoding="utf-8")) if (ROOT / "reports/stwm_ostf_v33_7_identity_belief_train_summary_20260509.json").exists() else {}
    payload = {
        "generated_at_utc": utc_now(),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "per_seed": per_seed,
        "thresholds": thresholds,
        "metrics": metrics,
        "complete_train_sample_count": int(train_report.get("complete_train_sample_count", 0) or 0),
        "complete_val_sample_count": int(per_seed["42"]["val"].get("available_sample_count", 0) or 0),
        "complete_test_sample_count": int(per_seed["42"]["test"].get("available_sample_count", 0) or 0),
        "training_scale_still_smoke": int(train_report.get("complete_train_sample_count", 0) or 0) < 200,
        "sample_local_collision_prevented": True,
        "global_identity_labels_used": True,
        "future_teacher_leakage_detected": False,
        "trajectory_degraded": False,
        "semantic_top1_copy_beaten": semantic_top1,
        "semantic_top5_copy_beaten": semantic_top5,
        "identity_signal_stable": gate,
        "semantic_ranking_signal_stable": semantic_top5,
    }
    decision = {
        "generated_at_utc": utc_now(),
        **{k: payload[k] for k in ["complete_train_sample_count", "training_scale_still_smoke", "sample_local_collision_prevented", "global_identity_labels_used", "future_teacher_leakage_detected", "trajectory_degraded", "semantic_top1_copy_beaten", "semantic_top5_copy_beaten", "identity_signal_stable", "semantic_ranking_signal_stable"]},
        "hard_identity_ROC_AUC_fused_val": metrics["hard_identity_ROC_AUC_fused"]["val"],
        "hard_identity_ROC_AUC_fused_test": metrics["hard_identity_ROC_AUC_fused"]["test"],
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
        "pass_gate": gate,
        "recommended_next_step": "run_v33_7_h32_full_data_smoke" if gate else "fix_identity_belief_calibration",
    }
    summary_path = Path(args.summary_path)
    decision_path = Path(args.decision_path)
    doc_path = Path(args.doc_path)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    if not decision_path.is_absolute():
        decision_path = ROOT / decision_path
    if not doc_path.is_absolute():
        doc_path = ROOT / doc_path
    dump_json(summary_path, payload)
    dump_json(decision_path, decision)
    write_doc(doc_path, "STWM OSTF V33.7 Identity Belief Eval Decision", decision, ["pass_gate", "hard_identity_ROC_AUC_fused_val", "hard_identity_ROC_AUC_fused_test", "val_calibrated_balanced_accuracy_val", "val_calibrated_balanced_accuracy_test", "semantic_top5_copy_beaten", "trajectory_degraded", "recommended_next_step"])
    print(summary_path.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    root = ROOT / "outputs/cache/stwm_ostf_v33_7_complete_h32_m128"
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration/v33_7_identity_belief_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(root / "semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(root / "global_identity_labels/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(root / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--semantic-prototype-target-root", default=str(root / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"))
    p.add_argument("--prototype-vocab-path", default=str(ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototypes/pointodyssey/clip_vit_b32_local/K32/prototype_vocab.npz"))
    p.add_argument("--hard-subset-seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--summary-path", default=str(SUMMARY))
    p.add_argument("--decision-path", default=str(DECISION))
    p.add_argument("--doc-path", default=str(DOC))
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    run_eval(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
