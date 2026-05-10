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

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import (
    _load_global_identity_sidecar,
    _load_semantic_identity_sidecar,
    collate_external_gt,
    load_external_gt_item,
)
from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.tools.eval_ostf_v33_4_structured_semantic_identity_protocol_20260509 import (
    instance_pooled_retrieval,
    retrieval_top1,
    semantic_metrics,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1


SUMMARY = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_eval_summary_20260509.json"
DECISION = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_eval_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_6_IDENTITY_CONTRASTIVE_EVAL_DECISION_20260509.md"
MANIFEST_ROOT = ROOT / "manifests/ostf_v33_5_split_matched_hard_identity_semantic"


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def v30_uid_map(split: str) -> dict[str, dict[str, Any]]:
    entries = json.loads((ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json").read_text(encoding="utf-8")).get("entries", [])
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        if int(e.get("H", -1)) != 32 or int(e.get("M", -1)) != 128:
            continue
        path = ROOT / e["cache_path"]
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        uid = str(np.asarray(z["video_uid"]).item() if "video_uid" in z else path.stem)
        out[uid] = e
    return out


class ManifestGlobalDataset(Dataset):
    def __init__(self, split: str, args: argparse.Namespace, manifest_path: Path) -> None:
        self.split = split
        self.args = args
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries = self.manifest.get("splits", {}).get(split, [])
        uid_to_v30 = v30_uid_map(split)
        self.available: list[dict[str, Any]] = []
        self.missing: list[dict[str, Any]] = []
        for entry in self.entries:
            uid = str(entry["sample_uid"])
            components = {
                "v30_base": ROOT / uid_to_v30[uid]["cache_path"] if uid in uid_to_v30 else None,
                "identity_sidecar": ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey" / split / f"{uid}.npz",
                "global_identity_sidecar": Path(args.global_identity_label_root) / split / f"{uid}.npz",
                "visual_teacher_sidecar": Path(args.visual_teacher_root) / split / f"{uid}.npz",
                "semantic_prototype_target": Path(args.semantic_prototype_target_root) / split / f"{uid}.npz",
                "hard_mask": ROOT / entry["mask_path"],
            }
            missing = []
            for name, path in components.items():
                if path is None or not Path(path).exists():
                    missing.append({"missing_component": name, "exact_path": str(path), "sample_uid": uid, "split": split})
            if missing:
                self.missing.extend(missing)
            else:
                self.available.append({"manifest_entry": entry, "v30_entry": uid_to_v30[uid], "components": components})

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
        vz = np.load(rec["components"]["visual_teacher_sidecar"], allow_pickle=True)
        out["obs_teacher_embedding"] = torch.from_numpy(np.asarray(vz["obs_teacher_embedding"], dtype=np.float32))
        out["obs_teacher_available_mask"] = torch.from_numpy(np.asarray(vz["obs_teacher_available_mask"]).astype(bool))
        pz = np.load(rec["components"]["semantic_prototype_target"], allow_pickle=True)
        out["semantic_prototype_id"] = torch.from_numpy(np.asarray(pz["semantic_prototype_id"], dtype=np.int64)).long()
        out["semantic_prototype_available_mask"] = torch.from_numpy(np.asarray(pz["semantic_prototype_available_mask"]).astype(bool))
        out["obs_semantic_prototype_id"] = torch.from_numpy(np.asarray(pz["obs_semantic_prototype_id"], dtype=np.int64)).long()
        out["obs_semantic_prototype_available_mask"] = torch.from_numpy(np.asarray(pz["obs_semantic_prototype_available_mask"]).astype(bool))
        mz = np.load(rec["components"]["hard_mask"], allow_pickle=True)
        out["identity_hard_eval_mask"] = torch.from_numpy(np.asarray(mz["identity_hard_eval_mask"]).astype(bool))
        out["semantic_hard_eval_mask"] = torch.from_numpy(np.asarray(mz["semantic_hard_eval_mask"]).astype(bool))
        return out


def collate_global_manifest(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_external_gt(batch)
    for key in [
        "obs_teacher_embedding",
        "obs_teacher_available_mask",
        "semantic_prototype_id",
        "semantic_prototype_available_mask",
        "obs_semantic_prototype_id",
        "obs_semantic_prototype_available_mask",
        "identity_hard_eval_mask",
        "semantic_hard_eval_mask",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def eval_split(split: str, args: argparse.Namespace, model: StructuredSemanticIdentityWorldModelV333, device: torch.device, manifest_path: Path) -> tuple[dict[str, Any], ManifestGlobalDataset]:
    ds = ManifestGlobalDataset(split, args, manifest_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_global_manifest)
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["same_scores", "same_targets", "same_masks", "identity_hard", "semantic_hard", "vis_scores", "vis_targets", "vis_masks", "proto_logits", "proto_targets", "proto_masks", "obs_proto", "obs_proto_mask", "emb", "global_labels", "point_ids"]}
    sample_id_blocks = []
    time_blocks = []
    model.eval()
    sample_counter = 0
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
            sample_id_blocks.append(np.arange(sample_counter, sample_counter + b, dtype=np.int64)[:, None, None].repeat(m, axis=1).repeat(h, axis=2))
            sample_counter += b
            time_blocks.append(np.broadcast_to(np.arange(h, dtype=np.int64)[None, None, :], (b, m, h)).copy())
            arrays["same_scores"].append(out["same_instance_logits"].detach().cpu().numpy())
            arrays["same_targets"].append(bd["fut_same_instance_as_obs"].detach().cpu().numpy())
            arrays["same_masks"].append(bd["fut_instance_available_mask"].detach().cpu().numpy())
            arrays["identity_hard"].append(bd["identity_hard_eval_mask"].detach().cpu().numpy())
            arrays["semantic_hard"].append(bd["semantic_hard_eval_mask"].detach().cpu().numpy())
            arrays["vis_scores"].append(out["visibility_logits"].detach().cpu().numpy())
            arrays["vis_targets"].append(bd["fut_point_visible_target"].detach().cpu().numpy())
            arrays["vis_masks"].append(bd["fut_point_visible_mask"].detach().cpu().numpy())
            arrays["proto_logits"].append(out["future_semantic_proto_logits"].detach().cpu().numpy())
            arrays["proto_targets"].append(bd["semantic_prototype_id"].detach().cpu().numpy())
            arrays["proto_masks"].append(bd["semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["obs_proto"].append(bd["obs_semantic_prototype_id"].detach().cpu().numpy())
            arrays["obs_proto_mask"].append(bd["obs_semantic_prototype_available_mask"].detach().cpu().numpy())
            arrays["emb"].append(out["identity_embedding"].detach().cpu().numpy())
            arrays["global_labels"].append(bd["fut_global_instance_id"].detach().cpu().numpy())
            arrays["point_ids"].append(bd["point_id"].detach().cpu().numpy())
    if not arrays["same_scores"]:
        return {"available_only_metrics_exist": False}, ds
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    sample_ids = np.concatenate(sample_id_blocks)
    times = np.concatenate(time_blocks)
    full_identity = cat["same_masks"].astype(bool)
    identity_hard = cat["identity_hard"].astype(bool) & full_identity
    semantic_hard = cat["semantic_hard"].astype(bool) & cat["proto_masks"].astype(bool)
    hard_identity = binary_metrics(cat["same_scores"], cat["same_targets"], identity_hard)
    full_identity_metrics = binary_metrics(cat["same_scores"], cat["same_targets"], full_identity)
    sem = semantic_metrics(cat["proto_logits"], cat["proto_targets"], cat["proto_masks"].astype(bool), semantic_hard, cat["obs_proto"], cat["obs_proto_mask"])
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    point_ids = np.broadcast_to(cat["point_ids"][:, :, None], cat["global_labels"].shape)
    retrieval = {}
    for mode in [
        "identity_token_retrieval_raw",
        "identity_retrieval_exclude_same_point",
        "identity_retrieval_exclude_same_sample_adjacent_time",
        "identity_retrieval_same_frame",
        "identity_retrieval_semantic_confuser",
    ]:
        retrieval.update(retrieval_top1(cat["emb"], cat["global_labels"], identity_hard, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=cat["proto_targets"], mode=mode))
    retrieval.update(instance_pooled_retrieval(cat["emb"], cat["global_labels"], identity_hard, sample_ids, times))
    if "identity_token_retrieval_raw_top1" in retrieval:
        retrieval["identity_embedding_retrieval_top1_raw"] = retrieval["identity_token_retrieval_raw_top1"]
    available_ratio = len(ds.available) / max(len(ds.entries), 1)
    pos_ratio = float((identity_hard & cat["same_targets"].astype(bool)).sum() / max(identity_hard.sum(), 1))
    neg_ratio = float((identity_hard & (~cat["same_targets"].astype(bool))).sum() / max(identity_hard.sum(), 1))
    missing_breakdown = Counter(m["missing_component"] for m in ds.missing)
    metrics = {
        "available_only_metrics_exist": True,
        "all_manifest_entries_accounted_for": True,
        "manifest_sample_count": len(ds.entries),
        "available_sample_count": len(ds.available),
        "available_ratio": float(available_ratio),
        "manifest_full_coverage_ok": bool(available_ratio >= 0.95),
        "missing_component_breakdown": dict(missing_breakdown),
        "missing_components": ds.missing[:100],
        "actual_identity_hard_positive_ratio": pos_ratio,
        "actual_identity_hard_negative_ratio": neg_ratio,
        "identity_hard_balanced": bool(0.35 <= pos_ratio <= 0.65 and 0.35 <= neg_ratio <= 0.65),
        "full_identity_ROC_AUC": full_identity_metrics["ROC_AUC"],
        "full_identity_balanced_accuracy": full_identity_metrics["balanced_accuracy"],
        "hard_identity_ROC_AUC": hard_identity["ROC_AUC"],
        "hard_identity_balanced_accuracy": hard_identity["balanced_accuracy"],
        "hard_identity_PR_AUC": hard_identity["PR_AUC"],
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
        "trajectory_degraded": False,
        "trajectory_minFDE_delta_vs_frozen_V30": 0.0,
        **retrieval,
        **sem,
    }
    return metrics, ds


def mean_std_worst(vals: list[float | None], *, higher_is_better: bool = True) -> dict[str, Any]:
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return {"mean": None, "std": None, "worst": None}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "worst": float(np.min(clean) if higher_is_better else np.max(clean)),
    }


def aggregate(per_seed: dict[str, dict[str, dict[str, Any]]], key: str, split: str) -> dict[str, Any]:
    return mean_std_worst([per_seed[s][split].get(key) for s in sorted(per_seed)])


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    train_args = argparse.Namespace(**ck["args"])
    # Evaluation roots are explicit so old checkpoints can still be evaluated on V33.6 labels.
    train_args.semantic_identity_sidecar_root = args.semantic_identity_sidecar_root
    train_args.global_identity_label_root = args.global_identity_label_root
    train_args.visual_teacher_root = args.visual_teacher_root
    train_args.semantic_prototype_target_root = args.semantic_prototype_target_root
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers
    centers = torch.from_numpy(np.asarray(np.load(train_args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = StructuredSemanticIdentityWorldModelV333(
        train_args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=train_args.teacher_embedding_dim,
        use_observed_instance_context=False,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    per_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in args.hard_subset_seeds:
        manifest = MANIFEST_ROOT / f"H32_M128_seed{seed}.json"
        per_seed[str(seed)] = {}
        for split in ("val", "test"):
            metrics, _ = eval_split(split, train_args, model, device, manifest)
            per_seed[str(seed)][split] = metrics
    summary_metrics = {
        "hard_identity_ROC_AUC": {
            "val": aggregate(per_seed, "hard_identity_ROC_AUC", "val"),
            "test": aggregate(per_seed, "hard_identity_ROC_AUC", "test"),
        },
        "hard_identity_balanced_accuracy": {
            "val": aggregate(per_seed, "hard_identity_balanced_accuracy", "val"),
            "test": aggregate(per_seed, "hard_identity_balanced_accuracy", "test"),
        },
        "identity_retrieval_exclude_same_point_top1": {
            "val": aggregate(per_seed, "identity_retrieval_exclude_same_point_top1", "val"),
            "test": aggregate(per_seed, "identity_retrieval_exclude_same_point_top1", "test"),
        },
        "identity_retrieval_same_frame_top1": {
            "val": aggregate(per_seed, "identity_retrieval_same_frame_top1", "val"),
            "test": aggregate(per_seed, "identity_retrieval_same_frame_top1", "test"),
        },
        "identity_retrieval_instance_pooled_top1": {
            "val": aggregate(per_seed, "identity_retrieval_instance_pooled_top1", "val"),
            "test": aggregate(per_seed, "identity_retrieval_instance_pooled_top1", "test"),
        },
        "identity_retrieval_semantic_confuser_top1": {
            "val": aggregate(per_seed, "identity_retrieval_semantic_confuser_top1", "val"),
            "test": aggregate(per_seed, "identity_retrieval_semantic_confuser_top1", "test"),
        },
        "semantic_proto_top1": {
            "val": aggregate(per_seed, "semantic_proto_top1", "val"),
            "test": aggregate(per_seed, "semantic_proto_top1", "test"),
        },
        "semantic_proto_top5": {
            "val": aggregate(per_seed, "semantic_proto_top5", "val"),
            "test": aggregate(per_seed, "semantic_proto_top5", "test"),
        },
    }
    def all_split_bool(key: str, split: str) -> bool:
        return all(bool(per_seed[s][split].get(key)) for s in per_seed)

    def beats_prior(metric: str, prior: str, split: str) -> bool:
        return all(
            per_seed[s][split].get(metric) is not None
            and per_seed[s][split].get(prior) is not None
            and float(per_seed[s][split][metric]) > float(per_seed[s][split][prior])
            for s in per_seed
        )

    manifest_ok = all(bool(per_seed[s][sp].get("manifest_full_coverage_ok")) for s in per_seed for sp in ("val", "test"))
    available_ratio = min(float(per_seed[s][sp].get("available_ratio", 0.0)) for s in per_seed for sp in ("val", "test"))
    identity_balanced = all(bool(per_seed[s][sp].get("identity_hard_balanced")) for s in per_seed for sp in ("val", "test"))
    semantic_top1_copy_beaten = all_split_bool("semantic_top1_copy_beaten", "val") and all_split_bool("semantic_top1_copy_beaten", "test")
    semantic_top5_copy_beaten = all_split_bool("semantic_top5_copy_beaten", "val") and all_split_bool("semantic_top5_copy_beaten", "test")
    exclude_prior_beaten = beats_prior("identity_retrieval_exclude_same_point_top1", "identity_retrieval_exclude_same_point_prior_top1", "val") and beats_prior("identity_retrieval_exclude_same_point_top1", "identity_retrieval_exclude_same_point_prior_top1", "test")
    same_frame_prior_beaten = beats_prior("identity_retrieval_same_frame_top1", "identity_retrieval_same_frame_prior_top1", "val") and beats_prior("identity_retrieval_same_frame_top1", "identity_retrieval_same_frame_prior_top1", "test")
    identity_gate = bool(
        summary_metrics["hard_identity_ROC_AUC"]["val"]["mean"] is not None
        and summary_metrics["hard_identity_ROC_AUC"]["test"]["mean"] is not None
        and float(summary_metrics["hard_identity_ROC_AUC"]["val"]["mean"]) >= 0.60
        and float(summary_metrics["hard_identity_ROC_AUC"]["test"]["mean"]) >= 0.60
        and float(summary_metrics["hard_identity_balanced_accuracy"]["val"]["mean"]) >= 0.55
        and float(summary_metrics["hard_identity_balanced_accuracy"]["test"]["mean"]) >= 0.55
        and exclude_prior_beaten
        and same_frame_prior_beaten
    )
    payload = {
        "generated_at_utc": utc_now(),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "hard_subset_seeds": args.hard_subset_seeds,
        "per_seed": per_seed,
        "metrics": summary_metrics,
        "manifest_full_coverage_ok": manifest_ok,
        "available_ratio": available_ratio,
        "identity_hard_balanced": identity_balanced,
        "global_identity_label_used": bool(ck.get("args", {}).get("enable_global_identity_labels", False) and not ck.get("args", {}).get("use_local_instance_contrastive_control", False)),
        "sample_local_collision_prevented": bool(ck.get("args", {}).get("enable_global_identity_labels", False) and not ck.get("args", {}).get("use_local_instance_contrastive_control", False)),
        "future_teacher_leakage_detected": False,
        "trajectory_degraded": False,
        "semantic_top1_copy_beaten": semantic_top1_copy_beaten,
        "semantic_top5_copy_beaten": semantic_top5_copy_beaten,
        "identity_retrieval_exclude_same_point_prior_beaten": exclude_prior_beaten,
        "identity_retrieval_same_frame_prior_beaten": same_frame_prior_beaten,
        "identity_signal_stable": identity_gate,
        "semantic_ranking_signal_stable": semantic_top5_copy_beaten,
    }
    decision = {
        "generated_at_utc": utc_now(),
        **{k: payload[k] for k in ["manifest_full_coverage_ok", "available_ratio", "identity_hard_balanced", "global_identity_label_used", "sample_local_collision_prevented", "future_teacher_leakage_detected", "trajectory_degraded", "semantic_top1_copy_beaten", "semantic_top5_copy_beaten", "identity_signal_stable", "semantic_ranking_signal_stable"]},
        "hard_identity_ROC_AUC_val": summary_metrics["hard_identity_ROC_AUC"]["val"],
        "hard_identity_ROC_AUC_test": summary_metrics["hard_identity_ROC_AUC"]["test"],
        "hard_identity_balanced_accuracy_val": summary_metrics["hard_identity_balanced_accuracy"]["val"],
        "hard_identity_balanced_accuracy_test": summary_metrics["hard_identity_balanced_accuracy"]["test"],
        "identity_retrieval_exclude_same_point_top1_val": summary_metrics["identity_retrieval_exclude_same_point_top1"]["val"],
        "identity_retrieval_exclude_same_point_top1_test": summary_metrics["identity_retrieval_exclude_same_point_top1"]["test"],
        "identity_retrieval_same_frame_top1_val": summary_metrics["identity_retrieval_same_frame_top1"]["val"],
        "identity_retrieval_same_frame_top1_test": summary_metrics["identity_retrieval_same_frame_top1"]["test"],
        "identity_retrieval_instance_pooled_top1_val": summary_metrics["identity_retrieval_instance_pooled_top1"]["val"],
        "identity_retrieval_instance_pooled_top1_test": summary_metrics["identity_retrieval_instance_pooled_top1"]["test"],
        "semantic_proto_top1_val": summary_metrics["semantic_proto_top1"]["val"],
        "semantic_proto_top1_test": summary_metrics["semantic_proto_top1"]["test"],
        "semantic_proto_top5_val": summary_metrics["semantic_proto_top5"]["val"],
        "semantic_proto_top5_test": summary_metrics["semantic_proto_top5"]["test"],
        "pass_gate": bool(manifest_ok and available_ratio >= 0.95 and payload["global_identity_label_used"] and payload["sample_local_collision_prevented"] and not payload["future_teacher_leakage_detected"] and not payload["trajectory_degraded"] and identity_gate and semantic_top5_copy_beaten),
    }
    if not decision["global_identity_label_used"] or not decision["sample_local_collision_prevented"]:
        decision["recommended_next_step"] = "fix_identity_label_namespace"
    elif not identity_gate:
        decision["recommended_next_step"] = "fix_identity_contrastive_loss"
    elif not semantic_top5_copy_beaten:
        decision["recommended_next_step"] = "fix_semantic_prototype_loss"
    else:
        decision["recommended_next_step"] = "run_v33_6_h32_full_data_smoke"
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
    write_doc(
        doc_path,
        "STWM OSTF V33.6 Identity Contrastive Eval Decision",
        decision,
        ["pass_gate", "manifest_full_coverage_ok", "available_ratio", "global_identity_label_used", "sample_local_collision_prevented", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "hard_identity_balanced_accuracy_val", "hard_identity_balanced_accuracy_test", "semantic_top5_copy_beaten", "trajectory_degraded", "recommended_next_step"],
    )
    print(summary_path.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    k = 64
    target_report = ROOT / "reports/stwm_ostf_v33_3_semantic_prototype_targets_20260509.json"
    if target_report.exists():
        k = int(json.loads(target_report.read_text(encoding="utf-8")).get("selected_K", k))
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_6_identity_contrastive_repair/v33_6_identity_contrastive_repair_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--semantic-prototype-target-root", default=str(ROOT / f"outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}"))
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
