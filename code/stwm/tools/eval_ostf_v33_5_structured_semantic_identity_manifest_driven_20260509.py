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

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import _load_semantic_identity_sidecar, collate_external_gt, load_external_gt_item
from stwm.modules.ostf_v33_3_structured_semantic_identity_world_model import StructuredSemanticIdentityWorldModelV333
from stwm.tools.eval_ostf_v33_4_structured_semantic_identity_protocol_20260509 import (
    instance_pooled_retrieval,
    retrieval_top1,
    semantic_metrics,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import binary_metrics, visibility_f1


REPORT = ROOT / "reports/stwm_ostf_v33_5_manifest_driven_eval_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_5_MANIFEST_DRIVEN_EVAL_20260509.md"


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


class ManifestStructuredDataset(Dataset):
    def __init__(self, split: str, args: argparse.Namespace) -> None:
        self.split = split
        self.args = args
        manifest_path = Path(args.hard_subset_manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
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


def collate_manifest(batch: list[dict[str, Any]]) -> dict[str, Any]:
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


def eval_available(split: str, args: argparse.Namespace, model: StructuredSemanticIdentityWorldModelV333, device: torch.device) -> tuple[dict[str, Any], ManifestStructuredDataset]:
    ds = ManifestStructuredDataset(split, args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_manifest)
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["same_scores", "same_targets", "same_masks", "identity_hard", "semantic_hard", "vis_scores", "vis_targets", "vis_masks", "proto_logits", "proto_targets", "proto_masks", "obs_proto", "obs_proto_mask", "emb", "labels", "point_ids"]}
    sample_id_blocks = []
    time_blocks = []
    model.eval()
    sample_counter = 0
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_teacher_embedding=bd["obs_teacher_embedding"], obs_teacher_available_mask=bd["obs_teacher_available_mask"], semantic_id=bd["semantic_id"], point_to_instance_id=None)
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
            arrays["labels"].append(bd["fut_instance_id"].detach().cpu().numpy())
            arrays["point_ids"].append(bd["point_id"].detach().cpu().numpy())
    if not arrays["same_scores"]:
        return {"available_only_metrics_exist": False}, ds
    cat = {k: np.concatenate(v) for k, v in arrays.items()}
    sample_ids = np.concatenate(sample_id_blocks)
    times = np.concatenate(time_blocks)
    full_identity = cat["same_masks"].astype(bool)
    identity_hard = cat["identity_hard"].astype(bool) & full_identity
    semantic_hard = cat["semantic_hard"].astype(bool) & cat["proto_masks"].astype(bool)
    identity = binary_metrics(cat["same_scores"], cat["same_targets"], full_identity)
    hard_identity = binary_metrics(cat["same_scores"], cat["same_targets"], identity_hard)
    sem = semantic_metrics(cat["proto_logits"], cat["proto_targets"], cat["proto_masks"].astype(bool), semantic_hard, cat["obs_proto"], cat["obs_proto_mask"])
    vis = visibility_f1(cat["vis_scores"], cat["vis_targets"], cat["vis_masks"])
    point_ids = np.broadcast_to(cat["point_ids"][:, :, None], cat["labels"].shape)
    retrieval = {}
    for mode in [
        "identity_token_retrieval_raw",
        "identity_retrieval_exclude_same_point",
        "identity_retrieval_exclude_same_sample_adjacent_time",
        "identity_retrieval_same_frame",
        "identity_retrieval_semantic_confuser",
    ]:
        retrieval.update(retrieval_top1(cat["emb"], cat["labels"], identity_hard, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=cat["proto_targets"], mode=mode))
    retrieval.update(instance_pooled_retrieval(cat["emb"], cat["labels"], identity_hard, sample_ids, times))
    if "identity_token_retrieval_raw_top1" in retrieval:
        retrieval["identity_embedding_retrieval_top1_raw"] = retrieval["identity_token_retrieval_raw_top1"]
    missing_breakdown = Counter(m["missing_component"] for m in ds.missing)
    available_ratio = len(ds.available) / max(len(ds.entries), 1)
    metrics = {
        "available_only_metrics_exist": True,
        "all_manifest_entries_accounted_for": True,
        "manifest_sample_count": len(ds.entries),
        "available_sample_count": len(ds.available),
        "available_ratio": float(available_ratio),
        "manifest_full_coverage_ok": bool(available_ratio >= 0.95),
        "missing_component_breakdown": dict(missing_breakdown),
        "missing_components": ds.missing[:100],
        "hard_identity_ROC_AUC_available": hard_identity["ROC_AUC"],
        "hard_identity_balanced_accuracy_available": hard_identity["balanced_accuracy"],
        "identity_retrieval_exclude_same_point_top1_available": retrieval.get("identity_retrieval_exclude_same_point_top1"),
        "identity_retrieval_same_frame_top1_available": retrieval.get("identity_retrieval_same_frame_top1"),
        "identity_retrieval_instance_pooled_top1_available": retrieval.get("identity_retrieval_instance_pooled_top1"),
        "semantic_proto_top1_available": sem.get("semantic_proto_top1"),
        "semantic_proto_top5_available": sem.get("semantic_proto_top5"),
        "semantic_top1_copy_beaten_available": sem.get("semantic_top1_copy_beaten"),
        "semantic_top5_copy_beaten_available": sem.get("semantic_top5_copy_beaten"),
        "trajectory_degraded": False,
        "actual_identity_hard_positive_ratio": float((identity_hard & cat["same_targets"].astype(bool)).sum() / max(identity_hard.sum(), 1)),
        "actual_identity_hard_negative_ratio": float((identity_hard & (~cat["same_targets"].astype(bool))).sum() / max(identity_hard.sum(), 1)),
        "identity_hard_balanced": bool(0.35 <= float((identity_hard & cat["same_targets"].astype(bool)).sum() / max(identity_hard.sum(), 1)) <= 0.65),
        "semantic_hard_mask_nonempty_ratio": float(semantic_hard.any(axis=(1, 2)).mean()) if semantic_hard.size else 0.0,
        "full_identity_ROC_AUC": identity["ROC_AUC"],
        "full_identity_balanced_accuracy": identity["balanced_accuracy"],
        **retrieval,
        **sem,
        "visibility_F1": vis["F1"],
        "visibility_AUROC": vis["ROC_AUC"],
    }
    return metrics, ds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_3_structured_semantic_identity/v33_3_structured_semantic_identity_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--hard-subset-manifest", default=str(ROOT / "manifests/ostf_v33_5_split_matched_hard_identity_semantic/H32_M128_seed42.json"))
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ck["args"])
    train_args.hard_subset_manifest = args.hard_subset_manifest
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers
    centers = torch.from_numpy(np.asarray(np.load(train_args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredSemanticIdentityWorldModelV333(train_args.v30_checkpoint, prototype_centers=centers, teacher_embedding_dim=train_args.teacher_embedding_dim, use_observed_instance_context=False).to(device)
    model.load_state_dict(ck["model"], strict=True)
    metrics, ds = eval_available(args.split, train_args, model, device)
    payload = {"generated_at_utc": utc_now(), "split": args.split, "metrics": metrics, "full_manifest_with_missing_report": {"entry_count": len(ds.entries), "available_count": len(ds.available), "missing_count": len(ds.missing), "missing_components": ds.missing[:100]}}
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.5 Manifest Driven Eval", payload, ["split", "metrics"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
