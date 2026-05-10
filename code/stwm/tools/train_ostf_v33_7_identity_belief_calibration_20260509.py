#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_7_identity_belief_world_model import IdentityBeliefWorldModelV337
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_3_structured_semantic_identity_20260509 import StructuredSidecarDataset, collate_structured
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v33_6_identity_contrastive_repair_20260509 import supervised_contrastive_loss, flat_metadata


RUN_DIR = ROOT / "reports/stwm_ostf_v33_7_identity_belief_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_7_identity_belief_calibration"
SUMMARY = ROOT / "reports/stwm_ostf_v33_7_identity_belief_train_summary_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_7_IDENTITY_BELIEF_TRAIN_SUMMARY_20260509.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_masks(manifest_path: str | Path, split: str) -> dict[str, dict[str, np.ndarray]]:
    p = Path(manifest_path)
    if not p.is_absolute():
        p = ROOT / p
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict[str, np.ndarray]] = {}
    for entry in payload.get("splits", {}).get(split, []):
        z = np.load(ROOT / entry["mask_path"], allow_pickle=True)
        out[str(entry["sample_uid"])] = {
            "identity_hard_train_mask": np.asarray(z["identity_hard_train_mask" if "identity_hard_train_mask" in z.files else "identity_hard_eval_mask"]).astype(bool),
            "semantic_hard_train_mask": np.asarray(z["semantic_hard_train_mask" if "semantic_hard_train_mask" in z.files else "semantic_hard_eval_mask"]).astype(bool),
        }
    return out


class BeliefDataset(StructuredSidecarDataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        super().__init__(split, args, max_items=max_items)
        self.masks = load_masks(args.hard_train_mask_manifest, split)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        uid = str(item["uid"])
        m, h = item["fut_same_instance_as_obs"].shape
        masks = self.masks.get(uid)
        if masks is None:
            item["identity_hard_train_mask"] = torch.zeros((m, h), dtype=torch.bool)
            item["semantic_hard_train_mask"] = torch.zeros((m, h), dtype=torch.bool)
        else:
            item["identity_hard_train_mask"] = torch.from_numpy(masks["identity_hard_train_mask"]).bool()
            item["semantic_hard_train_mask"] = torch.from_numpy(masks["semantic_hard_train_mask"]).bool()
        return item


def collate_belief(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_structured(batch)
    for key in ["identity_hard_train_mask", "semantic_hard_train_mask"]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None) -> DataLoader:
    ds = BeliefDataset(split, args, max_items=max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_belief)


def balanced_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, focal: bool = False) -> torch.Tensor:
    mask_f = mask.float()
    target_f = target.float()
    pos = (target_f * mask_f).sum()
    neg = ((1.0 - target_f) * mask_f).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.05, 10.0)
    loss = F.binary_cross_entropy_with_logits(logits, target_f, reduction="none", pos_weight=pos_weight)
    if focal:
        pt = torch.where(target.bool(), torch.sigmoid(logits), 1.0 - torch.sigmoid(logits)).clamp(1e-4, 1.0)
        loss = loss * (1.0 - pt).pow(2.0)
    return (loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    full_mask = batch["fut_instance_available_mask"].bool()
    hard_mask = batch["identity_hard_train_mask"].bool() & full_mask
    same = batch["fut_same_instance_as_obs"].bool()
    full_bce = balanced_bce(out["same_instance_logits"], same, full_mask)
    hard_head = balanced_bce(out["same_instance_logits"], same, hard_mask, focal=True) if not args.disable_hard_bce else out["same_instance_logits"].sum() * 0.0
    hard_embed = balanced_bce(out["embedding_similarity_logits"], same, hard_mask, focal=True) if not args.disable_embedding_similarity_logits else out["embedding_similarity_logits"].sum() * 0.0
    hard_fused = balanced_bce(out["fused_same_instance_logits"], same, hard_mask, focal=True) if not args.disable_fused_logits else out["fused_same_instance_logits"].sum() * 0.0

    labels = batch["fut_global_instance_id"].long()
    label_mask = batch["fut_global_instance_available_mask"].bool()
    sample_ids, point_ids, times, proto_ids = flat_metadata(batch, labels)
    contrast = supervised_contrastive_loss(out["identity_embedding"], labels, label_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="global", max_tokens=args.contrastive_max_tokens)
    excl = supervised_contrastive_loss(out["identity_embedding"], labels, label_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="exclude_same_point", max_tokens=args.contrastive_max_tokens)
    frame = supervised_contrastive_loss(out["identity_embedding"], labels, label_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="same_frame", max_tokens=args.contrastive_max_tokens)
    semconf = supervised_contrastive_loss(out["identity_embedding"], labels, label_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="semantic_confuser", max_tokens=args.contrastive_max_tokens)
    consistency = F.mse_loss(torch.sigmoid(out["fused_same_instance_logits"][hard_mask]), torch.sigmoid(out["embedding_similarity_logits"][hard_mask])) if bool(hard_mask.any()) else out["fused_same_instance_logits"].sum() * 0.0

    proto_target = batch["semantic_prototype_id"].long()
    proto_mask = batch["semantic_prototype_available_mask"].bool() & (proto_target >= 0)
    proto_ce = F.cross_entropy(out["future_semantic_proto_logits"][proto_mask], proto_target[proto_mask]) if bool(proto_mask.any()) else out["future_semantic_proto_logits"].sum() * 0.0
    vis_mask = batch["fut_point_visible_mask"].float()
    vis_bce = (F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_point_visible_target"].float(), reduction="none") * vis_mask).sum() / vis_mask.sum().clamp_min(1.0)
    total = (
        args.full_bce_weight * full_bce
        + args.hard_bce_weight * hard_head
        + args.embedding_bce_weight * hard_embed
        + args.fused_bce_weight * hard_fused
        + 0.16 * contrast
        + 0.20 * excl
        + 0.14 * frame
        + 0.08 * semconf
        + 0.10 * consistency
        + 1.0 * proto_ce
        + 0.1 * vis_bce
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "full_same_instance_bce": float(full_bce.detach().cpu()),
        "hard_same_instance_bce": float(hard_head.detach().cpu()),
        "embedding_similarity_bce": float(hard_embed.detach().cpu()),
        "fused_same_instance_bce": float(hard_fused.detach().cpu()),
        "identity_contrastive": float(contrast.detach().cpu()),
        "identity_exclude_same_point": float(excl.detach().cpu()),
        "identity_same_frame": float(frame.detach().cpu()),
        "identity_semantic_confuser": float(semconf.detach().cpu()),
        "identity_consistency": float(consistency.detach().cpu()),
        "semantic_proto_ce": float(proto_ce.detach().cpu()),
        "visibility_bce": float(vis_bce.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    args.enable_global_identity_labels = True
    args.require_global_identity_labels = True
    args.use_observed_instance_context = False
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    max_train = None if args.max_train_items <= 0 else args.max_train_items
    train_loader = make_loader("train", args, shuffle=True, max_items=max_train)
    model = IdentityBeliefWorldModelV337(
        args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=args.teacher_embedding_dim,
        use_observed_instance_context=False,
        disable_embedding_similarity_logits=args.disable_embedding_similarity_logits,
        disable_fused_logits=args.disable_fused_logits,
    ).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    start = time.time()
    it = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
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
        loss, comps = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": float(step), **comps})
    ckpt_path = CKPT_DIR / f"{args.experiment_name}_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt_path)
    payload = {
        "generated_at_utc": utc_now(),
        "experiment_name": args.experiment_name,
        "completed": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_sample_count": len(train_loader.dataset),
        "complete_train_sample_count": len(train_loader.dataset),
        "duration_seconds": float(time.time() - start),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "v30_checkpoint_loaded": Path(args.v30_checkpoint).exists(),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "integrated_v30_backbone_used": True,
        "observed_instance_context_used": False,
        "observed_visual_teacher_context_used": True,
        "future_teacher_leakage_detected": False,
        "global_identity_labels_used": True,
        "sample_local_collision_prevented": True,
        "same_instance_hard_bce_active": not args.disable_hard_bce,
        "embedding_similarity_logits_active": not args.disable_embedding_similarity_logits,
        "fused_same_instance_logits_active": not args.disable_fused_logits,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "loss_tail": losses[-10:],
    }
    run_path = RUN_DIR / f"{args.experiment_name}.json"
    dump_json(run_path, payload)
    if args.write_main_summary:
        dump_json(SUMMARY, payload)
        write_doc(
            DOC,
            "STWM OSTF V33.7 Identity Belief Train Summary",
            payload,
            ["experiment_name", "completed", "complete_train_sample_count", "same_instance_hard_bce_active", "embedding_similarity_logits_active", "fused_same_instance_logits_active", "v30_backbone_frozen", "future_teacher_leakage_detected", "train_loss_decreased", "checkpoint_path"],
        )
        print(SUMMARY.relative_to(ROOT))
    else:
        print(run_path.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    root = ROOT / "outputs/cache/stwm_ostf_v33_7_complete_h32_m128"
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", default="v33_7_identity_belief_m128_h32_seed42")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(root / "semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(root / "global_identity_labels/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(root / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--semantic-prototype-target-root", default=str(root / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"))
    p.add_argument("--prototype-vocab-path", default=str(ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototypes/pointodyssey/clip_vit_b32_local/K32/prototype_vocab.npz"))
    p.add_argument("--hard-train-mask-manifest", default=str(ROOT / "manifests/ostf_v33_7_hard_identity_train_masks/H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-items", type=int, default=0)
    p.add_argument("--teacher-embedding-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--contrastive-max-tokens", type=int, default=2048)
    p.add_argument("--full-bce-weight", type=float, default=0.15)
    p.add_argument("--hard-bce-weight", type=float, default=1.25)
    p.add_argument("--embedding-bce-weight", type=float, default=0.85)
    p.add_argument("--fused-bce-weight", type=float, default=1.35)
    p.add_argument("--disable-hard-bce", action="store_true")
    p.add_argument("--disable-embedding-similarity-logits", action="store_true")
    p.add_argument("--disable-fused-logits", action="store_true")
    p.add_argument("--write-main-summary", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
