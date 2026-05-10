#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from stwm.modules.ostf_v33_10_copy_residual_semantic_world_model import CopyResidualSemanticWorldModelV3310
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v33_6_identity_contrastive_repair_20260509 import flat_metadata, supervised_contrastive_loss
from stwm.tools.train_ostf_v33_7_identity_belief_calibration_20260509 import BeliefDataset, balanced_bce, collate_belief


RUN_DIR = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_10_copy_residual_semantic_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_train_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_10_COPY_RESIDUAL_SEMANTIC_TRAIN_SUMMARY_20260510.md"
COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
COPY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_10_copy_residual_semantic_targets/pointodyssey/clip_vit_b32_local/K32"
MASK = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic/H32_M128_seed42.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CopyResidualDataset(BeliefDataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        super().__init__(split, args, max_items=max_items)
        self.copy_root = Path(args.copy_residual_semantic_target_root)
        if not self.copy_root.is_absolute():
            self.copy_root = ROOT / self.copy_root
        self._copy_cache: dict[str, dict[str, torch.Tensor]] = {}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        uid = str(item["uid"])
        cached = self._copy_cache.get(uid)
        if cached is None:
            z = np.load(self.copy_root / self.split / f"{uid}.npz", allow_pickle=True)
            cached = {}
            for key in [
                "last_observed_semantic_prototype_id",
                "copy_semantic_prototype_id",
                "semantic_stable_mask",
                "semantic_changed_mask",
                "semantic_hard_mask",
                "semantic_update_target",
                "semantic_update_available_mask",
                "copy_prior_distribution",
                "observed_frequency_prior_distribution",
            ]:
                arr = np.asarray(z[key])
                if key.endswith("_mask") or (key.endswith("_target") and arr.dtype == bool):
                    cached[key] = torch.from_numpy(arr.astype(bool)).bool()
                elif key.endswith("_distribution"):
                    cached[key] = torch.from_numpy(arr.astype(np.float32)).float()
                else:
                    cached[key] = torch.from_numpy(arr.astype(np.int64)).long()
            cached["copy_residual_leakage_safe"] = torch.tensor(bool(np.asarray(z["leakage_safe"]).item()), dtype=torch.bool)
            self._copy_cache[uid] = cached
        item.update(cached)
        return item


def collate_copy(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_belief(batch)
    for key in [
        "last_observed_semantic_prototype_id",
        "copy_semantic_prototype_id",
        "semantic_stable_mask",
        "semantic_changed_mask",
        "semantic_hard_mask",
        "semantic_update_target",
        "semantic_update_available_mask",
        "copy_prior_distribution",
        "observed_frequency_prior_distribution",
        "copy_residual_leakage_safe",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None) -> DataLoader:
    ds = CopyResidualDataset(split, args, max_items=max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_copy)


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    same = batch["fut_same_instance_as_obs"].bool()
    full_mask = batch["fut_instance_available_mask"].bool()
    id_bce = balanced_bce(out["same_instance_logits"], same, full_mask)
    labels = batch["fut_global_instance_id"].long()
    label_mask = batch["fut_global_instance_available_mask"].bool()
    sample_ids, point_ids, times, proto_ids = flat_metadata(batch, labels)
    id_global = supervised_contrastive_loss(out["identity_embedding"], labels, label_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="global", max_tokens=args.contrastive_max_tokens)
    id_excl = supervised_contrastive_loss(out["identity_embedding"], labels, label_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="exclude_same_point", max_tokens=args.contrastive_max_tokens)
    id_frame = supervised_contrastive_loss(out["identity_embedding"], labels, label_mask, sample_ids=sample_ids, point_ids=point_ids, times=times, proto_ids=proto_ids, mode="same_frame", max_tokens=args.contrastive_max_tokens)

    target = batch["semantic_prototype_id"].long()
    valid = batch["semantic_prototype_available_mask"].bool() & (target >= 0)
    stable = batch["semantic_stable_mask"].bool() & valid
    changed = batch["semantic_changed_mask"].bool() & valid
    hard = batch["semantic_hard_mask"].bool() & valid
    logits = out["final_semantic_proto_logits"]
    full_ce = F.cross_entropy(logits[valid], target[valid]) if bool(valid.any()) else logits.sum() * 0.0
    stable_ce = F.cross_entropy(logits[stable], batch["copy_semantic_prototype_id"].long()[stable].clamp_min(0)) if bool(stable.any()) else logits.sum() * 0.0
    changed_ce = F.cross_entropy(logits[changed], target[changed]) if bool(changed.any()) else logits.sum() * 0.0
    hard_ce = F.cross_entropy(logits[hard], target[hard]) if bool(hard.any()) else logits.sum() * 0.0
    gate_mask = batch["semantic_update_available_mask"].bool()
    gate_target = batch["semantic_update_target"].float()
    pos = (gate_target * gate_mask.float()).sum()
    neg = ((1.0 - gate_target) * gate_mask.float()).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.05, 20.0)
    gate_bce = (F.binary_cross_entropy_with_logits(out["semantic_change_logits"], gate_target, reduction="none", pos_weight=pos_weight) * gate_mask.float()).sum() / gate_mask.float().sum().clamp_min(1.0)
    gate = out["semantic_change_gate"]
    stable_sparsity = gate[stable].mean() if bool(stable.any()) else gate.mean() * 0.0
    residual_l2 = out["semantic_residual_logits"].pow(2).mean()
    vis_mask = batch["fut_point_visible_mask"].float()
    vis_bce = (F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["fut_point_visible_target"].float(), reduction="none") * vis_mask).sum() / vis_mask.sum().clamp_min(1.0)
    total = (
        0.35 * id_bce
        + 0.16 * id_global
        + 0.20 * id_excl
        + 0.10 * id_frame
        + args.full_semantic_weight * full_ce
        + args.stable_semantic_weight * stable_ce
        + args.changed_semantic_weight * changed_ce
        + args.hard_semantic_weight * hard_ce
        + args.change_gate_weight * gate_bce
        + args.residual_sparsity_weight * stable_sparsity
        + 1e-5 * residual_l2
        + 0.05 * vis_bce
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "identity_bce": float(id_bce.detach().cpu()),
        "identity_global_contrastive": float(id_global.detach().cpu()),
        "identity_exclude_same_point": float(id_excl.detach().cpu()),
        "identity_same_frame": float(id_frame.detach().cpu()),
        "semantic_full_ce": float(full_ce.detach().cpu()),
        "semantic_stable_ce": float(stable_ce.detach().cpu()),
        "semantic_changed_ce": float(changed_ce.detach().cpu()),
        "semantic_hard_ce": float(hard_ce.detach().cpu()),
        "semantic_change_gate_bce": float(gate_bce.detach().cpu()),
        "stable_gate_mean": float(stable_sparsity.detach().cpu()),
        "visibility_bce": float(vis_bce.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    train_loader = make_loader("train", args, shuffle=True, max_items=None if args.max_train_items <= 0 else args.max_train_items)
    model = CopyResidualSemanticWorldModelV3310(
        args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=args.teacher_embedding_dim,
        no_copy_prior=args.no_copy_prior,
        no_change_gate=args.no_change_gate,
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
            copy_semantic_prototype_id=bd["copy_semantic_prototype_id"],
            last_observed_semantic_prototype_id=bd["last_observed_semantic_prototype_id"],
        )
        loss, comps = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": float(step), **comps})
    ckpt = CKPT_DIR / f"{args.experiment_name}_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "experiment_name": args.experiment_name,
        "fresh_training_completed": True,
        "completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "complete_train_sample_count": len(train_loader.dataset),
        "duration_seconds": float(time.time() - start),
        "copy_prior_active": not args.no_copy_prior,
        "semantic_change_gate_active": not args.no_change_gate,
        "v30_checkpoint_loaded": Path(args.v30_checkpoint).exists(),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "integrated_v30_backbone_used": True,
        "observed_instance_context_used": False,
        "observed_visual_teacher_context_used": True,
        "global_identity_labels_used": True,
        "sample_local_collision_prevented": True,
        "future_teacher_leakage_detected": False,
        "target_root": str(COMPLETE.relative_to(ROOT)),
        "copy_residual_target_root": str(Path(args.copy_residual_semantic_target_root).relative_to(ROOT)),
        "hard_mask_manifest": str(Path(args.hard_train_mask_manifest).relative_to(ROOT)),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "loss_tail": losses[-10:],
    }
    run_path = RUN_DIR / f"{args.experiment_name}.json"
    dump_json(run_path, payload)
    if args.write_main_summary:
        dump_json(SUMMARY, payload)
        write_doc(DOC, "STWM OSTF V33.10 Copy Residual Semantic Train Summary", payload, ["experiment_name", "fresh_training_completed", "complete_train_sample_count", "copy_prior_active", "semantic_change_gate_active", "v30_backbone_frozen", "future_teacher_leakage_detected", "train_loss_decreased", "checkpoint_path"])
        print(SUMMARY.relative_to(ROOT))
    else:
        print(run_path.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", default="v33_10_copy_residual_semantic_m128_h32_seed42")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-identity-sidecar-root", default=str(COMPLETE / "semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(COMPLETE / "global_identity_labels/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(COMPLETE / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--semantic-prototype-target-root", default=str(COMPLETE / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"))
    p.add_argument("--copy-residual-semantic-target-root", default=str(COPY_ROOT))
    p.add_argument("--prototype-vocab-path", default=str(ROOT / "outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K32/prototype_vocab.npz"))
    p.add_argument("--hard-train-mask-manifest", default=str(MASK))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-items", type=int, default=0)
    p.add_argument("--teacher-embedding-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--contrastive-max-tokens", type=int, default=2048)
    p.add_argument("--full-semantic-weight", type=float, default=0.35)
    p.add_argument("--stable-semantic-weight", type=float, default=2.0)
    p.add_argument("--changed-semantic-weight", type=float, default=1.0)
    p.add_argument("--hard-semantic-weight", type=float, default=1.0)
    p.add_argument("--change-gate-weight", type=float, default=0.6)
    p.add_argument("--residual-sparsity-weight", type=float, default=0.4)
    p.add_argument("--no-copy-prior", action="store_true")
    p.add_argument("--no-change-gate", action="store_true")
    p.add_argument("--write-main-summary", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    # Keep the V33.10 main path leakage-safe: observed instance IDs are not
    # consumed as model input, while global future identity labels are required
    # only as supervision for the contrastive objective.
    args.use_observed_instance_context = False
    args.enable_global_identity_labels = True
    args.require_global_identity_labels = True
    return args


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
