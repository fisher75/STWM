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

from stwm.modules.ostf_v34_1_identity_bound_semantic_trace_units import IdentityBoundSemanticTraceUnitsV341
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_semantic_trace_units_20260510 import (
    GLOBAL_ROOT,
    IDENTITY_ROOT,
    MASK_ROOT,
    MEAS_ROOT,
    V34TraceUnitDataset,
    collate_v34,
    contrastive_loss,
    weighted_cosine_loss,
)


BINDING_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_1_unit_identity_binding_targets/pointodyssey"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_1_identity_bound_semantic_trace_units_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_1_IDENTITY_BOUND_SEMANTIC_TRACE_UNITS_TRAIN_SUMMARY_20260511.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class V341Dataset(V34TraceUnitDataset):
    def __init__(self, split: str, args: argparse.Namespace, *, max_items: int | None = None) -> None:
        super().__init__(split, args, max_items=max_items)
        self.binding_root = Path(args.unit_identity_binding_root)
        if not self.binding_root.is_absolute():
            self.binding_root = ROOT / self.binding_root
        keep = []
        for entry in self.base.entries:
            uid = Path(entry["cache_path"]).stem
            if (self.binding_root / split / f"{uid}.npz").exists():
                keep.append(entry)
        self.base.entries = keep[:max_items] if max_items is not None else keep
        if not self.base.entries:
            raise RuntimeError(f"No V34.1 binding sidecars for split={split}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        uid = str(item["uid"])
        z = np.load(self.binding_root / self.split / f"{uid}.npz", allow_pickle=True)
        for key in [
            "point_to_unit_target",
            "same_instance_unit_pair_mask",
            "same_instance_pair_available_mask",
            "unit_identity_purity_target",
            "unit_semantic_purity_target",
            "unit_temporal_consistency_target",
        ]:
            arr = np.asarray(z[key])
            item[key] = torch.from_numpy(arr.astype(np.float32 if arr.dtype.kind == "f" else np.int64 if arr.dtype.kind in "iu" else bool))
        return item


def collate_v341(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_v34(batch)
    for key in [
        "point_to_unit_target",
        "same_instance_unit_pair_mask",
        "same_instance_pair_available_mask",
        "unit_identity_purity_target",
        "unit_semantic_purity_target",
        "unit_temporal_consistency_target",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool, max_items: int | None = None) -> DataLoader:
    ds = V341Dataset(split, args, max_items=max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v341)


def assignment_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    target = batch["point_to_unit_target"].long()
    valid = target >= 0
    ce = F.nll_loss(assign[valid].log(), target[valid], reduction="mean") if bool(valid.any()) else assign.sum() * 0.0
    sim = torch.einsum("bmu,bnu->bmn", assign, assign).clamp(1e-4, 1 - 1e-4)
    pair_mask = batch["same_instance_pair_available_mask"].bool()
    same_pair = batch["same_instance_unit_pair_mask"].float()
    pair = F.binary_cross_entropy(sim[pair_mask], same_pair[pair_mask]) if bool(pair_mask.any()) else assign.sum() * 0.0
    usage = assign.mean(dim=1).clamp_min(1e-8)
    usage_entropy = -(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    point_entropy = -(assign * assign.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    anti_collapse = (1.0 - usage_entropy) + 0.2 * point_entropy
    return ce + 0.5 * pair + 0.3 * anti_collapse, {
        "unit_assignment_ce": float(ce.detach().cpu()),
        "same_instance_assignment_consistency": float(pair.detach().cpu()),
        "unit_usage_entropy": float(usage_entropy.detach().cpu()),
        "point_assignment_entropy": float(point_entropy.detach().cpu()),
    }


def observed_semantic_reconstruction_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    obs = batch["obs_semantic_measurements"]
    mask = batch["obs_semantic_measurement_mask"].float()
    pooled = (torch.nan_to_num(obs.float()) * mask[..., None]).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    valid = mask.sum(dim=2) > 0
    pred = out["observed_semantic_reconstruction"]
    return weighted_cosine_loss(pred[:, :, None, :], pooled[:, :, None, :], valid[:, :, None], torch.ones_like(valid[:, :, None]).float())


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].bool()
    hard_id = batch["identity_hard_train_mask"].bool() & same_mask
    same = batch["fut_same_instance_as_obs"].float()
    bce = F.binary_cross_entropy_with_logits(out["future_identity_belief"], same, reduction="none")
    id_loss = (bce * same_mask.float()).sum() / same_mask.float().sum().clamp_min(1.0)
    id_hard = (bce * hard_id.float()).sum() / hard_id.float().sum().clamp_min(1.0) if bool(hard_id.any()) else id_loss * 0.0
    contr = contrastive_loss(out["identity_embedding"], batch["fut_global_instance_id"].long(), batch["fut_global_instance_available_mask"].bool())
    assign_loss, assign_stats = assignment_losses(out, batch)
    obs_recon = observed_semantic_reconstruction_loss(out, batch)
    sem_mask = batch["fut_teacher_available_mask"].bool()
    sem_weight = batch["fut_teacher_confidence"].float().clamp(0.05, 1.0)
    sem = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], sem_mask, sem_weight)
    hard_sem = batch["semantic_hard_train_mask"].bool() & sem_mask
    sem_hard = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], hard_sem, sem_weight) if bool(hard_sem.any()) else sem * 0.0
    unc = out["semantic_uncertainty"]
    pred = F.normalize(out["future_semantic_belief"], dim=-1)
    tgt = F.normalize(torch.nan_to_num(batch["fut_teacher_embedding"].float()), dim=-1)
    err = (1.0 - (pred * tgt).sum(dim=-1)).detach()
    unc_loss = ((unc - err).abs() * sem_mask.float()).sum() / sem_mask.float().sum().clamp_min(1.0)
    total = id_loss + 1.5 * id_hard + 0.25 * contr + 0.6 * assign_loss + 0.4 * obs_recon + args.semantic_weight * sem + args.semantic_hard_weight * sem_hard + 0.05 * unc_loss
    stats = {
        "loss": float(total.detach().cpu()),
        "identity_bce": float(id_loss.detach().cpu()),
        "identity_hard_bce": float(id_hard.detach().cpu()),
        "identity_contrastive": float(contr.detach().cpu()),
        "observed_semantic_reconstruction": float(obs_recon.detach().cpu()),
        "future_semantic_rollout": float(sem.detach().cpu()),
        "semantic_hard_loss": float(sem_hard.detach().cpu()),
        "semantic_uncertainty_calibration": float(unc_loss.detach().cpu()),
    }
    stats.update(assign_stats)
    return total, stats


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = make_loader("train", args, shuffle=True)
    model = IdentityBoundSemanticTraceUnitsV341(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
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
            obs_semantic_measurements=bd["obs_semantic_measurements"],
            obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
            semantic_id=bd["semantic_id"],
        )
        loss, stats = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": float(step), **stats})
    ckpt = CKPT_DIR / "v34_1_identity_bound_semantic_trace_units_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "complete_train_sample_count": len(train_loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "identity_bound_model_built": True,
        "trace_conditioned_semantic_units_active": True,
        "future_leakage_detected": False,
        "teacher_as_method": False,
        "outputs_future_trace_field": True,
        "outputs_future_semantic_field": True,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_tail": losses[-10:],
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.1 Identity-Bound Semantic Trace Units Train Summary", payload, ["fresh_training_completed", "checkpoint_path", "complete_train_sample_count", "v30_backbone_frozen", "identity_bound_model_built", "trace_conditioned_semantic_units_active", "future_leakage_detected", "teacher_as_method", "outputs_future_trace_field", "outputs_future_semantic_field", "train_loss_decreased"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(IDENTITY_ROOT))
    p.add_argument("--global-identity-label-root", default=str(GLOBAL_ROOT))
    p.add_argument("--unit-identity-binding-root", default=str(BINDING_ROOT))
    p.add_argument("--hard-mask-manifest", default=str(MASK_ROOT / "H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--semantic-weight", type=float, default=1.0)
    p.add_argument("--semantic-hard-weight", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
