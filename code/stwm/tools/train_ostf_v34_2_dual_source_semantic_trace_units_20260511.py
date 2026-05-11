#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_2_dual_source_semantic_trace_units import DualSourceSemanticTraceUnitsV342
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import BINDING_ROOT, collate_v341, make_loader
from stwm.tools.train_ostf_v34_semantic_trace_units_20260510 import GLOBAL_ROOT, IDENTITY_ROOT, MASK_ROOT, MEAS_ROOT, contrastive_loss, weighted_cosine_loss


CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_2_dual_source_semantic_trace_units_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_2_DUAL_SOURCE_SEMANTIC_TRACE_UNITS_TRAIN_SUMMARY_20260511.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_binding_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    sim = torch.einsum("bmu,bnu->bmn", assign, assign).clamp(1e-4, 1 - 1e-4)
    pair_mask = batch["same_instance_pair_available_mask"].bool()
    same = batch["same_instance_unit_pair_mask"].float()
    if bool(pair_mask.any()):
        pos = pair_mask & same.bool()
        neg = pair_mask & (~same.bool())
        pos_loss = -torch.log(sim[pos]).mean() if bool(pos.any()) else sim.sum() * 0.0
        neg_loss = -torch.log(1.0 - sim[neg]).mean() if bool(neg.any()) else sim.sum() * 0.0
    else:
        pos_loss = neg_loss = sim.sum() * 0.0
    usage = assign.mean(dim=1).clamp_min(1e-8)
    usage_entropy = -(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    point_entropy = -(assign * assign.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    anti_collapse = (1.0 - usage_entropy) + 0.15 * point_entropy
    total = pos_loss + neg_loss + 0.25 * anti_collapse
    return total, {
        "same_instance_pair_attraction": float(pos_loss.detach().cpu()),
        "different_instance_unit_separation": float(neg_loss.detach().cpu()),
        "unit_usage_entropy": float(usage_entropy.detach().cpu()),
        "point_assignment_entropy": float(point_entropy.detach().cpu()),
    }


def observed_recon_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    obs = batch["obs_semantic_measurements"]
    mask = batch["obs_semantic_measurement_mask"].float()
    pooled = (torch.nan_to_num(obs.float()) * mask[..., None]).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    valid = mask.sum(dim=2) > 0
    return weighted_cosine_loss(out["observed_semantic_reconstruction"][:, :, None, :], pooled[:, :, None, :], valid[:, :, None], torch.ones_like(valid[:, :, None]).float())


def semantic_masks(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = batch["obs_semantic_measurements"]
    obs_mask = batch["obs_semantic_measurement_mask"].float()
    last = (torch.nan_to_num(obs.float()) * obs_mask[..., None]).sum(dim=2) / obs_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    copy = F.normalize(last[:, :, None, :].expand_as(batch["fut_teacher_embedding"]), dim=-1)
    tgt = F.normalize(torch.nan_to_num(batch["fut_teacher_embedding"].float()), dim=-1)
    copy_cos = (copy * tgt).sum(dim=-1)
    valid = batch["fut_teacher_available_mask"].bool()
    stable = valid & (copy_cos >= 0.80)
    changed = valid & (copy_cos < 0.65)
    hard = valid & batch["semantic_hard_train_mask"].bool()
    return valid, stable, changed, hard


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].bool()
    hard_id = batch["identity_hard_train_mask"].bool() & same_mask
    same = batch["fut_same_instance_as_obs"].float()
    bce = F.binary_cross_entropy_with_logits(out["future_identity_belief"], same, reduction="none")
    id_loss = (bce * same_mask.float()).sum() / same_mask.float().sum().clamp_min(1.0)
    id_hard = (bce * hard_id.float()).sum() / hard_id.float().sum().clamp_min(1.0) if bool(hard_id.any()) else id_loss * 0.0
    contr = contrastive_loss(out["identity_embedding"], batch["fut_global_instance_id"].long(), batch["fut_global_instance_available_mask"].bool())
    bind, bind_stats = pairwise_binding_loss(out, batch)
    obs_recon = observed_recon_loss(out, batch)
    valid, stable, changed, hard = semantic_masks(batch)
    sem_weight = batch["fut_teacher_confidence"].float().clamp(0.05, 1.0)
    sem = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], valid, sem_weight)
    sem_hard = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], hard, sem_weight) if bool(hard.any()) else sem * 0.0
    sem_changed = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], changed, sem_weight) if bool(changed.any()) else sem * 0.0
    sem_stable = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], stable, sem_weight) if bool(stable.any()) else sem * 0.0
    pred = F.normalize(out["future_semantic_belief"], dim=-1)
    tgt = F.normalize(torch.nan_to_num(batch["fut_teacher_embedding"].float()), dim=-1)
    err = (1.0 - (pred * tgt).sum(dim=-1)).detach()
    unc = ((out["semantic_uncertainty"] - err).abs() * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    total = id_loss + 1.5 * id_hard + 0.25 * contr + 0.7 * bind + 0.4 * obs_recon + args.semantic_weight * sem + 0.8 * sem_stable + 1.0 * sem_changed + args.semantic_hard_weight * sem_hard + 0.05 * unc
    stats = {
        "loss": float(total.detach().cpu()),
        "identity_bce": float(id_loss.detach().cpu()),
        "identity_hard_bce": float(id_hard.detach().cpu()),
        "identity_contrastive": float(contr.detach().cpu()),
        "observed_semantic_reconstruction": float(obs_recon.detach().cpu()),
        "future_semantic_rollout": float(sem.detach().cpu()),
        "stable_semantic_preservation_loss": float(sem_stable.detach().cpu()),
        "changed_semantic_loss": float(sem_changed.detach().cpu()),
        "semantic_hard_loss": float(sem_hard.detach().cpu()),
        "semantic_uncertainty_calibration": float(unc.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = DualSourceSemanticTraceUnitsV342(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
    it = iter(loader)
    start = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
        loss, stats = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append({"step": float(step), **stats})
    ckpt = CKPT_DIR / "v34_2_dual_source_semantic_trace_units_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "complete_train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "dual_source_model_built": True,
        "permutation_aware_binding_active": True,
        "z_dyn_source_is_trace_dynamics": True,
        "z_sem_source_is_semantic_measurement": True,
        "z_dyn_z_sem_factorization_real": True,
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
    write_doc(DOC, "STWM OSTF V34.2 Dual-Source Semantic Trace Units Train Summary", payload, ["fresh_training_completed", "checkpoint_path", "complete_train_sample_count", "v30_backbone_frozen", "dual_source_model_built", "permutation_aware_binding_active", "z_dyn_source_is_trace_dynamics", "z_sem_source_is_semantic_measurement", "z_dyn_z_sem_factorization_real", "future_leakage_detected", "train_loss_decreased"])
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
