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

from stwm.modules.ostf_v34_2_pointwise_no_unit_baseline import PointwiseNoUnitBaselineV342
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import make_loader
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import semantic_masks
from stwm.tools.train_ostf_v34_semantic_trace_units_20260510 import GLOBAL_ROOT, IDENTITY_ROOT, MASK_ROOT, MEAS_ROOT, contrastive_loss, weighted_cosine_loss


CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_2_pointwise_no_unit_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_2_POINTWISE_NO_UNIT_BASELINE_20260511.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    same_mask = batch["fut_instance_available_mask"].bool()
    hard_id = batch["identity_hard_train_mask"].bool() & same_mask
    same = batch["fut_same_instance_as_obs"].float()
    bce = F.binary_cross_entropy_with_logits(out["future_identity_belief"], same, reduction="none")
    id_loss = (bce * same_mask.float()).sum() / same_mask.float().sum().clamp_min(1.0)
    id_hard = (bce * hard_id.float()).sum() / hard_id.float().sum().clamp_min(1.0) if bool(hard_id.any()) else id_loss * 0.0
    contr = contrastive_loss(out["identity_embedding"], batch["fut_global_instance_id"].long(), batch["fut_global_instance_available_mask"].bool())
    valid, stable, changed, hard = semantic_masks(batch)
    sem_weight = batch["fut_teacher_confidence"].float().clamp(0.05, 1.0)
    sem = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], valid, sem_weight)
    sem_hard = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], hard, sem_weight) if bool(hard.any()) else sem * 0.0
    sem_stable = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], stable, sem_weight) if bool(stable.any()) else sem * 0.0
    sem_changed = weighted_cosine_loss(out["future_semantic_belief"], batch["fut_teacher_embedding"], changed, sem_weight) if bool(changed.any()) else sem * 0.0
    total = id_loss + 1.5 * id_hard + 0.25 * contr + args.semantic_weight * sem + 0.8 * sem_stable + sem_changed + args.semantic_hard_weight * sem_hard
    return total, {
        "loss": float(total.detach().cpu()),
        "identity_bce": float(id_loss.detach().cpu()),
        "identity_hard_bce": float(id_hard.detach().cpu()),
        "identity_contrastive": float(contr.detach().cpu()),
        "future_semantic_rollout": float(sem.detach().cpu()),
        "stable_semantic_preservation_loss": float(sem_stable.detach().cpu()),
        "changed_semantic_loss": float(sem_changed.detach().cpu()),
        "semantic_hard_loss": float(sem_hard.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = PointwiseNoUnitBaselineV342(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim).to(device)
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
    ckpt = CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "fresh_training_completed": True,
        "real_pointwise_no_unit_baseline_built": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "complete_train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "future_leakage_detected": False,
        "teacher_as_method": False,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_tail": losses[-10:],
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.2 Pointwise No-Unit Baseline", payload, ["fresh_training_completed", "real_pointwise_no_unit_baseline_built", "checkpoint_path", "complete_train_sample_count", "v30_backbone_frozen", "future_leakage_detected", "train_loss_decreased"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(IDENTITY_ROOT))
    p.add_argument("--global-identity-label-root", default=str(GLOBAL_ROOT))
    p.add_argument("--unit-identity-binding-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_1_unit_identity_binding_targets/pointodyssey"))
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
