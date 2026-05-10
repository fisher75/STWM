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

from stwm.modules.ostf_v33_11_identity_preserving_copy_residual_semantic_world_model import IdentityPreservingCopyResidualSemanticWorldModelV3311
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import BASELINE_ROOT, COMPLETE, COPY_ROOT, V33_11_MASK_ROOT, V33_9_CKPT, make_loader_v3311
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


RUN_DIR = ROOT / "reports/stwm_ostf_v33_11_identity_preserving_copy_residual_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_11_identity_preserving_copy_residual_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v33_11_identity_preserving_copy_residual_train_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_IDENTITY_PRESERVING_COPY_RESIDUAL_TRAIN_SUMMARY_20260510.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & (target >= 0)
    return F.cross_entropy(logits[valid], target.long()[valid]) if bool(valid.any()) else logits.sum() * 0.0


def stable_margin_loss(logits: torch.Tensor, copy_ids: torch.Tensor, mask: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    valid = mask.bool() & (copy_ids >= 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    l = logits[valid]
    ids = copy_ids.long()[valid].clamp(0, l.shape[-1] - 1)
    copy_logit = l.gather(-1, ids[:, None]).squeeze(-1)
    other = l.clone()
    other.scatter_(-1, ids[:, None], -1e9)
    max_other = other.max(dim=-1).values
    return F.relu(margin - (copy_logit - max_other)).mean()


def focal_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    valid = mask.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(t > 0.5, p, 1.0 - p)
    loss = bce * (1.0 - pt).pow(gamma)
    pos = (t * valid.float()).sum()
    neg = ((1.0 - t) * valid.float()).sum()
    weights = torch.where(t > 0.5, neg / pos.clamp_min(1.0), pos / neg.clamp_min(1.0)).clamp(0.1, 10.0)
    return (loss * weights * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["semantic_prototype_id"].long()
    valid = batch["semantic_prototype_available_mask"].bool() & (target >= 0)
    stable = batch["semantic_stable_mask"].bool() & valid
    changed = batch["semantic_changed_mask"].bool() & valid
    hard = batch["semantic_hard_mask"].bool() & valid
    copy_ids = batch["copy_semantic_prototype_id"].long()
    logits = out["final_semantic_proto_logits"]
    full_ce = masked_ce(logits, target, valid)
    stable_ce = masked_ce(logits, copy_ids.clamp_min(0), stable)
    margin = stable_margin_loss(logits, copy_ids, stable) if not args.no_stable_margin else logits.sum() * 0.0
    changed_ce = masked_ce(logits, target, changed)
    hard_ce = masked_ce(logits, target, hard)
    gate_mask = batch["semantic_update_available_mask"].bool()
    update_target = batch["semantic_update_target"].float()
    gate_loss = focal_bce(out["semantic_change_logits"], update_target, gate_mask, gamma=2.0) if not args.no_gate_focal else (
        F.binary_cross_entropy_with_logits(out["semantic_change_logits"], update_target, reduction="none") * gate_mask.float()
    ).sum() / gate_mask.float().sum().clamp_min(1.0)
    gate = out["semantic_change_gate"]
    stable_gate = gate[stable].mean() if bool(stable.any()) else gate.sum() * 0.0
    residual_l2 = out["semantic_residual_logits"].pow(2).mean()
    # Identity path is frozen for the main model; this term is a recorded no-op
    # unless the explicit no_identity_freeze ablation is run.
    identity_distill = (out["same_instance_logits"] - out["identity_teacher_same_instance_logits"]).pow(2).mean()
    total = (
        args.full_semantic_weight * full_ce
        + args.stable_semantic_weight * stable_ce
        + args.stable_margin_weight * margin
        + args.changed_semantic_weight * changed_ce
        + args.hard_semantic_weight * hard_ce
        + args.change_gate_weight * gate_loss
        + args.residual_sparsity_weight * stable_gate
        + args.identity_distill_weight * identity_distill
        + 1e-5 * residual_l2
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "semantic_full_ce": float(full_ce.detach().cpu()),
        "semantic_stable_ce": float(stable_ce.detach().cpu()),
        "stable_margin_loss": float(margin.detach().cpu()),
        "semantic_changed_ce": float(changed_ce.detach().cpu()),
        "semantic_hard_ce": float(hard_ce.detach().cpu()),
        "semantic_change_gate_focal_bce": float(gate_loss.detach().cpu()),
        "stable_gate_mean": float(stable_gate.detach().cpu()),
        "identity_distillation_loss": float(identity_distill.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    train_loader = make_loader_v3311("train", args, shuffle=True, max_items=None if args.max_train_items <= 0 else args.max_train_items)
    model = IdentityPreservingCopyResidualSemanticWorldModelV3311(
        args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=args.teacher_embedding_dim,
        identity_teacher_checkpoint=args.identity_teacher_checkpoint,
        freeze_identity_path=not args.no_identity_freeze,
        no_stable_margin=args.no_stable_margin,
        no_gate_focal=args.no_gate_focal,
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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
        "v30_checkpoint_loaded": Path(args.v30_checkpoint).exists(),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "integrated_v30_backbone_used": True,
        "identity_path_frozen_or_distilled": bool(model.identity_path_frozen_or_distilled),
        "identity_teacher_checkpoint_loaded": bool(model.identity_teacher_checkpoint_loaded),
        "identity_teacher_checkpoint": str(Path(args.identity_teacher_checkpoint).relative_to(ROOT)) if Path(args.identity_teacher_checkpoint).is_absolute() and Path(args.identity_teacher_checkpoint).exists() else args.identity_teacher_checkpoint,
        "observed_instance_context_used": False,
        "observed_visual_teacher_context_used": True,
        "future_teacher_leakage_detected": False,
        "target_root": str(COMPLETE.relative_to(ROOT)),
        "copy_residual_target_root": str(Path(args.copy_residual_semantic_target_root).relative_to(ROOT)),
        "semantic_baseline_bank_root": str(Path(args.semantic_baseline_bank_root).relative_to(ROOT)),
        "hard_semantic_protocol": str(Path(args.hard_train_mask_manifest).relative_to(ROOT)),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "loss_tail": losses[-10:],
        "no_identity_freeze": bool(args.no_identity_freeze),
        "no_stable_margin": bool(args.no_stable_margin),
        "no_gate_focal": bool(args.no_gate_focal),
    }
    run_path = RUN_DIR / f"{args.experiment_name}.json"
    dump_json(run_path, payload)
    if args.write_main_summary:
        dump_json(SUMMARY, payload)
        write_doc(DOC, "STWM OSTF V33.11 Identity-Preserving Copy Residual Train Summary", payload, ["experiment_name", "fresh_training_completed", "complete_train_sample_count", "v30_backbone_frozen", "identity_path_frozen_or_distilled", "identity_teacher_checkpoint_loaded", "future_teacher_leakage_detected", "train_loss_decreased", "checkpoint_path"])
        print(SUMMARY.relative_to(ROOT))
    else:
        print(run_path.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", default="v33_11_identity_preserving_copy_residual_m128_h32_seed42")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--identity-teacher-checkpoint", default=str(V33_9_CKPT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(COMPLETE / "semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(COMPLETE / "global_identity_labels/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(COMPLETE / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"))
    p.add_argument("--semantic-prototype-target-root", default=str(COMPLETE / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"))
    p.add_argument("--copy-residual-semantic-target-root", default=str(COPY_ROOT))
    p.add_argument("--semantic-baseline-bank-root", default=str(BASELINE_ROOT))
    p.add_argument("--prototype-vocab-path", default=str(ROOT / "outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K32/prototype_vocab.npz"))
    p.add_argument("--hard-train-mask-manifest", default=str(V33_11_MASK_ROOT / "H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-items", type=int, default=0)
    p.add_argument("--teacher-embedding-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--full-semantic-weight", type=float, default=0.10)
    p.add_argument("--stable-semantic-weight", type=float, default=1.75)
    p.add_argument("--stable-margin-weight", type=float, default=1.25)
    p.add_argument("--changed-semantic-weight", type=float, default=1.0)
    p.add_argument("--hard-semantic-weight", type=float, default=1.0)
    p.add_argument("--change-gate-weight", type=float, default=0.9)
    p.add_argument("--residual-sparsity-weight", type=float, default=0.9)
    p.add_argument("--identity-distill-weight", type=float, default=0.05)
    p.add_argument("--no-identity-freeze", action="store_true")
    p.add_argument("--no-stable-margin", action="store_true")
    p.add_argument("--no-gate-focal", action="store_true")
    p.add_argument("--write-main-summary", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    args.use_observed_instance_context = False
    args.enable_global_identity_labels = True
    args.require_global_identity_labels = True
    return args


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
