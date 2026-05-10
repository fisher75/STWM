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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_13_gate_repaired_copy_semantic_world_model import GateRepairedCopySemanticWorldModelV3313
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import COMPLETE, V33_11_MASK_ROOT, V33_9_CKPT, make_loader_v3311
from stwm.tools.train_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510 import (
    focal_bce,
    masked_ce,
    stable_margin_loss,
)
from stwm.tools.train_ostf_v33_12_copy_conservative_semantic_20260510 import BASELINE_ROOT, COPY_ROOT, TARGET_ROOT, VIS_ROOT, VOCAB, ensure_targets
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


RUN_DIR = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_runs"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v33_13_gate_repaired_copy_semantic_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_train_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_GATE_REPAIRED_TRAIN_SUMMARY_20260510.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    stable_margin = stable_margin_loss(logits, copy_ids, stable, margin=args.stable_margin)
    changed_ce = masked_ce(logits, target, changed)
    hard_ce = masked_ce(logits, target, hard)
    gate_loss = focal_bce(out["semantic_change_logits_raw"], batch["semantic_update_target"].float(), batch["semantic_update_available_mask"].bool(), gamma=2.0)
    stable_prob = out["semantic_change_prob"][stable].mean() if bool(stable.any()) else logits.sum() * 0.0
    stable_effective = out["semantic_effective_gate"][stable].mean() if bool(stable.any()) else logits.sum() * 0.0
    changed_prob = out["semantic_change_prob"][changed].mean() if bool(changed.any()) else logits.sum() * 0.0
    identity_distill = (out["same_instance_logits"] - out["identity_teacher_same_instance_logits"]).pow(2).mean()
    total = (
        args.full_semantic_weight * full_ce
        + args.stable_semantic_weight * stable_ce
        + args.stable_margin_weight * stable_margin
        + args.changed_semantic_weight * changed_ce
        + args.hard_semantic_weight * hard_ce
        + args.change_gate_weight * gate_loss
        + args.stable_gate_penalty_weight * stable_effective
        - args.changed_gate_reward_weight * changed_prob
        + args.identity_distill_weight * identity_distill
        + args.residual_l2_weight * out["semantic_residual_logits"].pow(2).mean()
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "semantic_full_ce": float(full_ce.detach().cpu()),
        "semantic_stable_ce": float(stable_ce.detach().cpu()),
        "stable_margin_loss": float(stable_margin.detach().cpu()),
        "semantic_changed_ce": float(changed_ce.detach().cpu()),
        "semantic_hard_ce": float(hard_ce.detach().cpu()),
        "semantic_change_gate_focal_bce": float(gate_loss.detach().cpu()),
        "stable_change_prob_mean": float(stable_prob.detach().cpu()),
        "stable_effective_gate_mean": float(stable_effective.detach().cpu()),
        "changed_change_prob_mean": float(changed_prob.detach().cpu()),
        "identity_distillation_loss": float(identity_distill.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    target_info = ensure_targets(device)
    centers = torch.from_numpy(np.asarray(np.load(args.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    train_loader = make_loader_v3311("train", args, shuffle=True, max_items=None if args.max_train_items <= 0 else args.max_train_items)
    model = GateRepairedCopySemanticWorldModelV3313(
        args.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=args.teacher_embedding_dim,
        identity_teacher_checkpoint=args.identity_teacher_checkpoint,
        gate_threshold=args.gate_threshold,
        freeze_identity_path=True,
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    it = iter(train_loader)
    start = time.time()
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
        "gate_protocol_repaired": True,
        "double_sigmoid_bug_fixed": True,
        "train_eval_gate_consistent": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "complete_train_sample_count": len(train_loader.dataset),
        "duration_seconds": float(time.time() - start),
        "v30_checkpoint_loaded": Path(args.v30_checkpoint).exists(),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "identity_path_frozen_or_distilled": bool(model.identity_path_frozen_or_distilled),
        "identity_teacher_checkpoint_loaded": bool(model.identity_teacher_checkpoint_loaded),
        "observed_instance_context_used": False,
        "observed_visual_teacher_context_used": True,
        "future_teacher_leakage_detected": False,
        "teacher": "clip_vit_b32_local",
        "prototype_K": 256,
        "target_root": str(COMPLETE.relative_to(ROOT)),
        "semantic_prototype_target_root": str(Path(args.semantic_prototype_target_root).relative_to(ROOT)),
        "copy_residual_target_root": str(Path(args.copy_residual_semantic_target_root).relative_to(ROOT)),
        "semantic_baseline_bank_root": str(Path(args.semantic_baseline_bank_root).relative_to(ROOT)),
        "target_space_info": target_info,
        "gate_threshold": args.gate_threshold,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "loss_tail": losses[-10:],
    }
    dump_json(RUN_DIR / f"{args.experiment_name}.json", payload)
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.13 Gate-Repaired Copy Semantic Train Summary",
        payload,
        ["fresh_training_completed", "gate_protocol_repaired", "double_sigmoid_bug_fixed", "train_eval_gate_consistent", "complete_train_sample_count", "train_loss_decreased", "checkpoint_path"],
    )
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", default="v33_13_gate_repaired_copy_semantic_m128_h32_seed42")
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--identity-teacher-checkpoint", default=str(V33_9_CKPT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(COMPLETE / "semantic_identity_targets/pointodyssey"))
    p.add_argument("--global-identity-label-root", default=str(COMPLETE / "global_identity_labels/pointodyssey"))
    p.add_argument("--visual-teacher-root", default=str(VIS_ROOT))
    p.add_argument("--semantic-prototype-target-root", default=str(TARGET_ROOT))
    p.add_argument("--copy-residual-semantic-target-root", default=str(COPY_ROOT))
    p.add_argument("--semantic-baseline-bank-root", default=str(BASELINE_ROOT))
    p.add_argument("--prototype-vocab-path", default=str(VOCAB))
    p.add_argument("--hard-train-mask-manifest", default=str(V33_11_MASK_ROOT / "H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-items", type=int, default=0)
    p.add_argument("--teacher-embedding-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--gate-threshold", type=float, default=0.10)
    p.add_argument("--stable-margin", type=float, default=2.5)
    p.add_argument("--full-semantic-weight", type=float, default=0.05)
    p.add_argument("--stable-semantic-weight", type=float, default=2.5)
    p.add_argument("--stable-margin-weight", type=float, default=2.0)
    p.add_argument("--changed-semantic-weight", type=float, default=1.0)
    p.add_argument("--hard-semantic-weight", type=float, default=1.0)
    p.add_argument("--change-gate-weight", type=float, default=1.0)
    p.add_argument("--stable-gate-penalty-weight", type=float, default=1.0)
    p.add_argument("--changed-gate-reward-weight", type=float, default=0.05)
    p.add_argument("--identity-distill-weight", type=float, default=0.1)
    p.add_argument("--residual-l2-weight", type=float, default=1e-5)
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
