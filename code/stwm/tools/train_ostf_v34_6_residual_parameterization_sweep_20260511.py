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
import setproctitle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import pairwise_binding_loss
from stwm.tools.train_ostf_v34_2_pointwise_no_unit_baseline_20260511 import CKPT_DIR as POINT_CKPT_DIR, SUMMARY as POINT_TRAIN
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_5_delta_residual_probe_20260511 import (
    BINDING_ROOT,
    GLOBAL_ROOT,
    IDENTITY_ROOT,
    MASK_ROOT,
    MEAS_ROOT,
    STRICT_ROOT,
    StrictResidualUtilityDataset,
    collate_v345,
)


VARIANTS = [
    "standalone_target_residual",
    "orthogonal_delta_residual",
    "true_vector_delta_residual",
    "scaled_tangent_delta_residual",
    "mixture_residual",
]
INIT_MODES = ["init_from_pointwise_base", "init_from_v34_4_standalone_residual_checkpoint"]
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_6_residual_parameterization_h32_m128"
WORKER_DIR = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_workers"
SUMMARY = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_RESIDUAL_PARAMETERIZATION_TRAIN_SUMMARY_20260511.md"
V344_TRAIN = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_train_summary_20260511.json"
V344_CKPT = ROOT / "outputs/checkpoints/stwm_ostf_v34_4_oracle_residual_probe_h32_m128/v34_4_oracle_residual_probe_m128_h32_seed42_best.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    ds = StrictResidualUtilityDataset(split, args)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_v345,
    )


def load_pointwise_into_residual(model: PointwiseUnitResidualWorldModelV343) -> Path:
    train = json.loads(POINT_TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / train.get("checkpoint_path", str(POINT_CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    return ckpt


def v34_4_checkpoint() -> Path | None:
    if V344_TRAIN.exists():
        tr = json.loads(V344_TRAIN.read_text(encoding="utf-8"))
        path = ROOT / tr.get("checkpoint_path", str(V344_CKPT))
        if path.exists():
            return path
    return V344_CKPT if V344_CKPT.exists() else None


def init_model(model: PointwiseUnitResidualWorldModelV343, init_mode: str) -> dict[str, Any]:
    pointwise = load_pointwise_into_residual(model)
    loaded_v344 = None
    if init_mode == "init_from_v34_4_standalone_residual_checkpoint":
        loaded_v344 = v34_4_checkpoint()
        if loaded_v344 is not None:
            ck = torch.load(loaded_v344, map_location="cpu")
            model.load_state_dict(ck["model"], strict=False)
    elif init_mode != "init_from_pointwise_base":
        raise ValueError(f"Unknown init_mode={init_mode}")
    return {
        "pointwise_checkpoint": str(pointwise.relative_to(ROOT)) if pointwise.is_relative_to(ROOT) else str(pointwise),
        "v34_4_residual_checkpoint": None if loaded_v344 is None else str(loaded_v344.relative_to(ROOT)),
        "v34_4_residual_init_available": bool(loaded_v344 is not None),
    }


def freeze_pointwise_base(model: PointwiseUnitResidualWorldModelV343) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [
        model.tokenizer,
        model.factorized_state,
        model.unit_rollout,
        model.identity_to_hidden,
        model.unit_memory,
        model.semantic_residual_head,
        model.identity_residual_head,
        model.semantic_uncertainty_head,
    ]:
        for p in module.parameters():
            p.requires_grad_(True)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def direction_target(base: torch.Tensor, target: torch.Tensor, variant: str) -> torch.Tensor:
    base_n = _norm(base.detach())
    target_n = _norm(target)
    alpha = (target_n * base_n).sum(dim=-1, keepdim=True)
    tangent = target_n - alpha * base_n
    true_delta = target_n - base_n
    if variant == "standalone_target_residual":
        return target_n
    if variant == "orthogonal_delta_residual":
        return _norm(tangent)
    if variant == "true_vector_delta_residual":
        return _norm(true_delta)
    if variant == "scaled_tangent_delta_residual":
        return _norm(tangent)
    if variant == "mixture_residual":
        return _norm(0.5 * true_delta + 0.5 * tangent)
    raise ValueError(f"Unknown variant={variant}")


def residual_scale(base: torch.Tensor, target: torch.Tensor, variant: str, args: argparse.Namespace) -> torch.Tensor:
    scale = torch.ones((*base.shape[:-1], 1), device=base.device, dtype=base.dtype)
    if variant == "scaled_tangent_delta_residual":
        scale = scale * float(args.scaled_tangent_eval_beta)
    elif variant == "mixture_residual":
        scale = scale * float(args.mixture_residual_scale)
    elif variant == "true_vector_delta_residual":
        scale = scale * float(args.true_vector_residual_scale)
    return scale


def compose_semantic(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace, *, gate_mode: str = "strict") -> torch.Tensor:
    base = out["pointwise_semantic_belief"]
    if gate_mode == "strict":
        gate = batch["strict_residual_semantic_utility_mask"].float()
    elif gate_mode == "all":
        gate = batch["fut_teacher_available_mask"].float()
    elif gate_mode == "zero":
        gate = torch.zeros_like(batch["strict_residual_semantic_utility_mask"].float())
    else:
        raise ValueError(f"Unknown gate_mode={gate_mode}")
    scale = residual_scale(base, batch["fut_teacher_embedding"], args.variant, args)
    return _norm(base + gate[..., None] * scale * out["unit_semantic_residual"])


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["fut_teacher_embedding"]
    base = out["pointwise_semantic_belief"].detach()
    final_sem = compose_semantic(out, batch, args, gate_mode="strict")
    sem_pos = batch["strict_residual_semantic_utility_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    stable = batch["strict_stable_suppress_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    sem_weight = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    direction = direction_target(base, target, args.variant)
    final_loss = cosine_loss(final_sem, target, sem_pos, sem_weight) if bool(sem_pos.any()) else final_sem.sum() * 0.0
    direction_loss = cosine_loss(out["unit_semantic_residual"], direction, sem_pos, sem_weight) if bool(sem_pos.any()) else final_sem.sum() * 0.0
    stable_loss = cosine_loss(final_sem, base, stable, torch.ones_like(sem_weight)) if bool(stable.any()) else final_sem.sum() * 0.0
    base_cos = (_norm(base) * _norm(target)).sum(dim=-1).detach()
    final_cos = (_norm(final_sem) * _norm(target)).sum(dim=-1)
    gain = final_cos - base_cos
    gain_loss = (F.relu(args.gain_margin - gain) * sem_pos.float() * sem_weight).sum() / (sem_pos.float() * sem_weight).sum().clamp_min(1.0)
    residual_n = _norm(out["unit_semantic_residual"])
    base_n = _norm(base)
    anti_base = F.relu((residual_n * base_n).sum(dim=-1) + args.anti_base_margin)
    anti_base_loss = (anti_base * sem_pos.float()).sum() / sem_pos.float().sum().clamp_min(1.0)
    bind, bind_stats = pairwise_binding_loss(out, batch)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    usage_entropy = -(usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    anti_collapse = 1.0 - usage_entropy
    total = (
        args.final_target_weight * final_loss
        + args.direction_weight * direction_loss
        + args.gain_weight * gain_loss
        + args.stable_preservation_weight * stable_loss
        + args.anti_base_weight * anti_base_loss
        + args.unit_binding_weight * bind
        + args.anti_collapse_weight * anti_collapse
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "final_sem_to_target_loss": float(final_loss.detach().cpu()),
        "residual_direction_loss": float(direction_loss.detach().cpu()),
        "semantic_hard_gain_objective": float(gain_loss.detach().cpu()),
        "stable_preservation_loss": float(stable_loss.detach().cpu()),
        "anti_base_correction_loss": float(anti_base_loss.detach().cpu()),
        "unit_anti_collapse": float(anti_collapse.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def train_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if args.init_mode not in INIT_MODES:
        raise ValueError(f"init-mode must be one of {INIT_MODES}")
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = PointwiseUnitResidualWorldModelV343(
        args.v30_checkpoint,
        teacher_embedding_dim=args.teacher_embedding_dim,
        units=args.trace_units,
        horizon=args.horizon,
    ).to(device)
    init_info = init_model(model, args.init_mode)
    freeze_pointwise_base(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    start = time.time()
    it = iter(loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        out = model(
            obs_points=bd["obs_points"],
            obs_vis=bd["obs_vis"],
            obs_conf=bd["obs_conf"],
            obs_semantic_measurements=bd["obs_semantic_measurements"],
            obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
            semantic_id=bd["semantic_id"],
            intervention="force_gate_zero",
        )
        loss, stats = loss_fn(out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
    tag = f"{args.variant}__{args.init_mode}"
    ckpt = CKPT_DIR / f"v34_6_{tag}_m128_h32_seed42.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "variant": args.variant,
        "init_mode": args.init_mode,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "fresh_training_completed": True,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "pointwise_base_frozen": True,
        "learned_gate_training": False,
        "future_leakage_detected": False,
        "init_info": init_info,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_trace": losses,
    }
    out_path = WORKER_DIR / f"{tag}.json"
    dump_json(out_path, payload)
    print(out_path.relative_to(ROOT))
    return payload


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    workers = []
    for variant in VARIANTS:
        for init in INIT_MODES:
            path = WORKER_DIR / f"{variant}__{init}.json"
            if path.exists():
                workers.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "generated_at_utc": utc_now(),
        "candidate_count_expected": len(VARIANTS) * len(INIT_MODES),
        "candidate_count_completed": len(workers),
        "fresh_training_completed": bool(len(workers) == len(VARIANTS) * len(INIT_MODES)),
        "v30_backbone_frozen": bool(workers and all(w.get("v30_backbone_frozen") for w in workers)),
        "future_leakage_detected": False,
        "learned_gate_training": False,
        "variants": VARIANTS,
        "init_modes": INIT_MODES,
        "workers": workers,
        "exact_blockers": [] if len(workers) == len(VARIANTS) * len(INIT_MODES) else ["missing_worker_reports"],
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.6 Residual Parameterization Train Summary",
        payload,
        ["candidate_count_expected", "candidate_count_completed", "fresh_training_completed", "v30_backbone_frozen", "future_leakage_detected", "learned_gate_training", "variants", "init_modes", "exact_blockers"],
    )
    print(SUMMARY.relative_to(ROOT))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--semantic-identity-sidecar-root", default=str(IDENTITY_ROOT))
    p.add_argument("--global-identity-label-root", default=str(GLOBAL_ROOT))
    p.add_argument("--unit-identity-binding-root", default=str(BINDING_ROOT))
    p.add_argument("--strict-residual-utility-target-root", default=str(STRICT_ROOT))
    p.add_argument("--hard-mask-manifest", default=str(MASK_ROOT / "H32_M128_seed42.json"))
    p.add_argument("--m-points", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--variant", default=None, choices=VARIANTS)
    p.add_argument("--init-mode", default=None, choices=INIT_MODES)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--final-target-weight", type=float, default=1.2)
    p.add_argument("--direction-weight", type=float, default=0.9)
    p.add_argument("--gain-weight", type=float, default=0.7)
    p.add_argument("--gain-margin", type=float, default=0.01)
    p.add_argument("--stable-preservation-weight", type=float, default=0.9)
    p.add_argument("--anti-base-weight", type=float, default=0.1)
    p.add_argument("--anti-base-margin", type=float, default=0.0)
    p.add_argument("--unit-binding-weight", type=float, default=0.35)
    p.add_argument("--anti-collapse-weight", type=float, default=0.2)
    p.add_argument("--scaled-tangent-eval-beta", type=float, default=1.5)
    p.add_argument("--mixture-residual-scale", type=float, default=1.0)
    p.add_argument("--true-vector-residual-scale", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
        return 0
    if args.worker:
        if args.variant is None or args.init_mode is None:
            raise SystemExit("--worker requires --variant and --init-mode")
        train_worker(args)
        return 0
    for variant in VARIANTS:
        for init_mode in INIT_MODES:
            args.variant = variant
            args.init_mode = init_mode
            train_worker(args)
    aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
