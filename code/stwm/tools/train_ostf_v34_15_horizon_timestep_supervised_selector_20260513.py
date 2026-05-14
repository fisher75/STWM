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

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348
from stwm.modules.ostf_v34_14_horizon_conditioned_measurement_selector import HorizonConditionedMeasurementSelectorV3414
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import MEAS_ROOT, TARGET_ROOT, TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_14_horizon_conditioned_measurement_selector_20260513 import agreement, selector_inputs


V3414_TRAIN = ROOT / "reports/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_train_summary_20260513.json"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_15_horizon_timestep_supervised_selector_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_15_horizon_timestep_supervised_selector_train_summary_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_15_HORIZON_TIMESTEP_SUPERVISED_SELECTOR_TRAIN_SUMMARY_20260513.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        TraceContractResidualDataset(split, args),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_v3410,
    )


def local_cos(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (F.normalize(pred, dim=-1) * F.normalize(torch.nan_to_num(target.float()), dim=-1)).sum(dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    m = mask.float()
    if weight is not None:
        m = m * weight.float()
    return (x * m).sum() / m.sum().clamp_min(1.0)


def oracle_timestep_labels(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = F.normalize(torch.nan_to_num(batch["obs_semantic_measurements"].float()), dim=-1)
    target = F.normalize(torch.nan_to_num(batch["fut_teacher_embedding"].float()), dim=-1)
    mask = batch["obs_semantic_measurement_mask"].bool()
    sim = torch.einsum("bmtd,bmhd->bmht", obs, target)
    sim = sim.masked_fill(~mask[:, :, None, :], -1e4)
    labels = sim.argmax(dim=-1)
    oracle_cos = sim.max(dim=-1).values
    available = mask.any(dim=-1)[:, :, None].expand_as(labels)
    return labels, oracle_cos, available


def load_v3414_if_available(selector: HorizonConditionedMeasurementSelectorV3414) -> dict[str, Any]:
    train = json.loads(V3414_TRAIN.read_text(encoding="utf-8")) if V3414_TRAIN.exists() else {}
    ckpt = ROOT / train.get("checkpoint_path", "")
    if not ckpt.exists():
        return {"init_source": "random"}
    ck = torch.load(ckpt, map_location="cpu")
    src = ck.get("model", ck)
    missing, unexpected = selector.load_state_dict(src, strict=False)
    return {"init_source": "v34_14_horizon_conditioned", "checkpoint_path": str(ckpt.relative_to(ROOT)), "missing_key_count": len(missing), "unexpected_key_count": len(unexpected)}


def loss_fn(selector: HorizonConditionedMeasurementSelectorV3414, base_model: CausalAssignmentBoundResidualMemoryV348, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    out = selector(**selector_inputs(selector, base_model, batch))
    labels, oracle_cos, avail = oracle_timestep_labels(batch)
    valid = batch["fut_teacher_available_mask"].bool() & avail
    hard = batch["semantic_hard_mask"].bool() & valid
    changed = batch["changed_mask"].bool() & valid
    strict = batch["strict_residual_semantic_utility_mask"].bool() & valid
    causal = batch["causal_assignment_residual_semantic_mask"].bool() & valid
    target_mask = valid & (hard | changed | strict | causal)
    if not bool(target_mask.any()):
        target_mask = valid
    logits = torch.log(out["measurement_weight"].clamp_min(1e-8))
    ce = F.nll_loss(logits.permute(0, 1, 3, 2).reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape_as(labels)
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    hard_weight = 1.0 + 2.5 * hard.float() + 2.5 * changed.float() + 1.2 * strict.float() + 1.2 * causal.float()
    ce_loss = masked_mean(ce, target_mask, teacher_w * hard_weight)
    selected_cos = local_cos(out["selected_evidence"], batch["fut_teacher_embedding"])
    pointwise = batch["pointwise_semantic_cosine"].float()
    align = masked_mean(1.0 - selected_cos, target_mask, teacher_w * hard_weight)
    oracle_gap_loss = masked_mean(F.softplus(args.oracle_gap_margin - (selected_cos - oracle_cos.detach())), target_mask, teacher_w)
    hard_changed = (hard | changed) & valid
    pointwise_contrast = masked_mean(F.softplus(args.pointwise_margin - (selected_cos - pointwise)), hard_changed, teacher_w) if bool(hard_changed.any()) else selected_cos.sum() * 0.0
    shuffled = out["selected_evidence"][:, torch.randperm(out["selected_evidence"].shape[1], device=selected_cos.device)]
    shuf_cos = local_cos(shuffled, batch["fut_teacher_embedding"]).detach()
    random_contrast = masked_mean(F.softplus(args.random_margin - (selected_cos - shuf_cos)), target_mask, teacher_w)
    entropy = out["selector_entropy"]
    entropy_target = torch.full_like(entropy, args.entropy_target)
    entropy_reg = masked_mean((entropy - entropy_target).pow(2), target_mask)
    total = args.ce_weight * ce_loss + args.align_weight * align + args.oracle_gap_weight * oracle_gap_loss + args.pointwise_contrast_weight * pointwise_contrast + args.random_contrast_weight * random_contrast + args.entropy_weight * entropy_reg
    stats = {
        "loss": float(total.detach().cpu()),
        "timestep_ce_loss": float(ce_loss.detach().cpu()),
        "target_alignment_loss": float(align.detach().cpu()),
        "oracle_gap_loss": float(oracle_gap_loss.detach().cpu()),
        "pointwise_contrast_loss": float(pointwise_contrast.detach().cpu()),
        "random_contrast_loss": float(random_contrast.detach().cpu()),
        "selector_entropy_mean": float(entropy.mean().detach().cpu()),
        "selector_max_weight_mean": float(out["selector_max_weight"].mean().detach().cpu()),
    }
    return total, stats


def summarize(losses: list[dict[str, float]], key: str) -> dict[str, float | None]:
    vals = [float(x[key]) for x in losses if key in x]
    return {"first": vals[0] if vals else None, "last": vals[-1] if vals else None, "mean": float(np.mean(vals)) if vals else None}


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    base_model = CausalAssignmentBoundResidualMemoryV348(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    hidden = int(base_model.v30.cfg.hidden_dim)
    selector = HorizonConditionedMeasurementSelectorV3414(hidden, args.teacher_embedding_dim, args.hidden_dim).to(device)
    init = load_v3414_if_available(selector)
    opt = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    it = iter(loader)
    losses: list[dict[str, float]] = []
    start = time.time()
    selector.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        loss, stats = loss_fn(selector, base_model, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(100, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(f"训练进度: step={step}/{args.steps}, loss={stats['loss']:.6f}, ce={stats['timestep_ce_loss']:.6f}, align={stats['target_alignment_loss']:.6f}, maxw={stats['selector_max_weight_mean']:.4f}", flush=True)
    ckpt = CKPT_DIR / "v34_15_horizon_timestep_supervised_selector_m128_h32_seed42_best.pt"
    torch.save({"model": selector.state_dict(), "args": vars(args), "trace_hidden_dim": hidden, "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.15 horizon timestep-supervised selector 已完成训练；future target 只用于生成训练监督的 oracle observed timestep label，selector forward 仍只读 observed semantic memory 与 frozen V30 trace query。",
        "horizon_timestep_supervised_selector_built": True,
        "selector_was_trained": True,
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "init_checkpoint": init,
        "train_sample_count": len(loader.dataset),
        "steps": args.steps,
        "v30_backbone_frozen": True,
        "future_teacher_embedding_input_allowed": False,
        "oracle_timestep_label_supervision_only": True,
        "measurement_weight_shape": "B,M,H,Tobs",
        "selected_evidence_shape": "B,M,H,D",
        "loss_trace": losses,
        "timestep_ce_loss": summarize(losses, "timestep_ce_loss"),
        "target_alignment_loss": summarize(losses, "target_alignment_loss"),
        "oracle_gap_loss": summarize(losses, "oracle_gap_loss"),
        "selector_max_weight_mean": summarize(losses, "selector_max_weight_mean"),
        "duration_seconds": float(time.time() - start),
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "V34.15 horizon timestep-supervised selector 训练中文报告", payload, ["中文结论", "horizon_timestep_supervised_selector_built", "selector_was_trained", "fresh_training_completed", "checkpoint_path", "init_checkpoint", "train_sample_count", "steps", "v30_backbone_frozen", "future_teacher_embedding_input_allowed", "oracle_timestep_label_supervision_only"])
    print(f"已写出 V34.15 timestep-supervised selector 训练摘要: {SUMMARY.relative_to(ROOT)}")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--strict-residual-utility-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"))
    p.add_argument("--assignment-aware-residual-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"))
    p.add_argument("--causal-assignment-residual-target-root", default=str(TARGET_ROOT))
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--ce-weight", type=float, default=0.8)
    p.add_argument("--align-weight", type=float, default=0.6)
    p.add_argument("--oracle-gap-weight", type=float, default=0.15)
    p.add_argument("--pointwise-contrast-weight", type=float, default=0.7)
    p.add_argument("--random-contrast-weight", type=float, default=0.25)
    p.add_argument("--entropy-weight", type=float, default=0.03)
    p.add_argument("--oracle-gap-margin", type=float, default=0.0)
    p.add_argument("--pointwise-margin", type=float, default=0.004)
    p.add_argument("--random-margin", type=float, default=0.01)
    p.add_argument("--entropy-target", type=float, default=0.2)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
