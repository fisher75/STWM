#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from stwm.modules.ostf_v34_13_nonoracle_measurement_selector import NonOracleMeasurementSelectorV3413
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import (
    MEAS_ROOT,
    TARGET_ROOT,
    TraceContractResidualDataset,
    collate_v3410,
)


CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_13_nonoracle_measurement_selector_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_train_summary_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_13_NONORACLE_MEASUREMENT_SELECTOR_TRAIN_SUMMARY_20260513.md"


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


def selector_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    unit_purity = torch.maximum(batch["unit_instance_purity"].float(), batch["unit_semantic_purity"].float())
    agreement = batch.get("teacher_agreement_score", batch.get("semantic_measurement_agreement", batch["semantic_measurement_confidence"]))
    return {
        "obs_semantic_measurements": batch["obs_semantic_measurements"],
        "obs_semantic_measurement_mask": batch["obs_semantic_measurement_mask"],
        "obs_measurement_confidence": batch["semantic_measurement_confidence"],
        "teacher_agreement_score": agreement,
        "obs_vis": batch["obs_vis"],
        "obs_conf": batch["trace_obs_conf"],
        "obs_points": batch["obs_points"],
        "unit_assignment": batch["point_to_unit_assignment"],
        "unit_purity_proxy": unit_purity,
    }


def cosine_to_future(selected: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
    return (F.normalize(selected, dim=-1)[:, :, None, :] * F.normalize(torch.nan_to_num(fut.float()), dim=-1)).sum(dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    m = mask.float()
    if weight is not None:
        m = m * weight.float()
    return (x * m).sum() / m.sum().clamp_min(1.0)


def loss_fn(model: NonOracleMeasurementSelectorV3413, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(**selector_inputs(batch))
    selected = out["selected_measurement_embedding"]
    fut = batch["fut_teacher_embedding"]
    valid = batch["fut_teacher_available_mask"].bool()
    hard = batch["semantic_hard_mask"].bool() & valid
    changed = batch["changed_mask"].bool() & valid
    strict = batch["strict_residual_semantic_utility_mask"].bool() & valid
    causal = batch["causal_assignment_residual_semantic_mask"].bool() & valid
    target_mask = valid & (hard | changed | strict | causal)
    if not bool(target_mask.any()):
        target_mask = valid
    cos = cosine_to_future(selected, fut)
    pointwise = batch["pointwise_semantic_cosine"].float()
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    hard_weight = 1.0 + 2.0 * hard.float() + 2.0 * changed.float() + 1.0 * strict.float() + 1.0 * causal.float()
    target_loss = masked_mean(1.0 - cos, target_mask, teacher_w * hard_weight)

    shuffled = selected[:, torch.randperm(selected.shape[1], device=selected.device)]
    shuf_cos = cosine_to_future(shuffled, fut).detach()
    random_contrast = masked_mean(F.softplus(args.random_margin - (cos - shuf_cos)), target_mask, teacher_w)
    hard_changed = (hard | changed) & valid
    pointwise_contrast = masked_mean(F.softplus(args.pointwise_margin - (cos - pointwise)), hard_changed, teacher_w) if bool(hard_changed.any()) else cos.sum() * 0.0
    entropy = out["selector_entropy"]
    entropy_target = torch.full_like(entropy, args.entropy_target)
    coverage = batch["obs_semantic_measurement_mask"].float().mean(dim=-1)
    entropy_reg = (((entropy - entropy_target) ** 2) * (coverage > 0).float()).sum() / (coverage > 0).float().sum().clamp_min(1.0)
    conf_loss = F.binary_cross_entropy(out["selector_confidence"].clamp(1e-4, 1 - 1e-4), (coverage > 0).float())
    total = target_loss + args.random_contrast_weight * random_contrast + args.pointwise_contrast_weight * pointwise_contrast + args.entropy_weight * entropy_reg + 0.05 * conf_loss
    stats = {
        "loss": float(total.detach().cpu()),
        "target_alignment_loss": float(target_loss.detach().cpu()),
        "random_contrast_loss": float(random_contrast.detach().cpu()),
        "pointwise_contrast_loss": float(pointwise_contrast.detach().cpu()),
        "selector_entropy_reg": float(entropy_reg.detach().cpu()),
        "selector_entropy_mean": float(entropy.mean().detach().cpu()),
        "selector_confidence_mean": float(out["selector_confidence"].mean().detach().cpu()),
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
    model = NonOracleMeasurementSelectorV3413(args.teacher_embedding_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    it = iter(loader)
    losses: list[dict[str, float]] = []
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        loss, stats = loss_fn(model, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(100, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(f"训练进度: step={step}/{args.steps}, loss={stats['loss']:.6f}, target={stats['target_alignment_loss']:.6f}, entropy={stats['selector_entropy_mean']:.4f}", flush=True)
    ckpt = CKPT_DIR / "v34_13_nonoracle_measurement_selector_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.13 训练式 non-oracle measurement selector 已完成训练；selector forward 输入仅来自 observed measurements/trace/confidence/agreement/unit purity proxy，future target 只用于 loss supervision。",
        "nonoracle_selector_built": True,
        "selector_was_trained": True,
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "train_sample_count": len(loader.dataset),
        "steps": args.steps,
        "future_teacher_embedding_input_allowed": False,
        "v30_backbone_frozen": True,
        "loss_trace": losses,
        "target_alignment_loss": summarize(losses, "target_alignment_loss"),
        "pointwise_contrast_loss": summarize(losses, "pointwise_contrast_loss"),
        "random_contrast_loss": summarize(losses, "random_contrast_loss"),
        "selector_entropy_mean": summarize(losses, "selector_entropy_mean"),
        "duration_seconds": float(time.time() - start),
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "V34.13 训练式 non-oracle selector 训练中文报告",
        payload,
        ["中文结论", "nonoracle_selector_built", "selector_was_trained", "fresh_training_completed", "checkpoint_path", "train_sample_count", "steps", "future_teacher_embedding_input_allowed"],
    )
    print(f"已写出 V34.13 selector 训练摘要: {SUMMARY.relative_to(ROOT)}")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--strict-residual-utility-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"))
    p.add_argument("--assignment-aware-residual-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"))
    p.add_argument("--causal-assignment-residual-target-root", default=str(TARGET_ROOT))
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--random-margin", type=float, default=0.01)
    p.add_argument("--pointwise-margin", type=float, default=0.004)
    p.add_argument("--random-contrast-weight", type=float, default=0.3)
    p.add_argument("--pointwise-contrast-weight", type=float, default=0.5)
    p.add_argument("--entropy-target", type=float, default=0.35)
    p.add_argument("--entropy-weight", type=float, default=0.04)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
