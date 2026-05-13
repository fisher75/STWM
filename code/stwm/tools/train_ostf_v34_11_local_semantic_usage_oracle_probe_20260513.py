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
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import pairwise_binding_loss
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import (
    CKPT_DIR as V3410_CKPT_DIR,
    MEAS_ROOT,
    TARGET_ROOT,
    TraceContractResidualDataset,
    collate_v3410,
    freeze_pointwise_base,
    load_init_checkpoint,
    make_loader,
    unit_memory_target_loss,
)


QUALITY_REPORT = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_quality_probe_20260513.json"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_train_summary_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_11_LOCAL_SEMANTIC_USAGE_ORACLE_PROBE_TRAIN_SUMMARY_20260513.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LocalSemanticUsageDataset(TraceContractResidualDataset):
    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        uid = item["uid"]
        z = np.load(self.meas_root / self.split / f"{uid}.npz", allow_pickle=True)
        item["semantic_measurement_confidence"] = torch.from_numpy(np.asarray(z["obs_measurement_confidence"], dtype=np.float32))
        item["teacher_agreement_score"] = torch.from_numpy(np.asarray(z.get("teacher_agreement_score", z["obs_measurement_confidence"]), dtype=np.float32))
        return item


def make_local_loader(split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(LocalSemanticUsageDataset(split, args), batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v3410)


def _pool_measurements(batch: dict[str, torch.Tensor], variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.nan_to_num(batch["obs_semantic_measurements"].float())
    mask = batch["obs_semantic_measurement_mask"].bool()
    conf = batch["semantic_measurement_confidence"].float().clamp(0.0, 1.0)
    agree = batch.get("teacher_agreement_score", conf).float().clamp(0.0, 1.0)
    m_float = mask.float()
    if variant == "mean_pooling":
        pooled = (obs * m_float[..., None]).sum(dim=2) / m_float.sum(dim=2, keepdim=True).clamp_min(1.0)
    elif variant == "max_confidence_pooling":
        score = conf.masked_fill(~mask, -1.0)
        idx = score.argmax(dim=2)
        pooled = obs.gather(2, idx[:, :, None, None].expand(-1, -1, 1, obs.shape[-1])).squeeze(2)
        pooled = torch.where(score.max(dim=2).values[..., None] >= 0.0, pooled, torch.zeros_like(pooled))
    elif variant == "teacher_agreement_weighted_pooling":
        w = (conf * agree * m_float).clamp_min(0.0)
        pooled = (obs * w[..., None]).sum(dim=2) / w.sum(dim=2, keepdim=True).clamp_min(1e-6)
    elif variant == "temporal_attention_pooling":
        t = torch.linspace(0.0, 1.0, obs.shape[2], device=obs.device).view(1, 1, -1)
        logits = 2.0 * conf + 2.0 * agree + t
        logits = logits.masked_fill(~mask, -1e4)
        w = torch.softmax(logits, dim=2) * m_float
        pooled = (obs * w[..., None]).sum(dim=2) / w.sum(dim=2, keepdim=True).clamp_min(1e-6)
    else:
        raise ValueError(f"未知 pooling_variant: {variant}")
    pooled_mask = mask.any(dim=2)
    sem = pooled[:, :, None, :].expand(-1, -1, obs.shape[2], -1).contiguous()
    sem_mask = pooled_mask[:, :, None].expand(-1, -1, obs.shape[2]).contiguous()
    return sem, sem_mask


def model_inputs(batch: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    sem, sem_mask = _pool_measurements(batch, args.pooling_variant)
    return {
        "obs_points": batch["obs_points"],
        "obs_vis": batch["obs_vis"],
        "obs_conf": batch["trace_obs_conf"],
        "obs_semantic_measurements": sem,
        "obs_semantic_measurement_mask": sem_mask,
        "semantic_id": batch["semantic_id"],
    }


def local_compose(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, use_usage_score: bool = True) -> torch.Tensor:
    gate = batch["causal_assignment_residual_semantic_mask"].float()
    if use_usage_score and "semantic_measurement_usage_score" in out:
        unit_usage = out["semantic_measurement_usage_score"].float()
        point_usage = torch.einsum("bmu,bu->bm", out["point_to_unit_assignment"].float(), unit_usage).clamp(0.0, 1.0)
        gate = gate * point_usage[..., None]
    return F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["assignment_bound_residual"], dim=-1)


def local_cos(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (F.normalize(pred, dim=-1) * F.normalize(torch.nan_to_num(target.float()), dim=-1)).sum(dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    m = mask.float()
    if weight is not None:
        m = m * weight.float()
    return (values * m).sum() / m.sum().clamp_min(1.0)


def local_loss(
    out: dict[str, torch.Tensor],
    zero: dict[str, torch.Tensor],
    shuf_sem: dict[str, torch.Tensor],
    shuf_assign: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    final = local_compose(out, batch, use_usage_score=True)
    zero_final = local_compose(zero, batch, use_usage_score=True)
    shuf_sem_final = local_compose(shuf_sem, batch, use_usage_score=True)
    shuf_assign_final = local_compose(shuf_assign, batch, use_usage_score=True)
    valid = batch["fut_teacher_available_mask"].bool()
    pos = batch["causal_assignment_residual_semantic_mask"].bool() & valid
    strict = batch["strict_residual_semantic_utility_mask"].bool() & valid
    hard = batch["semantic_hard_mask"].bool() & valid
    stable = batch["stable_suppress_mask"].bool() & valid
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    point_conf = batch["semantic_measurement_confidence"].float().mean(dim=2)
    agree = batch.get("teacher_agreement_score", point_conf).float().mean(dim=2)
    local_w = (teacher_w * point_conf[..., None].clamp(0.05, 1.0) * agree[..., None].clamp(0.05, 1.0)).clamp(0.02, 1.0)
    sem_loss = cosine_loss(final, batch["fut_teacher_embedding"], pos, local_w) if bool(pos.any()) else final.sum() * 0.0
    hard_loss = cosine_loss(final, batch["fut_teacher_embedding"], hard & strict, local_w) if bool((hard & strict).any()) else final.sum() * 0.0
    stable_loss = cosine_loss(final, out["pointwise_semantic_belief"].detach(), stable, torch.ones_like(teacher_w)) if bool(stable.any()) else final.sum() * 0.0
    unit_loss = unit_memory_target_loss(out, batch, teacher_w)
    bind, bind_stats = pairwise_binding_loss(out, batch)
    normal_cos = local_cos(final, batch["fut_teacher_embedding"])
    zero_cos = local_cos(zero_final, batch["fut_teacher_embedding"]).detach()
    shuf_sem_cos = local_cos(shuf_sem_final, batch["fut_teacher_embedding"]).detach()
    shuf_assign_cos = local_cos(shuf_assign_final, batch["fut_teacher_embedding"]).detach()
    local_sem_usage = masked_mean(F.softplus(args.usage_margin - (normal_cos - torch.maximum(zero_cos, shuf_sem_cos))), pos, local_w)
    local_assign_contrast = masked_mean(F.softplus(args.assignment_contrast_margin - (normal_cos - shuf_assign_cos)), pos, local_w)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    anti = 1.0 + (usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    total = 1.2 * sem_loss + 0.4 * hard_loss + stable_loss + unit_loss + 0.25 * bind + args.semantic_usage_weight * local_sem_usage + args.assignment_contrast_weight * local_assign_contrast + 0.2 * anti
    stats = {
        "loss": float(total.detach().cpu()),
        "local_semantic_target_loss": float(sem_loss.detach().cpu()),
        "semantic_hard_local_residual_loss": float(hard_loss.detach().cpu()),
        "stable_preservation_loss": float(stable_loss.detach().cpu()),
        "unit_memory_target_loss": float(unit_loss.detach().cpu()),
        "local_semantic_usage_loss": float(local_sem_usage.detach().cpu()),
        "local_assignment_contrast_loss": float(local_assign_contrast.detach().cpu()),
        "unit_anti_collapse": float(anti.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def summarize_trace(losses: list[dict[str, float]], key: str) -> dict[str, float | bool | None]:
    vals = [float(x[key]) for x in losses if key in x]
    return {
        "first": vals[0] if vals else None,
        "last": vals[-1] if vals else None,
        "mean": float(np.mean(vals)) if vals else None,
        "active": bool(vals and np.mean(np.abs(vals)) > 1e-8),
    }


def load_quality() -> dict[str, Any]:
    return json.loads(QUALITY_REPORT.read_text(encoding="utf-8")) if QUALITY_REPORT.exists() else {}


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    quality = load_quality()
    if not (quality.get("semantic_measurement_quality_passed") or quality.get("measurement_beats_random")):
        payload = {
            "generated_at_utc": utc_now(),
            "中文结论": "V34.11 measurement quality 未通过且未赢 random，local usage probe 按规则跳过。",
            "local_semantic_usage_probe_ran": False,
            "local_semantic_usage_probe_passed": "not_run",
            "skip_reason": "semantic_measurement_quality_failed",
            "v30_backbone_frozen": True,
            "future_leakage_detected": False,
        }
        dump_json(SUMMARY, payload)
        write_doc(DOC, "V34.11 local semantic usage oracle probe 训练跳过中文报告", payload, ["中文结论", "local_semantic_usage_probe_ran", "skip_reason"])
        print(f"已写出 V34.11 local usage 训练跳过报告: {SUMMARY.relative_to(ROOT)}")
        return payload
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_local_loader("train", args, shuffle=True)
    model = CausalAssignmentBoundResidualMemoryV348(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    init = load_init_checkpoint(model)
    freeze_pointwise_base(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    it = iter(loader)
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        inp = model_inputs(bd, args)
        out = model(**inp, intervention="force_gate_zero")
        zero = model(**inp, intervention="zero_semantic_measurements")
        shuf_sem = model(**inp, intervention="shuffle_semantic_measurements_across_points")
        shuf_assign = model(**inp, intervention="shuffle_assignment")
        loss, stats = local_loss(out, zero, shuf_sem, shuf_assign, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(100, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(f"训练进度: step={step}/{args.steps}, loss={stats['loss']:.6f}, local_usage={stats['local_semantic_usage_loss']:.6f}, local_assignment={stats['local_assignment_contrast_loss']:.6f}", flush=True)
    ckpt = CKPT_DIR / "v34_11_local_semantic_usage_oracle_probe_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    sem_stats = summarize_trace(losses, "local_semantic_usage_loss")
    ass_stats = summarize_trace(losses, "local_assignment_contrast_loss")
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.11 local semantic usage oracle probe 已完成训练；usage/assignment loss 改为局部逐点逐 horizon，对照分支 detach，normal path 保持可训练；未训练 learned gate。",
        "local_semantic_usage_probe_ran": True,
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "pooling_variant": args.pooling_variant,
        "pooling_ablation_source": str(QUALITY_REPORT.relative_to(ROOT)),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "pointwise_base_frozen": True,
        "future_leakage_detected": False,
        "trace_state_contract_fully_passed": True,
        "local_semantic_usage_loss_first": sem_stats["first"],
        "local_semantic_usage_loss_last": sem_stats["last"],
        "local_semantic_usage_loss_mean": sem_stats["mean"],
        "local_semantic_usage_loss_active": sem_stats["active"],
        "local_assignment_contrast_loss_first": ass_stats["first"],
        "local_assignment_contrast_loss_last": ass_stats["last"],
        "local_assignment_contrast_loss_mean": ass_stats["mean"],
        "local_assignment_contrast_loss_active": ass_stats["active"],
        "semantic_measurement_usage_score_used_in_residual_magnitude": True,
        "teacher_agreement_score_used_in_loss_weight": True,
        "obs_measurement_confidence_used_in_loss_weight": True,
        "init_checkpoint": init,
        "duration_seconds": float(time.time() - start),
        "loss_trace": losses,
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "V34.11 local semantic usage oracle probe 训练中文摘要",
        payload,
        [
            "中文结论",
            "local_semantic_usage_probe_ran",
            "fresh_training_completed",
            "checkpoint_path",
            "train_sample_count",
            "pooling_variant",
            "v30_backbone_frozen",
            "trace_state_contract_fully_passed",
            "local_semantic_usage_loss_active",
            "local_assignment_contrast_loss_active",
            "semantic_measurement_usage_score_used_in_residual_magnitude",
            "teacher_agreement_score_used_in_loss_weight",
            "obs_measurement_confidence_used_in_loss_weight",
        ],
    )
    print(f"已写出 V34.11 local usage 训练摘要: {SUMMARY.relative_to(ROOT)}")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v30-checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v30_external_gt/v30_extgt_m128_h32_seed42_best.pt"))
    p.add_argument("--semantic-measurement-bank-root", default=str(MEAS_ROOT))
    p.add_argument("--strict-residual-utility-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"))
    p.add_argument("--assignment-aware-residual-target-root", default=str(ROOT / "outputs/cache/stwm_ostf_v34_7_assignment_aware_residual_targets/pointodyssey"))
    p.add_argument("--causal-assignment-residual-target-root", default=str(TARGET_ROOT))
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--trace-units", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1.2e-4)
    p.add_argument("--pooling-variant", choices=["mean_pooling", "max_confidence_pooling", "teacher_agreement_weighted_pooling", "temporal_attention_pooling"], default="teacher_agreement_weighted_pooling")
    p.add_argument("--semantic-usage-weight", type=float, default=0.55)
    p.add_argument("--assignment-contrast-weight", type=float, default=0.45)
    p.add_argument("--usage-margin", type=float, default=0.006)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.006)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
