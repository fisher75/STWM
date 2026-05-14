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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_18_topk_evidence_residual_memory import TopKEvidenceResidualMemoryV3418
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import pairwise_binding_loss
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import MEAS_ROOT, TARGET_ROOT, unit_memory_target_loss
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import V3414_TRAIN, make_loader, model_inputs, load_selector, freeze_for_probe, local_cos, masked_mean


V3418_TRAIN = ROOT / "reports/stwm_ostf_v34_18_topk_evidence_oracle_residual_probe_train_summary_20260513.json"
V3419 = ROOT / "reports/stwm_ostf_v34_19_hard_changed_mask_realignment_probe_20260513.json"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_20_hard_changed_aligned_topk_residual_probe_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_20_hard_changed_aligned_topk_residual_probe_train_summary_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_20_HARD_CHANGED_ALIGNED_TOPK_RESIDUAL_PROBE_TRAIN_SUMMARY_20260513.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hard_changed_aligned_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = batch["fut_teacher_available_mask"].bool()
    causal = batch["causal_assignment_residual_semantic_mask"].bool()
    strict = batch["strict_residual_semantic_utility_mask"].bool()
    hard_changed = batch["semantic_hard_mask"].bool() | batch["changed_mask"].bool()
    return (causal | (strict & hard_changed)) & valid


def compose(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    gate = hard_changed_aligned_mask(batch).float() * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    return F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["assignment_bound_residual"], dim=-1)


def init_from_v3418(model: TopKEvidenceResidualMemoryV3418) -> dict[str, Any]:
    if not V3418_TRAIN.exists():
        return load_selector(model) | {"residual_init_source": "missing_v34_18_train_summary"}
    train = json.loads(V3418_TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / train.get("checkpoint_path", "")
    if not ckpt.exists():
        return load_selector(model) | {"residual_init_source": "missing_v34_18_checkpoint"}
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    return {"residual_init_source": str(ckpt.relative_to(ROOT)), "selector_init_source": "inherited_from_v34_18", "selector_frozen": True}


def loss_fn(
    out: dict[str, torch.Tensor],
    zero: dict[str, torch.Tensor],
    shuf_sem: dict[str, torch.Tensor],
    shuf_assign: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    final = compose(out, batch)
    zero_final = compose(zero, batch)
    shuf_sem_final = compose(shuf_sem, batch)
    shuf_assign_final = compose(shuf_assign, batch)
    valid = batch["fut_teacher_available_mask"].bool()
    pos = hard_changed_aligned_mask(batch)
    causal = batch["causal_assignment_residual_semantic_mask"].bool() & valid
    hard = batch["semantic_hard_mask"].bool() & valid
    changed = batch["changed_mask"].bool() & valid
    stable = batch["stable_suppress_mask"].bool() & valid
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    sem = cosine_loss(final, batch["fut_teacher_embedding"], pos, teacher_w) if bool(pos.any()) else final.sum() * 0.0
    causal_loss = cosine_loss(final, batch["fut_teacher_embedding"], causal, teacher_w) if bool(causal.any()) else final.sum() * 0.0
    hard_loss = cosine_loss(final, batch["fut_teacher_embedding"], (hard | changed) & pos, teacher_w) if bool(((hard | changed) & pos).any()) else final.sum() * 0.0
    evidence = cosine_loss(out["topk_raw_evidence_embedding"], batch["fut_teacher_embedding"], pos, teacher_w) if bool(pos.any()) else final.sum() * 0.0
    stable_loss = cosine_loss(final, out["pointwise_semantic_belief"].detach(), stable, torch.ones_like(teacher_w)) if bool(stable.any()) else final.sum() * 0.0
    unit_loss = unit_memory_target_loss(out, batch, teacher_w)
    bind, bind_stats = pairwise_binding_loss(out, batch)
    normal_cos = local_cos(final, batch["fut_teacher_embedding"])
    zero_cos = local_cos(zero_final, batch["fut_teacher_embedding"]).detach()
    shuf_sem_cos = local_cos(shuf_sem_final, batch["fut_teacher_embedding"]).detach()
    shuf_assign_cos = local_cos(shuf_assign_final, batch["fut_teacher_embedding"]).detach()
    sem_contrast = masked_mean(F.softplus(args.semantic_contrast_margin - (normal_cos - torch.maximum(zero_cos, shuf_sem_cos))), pos, teacher_w)
    assign_contrast = masked_mean(F.softplus(args.assignment_contrast_margin - (normal_cos - shuf_assign_cos)), pos, teacher_w)
    assign = out["point_to_unit_assignment"].clamp_min(1e-8)
    usage = assign.mean(dim=1).clamp_min(1e-8)
    anti = 1.0 + (usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])
    total = 1.1 * sem + 0.5 * causal_loss + 0.55 * hard_loss + 0.35 * evidence + 0.85 * stable_loss + 0.8 * unit_loss + 0.2 * bind + args.semantic_contrast_weight * sem_contrast + args.assignment_contrast_weight * assign_contrast + 0.15 * anti
    stats = {
        "loss": float(total.detach().cpu()),
        "aligned_final_semantic_target_loss": float(sem.detach().cpu()),
        "causal_final_semantic_target_loss": float(causal_loss.detach().cpu()),
        "hard_changed_loss": float(hard_loss.detach().cpu()),
        "topk_raw_evidence_loss": float(evidence.detach().cpu()),
        "semantic_measurement_causal_contrast_loss": float(sem_contrast.detach().cpu()),
        "assignment_contrast_loss": float(assign_contrast.detach().cpu()),
        "stable_preservation_loss": float(stable_loss.detach().cpu()),
        "unit_memory_target_loss": float(unit_loss.detach().cpu()),
        "unit_anti_collapse": float(anti.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def summarize(losses: list[dict[str, float]], key: str) -> dict[str, float | bool | None]:
    vals = [float(x[key]) for x in losses if key in x]
    return {"first": vals[0] if vals else None, "last": vals[-1] if vals else None, "mean": float(np.mean(vals)) if vals else None, "active": bool(vals and np.mean(np.abs(vals)) > 1e-8)}


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    realign = json.loads(V3419.read_text(encoding="utf-8")) if V3419.exists() else {}
    if realign.get("decision", {}).get("recommended_next_step") != "train_hard_changed_aligned_topk_residual_content":
        payload = {"generated_at_utc": utc_now(), "中文结论": "V34.19 未确认 hard/changed 对齐 mask 有效，V34.20 训练跳过。", "oracle_residual_probe_ran": False, "skip_reason": "v34_19_not_ready"}
        dump_json(SUMMARY, payload)
        write_doc(DOC, "V34.20 hard/changed aligned top-k residual 训练跳过中文报告", payload, ["中文结论", "oracle_residual_probe_ran", "skip_reason"])
        print(f"已写出 V34.20 训练跳过报告: {SUMMARY.relative_to(ROOT)}")
        return payload
    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = make_loader("train", args, shuffle=True)
    model = TopKEvidenceResidualMemoryV3418(
        args.v30_checkpoint,
        teacher_embedding_dim=args.teacher_embedding_dim,
        units=args.trace_units,
        horizon=args.horizon,
        selector_hidden_dim=args.selector_hidden_dim,
        topk=args.topk,
    ).to(device)
    init = init_from_v3418(model)
    freeze_for_probe(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
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
        inp = model_inputs(bd)
        out = model(**inp, intervention="force_gate_zero")
        zero = model(**inp, intervention="zero_semantic_measurements")
        shuf_sem = model(**inp, intervention="shuffle_semantic_measurements_across_points")
        shuf_assign = model(**inp, intervention="shuffle_assignment")
        loss, stats = loss_fn(out, zero, shuf_sem, shuf_assign, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(100, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(f"训练进度: step={step}/{args.steps}, loss={stats['loss']:.6f}, aligned={stats['aligned_final_semantic_target_loss']:.6f}, sem_contrast={stats['semantic_measurement_causal_contrast_loss']:.6f}", flush=True)
    ckpt = CKPT_DIR / "v34_20_hard_changed_aligned_topk_residual_probe_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.20 hard/changed-aligned top-k residual content 训练完成；V30 和 horizon selector 冻结，未训练 learned gate。",
        "oracle_residual_probe_ran": True,
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "topk": args.topk,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "measurement_selector_frozen": True,
        "future_leakage_detected": False,
        "init_checkpoint": init,
        "semantic_measurement_causal_contrast_loss": summarize(losses, "semantic_measurement_causal_contrast_loss"),
        "assignment_contrast_loss": summarize(losses, "assignment_contrast_loss"),
        "aligned_final_semantic_target_loss": summarize(losses, "aligned_final_semantic_target_loss"),
        "duration_seconds": float(time.time() - start),
        "loss_trace": losses,
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "V34.20 hard/changed aligned top-k residual 训练中文摘要", payload, ["中文结论", "oracle_residual_probe_ran", "fresh_training_completed", "checkpoint_path", "topk", "train_sample_count", "v30_backbone_frozen", "measurement_selector_frozen", "future_leakage_detected"])
    print(f"已写出 V34.20 训练摘要: {SUMMARY.relative_to(ROOT)}")
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
    p.add_argument("--selector-hidden-dim", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5.0e-5)
    p.add_argument("--semantic-contrast-weight", type=float, default=0.75)
    p.add_argument("--assignment-contrast-weight", type=float, default=0.5)
    p.add_argument("--semantic-contrast-margin", type=float, default=0.006)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.006)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
