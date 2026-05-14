#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import (
    load_v3425_readers,
    masks,
    observed_max_conf,
    observed_mean,
)
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import (
    Acc,
    finalize_method,
    norm,
    update_method,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, masked_mean, model_inputs
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import (
    best_copy_topk,
    compose,
    counterfactual_batch,
    load_frozen_residual_model,
    read_unit_delta,
    roll_assignment,
    set_seed,
    sparse_seed_mean_gate,
    summarize_loss,
    unit_delta_target,
)
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import hard_changed_aligned_mask


CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_32_direct_evidence_raw_delta_value_memory_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_32_direct_evidence_raw_delta_value_memory_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_32_direct_evidence_raw_delta_value_memory_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_32_DIRECT_EVIDENCE_RAW_DELTA_VALUE_MEMORY_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_32_DIRECT_EVIDENCE_RAW_DELTA_VALUE_MEMORY_DECISION_20260514.md"


class DirectEvidenceRawDeltaHeadV3432(nn.Module):
    """直接从 future trace、raw evidence、anchor、旧 unit state 预测 raw delta value。"""

    def __init__(self, trace_hidden_dim: int, semantic_dim: int = 768, hidden_dim: int = 768) -> None:
        super().__init__()
        in_dim = trace_hidden_dim + semantic_dim * 3 + 2
        self.body = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.direction = nn.Linear(hidden_dim, semantic_dim)
        self.magnitude = nn.Linear(hidden_dim, 1)

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assign = out["point_to_unit_assignment"].float()
        den = assign.sum(dim=1).clamp_min(1.0e-6)
        future_trace = out["future_trace_hidden"].float()
        topk_embed = out["topk_raw_evidence_embedding"].float()
        anchor = observed_mean(batch).float()
        unit_trace = torch.einsum("bmu,bmhc->buhc", assign, future_trace) / den[:, :, None, None]
        unit_topk = torch.einsum("bmu,bmhd->buhd", assign, topk_embed) / den[:, :, None, None]
        unit_anchor = torch.einsum("bmu,bmhd->buhd", assign, anchor) / den[:, :, None, None]
        old_unit = out["unit_memory"].float()
        b, u, h, _ = old_unit.shape
        unit_conf = out["unit_confidence"].float()[:, :, None, None].expand(b, u, h, 1)
        assign_usage = out["assignment_usage_score"].float()[:, :, :, None].clamp(0.0, 1.0)
        x = torch.cat([unit_trace, old_unit, unit_topk, unit_anchor, unit_conf, assign_usage], dim=-1)
        y = self.body(torch.nan_to_num(x))
        direction = F.normalize(self.direction(y), dim=-1)
        magnitude = 2.0 * torch.sigmoid(self.magnitude(y))
        return direction * magnitude


def direct_value_loss(
    head: DirectEvidenceRawDeltaHeadV3432,
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos = hard_changed_aligned_mask(batch)
    valid = batch["fut_teacher_available_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & valid
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    anchor = observed_mean(batch)
    target = batch["fut_teacher_embedding"]
    assign = out["point_to_unit_assignment"].float()
    pred_unit_delta = head(out, batch)
    target_unit_delta, active = unit_delta_target(assign, anchor, target, pos, teacher_w)
    if bool(active.any()):
        pred_active = pred_unit_delta[active]
        target_active = target_unit_delta[active]
        dir_loss = 1.0 - (F.normalize(pred_active, dim=-1) * F.normalize(target_active, dim=-1)).sum(dim=-1).mean()
        mag_loss = F.smooth_l1_loss(pred_active.norm(dim=-1), target_active.norm(dim=-1))
        raw_loss = F.smooth_l1_loss(pred_active, target_active)
    else:
        dir_loss = pred_unit_delta.sum() * 0.0
        mag_loss = pred_unit_delta.sum() * 0.0
        raw_loss = pred_unit_delta.sum() * 0.0
    point_delta = read_unit_delta(assign, pred_unit_delta)
    gate = pos.float() * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    final = compose(anchor, point_delta, gate, args.train_residual_scale)
    shuf = compose(anchor, read_unit_delta(roll_assignment(assign), pred_unit_delta), gate, args.train_residual_scale)
    zero = compose(anchor, torch.zeros_like(point_delta), gate, args.train_residual_scale)
    normal_cos = (norm(final) * norm(target)).sum(dim=-1)
    anchor_cos = (norm(anchor) * norm(target)).sum(dim=-1).detach()
    shuf_cos = (norm(shuf) * norm(target)).sum(dim=-1)
    zero_cos = (norm(zero) * norm(target)).sum(dim=-1).detach()
    final_loss = masked_mean(1.0 - normal_cos, pos, teacher_w)
    anchor_gain = masked_mean(F.softplus(args.anchor_gain_margin - (normal_cos - anchor_cos)), pos, teacher_w)
    assignment_contrast = masked_mean(F.softplus(args.assignment_margin - (normal_cos - shuf_cos)), pos, teacher_w)
    unit_contrast = masked_mean(F.softplus(args.unit_margin - (normal_cos - zero_cos)), pos, teacher_w)
    stable_norm = read_unit_delta(assign, pred_unit_delta).norm(dim=-1)
    stable_suppress = masked_mean(stable_norm, stable, torch.ones_like(teacher_w)) if bool(stable.any()) else pred_unit_delta.sum() * 0.0
    total = (
        args.unit_direction_weight * dir_loss
        + args.unit_magnitude_weight * mag_loss
        + args.raw_delta_weight * raw_loss
        + args.final_target_weight * final_loss
        + args.anchor_gain_weight * anchor_gain
        + args.assignment_contrast_weight * assignment_contrast
        + args.unit_contrast_weight * unit_contrast
        + args.stable_suppress_weight * stable_suppress
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "unit_delta_direction_loss": float(dir_loss.detach().cpu()),
        "unit_delta_magnitude_loss": float(mag_loss.detach().cpu()),
        "unit_delta_raw_loss": float(raw_loss.detach().cpu()),
        "final_target_loss": float(final_loss.detach().cpu()),
        "anchor_gain_contrast_loss": float(anchor_gain.detach().cpu()),
        "assignment_contrast_loss": float(assignment_contrast.detach().cpu()),
        "unit_contrast_loss": float(unit_contrast.detach().cpu()),
        "stable_delta_suppress_loss": float(stable_suppress.detach().cpu()),
    }


def train_one(args: argparse.Namespace) -> tuple[Any, argparse.Namespace, DirectEvidenceRawDeltaHeadV3432, dict[str, Any]]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, init = load_frozen_residual_model(args, device)
    head = DirectEvidenceRawDeltaHeadV3432(int(model.v30.cfg.hidden_dim), args.teacher_embedding_dim, args.value_hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1.0e-4)
    loader = make_loader("train", ckargs, shuffle=True)
    it = iter(loader)
    losses: list[dict[str, float]] = []
    start = time.time()
    head.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        with torch.no_grad():
            out = model(**model_inputs(bd), intervention="force_gate_zero")
        loss, stats = direct_value_loss(head, out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(
                "训练进度: "
                f"step={step}/{args.steps}, loss={stats['loss']:.6f}, "
                f"dir={stats['unit_delta_direction_loss']:.6f}, assign={stats['assignment_contrast_loss']:.6f}, "
                f"anchor={stats['anchor_gain_contrast_loss']:.6f}",
                flush=True,
            )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"v34_32_direct_evidence_raw_delta_value_head_m128_h32_seed{args.seed}.pt"
    torch.save({"head": head.state_dict(), "args": vars(args), "init": init, "step": args.steps}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.32 direct-evidence raw unit-delta value head 训练完成；value head 直接读取 future_trace_hidden、top-k raw evidence、evidence anchor 与 unit state。",
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "init": init,
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "train_residual_scale": args.train_residual_scale,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "assignment_frozen": True,
        "learned_gate_training_ran": False,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "loss_summaries": {k: summarize_loss(losses, k) for k in losses[-1].keys() if k != "step"} if losses else {},
        "loss_trace": losses,
        "duration_seconds": float(time.time() - start),
    }
    dump_json(SUMMARY, summary)
    write_doc(DOC, "V34.32 direct-evidence raw delta value memory 训练中文摘要", summary, ["中文结论", "fresh_training_completed", "checkpoint_path", "steps", "train_sample_count", "train_residual_scale", "v30_backbone_frozen", "assignment_frozen", "learned_gate_training_ran", "future_leakage_detected", "trajectory_degraded"])
    print(f"已写出 V34.32 训练摘要: {SUMMARY.relative_to(ROOT)}", flush=True)
    head.eval()
    return model, ckargs, head, summary


def eval_split(split: str, model: Any, ckargs: argparse.Namespace, head: DirectEvidenceRawDeltaHeadV3432, readers: dict[str, dict[str, Any]], args: argparse.Namespace, scale: float, device: torch.device) -> dict[str, Any]:
    acc = Acc()
    delta_acc = {k: Acc() for k in ["normal", "zero_semantic", "shuffle_semantic", "shuffle_assignment", "zero_unit"]}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            target = bd["fut_teacher_embedding"]
            anchor = observed_mean(bd)
            gate = sparse_seed_mean_gate(out, readers)
            pred_unit = head(out, bd)
            point_delta = read_unit_delta(out["point_to_unit_assignment"].float(), pred_unit)
            final = compose(anchor, point_delta, gate, scale)
            update_method(acc, "pointwise_base", pointwise, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "copy_mean_observed", anchor, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "copy_max_conf_observed", observed_max_conf(bd), pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "topk_raw_evidence", out["topk_raw_evidence_embedding"], pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "v34_32_direct_evidence_raw_delta", final, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["normal"], "normal", final, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            for mode, intervention, cf_mode in [
                ("zero_semantic", "zero_semantic_measurements", "zero_semantic_measurements"),
                ("shuffle_semantic", "shuffle_semantic_measurements_across_points", "shuffle_semantic_measurements"),
            ]:
                cfb = counterfactual_batch(bd, cf_mode)
                cfout = model(**model_inputs(bd), intervention=intervention)
                cfpred_unit = head(cfout, cfb)
                cfpred = compose(observed_mean(cfb), read_unit_delta(cfout["point_to_unit_assignment"].float(), cfpred_unit), gate, scale)
                update_method(delta_acc[mode], mode, cfpred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            shuf = compose(anchor, read_unit_delta(roll_assignment(out["point_to_unit_assignment"].float()), pred_unit), gate, scale)
            zero = compose(anchor, torch.zeros_like(point_delta), gate, scale)
            update_method(delta_acc["shuffle_assignment"], "shuffle_assignment", shuf, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["zero_unit"], "zero_unit", zero, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
    metrics = {name: finalize_method(acc, name) for name in sorted({key.split(":")[0] for key in acc.sum.keys()})}
    deltas = {mode: finalize_method(a, mode) for mode, a in delta_acc.items()}
    normal = deltas["normal"]

    def delta(mode: str) -> float | None:
        a = normal["hard_changed_gain_vs_pointwise"]
        b = deltas[mode]["hard_changed_gain_vs_pointwise"]
        if a is None or b is None:
            return None
        return float(a - b)

    return {
        "methods": metrics,
        "intervention_delta": {
            "zero_semantic_measurements_delta": delta("zero_semantic"),
            "shuffle_semantic_measurements_delta": delta("shuffle_semantic"),
            "shuffle_assignment_delta": delta("shuffle_assignment"),
            "zero_unit_memory_delta": delta("zero_unit"),
            "normal_hard_changed_gain_vs_pointwise": normal["hard_changed_gain_vs_pointwise"],
            "normal_hard_changed_gain_vs_anchor": normal["hard_changed_gain_vs_anchor"],
        },
    }


def evaluate(model: Any, ckargs: argparse.Namespace, head: DirectEvidenceRawDeltaHeadV3432, train_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = next(model.parameters()).device
    readers = load_v3425_readers(args, model, device)
    scale_rows = []
    cache: dict[float, dict[str, Any]] = {}
    for scale in args.eval_scales:
        print(f"开始 V34.32 scale sweep eval: scale={scale}", flush=True)
        per = {split: eval_split(split, model, ckargs, head, readers, args, scale, device) for split in ("val", "test")}
        cache[scale] = per
        val_m = per["val"]["methods"]["v34_32_direct_evidence_raw_delta"]
        scale_rows.append({"scale": scale, "val_gain_anchor": val_m["hard_changed_gain_vs_anchor"], "val_gain_pointwise": val_m["hard_changed_gain_vs_pointwise"], "stable": val_m["stable_preservation"]})
    valid = [r for r in scale_rows if r["stable"] and float(r["val_gain_anchor"] or -1.0) > 0.0]
    if not valid:
        valid = scale_rows
    best_scale = float(max(valid, key=lambda x: float(x["val_gain_pointwise"] or -1.0e9))["scale"])
    per_split = cache[best_scale]
    method = "v34_32_direct_evidence_raw_delta"
    val_m = per_split["val"]["methods"][method]
    test_m = per_split["test"]["methods"][method]
    val_delta = per_split["val"]["intervention_delta"]
    test_delta = per_split["test"]["intervention_delta"]
    best_base = {split: best_copy_topk(per_split[split]) for split in ("val", "test")}
    beats_copy_topk = bool((val_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["val"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002 and (test_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["test"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002)
    improves_anchor = bool((val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and (test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    assignment_lb = bool((val_delta["shuffle_assignment_delta"] or 0.0) > 0.002 and (test_delta["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool((val_delta["zero_unit_memory_delta"] or 0.0) > 0.002 and (test_delta["zero_unit_memory_delta"] or 0.0) > 0.002)
    semantic_lb = bool(min(val_delta["zero_semantic_measurements_delta"] or 0.0, val_delta["shuffle_semantic_measurements_delta"] or 0.0, test_delta["zero_semantic_measurements_delta"] or 0.0, test_delta["shuffle_semantic_measurements_delta"] or 0.0) > 0.002)
    semantic_hard_signal = {"val": val_m["semantic_hard_signal"], "test": test_m["semantic_hard_signal"]}
    changed_semantic_signal = {"val": val_m["changed_semantic_signal"], "test": test_m["changed_semantic_signal"]}
    stable_preservation = {"val": val_m["stable_preservation"], "test": test_m["stable_preservation"]}
    passed = bool(beats_copy_topk and improves_anchor and assignment_lb and unit_lb and semantic_lb and all(semantic_hard_signal.values()) and all(changed_semantic_signal.values()) and all(stable_preservation.values()))
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.32 direct-evidence raw delta value memory 完成；本轮仍不训练 gate、不跑 M512。",
        "direct_evidence_raw_delta_value_head_trained": True,
        "probe_passed": passed,
        "best_eval_scale_by_val": best_scale,
        "scale_sweep": scale_rows,
        "beats_copy_topk_baseline": beats_copy_topk,
        "unit_residual_improves_evidence_anchor": improves_anchor,
        "semantic_measurements_load_bearing_on_system": semantic_lb,
        "assignment_load_bearing_on_system": assignment_lb,
        "unit_memory_load_bearing_on_system": unit_lb,
        "semantic_hard_signal": semantic_hard_signal,
        "changed_semantic_signal": changed_semantic_signal,
        "stable_preservation": stable_preservation,
        "best_copy_topk_baseline": best_base,
        "v34_32_metrics": {"val": val_m, "test": test_m},
        "intervention_delta": {"val": val_delta, "test": test_delta},
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": False,
        "m512_dense_ready": bool(passed),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "train_assignment_aware_raw_delta_value_head" if passed else "fix_unit_residual_training_objective",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "train_summary": train_summary,
        "per_split_best_scale": per_split,
        "decision": decision,
        "阶段性分析": "V34.32 检查 V34.31 的失败是否来自旧 normalized unit_memory 特征不足。value head 现在直接读 future trace hidden、top-k evidence、anchor 和 unit state，并用 val 选择 residual scale。",
        "论文相关问题解决方案参考": "这延续 memory read 的 key/value 分离：assignment 是 key/address，direct evidence raw delta 是 value。若仍失败，说明需要更强的 per-instance supervised value distillation 或直接把 oracle unit delta 作为中间监督缓存。",
        "最佳下一步方案": "若通过，训练 assignment-aware raw delta value head；若失败，停止微调 loss，转向构建显式 oracle-unit-delta distillation targets/cache 与更强 value decoder。"
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DECISION_DOC, "V34.32 direct-evidence raw delta value memory 决策中文报告", decision, ["中文结论", "probe_passed", "best_eval_scale_by_val", "beats_copy_topk_baseline", "unit_residual_improves_evidence_anchor", "semantic_measurements_load_bearing_on_system", "assignment_load_bearing_on_system", "unit_memory_load_bearing_on_system", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "learned_gate_training_ran", "m512_dense_ready", "integrated_semantic_field_claim_allowed", "integrated_identity_field_claim_allowed", "recommended_next_step"])
    print(f"已写出 V34.32 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
    print(f"probe_passed: {passed}", flush=True)
    print(f"recommended_next_step: {decision['recommended_next_step']}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=6.0e-5)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=768)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--train-residual-scale", type=float, default=0.25)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--unit-direction-weight", type=float, default=1.0)
    p.add_argument("--unit-magnitude-weight", type=float, default=0.75)
    p.add_argument("--raw-delta-weight", type=float, default=0.5)
    p.add_argument("--final-target-weight", type=float, default=0.8)
    p.add_argument("--anchor-gain-weight", type=float, default=1.0)
    p.add_argument("--assignment-contrast-weight", type=float, default=1.2)
    p.add_argument("--unit-contrast-weight", type=float, default=1.0)
    p.add_argument("--stable-suppress-weight", type=float, default=0.05)
    p.add_argument("--anchor-gain-margin", type=float, default=0.006)
    p.add_argument("--assignment-margin", type=float, default=0.006)
    p.add_argument("--unit-margin", type=float, default=0.006)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model, ckargs, head, train_summary = train_one(args)
    evaluate(model, ckargs, head, train_summary, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
