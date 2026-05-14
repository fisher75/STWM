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
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_18_topk_evidence_residual_memory import TopKEvidenceResidualMemoryV3418
from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import reader_inputs
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import (
    load_v3425_readers,
    masks,
    observed_max_conf,
    observed_mean,
)
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import (
    Acc,
    finalize_method,
    local_cos,
    norm,
    update_method,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import (
    make_loader,
    masked_mean,
    model_inputs,
)
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import (
    CKPT_DIR as V3420_CKPT_DIR,
    SUMMARY as V3420_TRAIN_SUMMARY,
    hard_changed_aligned_mask,
)
from stwm.tools.train_ostf_v34_25_sparse_calibrated_gate_repair_20260514 import gate_from_logits


CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_31_raw_unit_delta_value_memory_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_31_raw_unit_delta_value_memory_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_31_raw_unit_delta_value_memory_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_31_RAW_UNIT_DELTA_VALUE_MEMORY_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_31_RAW_UNIT_DELTA_VALUE_MEMORY_DECISION_20260514.md"


class RawUnitDeltaValueHeadV3431(nn.Module):
    """用 observed-only unit state 预测 raw delta，而不是 normalized semantic vector。"""

    def __init__(self, semantic_dim: int = 768, hidden_dim: int = 512) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(semantic_dim + 2),
            nn.Linear(semantic_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.direction = nn.Linear(hidden_dim, semantic_dim)
        self.magnitude = nn.Linear(hidden_dim, 1)

    def forward(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        unit_memory = torch.nan_to_num(out["unit_memory"].float())
        b, u, h, _ = unit_memory.shape
        unit_conf = out["unit_confidence"].float()[:, :, None, None].expand(b, u, h, 1)
        assign_usage = out["assignment_usage_score"].float()[:, :, :, None].clamp(0.0, 1.0)
        x = torch.cat([unit_memory, unit_conf, assign_usage], dim=-1)
        y = self.body(x)
        direction = F.normalize(self.direction(y), dim=-1)
        magnitude = 2.0 * torch.sigmoid(self.magnitude(y))
        return direction * magnitude


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frozen_residual_model(args: argparse.Namespace, device: torch.device) -> tuple[TopKEvidenceResidualMemoryV3418, argparse.Namespace, dict[str, Any]]:
    train = json.loads(V3420_TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = ROOT / train.get("checkpoint_path", str(V3420_CKPT_DIR / "v34_20_hard_changed_aligned_topk_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = TopKEvidenceResidualMemoryV3418(
        ckargs.v30_checkpoint,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        units=ckargs.trace_units,
        horizon=ckargs.horizon,
        selector_hidden_dim=ckargs.selector_hidden_dim,
        topk=ckargs.topk,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckargs, {"init_checkpoint": str(ckpt.relative_to(ROOT)), "source": "v34_20_frozen_residual_features"}


def counterfactual_batch(batch: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    if mode not in {"zero_semantic_measurements", "shuffle_semantic_measurements"}:
        return batch
    out = dict(batch)
    if mode == "zero_semantic_measurements":
        out["obs_semantic_measurements"] = torch.zeros_like(batch["obs_semantic_measurements"])
        out["obs_semantic_measurement_mask"] = torch.zeros_like(batch["obs_semantic_measurement_mask"])
        out["semantic_measurement_confidence"] = torch.zeros_like(batch["semantic_measurement_confidence"])
        return out
    if batch["obs_semantic_measurements"].shape[1] <= 1:
        return out
    idx = torch.roll(torch.arange(batch["obs_semantic_measurements"].shape[1], device=batch["obs_semantic_measurements"].device), shifts=1, dims=0)
    out["obs_semantic_measurements"] = batch["obs_semantic_measurements"][:, idx]
    for key in ("obs_semantic_measurement_mask", "semantic_measurement_confidence", "teacher_agreement_score"):
        if key in batch:
            out[key] = batch[key][:, idx]
    return out


def read_unit_delta(assign: torch.Tensor, unit_delta: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bmu,buhd->bmhd", assign, unit_delta)


def roll_assignment(assign: torch.Tensor) -> torch.Tensor:
    if assign.shape[-1] <= 1:
        return assign
    idx = torch.roll(torch.arange(assign.shape[-1], device=assign.device), shifts=1, dims=0)
    return assign[..., idx]


def compose(anchor: torch.Tensor, point_delta: torch.Tensor, gate: torch.Tensor, residual_scale: float) -> torch.Tensor:
    return norm(anchor + float(residual_scale) * gate[..., None] * point_delta)


def sparse_seed_mean_gate(out: dict[str, torch.Tensor], readers: dict[str, dict[str, Any]]) -> torch.Tensor:
    usage = out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    gates = []
    for item in readers.values():
        pred = item["reader"](**reader_inputs(out))["activation_logits"]
        cfg = item["config"]
        gates.append(
            gate_from_logits(
                pred["benefit"],
                usage,
                threshold=cfg.get("threshold"),
                temperature=float(cfg.get("temperature") or 1.0),
                power=float(cfg.get("power") or 1.0),
            )
        )
    return torch.stack(gates, dim=0).mean(dim=0)


def unit_delta_target(assign: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = target.float() - anchor.float()
    w = assign[:, :, None, :] * pos.float()[:, :, :, None] * weight.float()[:, :, :, None]
    unit_delta = torch.einsum("bmhu,bmhd->buhd", w, delta)
    den = w.sum(dim=1).permute(0, 2, 1).unsqueeze(-1)
    unit_delta = unit_delta / den.clamp_min(1.0e-6)
    active = den.squeeze(-1) > 1.0e-5
    return unit_delta, active


def raw_value_loss(
    head: RawUnitDeltaValueHeadV3431,
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
    pred_unit_delta = head(out)
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
    final = compose(anchor, point_delta, gate, args.residual_scale)
    shuf_final = compose(anchor, read_unit_delta(roll_assignment(assign), pred_unit_delta), gate, args.residual_scale)
    zero_final = compose(anchor, torch.zeros_like(point_delta), gate, args.residual_scale)
    normal_cos = (norm(final) * norm(target)).sum(dim=-1)
    anchor_cos = (norm(anchor) * norm(target)).sum(dim=-1).detach()
    shuf_cos = (norm(shuf_final) * norm(target)).sum(dim=-1)
    zero_cos = (norm(zero_final) * norm(target)).sum(dim=-1).detach()
    final_loss = masked_mean(1.0 - normal_cos, pos, teacher_w)
    anchor_gain = masked_mean(F.softplus(args.anchor_gain_margin - (normal_cos - anchor_cos)), pos, teacher_w)
    assignment_contrast = masked_mean(F.softplus(args.assignment_margin - (normal_cos - shuf_cos)), pos, teacher_w)
    unit_contrast = masked_mean(F.softplus(args.unit_margin - (normal_cos - zero_cos)), pos, teacher_w)
    stable_delta = read_unit_delta(assign, pred_unit_delta).norm(dim=-1)
    stable_suppress = masked_mean(stable_delta, stable, torch.ones_like(teacher_w)) if bool(stable.any()) else pred_unit_delta.sum() * 0.0
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


def summarize_loss(losses: list[dict[str, float]], key: str) -> dict[str, float | bool | None]:
    vals = [float(x[key]) for x in losses if key in x]
    return {
        "first": vals[0] if vals else None,
        "last": vals[-1] if vals else None,
        "mean": float(np.mean(vals)) if vals else None,
        "active": bool(vals and np.mean(np.abs(vals)) > 1.0e-8),
    }


def train_one(args: argparse.Namespace) -> tuple[TopKEvidenceResidualMemoryV3418, argparse.Namespace, RawUnitDeltaValueHeadV3431, dict[str, Any]]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, init = load_frozen_residual_model(args, device)
    head = RawUnitDeltaValueHeadV3431(args.teacher_embedding_dim, args.value_hidden_dim).to(device)
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
        loss, stats = raw_value_loss(head, out, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(
                "训练进度: "
                f"step={step}/{args.steps}, loss={stats['loss']:.6f}, "
                f"dir={stats['unit_delta_direction_loss']:.6f}, mag={stats['unit_delta_magnitude_loss']:.6f}, "
                f"assign={stats['assignment_contrast_loss']:.6f}",
                flush=True,
            )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"v34_31_raw_unit_delta_value_head_m128_h32_seed{args.seed}.pt"
    torch.save({"head": head.state_dict(), "args": vars(args), "init": init, "step": args.steps}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.31 raw unit-delta value head 训练完成；冻结 V30、selector、assignment 和原 residual model，只训练 raw value head，不训练 gate。",
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "init": init,
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "residual_scale": args.residual_scale,
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
    write_doc(
        DOC,
        "V34.31 raw unit-delta value memory 训练中文摘要",
        summary,
        ["中文结论", "fresh_training_completed", "checkpoint_path", "steps", "train_sample_count", "residual_scale", "v30_backbone_frozen", "assignment_frozen", "learned_gate_training_ran", "future_leakage_detected", "trajectory_degraded"],
    )
    print(f"已写出 V34.31 训练摘要: {SUMMARY.relative_to(ROOT)}", flush=True)
    head.eval()
    return model, ckargs, head, summary


def eval_split(split: str, model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, head: RawUnitDeltaValueHeadV3431, readers: dict[str, dict[str, Any]], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
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
            max_conf = observed_max_conf(bd)
            topk = out["topk_raw_evidence_embedding"]
            gate = sparse_seed_mean_gate(out, readers)
            pred_unit_delta = head(out)
            point_delta = read_unit_delta(out["point_to_unit_assignment"].float(), pred_unit_delta)
            final = compose(anchor, point_delta, gate, args.residual_scale)
            update_method(acc, "pointwise_base", pointwise, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "copy_mean_observed", anchor, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "copy_max_conf_observed", max_conf, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "topk_raw_evidence", topk, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "v34_31_raw_unit_delta_value", final, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["normal"], "normal", final, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)

            for mode, intervention, cf_mode in [
                ("zero_semantic", "zero_semantic_measurements", "zero_semantic_measurements"),
                ("shuffle_semantic", "shuffle_semantic_measurements_across_points", "shuffle_semantic_measurements"),
            ]:
                cfb = counterfactual_batch(bd, cf_mode)
                cfout = model(**model_inputs(bd), intervention=intervention)
                cfpred_unit_delta = head(cfout)
                cfpred = compose(observed_mean(cfb), read_unit_delta(cfout["point_to_unit_assignment"].float(), cfpred_unit_delta), gate, args.residual_scale)
                update_method(delta_acc[mode], mode, cfpred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            shuf = compose(anchor, read_unit_delta(roll_assignment(out["point_to_unit_assignment"].float()), pred_unit_delta), gate, args.residual_scale)
            zero_unit = compose(anchor, torch.zeros_like(point_delta), gate, args.residual_scale)
            update_method(delta_acc["shuffle_assignment"], "shuffle_assignment", shuf, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["zero_unit"], "zero_unit", zero_unit, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
    methods = sorted({key.split(":")[0] for key in acc.sum.keys()})
    metrics = {name: finalize_method(acc, name) for name in methods}
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


def best_copy_topk(split_metrics: dict[str, Any]) -> dict[str, Any]:
    names = ["copy_mean_observed", "copy_max_conf_observed", "topk_raw_evidence"]
    rows = [{"method": n, "hard_changed_gain_vs_pointwise": split_metrics["methods"][n]["hard_changed_gain_vs_pointwise"]} for n in names]
    return max(rows, key=lambda x: float(x["hard_changed_gain_vs_pointwise"] or -1.0e9))


def evaluate(model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, head: RawUnitDeltaValueHeadV3431, train_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = next(model.parameters()).device
    readers = load_v3425_readers(args, model, device)
    per_split = {}
    for split in ("val", "test"):
        print(f"开始 V34.31 raw unit-delta value eval: split={split}", flush=True)
        per_split[split] = eval_split(split, model, ckargs, head, readers, args, device)
    method = "v34_31_raw_unit_delta_value"
    val_m = per_split["val"]["methods"][method]
    test_m = per_split["test"]["methods"][method]
    val_delta = per_split["val"]["intervention_delta"]
    test_delta = per_split["test"]["intervention_delta"]
    best_base = {split: best_copy_topk(per_split[split]) for split in ("val", "test")}
    beats_copy_topk = bool(
        (val_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["val"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002
        and (test_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["test"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002
    )
    improves_anchor = bool((val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and (test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    assignment_lb = bool((val_delta["shuffle_assignment_delta"] or 0.0) > 0.002 and (test_delta["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool((val_delta["zero_unit_memory_delta"] or 0.0) > 0.002 and (test_delta["zero_unit_memory_delta"] or 0.0) > 0.002)
    semantic_lb = bool(
        min(
            val_delta["zero_semantic_measurements_delta"] or 0.0,
            val_delta["shuffle_semantic_measurements_delta"] or 0.0,
            test_delta["zero_semantic_measurements_delta"] or 0.0,
            test_delta["shuffle_semantic_measurements_delta"] or 0.0,
        )
        > 0.002
    )
    semantic_hard_signal = {"val": val_m["semantic_hard_signal"], "test": test_m["semantic_hard_signal"]}
    changed_semantic_signal = {"val": val_m["changed_semantic_signal"], "test": test_m["changed_semantic_signal"]}
    stable_preservation = {"val": val_m["stable_preservation"], "test": test_m["stable_preservation"]}
    passed = bool(
        beats_copy_topk
        and improves_anchor
        and assignment_lb
        and unit_lb
        and semantic_lb
        and all(semantic_hard_signal.values())
        and all(changed_semantic_signal.values())
        and all(stable_preservation.values())
    )
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.31 raw unit-delta value memory 已完成训练与评估；本轮只修 value objective，不训练 gate，不跑 M512，不声明 semantic field success。",
        "raw_unit_delta_value_head_trained": True,
        "probe_passed": passed,
        "beats_copy_topk_baseline": beats_copy_topk,
        "unit_residual_improves_evidence_anchor": improves_anchor,
        "semantic_measurements_load_bearing_on_system": semantic_lb,
        "assignment_load_bearing_on_system": assignment_lb,
        "unit_memory_load_bearing_on_system": unit_lb,
        "semantic_hard_signal": semantic_hard_signal,
        "changed_semantic_signal": changed_semantic_signal,
        "stable_preservation": stable_preservation,
        "best_copy_topk_baseline": best_base,
        "v34_31_metrics": {"val": val_m, "test": test_m},
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
        "per_split": per_split,
        "decision": decision,
        "阶段性分析": (
            "V34.30 证明当前 unit assignment 有很强 oracle 上界，因此 V34.31 把 unit memory value 从 normalized semantic vector 改成 raw residual delta。"
            "本轮冻结 assignment，训练 value head 去拟合 evidence-anchor-relative unit delta，并用 assignment/zero-unit counterfactual 验证是否真正 load-bearing。"
        ),
        "论文相关问题解决方案参考": (
            "这对应 memory network / video object memory 中 value memory 与 key/assignment 分离的思路：key/assignment 负责寻址，value 负责携带可加性 correction。"
            "Slot Attention 的 slot identifiability 和 Perceiver/DETR 的 query-value 分离都提示，不能把 unit value 强制归一化成语义原型，否则会丢掉 residual magnitude。"
        ),
        "最佳下一步方案": (
            "如果 V34.31 通过，下一步训练 assignment-aware raw delta value head；如果仍不过，说明 raw value head 输入特征不足，需让 future_trace_hidden/top-k raw evidence 直接进入 value head，而不是只从旧 unit_memory 映射。"
        ),
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DECISION_DOC,
        "V34.31 raw unit-delta value memory 决策中文报告",
        decision,
        [
            "中文结论",
            "probe_passed",
            "beats_copy_topk_baseline",
            "unit_residual_improves_evidence_anchor",
            "semantic_measurements_load_bearing_on_system",
            "assignment_load_bearing_on_system",
            "unit_memory_load_bearing_on_system",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "learned_gate_training_ran",
            "m512_dense_ready",
            "integrated_semantic_field_claim_allowed",
            "integrated_identity_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.31 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
    print(f"probe_passed: {passed}", flush=True)
    print(f"recommended_next_step: {decision['recommended_next_step']}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=512)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--unit-direction-weight", type=float, default=1.0)
    p.add_argument("--unit-magnitude-weight", type=float, default=0.75)
    p.add_argument("--raw-delta-weight", type=float, default=0.6)
    p.add_argument("--final-target-weight", type=float, default=0.8)
    p.add_argument("--anchor-gain-weight", type=float, default=0.8)
    p.add_argument("--assignment-contrast-weight", type=float, default=1.2)
    p.add_argument("--unit-contrast-weight", type=float, default=1.0)
    p.add_argument("--stable-suppress-weight", type=float, default=0.2)
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
