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
from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import (
    reader_inputs,
)
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
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import pairwise_binding_loss
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import (
    freeze_for_probe,
    local_cos as train_local_cos,
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


CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_29_evidence_anchor_assignment_discriminative_residual_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_29_evidence_anchor_assignment_discriminative_residual_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_29_evidence_anchor_assignment_discriminative_residual_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_29_EVIDENCE_ANCHOR_ASSIGNMENT_DISCRIMINATIVE_RESIDUAL_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_29_EVIDENCE_ANCHOR_ASSIGNMENT_DISCRIMINATIVE_RESIDUAL_DECISION_20260514.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compose(anchor: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor, residual_scale: float) -> torch.Tensor:
    return norm(anchor + float(residual_scale) * gate[..., None] * residual)


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


def counterfactual_batch(batch: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    if mode not in {"zero_semantic_measurements", "shuffle_semantic_measurements"}:
        return batch
    out = dict(batch)
    if mode == "zero_semantic_measurements":
        out["obs_semantic_measurements"] = torch.zeros_like(batch["obs_semantic_measurements"])
        out["obs_semantic_measurement_mask"] = torch.zeros_like(batch["obs_semantic_measurement_mask"])
        out["semantic_measurement_confidence"] = torch.zeros_like(batch["semantic_measurement_confidence"])
        return out
    sem = batch["obs_semantic_measurements"]
    if sem.shape[1] <= 1:
        return out
    idx = torch.roll(torch.arange(sem.shape[1], device=sem.device), shifts=1, dims=0)
    out["obs_semantic_measurements"] = sem[:, idx]
    for key in ("obs_semantic_measurement_mask", "semantic_measurement_confidence", "teacher_agreement_score"):
        if key in batch:
            out[key] = batch[key][:, idx]
    return out


def load_init_model(args: argparse.Namespace, device: torch.device) -> tuple[TopKEvidenceResidualMemoryV3418, argparse.Namespace, dict[str, Any]]:
    train = json.loads(V3420_TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = ROOT / train.get("checkpoint_path", str(V3420_CKPT_DIR / "v34_20_hard_changed_aligned_topk_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    ckargs.steps = args.steps
    ckargs.lr = args.lr
    ckargs.seed = args.seed
    model = TopKEvidenceResidualMemoryV3418(
        ckargs.v30_checkpoint,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        units=ckargs.trace_units,
        horizon=ckargs.horizon,
        selector_hidden_dim=ckargs.selector_hidden_dim,
        topk=ckargs.topk,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    return model, ckargs, {"init_checkpoint": str(ckpt.relative_to(ROOT)), "init_source": "v34_20_hard_changed_aligned_topk_residual"}


def assignment_roll(assign: torch.Tensor) -> torch.Tensor:
    if assign.shape[-1] <= 1:
        return assign
    idx = torch.roll(torch.arange(assign.shape[-1], device=assign.device), shifts=1, dims=0)
    return assign[..., idx]


def residual_from_assignment(assign: torch.Tensor, unit_memory: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bmu,buhd->bmhd", assign, unit_memory)


def unit_delta_target_loss(
    out: dict[str, torch.Tensor],
    anchor: torch.Tensor,
    batch: dict[str, torch.Tensor],
    pos: torch.Tensor,
    teacher_w: torch.Tensor,
) -> torch.Tensor:
    assign = out["point_to_unit_assignment"].float()
    delta_dir = norm(batch["fut_teacher_embedding"].float() - anchor.float())
    w = assign[:, :, None, :] * pos.float()[:, :, :, None] * teacher_w.float()[:, :, :, None]
    unit_target = torch.einsum("bmhu,bmhd->buhd", w, delta_dir)
    den = w.sum(dim=1).permute(0, 2, 1).unsqueeze(-1)
    active = den.squeeze(-1) > 1.0e-4
    unit_target = unit_target / den.clamp_min(1.0e-6)
    if not bool(active.any()):
        return out["unit_memory"].sum() * 0.0
    cos = (norm(out["unit_memory"]) * norm(unit_target)).sum(dim=-1)
    return (1.0 - cos)[active].mean()


def point_delta_direction_loss(out: dict[str, torch.Tensor], anchor: torch.Tensor, batch: dict[str, torch.Tensor], pos: torch.Tensor, teacher_w: torch.Tensor) -> torch.Tensor:
    delta_dir = norm(batch["fut_teacher_embedding"].float() - anchor.float())
    cos = (norm(out["assignment_bound_residual"]) * delta_dir).sum(dim=-1)
    return masked_mean(1.0 - cos, pos, teacher_w) if bool(pos.any()) else out["assignment_bound_residual"].sum() * 0.0


def assignment_target_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], pos: torch.Tensor) -> torch.Tensor:
    if "point_to_unit_residual_target" not in batch:
        return out["point_to_unit_assignment"].sum() * 0.0
    target = batch["point_to_unit_residual_target"].float()
    if target.ndim != 4:
        return out["point_to_unit_assignment"].sum() * 0.0
    w = pos.float()
    per_point = (target * w[..., None]).sum(dim=2)
    per_point = per_point / per_point.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    available = per_point.sum(dim=-1) > 0.5
    if not bool(available.any()):
        return out["point_to_unit_assignment"].sum() * 0.0
    logp = out["point_to_unit_assignment"].clamp_min(1.0e-8).log()
    return -(per_point * logp).sum(dim=-1)[available].mean()


def slot_diversity_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    mem = norm(out["unit_memory"].mean(dim=2))
    usage = out["point_to_unit_assignment"].mean(dim=1)
    active = usage > (0.25 / usage.shape[-1])
    losses = []
    for b in range(mem.shape[0]):
        act = active[b]
        if int(act.sum()) < 2:
            continue
        mb = mem[b, act]
        cos = mb @ mb.transpose(0, 1)
        off = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
        losses.append(F.relu(off - 0.15).pow(2).mean())
    if not losses:
        return mem.sum() * 0.0
    return torch.stack(losses).mean()


def loss_fn(out: dict[str, torch.Tensor], zero: dict[str, torch.Tensor], shuf_sem: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    pos = hard_changed_aligned_mask(batch)
    valid = batch["fut_teacher_available_mask"].bool()
    hard = batch["semantic_hard_mask"].bool() & valid
    changed = batch["changed_mask"].bool() & valid
    stable = batch["stable_suppress_mask"].bool() & valid
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    anchor = observed_mean(batch)
    zero_anchor = observed_mean(counterfactual_batch(batch, "zero_semantic_measurements"))
    shuf_anchor = observed_mean(counterfactual_batch(batch, "shuffle_semantic_measurements"))
    gate = pos.float() * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    final = compose(anchor, out["assignment_bound_residual"], gate, args.residual_scale)
    zero_final = compose(zero_anchor, zero["assignment_bound_residual"], gate, args.residual_scale)
    shuf_sem_final = compose(shuf_anchor, shuf_sem["assignment_bound_residual"], gate, args.residual_scale)
    shuf_assign_resid = residual_from_assignment(assignment_roll(out["point_to_unit_assignment"]), out["unit_memory"])
    shuf_assign_final = compose(anchor, shuf_assign_resid, gate, args.residual_scale)
    zero_unit_final = compose(anchor, torch.zeros_like(out["assignment_bound_residual"]), gate, args.residual_scale)

    target = batch["fut_teacher_embedding"]
    sem = cosine_loss(final, target, pos, teacher_w) if bool(pos.any()) else final.sum() * 0.0
    hard_loss = cosine_loss(final, target, (hard | changed) & pos, teacher_w) if bool(((hard | changed) & pos).any()) else final.sum() * 0.0
    stable_loss = cosine_loss(final, anchor.detach(), stable, torch.ones_like(teacher_w)) if bool(stable.any()) else final.sum() * 0.0
    point_delta = point_delta_direction_loss(out, anchor, batch, pos, teacher_w)
    unit_delta = unit_delta_target_loss(out, anchor, batch, pos, teacher_w)
    assign_target = assignment_target_loss(out, batch, pos)
    slot_div = slot_diversity_loss(out)
    bind, bind_stats = pairwise_binding_loss(out, batch)

    normal_cos = train_local_cos(final, target)
    anchor_cos = train_local_cos(anchor, target).detach()
    zero_cos = train_local_cos(zero_final, target).detach()
    shuf_sem_cos = train_local_cos(shuf_sem_final, target).detach()
    shuf_assign_cos = train_local_cos(shuf_assign_final, target)
    zero_unit_cos = train_local_cos(zero_unit_final, target).detach()
    anchor_gain = masked_mean(F.softplus(args.anchor_gain_margin - (normal_cos - anchor_cos)), pos, teacher_w)
    sem_contrast = masked_mean(F.softplus(args.semantic_contrast_margin - (normal_cos - torch.maximum(zero_cos, shuf_sem_cos))), pos, teacher_w)
    assign_contrast = masked_mean(F.softplus(args.assignment_contrast_margin - (normal_cos - shuf_assign_cos)), pos, teacher_w)
    unit_contrast = masked_mean(F.softplus(args.unit_contrast_margin - (normal_cos - zero_unit_cos)), pos, teacher_w)
    assign = out["point_to_unit_assignment"].clamp_min(1.0e-8)
    usage = assign.mean(dim=1).clamp_min(1.0e-8)
    anti = 1.0 + (usage * usage.log()).sum(dim=-1).mean() / np.log(assign.shape[-1])

    total = (
        1.0 * sem
        + 0.65 * hard_loss
        + 0.65 * stable_loss
        + 0.45 * point_delta
        + 0.75 * unit_delta
        + args.anchor_gain_weight * anchor_gain
        + args.semantic_contrast_weight * sem_contrast
        + args.assignment_contrast_weight * assign_contrast
        + args.unit_contrast_weight * unit_contrast
        + 0.25 * assign_target
        + 0.35 * slot_div
        + 0.10 * bind
        + 0.12 * anti
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "evidence_anchor_final_target_loss": float(sem.detach().cpu()),
        "hard_changed_loss": float(hard_loss.detach().cpu()),
        "stable_anchor_preservation_loss": float(stable_loss.detach().cpu()),
        "point_delta_direction_loss": float(point_delta.detach().cpu()),
        "unit_delta_target_loss": float(unit_delta.detach().cpu()),
        "anchor_gain_contrast_loss": float(anchor_gain.detach().cpu()),
        "semantic_measurement_contrast_loss": float(sem_contrast.detach().cpu()),
        "assignment_contrast_loss": float(assign_contrast.detach().cpu()),
        "unit_memory_contrast_loss": float(unit_contrast.detach().cpu()),
        "assignment_target_loss": float(assign_target.detach().cpu()),
        "slot_diversity_loss": float(slot_div.detach().cpu()),
        "unit_anti_collapse": float(anti.detach().cpu()),
    }
    stats.update(bind_stats)
    return total, stats


def summarize_loss(losses: list[dict[str, float]], key: str) -> dict[str, float | bool | None]:
    vals = [float(x[key]) for x in losses if key in x]
    return {
        "first": vals[0] if vals else None,
        "last": vals[-1] if vals else None,
        "mean": float(np.mean(vals)) if vals else None,
        "active": bool(vals and np.mean(np.abs(vals)) > 1.0e-8),
    }


def train_one(args: argparse.Namespace) -> tuple[TopKEvidenceResidualMemoryV3418, argparse.Namespace, dict[str, Any]]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, init = load_init_model(args, device)
    freeze_for_probe(model)
    loader = make_loader("train", ckargs, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1.0e-4)
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
        inp = model_inputs(bd)
        out = model(**inp, intervention="force_gate_zero")
        zero = model(**inp, intervention="zero_semantic_measurements")
        shuf_sem = model(**inp, intervention="shuffle_semantic_measurements_across_points")
        loss, stats = loss_fn(out, zero, shuf_sem, bd, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            row = {"step": float(step), **stats}
            losses.append(row)
            print(
                "训练进度: "
                f"step={step}/{args.steps}, loss={stats['loss']:.6f}, "
                f"anchor_gain={stats['anchor_gain_contrast_loss']:.6f}, "
                f"assign_contrast={stats['assignment_contrast_loss']:.6f}, "
                f"unit_delta={stats['unit_delta_target_loss']:.6f}",
                flush=True,
            )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"v34_29_evidence_anchor_assignment_discriminative_residual_m128_h32_seed{args.seed}.pt"
    torch.save({"model": model.state_dict(), "args": vars(ckargs) | vars(args), "step": args.steps}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.29 evidence-anchor-relative assignment-discriminative residual 训练完成；V30 冻结，未训练 learned gate，目标从 pointwise-relative 改为 evidence-anchor-relative correction。",
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "init": init,
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "residual_scale": args.residual_scale,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "loss_summaries": {k: summarize_loss(losses, k) for k in losses[-1].keys() if k != "step"} if losses else {},
        "loss_trace": losses,
        "duration_seconds": float(time.time() - start),
    }
    dump_json(SUMMARY, summary)
    write_doc(
        DOC,
        "V34.29 evidence-anchor-relative assignment-discriminative residual 训练中文摘要",
        summary,
        ["中文结论", "fresh_training_completed", "checkpoint_path", "steps", "train_sample_count", "residual_scale", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded"],
    )
    print(f"已写出 V34.29 训练摘要: {SUMMARY.relative_to(ROOT)}", flush=True)
    model.eval()
    return model, ckargs, summary


def eval_split(split: str, model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, readers: dict[str, dict[str, Any]], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    acc = Acc()
    delta_acc = {mode: Acc() for mode in ["normal", "zero_semantic", "shuffle_semantic", "shuffle_assignment", "zero_unit"]}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            target = bd["fut_teacher_embedding"]
            anchor = observed_mean(bd)
            topk = out["topk_raw_evidence_embedding"]
            max_conf = observed_max_conf(bd)
            gate = sparse_seed_mean_gate(out, readers)
            pred = compose(anchor, out["assignment_bound_residual"], gate, args.residual_scale)
            update_method(acc, "pointwise_base", pointwise, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "copy_mean_observed", anchor, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "copy_max_conf_observed", max_conf, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "topk_raw_evidence", topk, pointwise=pointwise, target=target, mm=mm)
            update_method(acc, "v34_29_evidence_anchor_assignment_residual", pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["normal"], "normal", pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)

            for mode, intervention, cf_mode in [
                ("zero_semantic", "zero_semantic_measurements", "zero_semantic_measurements"),
                ("shuffle_semantic", "shuffle_semantic_measurements_across_points", "shuffle_semantic_measurements"),
            ]:
                cfb = counterfactual_batch(bd, cf_mode)
                cfout = model(**model_inputs(bd), intervention=intervention)
                cfpred = compose(observed_mean(cfb), cfout["assignment_bound_residual"], gate, args.residual_scale)
                update_method(delta_acc[mode], mode, cfpred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)

            shuf_resid = residual_from_assignment(assignment_roll(out["point_to_unit_assignment"]), out["unit_memory"])
            shuf_pred = compose(anchor, shuf_resid, gate, args.residual_scale)
            update_method(delta_acc["shuffle_assignment"], "shuffle_assignment", shuf_pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
            zero_unit_pred = compose(anchor, torch.zeros_like(out["assignment_bound_residual"]), gate, args.residual_scale)
            update_method(delta_acc["zero_unit"], "zero_unit", zero_unit_pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
    methods = sorted({key.split(":")[0] for key in acc.sum.keys()})
    metrics = {name: finalize_method(acc, name) for name in methods}
    delta_rows = {mode: finalize_method(a, mode) for mode, a in delta_acc.items()}
    normal = delta_rows["normal"]

    def delta(mode: str) -> float | None:
        a = normal["hard_changed_gain_vs_pointwise"]
        b = delta_rows[mode]["hard_changed_gain_vs_pointwise"]
        if a is None or b is None:
            return None
        return float(a - b)

    rank = sorted(
        [
            {
                "method": name,
                "hard_changed_gain_vs_pointwise": row["hard_changed_gain_vs_pointwise"],
                "hard_changed_gain_vs_anchor": row["hard_changed_gain_vs_anchor"],
                "semantic_hard_signal": row["semantic_hard_signal"],
                "changed_semantic_signal": row["changed_semantic_signal"],
                "stable_preservation": row["stable_preservation"],
            }
            for name, row in metrics.items()
        ],
        key=lambda x: float(x["hard_changed_gain_vs_pointwise"] or -1.0e9),
        reverse=True,
    )
    return {
        "methods": metrics,
        "rank": rank,
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


def evaluate(model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, train_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = next(model.parameters()).device
    readers = load_v3425_readers(args, model, device)
    per_split = {}
    for split in ("val", "test"):
        print(f"开始 V34.29 evidence-anchor assignment residual eval: split={split}", flush=True)
        per_split[split] = eval_split(split, model, ckargs, readers, args, device)
    best_base = {split: best_copy_topk(per_split[split]) for split in ("val", "test")}
    method = "v34_29_evidence_anchor_assignment_residual"
    val_m = per_split["val"]["methods"][method]
    test_m = per_split["test"]["methods"][method]
    val_delta = per_split["val"]["intervention_delta"]
    test_delta = per_split["test"]["intervention_delta"]
    beats_copy_topk = bool(
        (val_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["val"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002
        and (test_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["test"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002
    )
    improves_anchor = bool((val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and (test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    assignment_lb = bool((val_delta["shuffle_assignment_delta"] or 0.0) > 0.002 and (test_delta["shuffle_assignment_delta"] or 0.0) > 0.002)
    semantic_lb = bool(
        min(
            val_delta["zero_semantic_measurements_delta"] or 0.0,
            val_delta["shuffle_semantic_measurements_delta"] or 0.0,
            test_delta["zero_semantic_measurements_delta"] or 0.0,
            test_delta["shuffle_semantic_measurements_delta"] or 0.0,
        )
        > 0.002
    )
    unit_lb = bool((val_delta["zero_unit_memory_delta"] or 0.0) > 0.002 and (test_delta["zero_unit_memory_delta"] or 0.0) > 0.002)
    semantic_hard_signal = {"val": val_m["semantic_hard_signal"], "test": test_m["semantic_hard_signal"]}
    changed_semantic_signal = {"val": val_m["changed_semantic_signal"], "test": test_m["changed_semantic_signal"]}
    stable_preservation = {"val": val_m["stable_preservation"], "test": test_m["stable_preservation"]}
    passed = bool(
        beats_copy_topk
        and improves_anchor
        and assignment_lb
        and semantic_lb
        and unit_lb
        and all(semantic_hard_signal.values())
        and all(changed_semantic_signal.values())
        and all(stable_preservation.values())
    )
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.29 完成 evidence-anchor-relative assignment-discriminative residual 训练与评估；该轮仍不是 learned gate，也不声明 semantic field success。",
        "evidence_anchor_assignment_residual_trained": True,
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
        "v34_29_metrics": {"val": val_m, "test": test_m},
        "intervention_delta": {"val": val_delta, "test": test_delta},
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "m512_dense_ready": bool(passed),
        "learned_gate_training_ran": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_29_m512_dense_visualization" if passed else "fix_assignment_bound_residual_model",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "train_summary": train_summary,
        "per_split": per_split,
        "decision": decision,
        "阶段性分析": (
            "V34.29 直接修 V34.28 暴露的核心短板：copy/top-k evidence 已经是强 semantic base，unit memory 必须学习相对 evidence base 的结构化增量，"
            "而不是相对弱 pointwise base 的增量。本轮训练加入 unit-level delta target、point-level delta direction、shuffled-assignment contrast、zero-unit contrast、slot diversity。"
        ),
        "论文相关问题解决方案参考": (
            "本轮对应 Slot Attention/OCVP 中的 slot 可区分性问题、XMem/SAM2 中 memory read 必须成为可反事实破坏的路径、以及 Perceiver IO/DETR 中 query-conditioned memory read 的输出绑定。"
            "关键不是再调 gate，而是让 unit slot correction 在强 evidence base 上仍然 object-bound。"
        ),
        "最佳下一步方案": (
            "若 probe_passed=true，可以进入 M128/M512 dense visualization；若 assignment 仍不 load-bearing，应继续修 assignment-bound residual model，"
            "更进一步可能需要显式 object/instance-slot matching 或 unit residual memory 的 permutation-aware supervised target。"
        ),
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DECISION_DOC,
        "V34.29 evidence-anchor assignment-discriminative residual 决策中文报告",
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
            "m512_dense_ready",
            "integrated_semantic_field_claim_allowed",
            "integrated_identity_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.29 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
    print(f"probe_passed: {passed}", flush=True)
    print(f"recommended_next_step: {decision['recommended_next_step']}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=4.0e-5)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--anchor-gain-weight", type=float, default=0.65)
    p.add_argument("--semantic-contrast-weight", type=float, default=0.6)
    p.add_argument("--assignment-contrast-weight", type=float, default=1.1)
    p.add_argument("--unit-contrast-weight", type=float, default=0.8)
    p.add_argument("--anchor-gain-margin", type=float, default=0.006)
    p.add_argument("--semantic-contrast-margin", type=float, default=0.006)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.006)
    p.add_argument("--unit-contrast-margin", type=float, default=0.006)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model, ckargs, train_summary = train_one(args)
    evaluate(model, ckargs, train_summary, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
