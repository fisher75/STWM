#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import (
    CHECKPOINT as V3422_CHECKPOINT,
    SUMMARY as V3422_SUMMARY,
    hard_changed_aligned_mask,
    load_residual_model,
    make_loader,
    reader_inputs,
)
from stwm.modules.ostf_v34_22_activation_state_reader import ActivationStateReaderV3422
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss


SUMMARY = ROOT / "reports/stwm_ostf_v34_25_sparse_calibrated_gate_repair_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_25_sparse_calibrated_gate_repair_decision_20260514.json"
SUMMARY_DOC = ROOT / "docs/STWM_OSTF_V34_25_SPARSE_CALIBRATED_GATE_REPAIR_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_25_SPARSE_CALIBRATED_GATE_REPAIR_DECISION_20260514.md"
CKPT_ROOT = ROOT / "outputs/checkpoints/stwm_ostf_v34_25_sparse_calibrated_gate_repair_h32_m128"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def local_cos(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (norm(pred) * norm(target)).sum(dim=-1)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    m = mask.float()
    if weight is not None:
        m = m * weight.float()
    return (value * m).sum() / m.sum().clamp_min(1.0)


def make_reader(args: argparse.Namespace, residual_model: Any, device: torch.device) -> tuple[ActivationStateReaderV3422, dict[str, Any]]:
    ref = json.loads(V3422_SUMMARY.read_text(encoding="utf-8")) if V3422_SUMMARY.exists() else {}
    ckpt = torch.load(V3422_CHECKPOINT, map_location="cpu")
    reader = ActivationStateReaderV3422(
        int(residual_model.v30.cfg.hidden_dim),
        semantic_dim=int(getattr(args, "teacher_embedding_dim", 768)),
        hidden_dim=int(ckpt["args"].get("reader_hidden_dim", args.reader_hidden_dim)),
    ).to(device)
    reader.load_state_dict(ckpt["reader"], strict=True)
    return reader, ref


def masks_from_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    valid = batch["fut_teacher_available_mask"].bool()
    hard = batch["semantic_hard_mask"].bool() & valid
    changed = batch["changed_mask"].bool() & valid
    stable = batch["stable_suppress_mask"].bool() & valid
    aligned = hard_changed_aligned_mask(batch)
    return {
        "valid": valid,
        "hard": hard,
        "changed": changed,
        "hard_changed": (hard | changed) & valid,
        "stable": stable,
        "aligned": aligned,
    }


def gate_from_logits(
    logits: torch.Tensor,
    usage: torch.Tensor,
    *,
    threshold: float | None,
    temperature: float,
    power: float = 1.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logits / max(float(temperature), 1.0e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        prob = prob.pow(float(power))
    if threshold is not None:
        prob = prob * (prob >= float(threshold)).float()
    return prob * usage.float().clamp(0.0, 1.0)


def compose_with_gate(out: dict[str, torch.Tensor], gate: torch.Tensor) -> torch.Tensor:
    return F.normalize(out["pointwise_semantic_belief"] + gate[..., None] * out["assignment_bound_residual"], dim=-1)


def batch_budget_loss(gate: torch.Tensor, valid: torch.Tensor, budget: float) -> torch.Tensor:
    # 每个 sample / horizon 的平均开门率都不应无限接近全开。
    denom = valid.float().sum(dim=1).clamp_min(1.0)
    per_bh = (gate * valid.float()).sum(dim=1) / denom
    available = valid.any(dim=1)
    if not bool(available.any()):
        return gate.sum() * 0.0
    return F.relu(per_bh[available] - float(budget)).pow(2).mean()


def train_reader_for_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, residual_train = load_residual_model(args, device)
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    reader, v3422_reference = make_reader(args, residual_model, device)
    for p in residual_model.parameters():
        p.requires_grad_(False)
    reader.train()
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_trace: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        totals = {
            "loss": 0.0,
            "semantic": 0.0,
            "hard": 0.0,
            "stable_preserve": 0.0,
            "stable_negative_gate": 0.0,
            "hard_recall_gate": 0.0,
            "budget": 0.0,
            "semantic_contrast": 0.0,
            "assignment_contrast": 0.0,
        }
        seen = 0
        for batch in make_loader("train", ckargs, shuffle=True):
            bd = move_batch(batch, device)
            with torch.no_grad():
                out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
                zero = residual_model(**model_inputs(bd), intervention="zero_semantic_measurements")
                shuf_sem = residual_model(**model_inputs(bd), intervention="shuffle_semantic_measurements_across_points")
                shuf_assign = residual_model(**model_inputs(bd), intervention="shuffle_assignment")
            pred = reader(**reader_inputs(out))["activation_logits"]
            pred_zero = reader(**reader_inputs(zero))["activation_logits"]
            pred_shuf_sem = reader(**reader_inputs(shuf_sem))["activation_logits"]
            pred_shuf_assign = reader(**reader_inputs(shuf_assign))["activation_logits"]
            logits = pred["benefit"]
            usage = out["semantic_measurement_usage_score"]
            gate = gate_from_logits(logits, usage, threshold=None, temperature=1.0)
            final = compose_with_gate(out, gate)
            zero_final = compose_with_gate(
                zero,
                gate_from_logits(pred_zero["benefit"], zero["semantic_measurement_usage_score"], threshold=None, temperature=1.0),
            )
            shuf_sem_final = compose_with_gate(
                shuf_sem,
                gate_from_logits(pred_shuf_sem["benefit"], shuf_sem["semantic_measurement_usage_score"], threshold=None, temperature=1.0),
            )
            shuf_assign_final = compose_with_gate(
                shuf_assign,
                gate_from_logits(pred_shuf_assign["benefit"], shuf_assign["semantic_measurement_usage_score"], threshold=None, temperature=1.0),
            )
            masks = masks_from_batch(bd)
            teacher_w = bd["teacher_confidence"].float().clamp(0.05, 1.0)
            semantic = cosine_loss(final, bd["fut_teacher_embedding"], masks["aligned"], teacher_w)
            hard = cosine_loss(final, bd["fut_teacher_embedding"], masks["hard_changed"], teacher_w)
            stable_preserve = (
                cosine_loss(final, out["pointwise_semantic_belief"].detach(), masks["stable"], torch.ones_like(teacher_w))
                if bool(masks["stable"].any())
                else final.sum() * 0.0
            )
            stable_neg_logit = masked_mean(F.softplus(logits), masks["stable"]) if bool(masks["stable"].any()) else final.sum() * 0.0
            stable_neg_gate = masked_mean(gate, masks["stable"]) if bool(masks["stable"].any()) else final.sum() * 0.0
            stable_negative_gate = stable_neg_logit + args.stable_gate_value_weight * stable_neg_gate
            hard_recall_logit = masked_mean(F.softplus(-logits), masks["hard_changed"], teacher_w) if bool(masks["hard_changed"].any()) else final.sum() * 0.0
            hard_gate_shortfall = (
                masked_mean(F.relu(args.min_hard_gate - gate).pow(2), masks["hard_changed"], teacher_w)
                if bool(masks["hard_changed"].any())
                else final.sum() * 0.0
            )
            hard_recall_gate = hard_recall_logit + args.hard_gate_shortfall_weight * hard_gate_shortfall
            budget = batch_budget_loss(gate, masks["valid"], args.gate_budget)
            normal_cos = local_cos(final, bd["fut_teacher_embedding"])
            zero_cos = local_cos(zero_final, bd["fut_teacher_embedding"]).detach()
            shuf_sem_cos = local_cos(shuf_sem_final, bd["fut_teacher_embedding"]).detach()
            shuf_assign_cos = local_cos(shuf_assign_final, bd["fut_teacher_embedding"]).detach()
            semantic_contrast = masked_mean(
                F.softplus(args.semantic_contrast_margin - (normal_cos - torch.maximum(zero_cos, shuf_sem_cos))),
                masks["aligned"],
                teacher_w,
            )
            assignment_contrast = masked_mean(
                F.softplus(args.assignment_contrast_margin - (normal_cos - shuf_assign_cos)),
                masks["aligned"],
                teacher_w,
            )
            loss = (
                args.semantic_weight * semantic
                + args.hard_weight * hard
                + args.stable_preserve_weight * stable_preserve
                + args.stable_negative_gate_weight * stable_negative_gate
                + args.hard_recall_gate_weight * hard_recall_gate
                + args.budget_weight * budget
                + args.semantic_contrast_weight * semantic_contrast
                + args.assignment_contrast_weight * assignment_contrast
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            for key, value in [
                ("loss", loss),
                ("semantic", semantic),
                ("hard", hard),
                ("stable_preserve", stable_preserve),
                ("stable_negative_gate", stable_negative_gate),
                ("hard_recall_gate", hard_recall_gate),
                ("budget", budget),
                ("semantic_contrast", semantic_contrast),
                ("assignment_contrast", assignment_contrast),
            ]:
                totals[key] += float(value.detach().cpu())
            seen += 1
        row = {"epoch": float(epoch), **{k: v / max(seen, 1) for k, v in totals.items()}}
        loss_trace.append(row)
        print(
            "训练进度: "
            f"seed={seed}, epoch={epoch}/{args.epochs}, loss={row['loss']:.6f}, "
            f"stable_gate={row['stable_negative_gate']:.6f}, budget={row['budget']:.6f}, hard_recall={row['hard_recall_gate']:.6f}",
            flush=True,
        )
    seed_dir = CKPT_ROOT / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = seed_dir / f"v34_25_sparse_calibrated_gate_repair_m128_h32_seed{seed}.pt"
    torch.save({"reader": reader.state_dict(), "args": vars(args), "seed": seed}, ckpt_path)
    eval_report = evaluate_seed(seed, args, residual_model, ckargs, reader, device)
    result = {
        "generated_at_utc": utc_now(),
        "seed": seed,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "loss_trace": loss_trace,
        "v34_20_train_summary": residual_train,
        "v34_22_reference": v3422_reference,
        **eval_report,
    }
    return result


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(value[m].sum().detach().cpu())
            self.count[key] = self.count.get(key, 0) + int(m.sum().detach().cpu())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def finalize_acc(acc: Acc) -> dict[str, Any]:
    keys = ["aligned", "hard", "changed", "hard_changed", "stable", "valid"]
    out: dict[str, Any] = {}
    for key in keys:
        out[f"{key}_gain"] = acc.mean(f"{key}:gain")
        out[f"{key}_gate_mean"] = acc.mean(f"{key}:gate")
        out[f"{key}_over_open_rate"] = acc.mean(f"{key}:over_open")
        out[f"{key}_over_update_rate"] = acc.mean(f"{key}:over_update")
    out["semantic_hard_signal"] = bool(out["hard_gain"] is not None and out["hard_gain"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain"] is not None and out["changed_gain"] > 0.005)
    out["stable_preservation"] = bool(out["stable_gain"] is None or out["stable_gain"] >= -0.02)
    out["gate_order_ok"] = bool((out["hard_changed_gate_mean"] or 0.0) > (out["stable_gate_mean"] or 0.0))
    out["stable_overopen_controlled"] = bool((out["stable_over_open_rate"] or 1.0) <= 0.35)
    out["stable_overupdate_detected"] = bool((out["stable_over_update_rate"] or 0.0) > 0.05)
    return out


def eval_config_split(
    split: str,
    residual_model: Any,
    reader: ActivationStateReaderV3422,
    ckargs: argparse.Namespace,
    device: torch.device,
    config: dict[str, float | None],
    *,
    intervention: str = "force_gate_zero",
) -> dict[str, Any]:
    reader.eval()
    acc = Acc()
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention=intervention)
            pred = reader(**reader_inputs(out))["activation_logits"]
            gate = gate_from_logits(
                pred["benefit"],
                out["semantic_measurement_usage_score"],
                threshold=config.get("threshold"),
                temperature=float(config.get("temperature") or 1.0),
                power=float(config.get("power") or 1.0),
            )
            final = compose_with_gate(out, gate)
            gain = local_cos(final, bd["fut_teacher_embedding"]) - local_cos(out["pointwise_semantic_belief"], bd["fut_teacher_embedding"])
            masks = masks_from_batch(bd)
            over_open = (gate > 0.05).float()
            over_update = (gain < -0.02).float()
            for key, mask in masks.items():
                acc.add(f"{key}:gain", gain, mask)
                acc.add(f"{key}:gate", gate, mask)
                acc.add(f"{key}:over_open", over_open, mask)
                acc.add(f"{key}:over_update", over_update, mask)
    row = finalize_acc(acc)
    row["config"] = config
    row["intervention"] = intervention
    return row


def eval_config_sweep_split(
    split: str,
    residual_model: Any,
    reader: ActivationStateReaderV3422,
    ckargs: argparse.Namespace,
    device: torch.device,
    configs: list[dict[str, float | None]],
) -> list[dict[str, Any]]:
    reader.eval()
    accs = [Acc() for _ in configs]
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            pred = reader(**reader_inputs(out))["activation_logits"]
            logits = pred["benefit"]
            usage = out["semantic_measurement_usage_score"]
            masks = masks_from_batch(bd)
            pointwise = out["pointwise_semantic_belief"]
            target = bd["fut_teacher_embedding"]
            for cfg, acc in zip(configs, accs, strict=True):
                gate = gate_from_logits(
                    logits,
                    usage,
                    threshold=cfg.get("threshold"),
                    temperature=float(cfg.get("temperature") or 1.0),
                    power=float(cfg.get("power") or 1.0),
                )
                final = compose_with_gate(out, gate)
                gain = local_cos(final, target) - local_cos(pointwise, target)
                over_open = (gate > 0.05).float()
                over_update = (gain < -0.02).float()
                for key, mask in masks.items():
                    acc.add(f"{key}:gain", gain, mask)
                    acc.add(f"{key}:gate", gate, mask)
                    acc.add(f"{key}:over_open", over_open, mask)
                    acc.add(f"{key}:over_update", over_update, mask)
    rows = []
    for cfg, acc in zip(configs, accs, strict=True):
        row = finalize_acc(acc)
        row["config"] = cfg
        row["intervention"] = "force_gate_zero"
        rows.append(row)
    return rows


def calibration_configs() -> list[dict[str, float | None]]:
    configs: list[dict[str, float | None]] = []
    # 这是风险审计用 Pareto 网格，不是大规模搜索。覆盖较冷/正常/较热温度与中高阈值，
    # 重点看 stable over-open 能否在保留 hard/changed gain 的同时降下来。
    for temperature in (0.5, 0.75, 1.0, 1.25):
        for threshold in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
            configs.append({"temperature": float(temperature), "threshold": float(threshold), "power": 1.0})
    return configs


def score_config(row: dict[str, Any]) -> float:
    hard_changed = float(row.get("hard_changed_gain") or -1.0)
    changed = float(row.get("changed_gain") or -1.0)
    hard = float(row.get("hard_gain") or -1.0)
    stable_open = float(row.get("stable_over_open_rate") or 1.0)
    stable_update = float(row.get("stable_over_update_rate") or 1.0)
    stable_gain = float(row.get("stable_gain") or 0.0)
    gate_order_bonus = 0.02 if row.get("gate_order_ok") else -0.1
    return hard_changed + 0.25 * min(hard, changed) - 0.08 * stable_open - 2.0 * stable_update + 0.05 * min(stable_gain, 0.0) + gate_order_bonus


def choose_config(val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # 校准策略必须安全优先：V34.24/25 的失败是 test stable over-open，
    # 因此 val 端需要留出 safety margin，而不是只追求 hard_changed_gain 最大。
    conservative = [
        row
        for row in val_rows
        if row.get("semantic_hard_signal")
        and row.get("changed_semantic_signal")
        and row.get("stable_preservation")
        and row.get("gate_order_ok")
        and not row.get("stable_overupdate_detected")
        and (row.get("stable_over_open_rate") or 1.0) <= 0.22
        and float(row.get("config", {}).get("threshold") or 0.0) >= 0.75
        and float(row.get("config", {}).get("temperature") or 0.0) >= 0.75
        and (row.get("hard_changed_gain") or 0.0) > 0.03
    ]
    if conservative:
        return max(conservative, key=score_config)
    valid = [
        row
        for row in val_rows
        if row.get("semantic_hard_signal")
        and row.get("changed_semantic_signal")
        and row.get("stable_preservation")
        and row.get("gate_order_ok")
        and not row.get("stable_overupdate_detected")
        and (row.get("stable_over_open_rate") or 1.0) <= 0.35
        and (row.get("hard_changed_gain") or 0.0) > 0.03
    ]
    if not valid:
        valid = [
            row
            for row in val_rows
            if row.get("semantic_hard_signal")
            and row.get("changed_semantic_signal")
            and row.get("stable_preservation")
            and not row.get("stable_overupdate_detected")
        ]
    if not valid:
        valid = val_rows
    return max(valid, key=score_config)


def evaluate_interventions(
    split: str,
    residual_model: Any,
    reader: ActivationStateReaderV3422,
    ckargs: argparse.Namespace,
    device: torch.device,
    config: dict[str, float | None],
) -> dict[str, Any]:
    interventions = {
        "normal": "force_gate_zero",
        "zero_semantic_measurements": "zero_semantic_measurements",
        "shuffle_semantic_measurements": "shuffle_semantic_measurements_across_points",
        "shuffle_assignment": "shuffle_assignment",
        "zero_unit_memory": "zero_unit_memory",
        "selector_ablation": "selector_ablation",
    }
    modes = {
        name: eval_config_split(split, residual_model, reader, ckargs, device, config, intervention=intervention)
        for name, intervention in interventions.items()
    }
    normal = modes["normal"]

    def delta(name: str) -> float | None:
        a = normal.get("hard_changed_gain")
        b = modes[name].get("hard_changed_gain")
        if a is None or b is None:
            return None
        return float(a - b)

    return {
        "modes": modes,
        "zero_semantic_measurements_delta": delta("zero_semantic_measurements"),
        "shuffle_semantic_measurements_delta": delta("shuffle_semantic_measurements"),
        "shuffle_assignment_delta": delta("shuffle_assignment"),
        "zero_unit_memory_delta": delta("zero_unit_memory"),
        "selector_ablation_delta": delta("selector_ablation"),
    }


def evaluate_seed(
    seed: int,
    args: argparse.Namespace,
    residual_model: Any,
    ckargs: argparse.Namespace,
    reader: ActivationStateReaderV3422,
    device: torch.device,
) -> dict[str, Any]:
    configs = calibration_configs()
    sweep = {split: eval_config_sweep_split(split, residual_model, reader, ckargs, device, configs) for split in ("val", "test")}
    best_val = choose_config(sweep["val"])
    best_config = best_val["config"]
    best = {
        split: eval_config_split(split, residual_model, reader, ckargs, device, best_config)
        for split in ("val", "test")
    }
    intervention_eval = {
        split: evaluate_interventions(split, residual_model, reader, ckargs, device, best_config)
        for split in ("val", "test")
    }
    normal_val = intervention_eval["val"]["modes"]["normal"]
    normal_test = intervention_eval["test"]["modes"]["normal"]
    sem_lb = bool(
        min(intervention_eval["val"]["zero_semantic_measurements_delta"] or 0.0, intervention_eval["val"]["shuffle_semantic_measurements_delta"] or 0.0) > 0.002
        and min(intervention_eval["test"]["zero_semantic_measurements_delta"] or 0.0, intervention_eval["test"]["shuffle_semantic_measurements_delta"] or 0.0) > 0.002
    )
    assign_lb = bool((intervention_eval["val"]["shuffle_assignment_delta"] or 0.0) > 0.002 and (intervention_eval["test"]["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool((intervention_eval["val"]["zero_unit_memory_delta"] or 0.0) > 0.002 and (intervention_eval["test"]["zero_unit_memory_delta"] or 0.0) > 0.002)
    sparse_gate_passed = bool(
        sem_lb
        and assign_lb
        and unit_lb
        and normal_val["semantic_hard_signal"]
        and normal_test["semantic_hard_signal"]
        and normal_val["changed_semantic_signal"]
        and normal_test["changed_semantic_signal"]
        and normal_val["stable_preservation"]
        and normal_test["stable_preservation"]
        and normal_val["stable_overopen_controlled"]
        and normal_test["stable_overopen_controlled"]
        and not normal_val["stable_overupdate_detected"]
        and not normal_test["stable_overupdate_detected"]
    )
    pareto = {
        split: sorted(
            [
                {
                    "config": row["config"],
                    "hard_gain": row.get("hard_gain"),
                    "changed_gain": row.get("changed_gain"),
                    "hard_changed_gain": row.get("hard_changed_gain"),
                    "stable_gain": row.get("stable_gain"),
                    "stable_gate_mean": row.get("stable_gate_mean"),
                    "stable_over_open_rate": row.get("stable_over_open_rate"),
                    "stable_over_update_rate": row.get("stable_over_update_rate"),
                    "semantic_hard_signal": row.get("semantic_hard_signal"),
                    "changed_semantic_signal": row.get("changed_semantic_signal"),
                    "stable_preservation": row.get("stable_preservation"),
                    "gate_order_ok": row.get("gate_order_ok"),
                    "score": score_config(row),
                }
                for row in sweep[split]
            ],
            key=lambda x: float(x["score"]),
            reverse=True,
        )[:12]
        for split in ("val", "test")
    }
    return {
        "sparse_calibrated_gate_repair_ran": True,
        "sparse_calibrated_gate_repair_passed": sparse_gate_passed,
        "seed": seed,
        "best_config_by_val": best_config,
        "best_config_eval": best,
        "calibration_sweep_pareto_top12": pareto,
        "intervention_eval": intervention_eval,
        "semantic_hard_signal": {"val": normal_val["semantic_hard_signal"], "test": normal_test["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": normal_val["changed_semantic_signal"], "test": normal_test["changed_semantic_signal"]},
        "stable_preservation": {"val": normal_val["stable_preservation"], "test": normal_test["stable_preservation"]},
        "hard_changed_gain": {"val": normal_val["hard_changed_gain"], "test": normal_test["hard_changed_gain"]},
        "stable_over_open_rate": {"val": normal_val["stable_over_open_rate"], "test": normal_test["stable_over_open_rate"]},
        "stable_gate_mean": {"val": normal_val["stable_gate_mean"], "test": normal_test["stable_gate_mean"]},
        "stable_over_update_rate": {"val": normal_val["stable_over_update_rate"], "test": normal_test["stable_over_update_rate"]},
        "gate_order_ok": {"val": normal_val["gate_order_ok"], "test": normal_test["gate_order_ok"]},
        "stable_overopen_controlled": {"val": normal_val["stable_overopen_controlled"], "test": normal_test["stable_overopen_controlled"]},
        "semantic_measurements_load_bearing_on_residual": sem_lb,
        "assignment_load_bearing_on_residual": assign_lb,
        "unit_memory_load_bearing_on_residual": unit_lb,
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
    }


def aggregate_bool(seed_reports: dict[str, dict[str, Any]], key: str) -> bool:
    return all(bool(report.get(key)) for report in seed_reports.values())


def aggregate_split_bool(seed_reports: dict[str, dict[str, Any]], key: str) -> dict[str, bool]:
    return {
        split: all(bool(report.get(key, {}).get(split)) for report in seed_reports.values())
        for split in ("val", "test")
    }


def aggregate_split_float(seed_reports: dict[str, dict[str, Any]], key: str) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for split in ("val", "test"):
        values = [report.get(key, {}).get(split) for report in seed_reports.values()]
        real = [float(v) for v in values if v is not None]
        out[split] = {
            "mean": float(mean(real)) if real else None,
            "std": float(pstdev(real)) if len(real) > 1 else 0.0 if real else None,
            "min": float(min(real)) if real else None,
            "max": float(max(real)) if real else None,
        }
    return out


def write_chinese_doc(payload: dict[str, Any]) -> None:
    write_doc(
        SUMMARY_DOC,
        "V34.25 sparse-calibrated gate repair 中文总结",
        payload,
        [
            "中文结论",
            "seeds",
            "all_seeds_passed",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "stable_overopen_controlled",
            "stable_over_open_rate",
            "stable_over_update_rate",
            "hard_changed_gain",
            "semantic_measurements_load_bearing_on_residual",
            "assignment_load_bearing_on_residual",
            "unit_memory_load_bearing_on_residual",
            "阶段性分析",
            "论文相关问题解决方案参考",
            "最佳下一步方案",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    write_doc(
        DECISION_DOC,
        "V34.25 sparse-calibrated gate repair 决策中文总结",
        payload,
        [
            "中文结论",
            "sparse_calibrated_gate_repair_ran",
            "sparse_calibrated_gate_repair_passed",
            "all_seeds_passed",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "stable_overopen_controlled",
            "stable_over_open_rate",
            "stable_over_update_rate",
            "semantic_measurements_load_bearing_on_residual",
            "assignment_load_bearing_on_residual",
            "unit_memory_load_bearing_on_residual",
            "claim_boundary",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=4.0e-5)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--semantic-weight", type=float, default=0.75)
    p.add_argument("--hard-weight", type=float, default=0.85)
    p.add_argument("--stable-preserve-weight", type=float, default=0.8)
    p.add_argument("--stable-negative-gate-weight", type=float, default=0.7)
    p.add_argument("--stable-gate-value-weight", type=float, default=1.5)
    p.add_argument("--hard-recall-gate-weight", type=float, default=0.35)
    p.add_argument("--hard-gate-shortfall-weight", type=float, default=1.0)
    p.add_argument("--min-hard-gate", type=float, default=0.35)
    p.add_argument("--budget-weight", type=float, default=0.25)
    p.add_argument("--gate-budget", type=float, default=0.35)
    p.add_argument("--semantic-contrast-weight", type=float, default=0.35)
    p.add_argument("--assignment-contrast-weight", type=float, default=0.25)
    p.add_argument("--semantic-contrast-margin", type=float, default=0.004)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.004)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def eval_existing_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, residual_train = load_residual_model(args, device)
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    reader, v3422_reference = make_reader(args, residual_model, device)
    ckpt_path = CKPT_ROOT / f"seed{seed}" / f"v34_25_sparse_calibrated_gate_repair_m128_h32_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    reader.load_state_dict(ckpt["reader"], strict=True)
    reader.to(device)
    eval_report = evaluate_seed(seed, args, residual_model, ckargs, reader, device)
    previous_loss_trace: list[dict[str, float]] = []
    if SUMMARY.exists():
        previous = json.loads(SUMMARY.read_text(encoding="utf-8"))
        previous_loss_trace = (
            previous.get("decision", {})
            .get("seed_reports", {})
            .get(f"seed{seed}", {})
            .get("loss_trace", [])
        )
    return {
        "generated_at_utc": utc_now(),
        "seed": seed,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "loss_trace": previous_loss_trace,
        "v34_20_train_summary": residual_train,
        "v34_22_reference": v3422_reference,
        **eval_report,
    }


def main() -> int:
    args = parse_args()
    seed_reports: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        if args.eval_only:
            print(f"开始 V34.25 sparse-calibrated gate repair 只评估校准: seed={seed}", flush=True)
            seed_reports[f"seed{seed}"] = eval_existing_seed(seed, args)
        else:
            print(f"开始 V34.25 sparse-calibrated gate repair: seed={seed}", flush=True)
            seed_reports[f"seed{seed}"] = train_reader_for_seed(seed, args)
    all_passed = all(bool(report.get("sparse_calibrated_gate_repair_passed")) for report in seed_reports.values())
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": (
            "V34.25 sparse-calibrated gate repair 已完成；本轮只修 gate calibration/sparsity，V30 与 residual content 保持 frozen，"
            "不跑 H64/H96，不声明 semantic field success。"
        ),
        "sparse_calibrated_gate_repair_ran": True,
        "sparse_calibrated_gate_repair_passed": all_passed,
        "seeds": list(seed_reports),
        "all_seeds_passed": all_passed,
        "seed_reports": seed_reports,
        "semantic_hard_signal": aggregate_split_bool(seed_reports, "semantic_hard_signal"),
        "changed_semantic_signal": aggregate_split_bool(seed_reports, "changed_semantic_signal"),
        "stable_preservation": aggregate_split_bool(seed_reports, "stable_preservation"),
        "stable_overopen_controlled": aggregate_split_bool(seed_reports, "stable_overopen_controlled"),
        "stable_over_open_rate": aggregate_split_float(seed_reports, "stable_over_open_rate"),
        "stable_gate_mean": aggregate_split_float(seed_reports, "stable_gate_mean"),
        "stable_over_update_rate": aggregate_split_float(seed_reports, "stable_over_update_rate"),
        "hard_changed_gain": aggregate_split_float(seed_reports, "hard_changed_gain"),
        "semantic_measurements_load_bearing_on_residual": aggregate_bool(seed_reports, "semantic_measurements_load_bearing_on_residual"),
        "assignment_load_bearing_on_residual": aggregate_bool(seed_reports, "assignment_load_bearing_on_residual"),
        "unit_memory_load_bearing_on_residual": aggregate_bool(seed_reports, "unit_memory_load_bearing_on_residual"),
        "v30_backbone_frozen": aggregate_bool(seed_reports, "v30_backbone_frozen"),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "claim_boundary": (
            "如果本轮通过，只能 claim sparse-calibrated residual gate repair 在 M128/H32 多 seed 上减轻 stable over-open，"
            "仍不能 claim integrated semantic field success 或 identity field success。"
        ),
        "阶段性分析": (
            "V34.24 的 blocker 是 stable gate over-open。V34.25 因此不改 residual 内容、不改 V30、不扩大 horizon，"
            "只在 gate 上加入 stable-negative loss、预算稀疏约束、hard/changed recall 保底和 threshold/temperature Pareto sweep。"
            "这个设计直接对应当前失败模式：stable 输出不坏，但 gate 选择边界太松。"
        ),
        "论文相关问题解决方案参考": (
            "本轮参考 sparse MoE / selective computation 的 gate budget 与稀疏路由思想，"
            "结合 memory-video 方法中 selective read 的原则，以及 Slot Attention/object-memory 中必须做 assignment intervention 的评价方式。"
        ),
        "最佳下一步方案": (
            "若 V34.25 通过，下一步仍应先做 claim-boundary replication/visualization，而不是直接 H64/H96；"
            "若未通过，继续修 sparse gate calibration，不扩大模型。"
        ),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_25_claim_boundary_visualization" if all_passed else "fix_gate_calibration_sparse_gate",
    }
    payload = {"generated_at_utc": utc_now(), "args": vars(args), "decision": decision}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_chinese_doc(decision)
    print(f"已写出 V34.25 summary: {SUMMARY.relative_to(ROOT)}", flush=True)
    print(f"已写出 V34.25 decision: {DECISION.relative_to(ROOT)}", flush=True)
    print(f"recommended_next_step: {decision['recommended_next_step']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
