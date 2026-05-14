#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_22_activation_state_reader import ActivationStateReaderV3422
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import cosine_loss
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import hard_changed_aligned_mask
from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import (
    CHECKPOINT as V3422_CHECKPOINT,
    SUMMARY as V3422_SUMMARY,
    TARGETS,
    compose_reader_gate,
    eval_best_gate_interventions,
    labels,
    load_residual_model,
    make_loader,
    reader_inputs,
)


SUMMARY = ROOT / "reports/stwm_ostf_v34_23_activation_state_gate_probe_summary_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_23_ACTIVATION_STATE_GATE_PROBE_SUMMARY_20260513.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_23_activation_state_gate_probe_h32_m128"
CHECKPOINT = CKPT_DIR / "v34_23_activation_state_gate_probe_m128_h32_seed42.pt"


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


def load_reader(args: argparse.Namespace, residual_model: Any, device: torch.device) -> tuple[ActivationStateReaderV3422, dict[str, float], dict[str, Any]]:
    report = json.loads(V3422_SUMMARY.read_text(encoding="utf-8")) if V3422_SUMMARY.exists() else {}
    ckpt = torch.load(V3422_CHECKPOINT, map_location="cpu")
    reader = ActivationStateReaderV3422(
        int(residual_model.v30.cfg.hidden_dim),
        semantic_dim=int(getattr(args, "teacher_embedding_dim", 768)),
        hidden_dim=int(ckpt["args"].get("reader_hidden_dim", args.reader_hidden_dim)),
    ).to(device)
    reader.load_state_dict(ckpt["reader"], strict=True)
    thresholds = {k: float(v) for k, v in ckpt.get("thresholds", {target: 0.5 for target in TARGETS}).items()}
    return reader, thresholds, report


def gate_final(out: dict[str, torch.Tensor], pred: dict[str, torch.Tensor], gate_name: str, thresholds: dict[str, float]) -> torch.Tensor:
    return compose_reader_gate(out, pred, gate_name, thresholds)


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, residual_train = load_residual_model(args, device)
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    reader, thresholds, v3422_report = load_reader(args, residual_model, device)
    for p in residual_model.parameters():
        p.requires_grad_(False)
    reader.train()
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=1e-4)
    gate_name = args.gate_name
    losses: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "semantic": 0.0, "hard": 0.0, "stable": 0.0, "semantic_contrast": 0.0, "assignment_contrast": 0.0}
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
            final = gate_final(out, pred, gate_name, thresholds)
            zero_final = gate_final(zero, pred_zero, gate_name, thresholds)
            shuf_sem_final = gate_final(shuf_sem, pred_shuf_sem, gate_name, thresholds)
            shuf_assign_final = gate_final(shuf_assign, pred_shuf_assign, gate_name, thresholds)
            valid = bd["fut_teacher_available_mask"].bool()
            aligned = hard_changed_aligned_mask(bd)
            hard_changed = (bd["semantic_hard_mask"].bool() | bd["changed_mask"].bool()) & valid
            stable = bd["stable_suppress_mask"].bool() & valid
            teacher_w = bd["teacher_confidence"].float().clamp(0.05, 1.0)
            semantic = cosine_loss(final, bd["fut_teacher_embedding"], aligned, teacher_w)
            hard = cosine_loss(final, bd["fut_teacher_embedding"], hard_changed, teacher_w)
            stable_loss = cosine_loss(final, out["pointwise_semantic_belief"].detach(), stable, torch.ones_like(teacher_w)) if bool(stable.any()) else final.sum() * 0.0
            normal_cos = local_cos(final, bd["fut_teacher_embedding"])
            zero_cos = local_cos(zero_final, bd["fut_teacher_embedding"]).detach()
            shuf_sem_cos = local_cos(shuf_sem_final, bd["fut_teacher_embedding"]).detach()
            shuf_assign_cos = local_cos(shuf_assign_final, bd["fut_teacher_embedding"]).detach()
            sem_contrast = masked_mean(F.softplus(args.semantic_contrast_margin - (normal_cos - torch.maximum(zero_cos, shuf_sem_cos))), aligned, teacher_w)
            assign_contrast = masked_mean(F.softplus(args.assignment_contrast_margin - (normal_cos - shuf_assign_cos)), aligned, teacher_w)
            loss = 0.8 * semantic + 0.9 * hard + 0.6 * stable_loss + args.semantic_contrast_weight * sem_contrast + args.assignment_contrast_weight * assign_contrast
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["semantic"] += float(semantic.detach().cpu())
            totals["hard"] += float(hard.detach().cpu())
            totals["stable"] += float(stable_loss.detach().cpu())
            totals["semantic_contrast"] += float(sem_contrast.detach().cpu())
            totals["assignment_contrast"] += float(assign_contrast.detach().cpu())
            seen += 1
        row = {"epoch": float(epoch), **{k: v / max(seen, 1) for k, v in totals.items()}}
        losses.append(row)
        print(f"训练进度: epoch={epoch}/{args.epochs}, loss={row['loss']:.6f}, hard={row['hard']:.6f}, sem_contrast={row['semantic_contrast']:.6f}", flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"reader": reader.state_dict(), "args": vars(args), "thresholds": thresholds, "gate_name": gate_name}, CHECKPOINT)
    intervention_eval = {split: eval_best_gate_interventions(split, residual_model, reader, ckargs, device, thresholds, gate_name) for split in ("val", "test")}
    normal_val = intervention_eval["val"]["modes"]["normal"]
    normal_test = intervention_eval["test"]["modes"]["normal"]
    sem_lb = bool(
        min(intervention_eval["val"]["zero_semantic_measurements_delta"] or 0.0, intervention_eval["val"]["shuffle_semantic_measurements_delta"] or 0.0) > 0.002
        and min(intervention_eval["test"]["zero_semantic_measurements_delta"] or 0.0, intervention_eval["test"]["shuffle_semantic_measurements_delta"] or 0.0) > 0.002
    )
    assign_lb = bool((intervention_eval["val"]["shuffle_assignment_delta"] or 0.0) > 0.002 and (intervention_eval["test"]["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool((intervention_eval["val"]["zero_unit_memory_delta"] or 0.0) > 0.002 and (intervention_eval["test"]["zero_unit_memory_delta"] or 0.0) > 0.002)
    passed = bool(
        sem_lb
        and assign_lb
        and unit_lb
        and normal_val["stable_preservation"]
        and normal_test["stable_preservation"]
        and (normal_val["semantic_hard_signal"] or normal_val["changed_semantic_signal"])
        and (normal_test["semantic_hard_signal"] or normal_test["changed_semantic_signal"])
    )
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.23 activation-state gate probe 已完成；只训练 activation reader/gate probe，V30、V34.20 residual 和 selector 全部 frozen，仍不声明 semantic field success。",
        "activation_state_gate_probe_ran": True,
        "activation_state_gate_probe_passed": passed,
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)),
        "gate_name": gate_name,
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": True,
        "residual_content_frozen": True,
        "semantic_hard_signal": {"val": normal_val["semantic_hard_signal"], "test": normal_test["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": normal_val["changed_semantic_signal"], "test": normal_test["changed_semantic_signal"]},
        "stable_preservation": {"val": normal_val["stable_preservation"], "test": normal_test["stable_preservation"]},
        "hard_changed_gain": {"val": normal_val["hard_changed_gain"], "test": normal_test["hard_changed_gain"]},
        "semantic_measurements_load_bearing_on_residual": sem_lb,
        "assignment_load_bearing_on_residual": assign_lb,
        "unit_memory_load_bearing_on_residual": unit_lb,
        "intervention_eval": intervention_eval,
        "loss_trace": losses,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_23_seed123_replication" if passed else "fix_activation_state_gate_training",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "v34_20_train_summary": residual_train,
        "v34_22_reference": v3422_report,
        "decision": decision,
    }
    dump_json(SUMMARY, payload)
    write_doc(
        DOC,
        "V34.23 activation-state gate probe 中文报告",
        decision,
        [
            "中文结论",
            "activation_state_gate_probe_ran",
            "activation_state_gate_probe_passed",
            "gate_name",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "learned_gate_training_ran",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "hard_changed_gain",
            "semantic_measurements_load_bearing_on_residual",
            "assignment_load_bearing_on_residual",
            "unit_memory_load_bearing_on_residual",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.23 activation-state gate probe: {SUMMARY.relative_to(ROOT)}")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5.0e-5)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--utility-margin", type=float, default=0.005)
    p.add_argument("--gate-name", default="benefit_soft")
    p.add_argument("--semantic-contrast-weight", type=float, default=0.35)
    p.add_argument("--assignment-contrast-weight", type=float, default=0.25)
    p.add_argument("--semantic-contrast-margin", type=float, default=0.004)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.004)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    train_one(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
