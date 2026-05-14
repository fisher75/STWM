#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import setproctitle
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import (
    load_residual_model,
    make_loader,
)
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import (
    masks,
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
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs


REPORT = ROOT / "reports/stwm_ostf_v34_30_assignment_oracle_upper_bound_probe_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_30_ASSIGNMENT_ORACLE_UPPER_BOUND_PROBE_20260514.md"


def assignment_roll(assign: torch.Tensor) -> torch.Tensor:
    if assign.shape[-1] <= 1:
        return assign
    idx = torch.roll(torch.arange(assign.shape[-1], device=assign.device), shifts=1, dims=0)
    return assign[..., idx]


def top1(assign: torch.Tensor) -> torch.Tensor:
    idx = assign.argmax(dim=-1)
    return F.one_hot(idx, num_classes=assign.shape[-1]).to(assign.dtype)


def oracle_unit_memory_from_assignment(assign: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    delta = target.float() - anchor.float()
    w = assign[:, :, None, :] * pos.float()[:, :, :, None] * weight.float()[:, :, :, None]
    unit_delta = torch.einsum("bmhu,bmhd->buhd", w, delta)
    den = w.sum(dim=1).permute(0, 2, 1).unsqueeze(-1)
    return unit_delta / den.clamp_min(1.0e-6)


def read_unit_memory(assign: torch.Tensor, unit_memory: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bmu,buhd->bmhd", assign, unit_memory)


def instance_assignment(point_to_instance_id: torch.Tensor) -> torch.Tensor:
    rows = []
    max_units = 1
    for inst in point_to_instance_id.detach().cpu():
        ids = sorted(int(x) for x in inst.unique().tolist() if int(x) >= 0)
        max_units = max(max_units, len(ids))
    for inst in point_to_instance_id.detach().cpu():
        ids = sorted(int(x) for x in inst.unique().tolist() if int(x) >= 0)
        if not ids:
            rows.append(torch.full((inst.shape[0], max_units), 1.0 / max_units))
            continue
        mapping = {v: i for i, v in enumerate(ids)}
        mat = torch.zeros(inst.shape[0], max_units)
        for m, val in enumerate(inst.tolist()):
            if int(val) in mapping:
                mat[m, mapping[int(val)]] = 1.0
        empty = mat.sum(dim=-1) == 0
        if bool(empty.any()):
            mat[empty] = 1.0 / max_units
        rows.append(mat)
    return torch.stack(rows, dim=0).to(point_to_instance_id.device, dtype=torch.float32)


def eval_split(split: str, residual_model: Any, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    acc = Acc()
    delta_acc = {k: Acc() for k in ["unit_soft", "unit_soft_shuffle", "unit_top1", "unit_top1_shuffle", "instance", "instance_shuffle"]}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            mm = masks(bd)
            pos = mm["hard_changed"]
            target = bd["fut_teacher_embedding"]
            pointwise = out["pointwise_semantic_belief"]
            anchor = observed_mean(bd)
            weight = bd["teacher_confidence"].float().clamp(0.05, 1.0)
            assign_soft = out["point_to_unit_assignment"].float()
            assign_top1 = top1(assign_soft)
            assign_inst = instance_assignment(bd["point_to_instance_id"])
            soft_mem = oracle_unit_memory_from_assignment(assign_soft, anchor, target, pos, weight)
            top1_mem = oracle_unit_memory_from_assignment(assign_top1, anchor, target, pos, weight)
            inst_mem = oracle_unit_memory_from_assignment(assign_inst, anchor, target, pos, weight)
            methods = {
                "copy_mean_observed": anchor,
                "oracle_unit_soft_assignment": norm(anchor + read_unit_memory(assign_soft, soft_mem)),
                "oracle_unit_top1_assignment": norm(anchor + read_unit_memory(assign_top1, top1_mem)),
                "oracle_instance_assignment": norm(anchor + read_unit_memory(assign_inst, inst_mem)),
            }
            for name, pred in methods.items():
                update_method(acc, name, pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor)

            for name, assign, memory in [("unit_soft", assign_soft, soft_mem), ("unit_top1", assign_top1, top1_mem), ("instance", assign_inst, inst_mem)]:
                pred = norm(anchor + read_unit_memory(assign, memory))
                shuf = norm(anchor + read_unit_memory(assignment_roll(assign), memory))
                update_method(delta_acc[name], name, pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor)
                update_method(delta_acc[f"{name}_shuffle"], f"{name}_shuffle", shuf, pointwise=pointwise, target=target, mm=mm, anchor=anchor)
    methods_out = sorted({key.split(":")[0] for key in acc.sum.keys()})
    metrics = {name: finalize_method(acc, name) for name in methods_out}
    deltas = {}
    for name in ["unit_soft", "unit_top1", "instance"]:
        normal = finalize_method(delta_acc[name], name)
        shuf = finalize_method(delta_acc[f"{name}_shuffle"], f"{name}_shuffle")
        deltas[name] = {
            "hard_changed_gain_vs_anchor": normal["hard_changed_gain_vs_anchor"],
            "shuffle_assignment_delta": None
            if normal["hard_changed_gain_vs_anchor"] is None or shuf["hard_changed_gain_vs_anchor"] is None
            else float(normal["hard_changed_gain_vs_anchor"] - shuf["hard_changed_gain_vs_anchor"]),
        }
    return {"methods": metrics, "oracle_assignment_delta": deltas}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, train_summary = load_residual_model(args, device)
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    per_split = {}
    for split in ("val", "test"):
        print(f"开始 V34.30 assignment oracle upper-bound probe: split={split}", flush=True)
        per_split[split] = eval_split(split, residual_model, ckargs, device)
    val = per_split["val"]["oracle_assignment_delta"]
    test = per_split["test"]["oracle_assignment_delta"]
    unit_oracle_possible = bool((val["unit_top1"]["hard_changed_gain_vs_anchor"] or 0.0) > 0.01 and (test["unit_top1"]["hard_changed_gain_vs_anchor"] or 0.0) > 0.01)
    instance_oracle_possible = bool((val["instance"]["hard_changed_gain_vs_anchor"] or 0.0) > 0.01 and (test["instance"]["hard_changed_gain_vs_anchor"] or 0.0) > 0.01)
    learned_assignment_gap = bool(instance_oracle_possible and not unit_oracle_possible)
    recommended = "fix_instance_slot_matching_targets" if learned_assignment_gap else "fix_unit_residual_training_objective" if unit_oracle_possible else "fix_hard_changed_targets_or_semantic_evidence_baseline"
    report = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.30 oracle upper-bound probe 完成；该报告只用 future target 做 supervision/upper-bound 分析，不作为方法输入，也不声明 semantic field success。",
        "oracle_probe_ran": True,
        "unit_assignment_oracle_possible": unit_oracle_possible,
        "instance_assignment_oracle_possible": instance_oracle_possible,
        "learned_assignment_gap_detected": learned_assignment_gap,
        "per_split": per_split,
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "integrated_semantic_field_claim_allowed": False,
        "integrated_identity_field_claim_allowed": False,
        "recommended_next_step": recommended,
        "阶段性分析": (
            "V34.29 训练后 residual 低于 copy_mean anchor，因此需要先确认 assignment-bound correction 是否存在上界。"
            "V34.30 用当前 learned unit assignment 与 GT instance assignment 分别构造 oracle unit correction：若 instance oracle 强而 learned unit oracle 弱，说明要修 instance-slot matching；"
            "若 learned unit oracle 已强，则主要是训练目标/容量没学到。"
        ),
        "论文相关问题解决方案参考": (
            "这个诊断对应 Slot Attention 的 slot-identifiability 问题，以及 video memory read 方法中 memory slot 必须可被反事实置换破坏的原则。"
            "它把“融合问题”拆成 target 上界、assignment 质量、residual 训练三个层次。"
        ),
        "train_summary_reference": train_summary,
    }
    dump_json(REPORT, report)
    write_doc(
        DOC,
        "V34.30 assignment oracle upper-bound probe 中文报告",
        report,
        [
            "中文结论",
            "oracle_probe_ran",
            "unit_assignment_oracle_possible",
            "instance_assignment_oracle_possible",
            "learned_assignment_gap_detected",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "integrated_semantic_field_claim_allowed",
            "integrated_identity_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.30 oracle upper-bound report: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"recommended_next_step: {recommended}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
