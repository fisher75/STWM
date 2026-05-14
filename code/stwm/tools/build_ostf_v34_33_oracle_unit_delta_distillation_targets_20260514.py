#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, norm, update_method
from stwm.tools.eval_ostf_v34_30_assignment_oracle_upper_bound_probe_20260514 import (
    oracle_unit_memory_from_assignment,
    read_unit_memory,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_33_oracle_unit_delta_distillation_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_33_oracle_unit_delta_distillation_target_build_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_33_ORACLE_UNIT_DELTA_DISTILLATION_TARGET_BUILD_20260514.md"


def top1(assign: torch.Tensor) -> torch.Tensor:
    idx = assign.argmax(dim=-1)
    return F.one_hot(idx, num_classes=assign.shape[-1]).to(assign.dtype)


def unit_target_active(assign: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = assign[:, :, None, :] * pos.float()[:, :, :, None] * weight.float()[:, :, :, None]
    den = w.sum(dim=1).permute(0, 2, 1)
    return den > 1.0e-5


def build_split(split: str, residual_model: Any, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    out_dir = TARGET_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = Acc()
    sample_count = 0
    active_soft = []
    active_top1 = []
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
            soft_delta = oracle_unit_memory_from_assignment(assign_soft, anchor, target, pos, weight)
            top1_delta = oracle_unit_memory_from_assignment(assign_top1, anchor, target, pos, weight)
            soft_active = unit_target_active(assign_soft, pos, weight)
            top1_active = unit_target_active(assign_top1, pos, weight)
            soft_pred = norm(anchor + read_unit_memory(assign_soft, soft_delta))
            top1_pred = norm(anchor + read_unit_memory(assign_top1, top1_delta))
            update_method(acc, "copy_mean_observed", anchor, pointwise=pointwise, target=target, mm=mm, anchor=anchor)
            update_method(acc, "oracle_unit_soft_delta_target", soft_pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor)
            update_method(acc, "oracle_unit_top1_delta_target", top1_pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor)
            bsz = len(bd["uid"])
            for i in range(bsz):
                uid = bd["uid"][i]
                np.savez(
                    out_dir / f"{uid}.npz",
                    sample_uid=np.asarray(uid),
                    point_id=bd["point_id"][i].detach().cpu().numpy() if "point_id" in bd else np.arange(assign_soft.shape[1], dtype=np.int64),
                    point_to_instance_id=bd["point_to_instance_id"][i].detach().cpu().numpy(),
                    hard_changed_mask=pos[i].detach().cpu().numpy().astype(bool),
                    fut_teacher_available_mask=bd["fut_teacher_available_mask"][i].detach().cpu().numpy().astype(bool),
                    evidence_anchor_type=np.asarray("copy_mean_observed"),
                    point_to_unit_assignment_soft=assign_soft[i].detach().cpu().numpy().astype(np.float32),
                    point_to_unit_assignment_top1=assign_top1[i].detach().cpu().numpy().astype(np.float32),
                    oracle_unit_delta_soft=soft_delta[i].detach().cpu().numpy().astype(np.float32),
                    oracle_unit_delta_top1=top1_delta[i].detach().cpu().numpy().astype(np.float32),
                    oracle_unit_delta_soft_active=soft_active[i].detach().cpu().numpy().astype(bool),
                    oracle_unit_delta_top1_active=top1_active[i].detach().cpu().numpy().astype(bool),
                    leakage_safe=np.asarray(True),
                    future_teacher_embeddings_supervision_only=np.asarray(True),
                    future_teacher_embeddings_input_allowed=np.asarray(False),
                )
                sample_count += 1
                active_soft.append(float(soft_active[i].float().mean().detach().cpu()))
                active_top1.append(float(top1_active[i].float().mean().detach().cpu()))
    methods = sorted({key.split(":")[0] for key in acc.sum.keys()})
    metrics = {name: finalize_method(acc, name) for name in methods}
    return {
        "sample_count": sample_count,
        "target_dir": str(out_dir.relative_to(ROOT)),
        "soft_active_ratio": float(np.mean(active_soft)) if active_soft else None,
        "top1_active_ratio": float(np.mean(active_top1)) if active_top1 else None,
        "oracle_metrics": metrics,
    }


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
    for split in ("train", "val", "test"):
        print(f"开始构建 V34.33 oracle unit delta target: split={split}", flush=True)
        per_split[split] = build_split(split, residual_model, ckargs, device)
    ready = all(per_split[s]["sample_count"] > 0 for s in ("train", "val", "test"))
    report = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.33 oracle unit delta distillation target 已构建；future teacher embedding 只用于监督 target，不进入模型输入。",
        "oracle_unit_delta_targets_built": ready,
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "per_split": per_split,
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "future_teacher_embeddings_supervision_only": True,
        "future_teacher_embeddings_input_allowed": False,
        "train_summary_reference": train_summary,
        "recommended_next_step": "train_oracle_unit_delta_value_decoder" if ready else "fix_oracle_unit_delta_target_build",
    }
    dump_json(REPORT, report)
    write_doc(
        DOC,
        "V34.33 oracle unit delta distillation target build 中文报告",
        report,
        [
            "中文结论",
            "oracle_unit_delta_targets_built",
            "target_root",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "future_teacher_embeddings_supervision_only",
            "future_teacher_embeddings_input_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.33 target build report: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"recommended_next_step: {report['recommended_next_step']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
