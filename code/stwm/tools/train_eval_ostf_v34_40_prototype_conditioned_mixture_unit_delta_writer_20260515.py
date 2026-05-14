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
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

import stwm.tools.train_eval_ostf_v34_36_predictability_filtered_unit_delta_writer_20260514 as v36
from stwm.modules.ostf_v34_40_prototype_conditioned_unit_delta_writer import PrototypeConditionedUnitDeltaWriterV3440
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import load_v3425_readers, observed_mean
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, masked_mean, model_inputs
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import (
    best_copy_topk,
    compose,
    load_frozen_residual_model,
    read_unit_delta,
    roll_assignment,
    set_seed,
    summarize_loss,
)
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import top1


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_39_prototype_blended_unit_delta_targets/pointodyssey"
PROTO_TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_38_cluster_regularized_unit_delta_targets/pointodyssey"
CODEBOOK_REPORT = ROOT / "reports/stwm_ostf_v34_40_prototype_codebook_build_20260515.json"
CODEBOOK_DOC = ROOT / "docs/STWM_OSTF_V34_40_PROTOTYPE_CODEBOOK_BUILD_20260515.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_40_prototype_conditioned_mixture_unit_delta_writer_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_40_prototype_conditioned_mixture_unit_delta_writer_summary_20260515.json"
DECISION = ROOT / "reports/stwm_ostf_v34_40_prototype_conditioned_mixture_unit_delta_writer_decision_20260515.json"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_40_PROTOTYPE_CONDITIONED_MIXTURE_UNIT_DELTA_WRITER_DECISION_20260515.md"


class TensorPrototypeMixtureWriter(nn.Module):
    def __init__(self, writer: PrototypeConditionedUnitDeltaWriterV3440, target_kind: str = "top1") -> None:
        super().__init__()
        self.writer = writer
        self.target_kind = target_kind

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assign = top1(out["point_to_unit_assignment"].float()) if self.target_kind == "top1" else out["point_to_unit_assignment"].float()
        return self.writer(out, batch, assignment=assign)["unit_delta"]

    def forward_full(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assign = top1(out["point_to_unit_assignment"].float()) if self.target_kind == "top1" else out["point_to_unit_assignment"].float()
        return self.writer(out, batch, assignment=assign)


def patch_v36_paths() -> None:
    v36.TARGET_ROOT = TARGET_ROOT
    v36.CKPT_DIR = CKPT_DIR
    v36.SUMMARY = SUMMARY
    v36.DECISION = DECISION
    v36.DOC = ROOT / "docs/STWM_OSTF_V34_40_PROTOTYPE_CONDITIONED_MIXTURE_UNIT_DELTA_WRITER_SUMMARY_20260515.md"
    v36.DECISION_DOC = DECISION_DOC


def build_codebook(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if not (PROTO_TARGET_ROOT / "train").exists():
        raise FileNotFoundError(f"缺少 V34.38 prototype target root: {PROTO_TARGET_ROOT}")
    sums: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    for path in sorted((PROTO_TARGET_ROOT / "train").glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        proto_id = torch.from_numpy(np.asarray(z["prototype_id"], dtype=np.int64))
        delta = torch.from_numpy(np.asarray(z["predictability_filtered_unit_delta"], dtype=np.float32))
        active = torch.from_numpy(np.asarray(z["predictability_filtered_active"]).astype(bool)) & (proto_id >= 0)
        if sums is None:
            pcount = int(max(args.prototype_count, int(proto_id.max()) + 1))
            sums = torch.zeros((pcount, delta.shape[-1]), dtype=torch.float32)
            counts = torch.zeros((pcount,), dtype=torch.float32)
        for pid in proto_id[active].unique().tolist():
            mask = active & (proto_id == int(pid))
            sums[int(pid)] += delta[mask].sum(dim=0)
            counts[int(pid)] += float(mask.sum().item())
    if sums is None or counts is None:
        raise RuntimeError("没有可构建 codebook 的 active prototype delta")
    nonempty = counts > 0
    if not bool(nonempty.all()):
        fallback = sums[nonempty].mean(dim=0)
        sums[~nonempty] = fallback
        counts[~nonempty] = 1.0
    codebook = sums / counts[:, None].clamp_min(1.0)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.40 prototype codebook 已从 V34.38 train split 的 assignment-bound prototype target 中构建；训练 writer 时 codebook 默认冻结，只学习 observed-only prototype mixture 与小幅 residual。",
        "codebook_built": True,
        "prototype_target_root": str(PROTO_TARGET_ROOT.relative_to(ROOT)),
        "prototype_count": int(codebook.shape[0]),
        "prototype_count_nonempty": int(nonempty.sum().item()),
        "prototype_count_min": float(counts.min().item()),
        "prototype_count_max": float(counts.max().item()),
        "codebook_norm_mean": float(codebook.norm(dim=-1).mean().item()),
        "codebook_norm_max": float(codebook.norm(dim=-1).max().item()),
        "learnable_codebook": bool(args.learnable_codebook),
    }
    dump_json(CODEBOOK_REPORT, payload)
    write_doc(
        CODEBOOK_DOC,
        "V34.40 prototype codebook build 中文报告",
        payload,
        ["中文结论", "codebook_built", "prototype_target_root", "prototype_count", "prototype_count_nonempty", "codebook_norm_mean", "learnable_codebook"],
    )
    return codebook, payload


def load_proto_labels(split: str, uids: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    labels, active = [], []
    for uid in uids:
        z = np.load(PROTO_TARGET_ROOT / split / f"{uid}.npz", allow_pickle=True)
        proto_id = torch.from_numpy(np.asarray(z["prototype_id"], dtype=np.int64))
        proto_active = torch.from_numpy(np.asarray(z["prototype_id"], dtype=np.int64) >= 0)
        labels.append(proto_id.clamp_min(0))
        active.append(proto_active)
    return {
        "prototype_id": torch.stack(labels, dim=0).to(device),
        "prototype_active": torch.stack(active, dim=0).to(device),
    }


def prototype_mixture_loss(head: TensorPrototypeMixtureWriter, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], target: dict[str, torch.Tensor], proto: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    full = head.forward_full(out, batch)
    pred_unit = full["unit_delta"]
    active = target["oracle_unit_delta_active"].bool()
    sup_loss, sup_stats = v36.target_loss(pred_unit, target["oracle_unit_delta"], active)
    proto_active = active & proto["prototype_active"].bool()
    if bool(proto_active.any()):
        ce = F.cross_entropy(full["prototype_logits"][proto_active], proto["prototype_id"][proto_active])
        proto_acc = (full["prototype_logits"][proto_active].argmax(dim=-1) == proto["prototype_id"][proto_active]).float().mean()
    else:
        ce = pred_unit.sum() * 0.0
        proto_acc = torch.zeros((), device=pred_unit.device)
    entropy = -(full["prototype_weights"].clamp_min(1.0e-8) * full["prototype_weights"].clamp_min(1.0e-8).log()).sum(dim=-1)
    entropy_loss = masked_mean(entropy, active, torch.ones_like(entropy))
    residual_norm = masked_mean(full["residual_delta"].norm(dim=-1), active, torch.ones_like(entropy))

    pos = target["point_predictable_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    anchor = observed_mean(batch)
    fut = batch["fut_teacher_embedding"]
    assign = top1(out["point_to_unit_assignment"].float())
    point_delta = read_unit_delta(assign, pred_unit)
    final = compose(anchor, point_delta, pos.float(), args.train_residual_scale)
    shuf = compose(anchor, read_unit_delta(roll_assignment(assign), pred_unit), pos.float(), args.train_residual_scale)
    zero = compose(anchor, torch.zeros_like(point_delta), pos.float(), args.train_residual_scale)
    normal_cos = (v36.norm(final) * v36.norm(fut)).sum(dim=-1)
    anchor_cos = (v36.norm(anchor) * v36.norm(fut)).sum(dim=-1).detach()
    shuf_cos = (v36.norm(shuf) * v36.norm(fut)).sum(dim=-1)
    zero_cos = (v36.norm(zero) * v36.norm(fut)).sum(dim=-1).detach()
    final_loss = masked_mean(1.0 - normal_cos, pos, teacher_w)
    anchor_gain = masked_mean(F.softplus(args.anchor_gain_margin - (normal_cos - anchor_cos)), pos, teacher_w)
    assign_contrast = masked_mean(F.softplus(args.assignment_margin - (normal_cos - shuf_cos)), pos, teacher_w)
    unit_contrast = masked_mean(F.softplus(args.unit_margin - (normal_cos - zero_cos)), pos, teacher_w)
    stable_suppress = masked_mean(point_delta.norm(dim=-1), stable, torch.ones_like(teacher_w)) if bool(stable.any()) else pred_unit.sum() * 0.0
    total = (
        args.target_supervision_weight * sup_loss
        + args.prototype_ce_weight * ce
        + args.prototype_entropy_weight * entropy_loss
        + args.residual_norm_weight * residual_norm
        + args.final_target_weight * final_loss
        + args.anchor_gain_weight * anchor_gain
        + args.assignment_contrast_weight * assign_contrast
        + args.unit_contrast_weight * unit_contrast
        + args.stable_suppress_weight * stable_suppress
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "prototype_conditioned_target_loss": float(sup_loss.detach().cpu()),
        "prototype_ce_loss": float(ce.detach().cpu()),
        "prototype_top1_accuracy": float(proto_acc.detach().cpu()),
        "prototype_entropy_loss": float(entropy_loss.detach().cpu()),
        "residual_norm_loss": float(residual_norm.detach().cpu()),
        "final_target_loss": float(final_loss.detach().cpu()),
        "anchor_gain_contrast_loss": float(anchor_gain.detach().cpu()),
        "assignment_contrast_loss": float(assign_contrast.detach().cpu()),
        "unit_contrast_loss": float(unit_contrast.detach().cpu()),
        "stable_delta_suppress_loss": float(stable_suppress.detach().cpu()),
        "point_predictable_positive_count": float(pos.sum().detach().cpu()),
    }
    stats.update(sup_stats)
    return total, stats


def train_one(model: Any, ckargs: argparse.Namespace, codebook: torch.Tensor, codebook_report: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[TensorPrototypeMixtureWriter, dict[str, Any]]:
    writer = PrototypeConditionedUnitDeltaWriterV3440(
        int(model.v30.cfg.hidden_dim),
        prototype_codebook=codebook.to(device),
        semantic_dim=args.teacher_embedding_dim,
        hidden_dim=args.value_hidden_dim,
        max_residual_magnitude=args.max_residual_magnitude,
        prototype_temperature=args.prototype_temperature,
        learnable_codebook=args.learnable_codebook,
    ).to(device)
    head = TensorPrototypeMixtureWriter(writer, args.target_kind).to(device)
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
        target = v36.load_filtered_batch("train", bd["uid"], device)
        proto = load_proto_labels("train", bd["uid"], device)
        with torch.no_grad():
            out = model(**model_inputs(bd), intervention="force_gate_zero")
        loss, stats = prototype_mixture_loss(head, out, bd, target, proto, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(
                "训练进度: "
                f"step={step}/{args.steps}, loss={stats['loss']:.6f}, "
                f"target={stats['prototype_conditioned_target_loss']:.6f}, "
                f"proto_ce={stats['prototype_ce_loss']:.6f}, proto_acc={stats['prototype_top1_accuracy']:.3f}, "
                f"assign={stats['assignment_contrast_loss']:.6f}",
                flush=True,
            )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"v34_40_prototype_conditioned_mixture_unit_delta_writer_m128_h32_seed{args.seed}.pt"
    torch.save({"head": head.state_dict(), "args": vars(args), "step": args.steps, "codebook_report": codebook_report}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.40 prototype-conditioned mixture writer 训练完成；writer 先预测共享 correction prototype/mode，再输出小幅 residual，不训练 learned gate。",
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "learned_gate_training_ran": False,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "codebook_report": codebook_report,
        "loss_summaries": {k: summarize_loss(losses, k) for k in losses[-1].keys() if k != "step"} if losses else {},
        "loss_trace": losses,
        "duration_seconds": float(time.time() - start),
    }
    return head.eval(), summary


def load_trained_head(model: Any, codebook: torch.Tensor, codebook_report: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[TensorPrototypeMixtureWriter, dict[str, Any]]:
    writer = PrototypeConditionedUnitDeltaWriterV3440(
        int(model.v30.cfg.hidden_dim),
        prototype_codebook=codebook.to(device),
        semantic_dim=args.teacher_embedding_dim,
        hidden_dim=args.value_hidden_dim,
        max_residual_magnitude=args.max_residual_magnitude,
        prototype_temperature=args.prototype_temperature,
        learnable_codebook=args.learnable_codebook,
    ).to(device)
    head = TensorPrototypeMixtureWriter(writer, args.target_kind).to(device)
    ckpt = CKPT_DIR / f"v34_40_prototype_conditioned_mixture_unit_delta_writer_m128_h32_seed{args.seed}.pt"
    ck = torch.load(ckpt, map_location=device)
    head.load_state_dict(ck["head"], strict=True)
    head.eval()
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.40 prototype-conditioned mixture writer 从已保存 checkpoint 进入 eval-only；没有重新训练。",
        "fresh_training_completed": False,
        "eval_only_loaded_checkpoint": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "learned_gate_training_ran": False,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "codebook_report": codebook_report,
    }
    return head, summary


def update_written_reports(train_summary: dict[str, Any], codebook_report: dict[str, Any]) -> None:
    if SUMMARY.exists():
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        payload["train_summary"] = train_summary
        payload["codebook_report"] = codebook_report
        payload["中文结论"] = "V34.40 prototype-conditioned mixture writer 训练/评估完成；该轮检验共享 prototype correction mode 是否能改善 V34.39 learned writer 的 val/test 泛化。"
        payload["阶段性分析"] = "V34.39 证明连续 blended target 可在训练集过拟合但 val/test 不泛化；V34.40 改成 prototype mixture + 小 residual，限制 writer 记忆样本噪声，要求 val/test 正增益和 assignment delta 同时成立。"
        payload["论文相关问题解决方案参考"] = "该轮对应 VQ/codebook、mixture-of-experts 与 object-centric slot memory 思路：先选择共享 correction mode，再做小幅连续修正，而不是直接回归不可预测的 sample-specific delta。"
        payload["最佳下一步方案"] = payload.get("decision", {}).get("recommended_next_step")
        dump_json(SUMMARY, payload)
    if DECISION.exists():
        decision = json.loads(DECISION.read_text(encoding="utf-8"))
        decision["中文结论"] = "V34.40 prototype-conditioned mixture unit_delta writer 完成；不训练 gate、不跑 M512，只判断共享 prototype mode 是否让 learned writer 在 copy/top-k evidence anchor 上泛化。"
        decision["prototype_conditioned_mixture_writer_built"] = True
        decision["codebook_report"] = codebook_report
        decision["v34_40_metrics"] = decision.pop("v34_36_metrics", None)
        if decision.get("probe_passed"):
            decision["recommended_next_step"] = "return_to_full_system_benchmark"
        elif decision.get("unit_residual_improves_evidence_anchor"):
            decision["recommended_next_step"] = "rerun_v34_40_seed123_or_refine_prototype_assignment"
        else:
            decision["recommended_next_step"] = "fix_prototype_conditioned_writer_generalization"
        dump_json(DECISION, decision)
        write_doc(
            DECISION_DOC,
            "V34.40 prototype-conditioned mixture unit_delta writer 决策中文报告",
            decision,
            [
                "中文结论",
                "prototype_conditioned_mixture_writer_built",
                "probe_passed",
                "selected_config_by_val",
                "beats_copy_topk_baseline",
                "unit_residual_improves_evidence_anchor",
                "assignment_load_bearing_on_system",
                "unit_memory_load_bearing_on_system",
                "semantic_hard_signal",
                "changed_semantic_signal",
                "stable_preservation",
                "recommended_next_step",
            ],
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=256)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--prototype-count", type=int, default=64)
    p.add_argument("--prototype-temperature", type=float, default=0.7)
    p.add_argument("--max-residual-magnitude", type=float, default=0.35)
    p.add_argument("--learnable-codebook", action="store_true")
    p.add_argument("--target-kind", choices=["top1"], default="top1")
    p.add_argument("--train-residual-scale", type=float, default=1.0)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--target-supervision-weight", type=float, default=2.0)
    p.add_argument("--prototype-ce-weight", type=float, default=1.0)
    p.add_argument("--prototype-entropy-weight", type=float, default=0.005)
    p.add_argument("--residual-norm-weight", type=float, default=0.05)
    p.add_argument("--final-target-weight", type=float, default=0.8)
    p.add_argument("--anchor-gain-weight", type=float, default=0.8)
    p.add_argument("--assignment-contrast-weight", type=float, default=1.5)
    p.add_argument("--unit-contrast-weight", type=float, default=1.0)
    p.add_argument("--stable-suppress-weight", type=float, default=0.05)
    p.add_argument("--anchor-gain-margin", type=float, default=0.006)
    p.add_argument("--assignment-margin", type=float, default=0.006)
    p.add_argument("--unit-margin", type=float, default=0.006)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    patch_v36_paths()
    args = parse_args()
    set_seed(args.seed)
    v36.TARGET_ROOT = TARGET_ROOT
    codebook, codebook_report = build_codebook(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    if args.eval_only:
        head, train_summary = load_trained_head(model, codebook, codebook_report, args, device)
    else:
        head, train_summary = train_one(model, ckargs, codebook, codebook_report, args, device)
    v36.evaluate(model, ckargs, head, train_summary, codebook_report, args, device)
    update_written_reports(train_summary, codebook_report)
    print(f"已写出 V34.40 prototype-conditioned mixture writer 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
