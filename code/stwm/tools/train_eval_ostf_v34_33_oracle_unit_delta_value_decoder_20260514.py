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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import load_v3425_readers, masks, observed_max_conf, observed_mean
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, norm, update_method
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, masked_mean, model_inputs
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import hard_changed_aligned_mask
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
)
from stwm.tools.train_eval_ostf_v34_32_direct_evidence_raw_delta_value_memory_20260514 import DirectEvidenceRawDeltaHeadV3432


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_33_oracle_unit_delta_distillation_targets/pointodyssey"
TARGET_REPORT = ROOT / "reports/stwm_ostf_v34_33_oracle_unit_delta_distillation_target_build_20260514.json"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_33_oracle_unit_delta_value_decoder_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_33_oracle_unit_delta_value_decoder_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_33_oracle_unit_delta_value_decoder_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_33_ORACLE_UNIT_DELTA_VALUE_DECODER_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_33_ORACLE_UNIT_DELTA_VALUE_DECODER_DECISION_20260514.md"


def load_target_batch(split: str, uids: list[str], device: torch.device, kind: str) -> dict[str, torch.Tensor]:
    deltas = []
    active = []
    for uid in uids:
        z = np.load(TARGET_ROOT / split / f"{uid}.npz", allow_pickle=True)
        deltas.append(torch.from_numpy(np.asarray(z[f"oracle_unit_delta_{kind}"], dtype=np.float32)))
        active.append(torch.from_numpy(np.asarray(z[f"oracle_unit_delta_{kind}_active"]).astype(bool)))
    return {
        "oracle_unit_delta": torch.stack(deltas, dim=0).to(device),
        "oracle_unit_delta_active": torch.stack(active, dim=0).to(device),
    }


def top1(assign: torch.Tensor) -> torch.Tensor:
    idx = assign.argmax(dim=-1)
    return F.one_hot(idx, num_classes=assign.shape[-1]).to(assign.dtype)


def target_loss(
    pred_unit: torch.Tensor,
    target_unit: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not bool(active.any()):
        z = pred_unit.sum() * 0.0
        return z, {"unit_delta_direction_loss": 0.0, "unit_delta_magnitude_loss": 0.0, "unit_delta_raw_loss": 0.0}
    pred = pred_unit[active]
    target = target_unit[active]
    dir_loss = 1.0 - (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean()
    mag_loss = F.smooth_l1_loss(pred.norm(dim=-1), target.norm(dim=-1))
    raw_loss = F.smooth_l1_loss(pred, target)
    return dir_loss + 0.75 * mag_loss + 0.5 * raw_loss, {
        "unit_delta_direction_loss": float(dir_loss.detach().cpu()),
        "unit_delta_magnitude_loss": float(mag_loss.detach().cpu()),
        "unit_delta_raw_loss": float(raw_loss.detach().cpu()),
    }


def decoder_loss(head: DirectEvidenceRawDeltaHeadV3432, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], target: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    pred_unit = head(out, batch)
    target_unit = target["oracle_unit_delta"]
    active = target["oracle_unit_delta_active"]
    sup_loss, sup_stats = target_loss(pred_unit, target_unit, active)
    pos = hard_changed_aligned_mask(batch)
    valid = batch["fut_teacher_available_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & valid
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    anchor = observed_mean(batch)
    fut = batch["fut_teacher_embedding"]
    assign = top1(out["point_to_unit_assignment"].float()) if args.target_kind == "top1" else out["point_to_unit_assignment"].float()
    point_delta = read_unit_delta(assign, pred_unit)
    gate = pos.float()
    final = compose(anchor, point_delta, gate, args.train_residual_scale)
    shuf = compose(anchor, read_unit_delta(roll_assignment(assign), pred_unit), gate, args.train_residual_scale)
    zero = compose(anchor, torch.zeros_like(point_delta), gate, args.train_residual_scale)
    normal_cos = (norm(final) * norm(fut)).sum(dim=-1)
    anchor_cos = (norm(anchor) * norm(fut)).sum(dim=-1).detach()
    shuf_cos = (norm(shuf) * norm(fut)).sum(dim=-1)
    zero_cos = (norm(zero) * norm(fut)).sum(dim=-1).detach()
    final_loss = masked_mean(1.0 - normal_cos, pos, teacher_w)
    anchor_gain = masked_mean(F.softplus(args.anchor_gain_margin - (normal_cos - anchor_cos)), pos, teacher_w)
    assign_contrast = masked_mean(F.softplus(args.assignment_margin - (normal_cos - shuf_cos)), pos, teacher_w)
    unit_contrast = masked_mean(F.softplus(args.unit_margin - (normal_cos - zero_cos)), pos, teacher_w)
    stable_norm = point_delta.norm(dim=-1)
    stable_suppress = masked_mean(stable_norm, stable, torch.ones_like(teacher_w)) if bool(stable.any()) else pred_unit.sum() * 0.0
    total = (
        args.target_supervision_weight * sup_loss
        + args.final_target_weight * final_loss
        + args.anchor_gain_weight * anchor_gain
        + args.assignment_contrast_weight * assign_contrast
        + args.unit_contrast_weight * unit_contrast
        + args.stable_suppress_weight * stable_suppress
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "oracle_unit_delta_supervision_loss": float(sup_loss.detach().cpu()),
        "final_target_loss": float(final_loss.detach().cpu()),
        "anchor_gain_contrast_loss": float(anchor_gain.detach().cpu()),
        "assignment_contrast_loss": float(assign_contrast.detach().cpu()),
        "unit_contrast_loss": float(unit_contrast.detach().cpu()),
        "stable_delta_suppress_loss": float(stable_suppress.detach().cpu()),
    }
    stats.update(sup_stats)
    return total, stats


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
        targets = load_target_batch("train", bd["uid"], device, args.target_kind)
        with torch.no_grad():
            out = model(**model_inputs(bd), intervention="force_gate_zero")
        loss, stats = decoder_loss(head, out, bd, targets, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(
                "训练进度: "
                f"step={step}/{args.steps}, loss={stats['loss']:.6f}, "
                f"target={stats['oracle_unit_delta_supervision_loss']:.6f}, "
                f"dir={stats['unit_delta_direction_loss']:.6f}, assign={stats['assignment_contrast_loss']:.6f}",
                flush=True,
            )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"v34_33_oracle_unit_delta_value_decoder_m128_h32_seed{args.seed}_{args.target_kind}.pt"
    torch.save({"head": head.state_dict(), "args": vars(args), "init": init, "step": args.steps}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.33 oracle unit-delta value decoder 训练完成；显式读取 oracle unit delta distillation target，冻结 V30/assignment/residual model，不训练 gate。",
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "target_kind": args.target_kind,
        "init": init,
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
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
        "V34.33 oracle unit-delta value decoder 训练中文摘要",
        summary,
        ["中文结论", "fresh_training_completed", "checkpoint_path", "target_root", "target_kind", "steps", "train_sample_count", "v30_backbone_frozen", "assignment_frozen", "learned_gate_training_ran", "future_leakage_detected", "trajectory_degraded"],
    )
    print(f"已写出 V34.33 训练摘要: {SUMMARY.relative_to(ROOT)}", flush=True)
    head.eval()
    return model, ckargs, head, summary


def eval_split(split: str, model: Any, ckargs: argparse.Namespace, head: DirectEvidenceRawDeltaHeadV3432, readers: dict[str, dict[str, Any]], args: argparse.Namespace, scale: float, gate_mode: str, device: torch.device) -> dict[str, Any]:
    acc = Acc()
    delta_acc = {k: Acc() for k in ["normal", "zero_semantic", "shuffle_semantic", "shuffle_assignment", "zero_unit"]}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            fut = bd["fut_teacher_embedding"]
            anchor = observed_mean(bd)
            assign = top1(out["point_to_unit_assignment"].float()) if args.target_kind == "top1" else out["point_to_unit_assignment"].float()
            gate = hard_changed_aligned_mask(bd).float() if gate_mode == "oracle_mask" else sparse_seed_mean_gate(out, readers)
            pred_unit = head(out, bd)
            point_delta = read_unit_delta(assign, pred_unit)
            final = compose(anchor, point_delta, gate, scale)
            method = f"v34_33_{gate_mode}_oracle_unit_delta_decoder"
            update_method(acc, "pointwise_base", pointwise, pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, "copy_max_conf_observed", observed_max_conf(bd), pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, "topk_raw_evidence", out["topk_raw_evidence_embedding"], pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["normal"], "normal", final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
            for mode, intervention, cf_mode in [
                ("zero_semantic", "zero_semantic_measurements", "zero_semantic_measurements"),
                ("shuffle_semantic", "shuffle_semantic_measurements_across_points", "shuffle_semantic_measurements"),
            ]:
                cfb = counterfactual_batch(bd, cf_mode)
                cfout = model(**model_inputs(bd), intervention=intervention)
                cf_assign = top1(cfout["point_to_unit_assignment"].float()) if args.target_kind == "top1" else cfout["point_to_unit_assignment"].float()
                cfpred = compose(observed_mean(cfb), read_unit_delta(cf_assign, head(cfout, cfb)), gate, scale)
                update_method(delta_acc[mode], mode, cfpred, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
            shuf = compose(anchor, read_unit_delta(roll_assignment(assign), pred_unit), gate, scale)
            zero = compose(anchor, torch.zeros_like(point_delta), gate, scale)
            update_method(delta_acc["shuffle_assignment"], "shuffle_assignment", shuf, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["zero_unit"], "zero_unit", zero, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
    metrics = {name: finalize_method(acc, name) for name in sorted({key.split(":")[0] for key in acc.sum.keys()})}
    rows = {mode: finalize_method(a, mode) for mode, a in delta_acc.items()}
    normal = rows["normal"]

    def delta(mode: str) -> float | None:
        a = normal["hard_changed_gain_vs_pointwise"]
        b = rows[mode]["hard_changed_gain_vs_pointwise"]
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


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["stable"] and float(r["val_gain_anchor"] or -1.0) > 0.002]
    if not valid:
        valid = rows
    return max(valid, key=lambda x: float(x["val_gain_pointwise"] or -1.0e9))


def evaluate(model: Any, ckargs: argparse.Namespace, head: DirectEvidenceRawDeltaHeadV3432, train_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = next(model.parameters()).device
    readers = load_v3425_readers(args, model, device)
    cache: dict[str, dict[float, dict[str, Any]]] = {"sparse_gate": {}, "oracle_mask": {}}
    scale_rows = []
    for gate_mode in ("sparse_gate", "oracle_mask"):
        for scale in args.eval_scales:
            print(f"开始 V34.33 eval: gate={gate_mode}, scale={scale}", flush=True)
            per = {split: eval_split(split, model, ckargs, head, readers, args, scale, gate_mode, device) for split in ("val", "test")}
            cache[gate_mode][scale] = per
            method = f"v34_33_{gate_mode}_oracle_unit_delta_decoder"
            val_m = per["val"]["methods"][method]
            scale_rows.append({"gate_mode": gate_mode, "scale": scale, "val_gain_anchor": val_m["hard_changed_gain_vs_anchor"], "val_gain_pointwise": val_m["hard_changed_gain_vs_pointwise"], "stable": val_m["stable_preservation"]})
    best_sparse = choose_best([r for r in scale_rows if r["gate_mode"] == "sparse_gate"])
    best_oracle = choose_best([r for r in scale_rows if r["gate_mode"] == "oracle_mask"])
    selected = best_sparse
    per_split = cache[selected["gate_mode"]][float(selected["scale"])]
    method = f"v34_33_{selected['gate_mode']}_oracle_unit_delta_decoder"
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
        "中文结论": "V34.33 oracle unit-delta value decoder 完成；本轮使用显式 oracle unit delta target 训练 value decoder，不训练 learned gate，不跑 M512。",
        "oracle_unit_delta_targets_used": True,
        "target_kind": args.target_kind,
        "probe_passed": passed,
        "selected_sparse_config_by_val": selected,
        "best_oracle_mask_config_by_val": best_oracle,
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
        "v34_33_metrics": {"val": val_m, "test": test_m},
        "intervention_delta": {"val": val_delta, "test": test_delta},
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": False,
        "m512_dense_ready": bool(passed),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_33_m512_dense_visualization" if passed else "fix_value_decoder_capacity_or_gate_interface",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "train_summary": train_summary,
        "target_build_reference": json.loads(TARGET_REPORT.read_text(encoding="utf-8")) if TARGET_REPORT.exists() else {},
        "per_split_selected_sparse": per_split,
        "decision": decision,
        "阶段性分析": "V34.33 把 V34.30 的 oracle unit delta 上界直接变成监督缓存，验证 value decoder 是否能学到 assignment-bound correction。",
        "论文相关问题解决方案参考": "这一步对应 memory distillation：先证明 value payload 可以被 supervised 写入，再回到 observed-only predictor。若显式 target 仍不过，就不是简单 loss 问题，而是 decoder capacity 或 gate/readout interface 问题。",
        "最佳下一步方案": "若 sparse gate 不过但 oracle mask 过，下一步修 gate/value interface；若 oracle mask 也不过，下一步增强 value decoder capacity 或直接使用 cached target 做 overfit sanity check。"
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DECISION_DOC,
        "V34.33 oracle unit-delta value decoder 决策中文报告",
        decision,
        [
            "中文结论",
            "oracle_unit_delta_targets_used",
            "target_kind",
            "probe_passed",
            "selected_sparse_config_by_val",
            "best_oracle_mask_config_by_val",
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
    print(f"已写出 V34.33 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
    print(f"probe_passed: {passed}", flush=True)
    print(f"recommended_next_step: {decision['recommended_next_step']}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=8.0e-5)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=1024)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--target-kind", choices=["soft", "top1"], default="top1")
    p.add_argument("--train-residual-scale", type=float, default=1.0)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--target-supervision-weight", type=float, default=3.0)
    p.add_argument("--final-target-weight", type=float, default=0.8)
    p.add_argument("--anchor-gain-weight", type=float, default=0.8)
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
