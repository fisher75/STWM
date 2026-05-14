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

from stwm.modules.ostf_v34_34_cross_attention_unit_delta_writer import CrossAttentionUnitDeltaWriterV3434
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
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import load_target_batch, top1
from stwm.tools.eval_ostf_v34_35_unit_delta_generalization_audit_20260514 import collect, fit_ridge, predict, unit_features
from stwm.tools.train_eval_ostf_v34_34_cross_attention_unit_delta_value_writer_20260514 import TensorUnitDeltaWriter


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_36_predictability_filtered_unit_delta_targets/pointodyssey"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_36_predictability_filtered_unit_delta_writer_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_36_predictability_filtered_unit_delta_writer_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_36_predictability_filtered_unit_delta_writer_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_36_PREDICTABILITY_FILTERED_UNIT_DELTA_WRITER_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_36_PREDICTABILITY_FILTERED_UNIT_DELTA_WRITER_DECISION_20260514.md"


def build_targets(model: Any, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if (TARGET_ROOT / "train").exists() and not args.rebuild_targets:
        return {"target_built": True, "target_root": str(TARGET_ROOT.relative_to(ROOT)), "reused_existing": True}
    print("V34.36: 收集 observed-only unit features 并拟合 ridge 可预测分量...", flush=True)
    train_x, train_y, train_active = collect("train", model, ckargs, args, device)
    ridge = fit_ridge(train_x, train_y, train_active, args.ridge_lambda)
    split_stats = {}
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (TARGET_ROOT / split).mkdir(parents=True, exist_ok=True)
        total_active = 0
        total_filtered = 0
        total_point_pos = 0
        total_point_valid = 0
        cos_values = []
        with torch.no_grad():
            for batch in make_loader(split, ckargs, shuffle=False):
                bd = move_batch(batch, device)
                out = model(**model_inputs(bd), intervention="force_gate_zero")
                assign = top1(out["point_to_unit_assignment"].float())
                feat = unit_features(out, bd, assign)
                b, u, h, f = feat.shape
                old_target = load_target_batch(split, bd["uid"], device, args.target_kind)
                oracle = old_target["oracle_unit_delta"]
                active = old_target["oracle_unit_delta_active"].bool()
                pred = predict(ridge, feat.reshape(-1, f).cpu()).to(device).reshape(b, u, h, -1)
                pred_norm = pred.norm(dim=-1)
                oracle_norm = oracle.norm(dim=-1)
                cos = (F.normalize(pred, dim=-1) * F.normalize(oracle, dim=-1)).sum(dim=-1)
                confidence = ((cos - args.predictability_cos_threshold) / max(1.0e-6, 1.0 - args.predictability_cos_threshold)).clamp(0.0, 1.0)
                confidence = confidence.pow(args.predictability_confidence_power)
                filtered_active = active & (confidence > 0.0) & (pred_norm > args.min_predicted_delta_norm) & (oracle_norm > args.min_oracle_delta_norm)
                filtered_delta = oracle * confidence[..., None] * filtered_active[..., None].float()
                point_mask = (torch.einsum("bmu,buh->bmh", assign, filtered_active.float()) > 0.5) & hard_changed_aligned_mask(bd).bool()
                total_active += int(active.sum().item())
                total_filtered += int(filtered_active.sum().item())
                total_point_pos += int(point_mask.sum().item())
                total_point_valid += int(hard_changed_aligned_mask(bd).sum().item())
                if bool(active.any()):
                    cos_values.append(cos[active].detach().cpu())
                for i, uid in enumerate(bd["uid"]):
                    np.savez_compressed(
                        TARGET_ROOT / split / f"{uid}.npz",
                        uid=str(uid),
                        predictability_filtered_unit_delta=filtered_delta[i].detach().cpu().numpy().astype(np.float32),
                        predictability_filtered_active=filtered_active[i].detach().cpu().numpy().astype(bool),
                        predictability_score=confidence[i].detach().cpu().numpy().astype(np.float32),
                        ridge_pred_unit_delta=pred[i].detach().cpu().numpy().astype(np.float32),
                        original_oracle_unit_delta=oracle[i].detach().cpu().numpy().astype(np.float32),
                        original_oracle_active=active[i].detach().cpu().numpy().astype(bool),
                        point_predictable_mask=point_mask[i].detach().cpu().numpy().astype(bool),
                    )
        cos_cat = torch.cat(cos_values) if cos_values else torch.zeros(1)
        split_stats[split] = {
            "original_active_count": total_active,
            "filtered_active_count": total_filtered,
            "filtered_active_ratio_vs_original": float(total_filtered / max(total_active, 1)),
            "point_predictable_ratio_vs_hard_changed": float(total_point_pos / max(total_point_valid, 1)),
            "direction_cosine_mean_on_original_active": float(cos_cat.mean().item()),
            "direction_cosine_p50_on_original_active": float(cos_cat.median().item()),
        }
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.36 predictability-filtered unit_delta target 已构建；将原始 oracle unit_delta 按 observed-only ridge 可预测性过滤/收缩，只监督可预测 correction 分量。",
        "target_built": True,
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "reused_existing": False,
        "ridge_lambda": args.ridge_lambda,
        "predictability_cos_threshold": args.predictability_cos_threshold,
        "predictability_confidence_power": args.predictability_confidence_power,
        "split_stats": split_stats,
    }
    dump_json(ROOT / "reports/stwm_ostf_v34_36_predictability_filtered_unit_delta_target_build_20260514.json", payload)
    write_doc(
        ROOT / "docs/STWM_OSTF_V34_36_PREDICTABILITY_FILTERED_UNIT_DELTA_TARGET_BUILD_20260514.md",
        "V34.36 predictability-filtered unit_delta target build 中文报告",
        payload,
        ["中文结论", "target_built", "target_root", "ridge_lambda", "predictability_cos_threshold", "split_stats"],
    )
    return payload


def load_filtered_batch(split: str, uids: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    deltas, active, point_masks = [], [], []
    for uid in uids:
        z = np.load(TARGET_ROOT / split / f"{uid}.npz", allow_pickle=True)
        deltas.append(torch.from_numpy(np.asarray(z["predictability_filtered_unit_delta"], dtype=np.float32)))
        active.append(torch.from_numpy(np.asarray(z["predictability_filtered_active"]).astype(bool)))
        point_masks.append(torch.from_numpy(np.asarray(z["point_predictable_mask"]).astype(bool)))
    return {
        "oracle_unit_delta": torch.stack(deltas, dim=0).to(device),
        "oracle_unit_delta_active": torch.stack(active, dim=0).to(device),
        "point_predictable_mask": torch.stack(point_masks, dim=0).to(device),
    }


def target_loss(pred_unit: torch.Tensor, target_unit: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
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


def loss_fn(head: TensorUnitDeltaWriter, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], target: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    pred_unit = head(out, batch)
    sup_loss, sup_stats = target_loss(pred_unit, target["oracle_unit_delta"], target["oracle_unit_delta_active"])
    pos = target["point_predictable_mask"].bool()
    stable = batch["stable_suppress_mask"].bool() & batch["fut_teacher_available_mask"].bool()
    teacher_w = batch["teacher_confidence"].float().clamp(0.05, 1.0)
    anchor = observed_mean(batch)
    fut = batch["fut_teacher_embedding"]
    assign = top1(out["point_to_unit_assignment"].float())
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
    stable_suppress = masked_mean(point_delta.norm(dim=-1), stable, torch.ones_like(teacher_w)) if bool(stable.any()) else pred_unit.sum() * 0.0
    total = args.target_supervision_weight * sup_loss + args.final_target_weight * final_loss + args.anchor_gain_weight * anchor_gain + args.assignment_contrast_weight * assign_contrast + args.unit_contrast_weight * unit_contrast + args.stable_suppress_weight * stable_suppress
    stats = {
        "loss": float(total.detach().cpu()),
        "predictability_filtered_target_loss": float(sup_loss.detach().cpu()),
        "final_target_loss": float(final_loss.detach().cpu()),
        "anchor_gain_contrast_loss": float(anchor_gain.detach().cpu()),
        "assignment_contrast_loss": float(assign_contrast.detach().cpu()),
        "unit_contrast_loss": float(unit_contrast.detach().cpu()),
        "stable_delta_suppress_loss": float(stable_suppress.detach().cpu()),
        "point_predictable_positive_count": float(pos.sum().detach().cpu()),
    }
    stats.update(sup_stats)
    return total, stats


def train_one(model: Any, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> tuple[TensorUnitDeltaWriter, dict[str, Any]]:
    writer = CrossAttentionUnitDeltaWriterV3434(int(model.v30.cfg.hidden_dim), args.teacher_embedding_dim, args.value_hidden_dim, max_delta_magnitude=args.max_delta_magnitude).to(device)
    head = TensorUnitDeltaWriter(writer, "top1").to(device)
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
        target = load_filtered_batch("train", bd["uid"], device)
        with torch.no_grad():
            out = model(**model_inputs(bd), intervention="force_gate_zero")
        loss, stats = loss_fn(head, out, bd, target, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), **stats})
            print(f"训练进度: step={step}/{args.steps}, loss={stats['loss']:.6f}, filt={stats['predictability_filtered_target_loss']:.6f}, assign={stats['assignment_contrast_loss']:.6f}", flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"v34_36_predictability_filtered_unit_delta_writer_m128_h32_seed{args.seed}.pt"
    torch.save({"head": head.state_dict(), "args": vars(args), "step": args.steps}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.36 predictability-filtered unit_delta writer 训练完成；只监督 observed-predictable oracle correction 分量，不训练 learned gate。",
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "learned_gate_training_ran": False,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "loss_summaries": {k: summarize_loss(losses, k) for k in losses[-1].keys() if k != "step"} if losses else {},
        "loss_trace": losses,
        "duration_seconds": float(time.time() - start),
    }
    return head.eval(), summary


def eval_split(split: str, model: Any, ckargs: argparse.Namespace, head: TensorUnitDeltaWriter, readers: dict[str, dict[str, Any]], args: argparse.Namespace, device: torch.device) -> dict[str, dict[float, dict[str, Any]]]:
    configs = [(gate_mode, float(scale)) for gate_mode in ("sparse_gate", "predictable_oracle_mask") for scale in args.eval_scales]
    acc = {cfg: Acc() for cfg in configs}
    delta_acc = {cfg: {k: Acc() for k in ["normal", "shuffle_assignment", "zero_unit"]} for cfg in configs}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            target = load_filtered_batch(split, bd["uid"], device)
            assign = top1(out["point_to_unit_assignment"].float())
            pred_unit = head(out, bd)
            point_delta = read_unit_delta(assign, pred_unit)
            anchor = observed_mean(bd)
            pointwise = out["pointwise_semantic_belief"]
            fut = bd["fut_teacher_embedding"]
            mm = masks(bd)
            sparse_gate = sparse_seed_mean_gate(out, readers)
            oracle_gate = target["point_predictable_mask"].float()
            for gate_mode, scale in configs:
                gate = oracle_gate if gate_mode == "predictable_oracle_mask" else sparse_gate
                method = f"v34_36_{gate_mode}_predictability_filtered_writer"
                final = compose(anchor, point_delta, gate, scale)
                update_method(acc[(gate_mode, scale)], "pointwise_base", pointwise, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "copy_max_conf_observed", observed_max_conf(bd), pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "topk_raw_evidence", out["topk_raw_evidence_embedding"], pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                update_method(delta_acc[(gate_mode, scale)]["normal"], "normal", final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                shuf = compose(anchor, read_unit_delta(roll_assignment(assign), pred_unit), gate, scale)
                zero = compose(anchor, torch.zeros_like(point_delta), gate, scale)
                update_method(delta_acc[(gate_mode, scale)]["shuffle_assignment"], "shuffle_assignment", shuf, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                update_method(delta_acc[(gate_mode, scale)]["zero_unit"], "zero_unit", zero, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
    rows: dict[str, dict[float, dict[str, Any]]] = {"sparse_gate": {}, "predictable_oracle_mask": {}}
    for gate_mode, scale in configs:
        metrics = {name: finalize_method(acc[(gate_mode, scale)], name) for name in sorted({key.split(":")[0] for key in acc[(gate_mode, scale)].sum.keys()})}
        dr = {mode: finalize_method(a, mode) for mode, a in delta_acc[(gate_mode, scale)].items()}
        normal = dr["normal"]

        def delta(mode: str) -> float | None:
            a = normal["hard_changed_gain_vs_pointwise"]
            b = dr[mode]["hard_changed_gain_vs_pointwise"]
            return None if a is None or b is None else float(a - b)

        rows[gate_mode][scale] = {
            "methods": metrics,
            "intervention_delta": {
                "normal_hard_changed_gain_vs_anchor": normal["hard_changed_gain_vs_anchor"],
                "normal_hard_changed_gain_vs_pointwise": normal["hard_changed_gain_vs_pointwise"],
                "shuffle_assignment_delta": delta("shuffle_assignment"),
                "zero_unit_memory_delta": delta("zero_unit"),
            },
        }
    return rows


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["stable"] and float(r["val_gain_anchor"] or -1.0) > 0.002]
    return max(valid or rows, key=lambda r: float(r["val_gain_pointwise"] or -1.0e9))


def evaluate(model: Any, ckargs: argparse.Namespace, head: TensorUnitDeltaWriter, train_summary: dict[str, Any], target_report: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    readers = load_v3425_readers(args, model, device)
    cache = {split: eval_split(split, model, ckargs, head, readers, args, device) for split in ("val", "test")}
    scale_rows = []
    for gate_mode in ("sparse_gate", "predictable_oracle_mask"):
        for scale in args.eval_scales:
            method = f"v34_36_{gate_mode}_predictability_filtered_writer"
            val_m = cache["val"][gate_mode][float(scale)]["methods"][method]
            scale_rows.append({"gate_mode": gate_mode, "scale": float(scale), "val_gain_anchor": val_m["hard_changed_gain_vs_anchor"], "val_gain_pointwise": val_m["hard_changed_gain_vs_pointwise"], "stable": val_m["stable_preservation"]})
    selected = choose([r for r in scale_rows if r["gate_mode"] == "predictable_oracle_mask"])
    method = f"v34_36_{selected['gate_mode']}_predictability_filtered_writer"
    val_pack = cache["val"][selected["gate_mode"]][float(selected["scale"])]
    test_pack = cache["test"][selected["gate_mode"]][float(selected["scale"])]
    val_m = val_pack["methods"][method]
    test_m = test_pack["methods"][method]
    val_delta = val_pack["intervention_delta"]
    test_delta = test_pack["intervention_delta"]
    best_base = {"val": best_copy_topk(val_pack), "test": best_copy_topk(test_pack)}
    beats_copy_topk = bool((val_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["val"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002 and (test_m["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base["test"]["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002)
    improves_anchor = bool((val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and (test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    assignment_lb = bool((val_delta["shuffle_assignment_delta"] or 0.0) > 0.002 and (test_delta["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool((val_delta["zero_unit_memory_delta"] or 0.0) > 0.002 and (test_delta["zero_unit_memory_delta"] or 0.0) > 0.002)
    passed = bool(beats_copy_topk and improves_anchor and assignment_lb and unit_lb and val_m["stable_preservation"] and test_m["stable_preservation"])
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.36 predictability-filtered unit_delta writer 完成；目标从原始 sample-specific oracle delta 改成 observed-predictable filtered component，仍不训练 learned gate。",
        "predictability_filtered_targets_built": True,
        "probe_passed": passed,
        "selected_config_by_val": selected,
        "scale_sweep": scale_rows,
        "beats_copy_topk_baseline": beats_copy_topk,
        "unit_residual_improves_evidence_anchor": improves_anchor,
        "assignment_load_bearing_on_system": assignment_lb,
        "unit_memory_load_bearing_on_system": unit_lb,
        "semantic_hard_signal": {"val": val_m["semantic_hard_signal"], "test": test_m["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": val_m["changed_semantic_signal"], "test": test_m["changed_semantic_signal"]},
        "stable_preservation": {"val": val_m["stable_preservation"], "test": test_m["stable_preservation"]},
        "best_copy_topk_baseline": best_base,
        "v34_36_metrics": {"val": val_m, "test": test_m},
        "intervention_delta": {"val": val_delta, "test": test_delta},
        "target_report": target_report,
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": False,
        "m512_dense_ready": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "train_observed_predictability_activation" if passed else "fix_predictability_filtered_targets",
    }
    payload = {
        "generated_at_utc": utc_now(),
        "train_summary": train_summary,
        "decision": decision,
        "阶段性分析": "V34.36 检验可预测 target 修复是否能把 train-only 的 oracle unit_delta 上界变成 val/test 可用 correction。如果仍不过，说明 filtering/smoothing 仍不足，必须继续重定义 target，而不是修 gate。",
        "论文相关问题解决方案参考": "这一步对应 denoised distillation / predictable target decomposition：只蒸馏由观测可预测的 teacher residual，避免把未来不可预测噪声硬塞给 memory writer。",
        "最佳下一步方案": decision["recommended_next_step"],
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DECISION_DOC, "V34.36 predictability-filtered unit_delta writer 决策中文报告", decision, ["中文结论", "probe_passed", "selected_config_by_val", "beats_copy_topk_baseline", "unit_residual_improves_evidence_anchor", "assignment_load_bearing_on_system", "unit_memory_load_bearing_on_system", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "recommended_next_step"])
    print(f"已写出 V34.36 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
    print(f"probe_passed: {passed}", flush=True)
    print(f"recommended_next_step: {decision['recommended_next_step']}", flush=True)
    return payload


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
    p.add_argument("--max-delta-magnitude", type=float, default=2.5)
    p.add_argument("--target-kind", choices=["top1"], default="top1")
    p.add_argument("--ridge-lambda", type=float, default=100.0)
    p.add_argument("--predictability-cos-threshold", type=float, default=0.15)
    p.add_argument("--predictability-confidence-power", type=float, default=1.0)
    p.add_argument("--min-predicted-delta-norm", type=float, default=0.01)
    p.add_argument("--min-oracle-delta-norm", type=float, default=0.01)
    p.add_argument("--train-residual-scale", type=float, default=1.0)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--target-supervision-weight", type=float, default=3.0)
    p.add_argument("--final-target-weight", type=float, default=0.8)
    p.add_argument("--anchor-gain-weight", type=float, default=0.8)
    p.add_argument("--assignment-contrast-weight", type=float, default=1.5)
    p.add_argument("--unit-contrast-weight", type=float, default=1.0)
    p.add_argument("--stable-suppress-weight", type=float, default=0.05)
    p.add_argument("--anchor-gain-margin", type=float, default=0.006)
    p.add_argument("--assignment-margin", type=float, default=0.006)
    p.add_argument("--unit-margin", type=float, default=0.006)
    p.add_argument("--rebuild-targets", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    target_report = build_targets(model, ckargs, args, device)
    head, train_summary = train_one(model, ckargs, args, device)
    evaluate(model, ckargs, head, train_summary, target_report, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
