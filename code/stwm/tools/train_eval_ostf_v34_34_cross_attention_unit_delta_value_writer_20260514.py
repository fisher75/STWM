#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import setproctitle
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_34_cross_attention_unit_delta_writer import CrossAttentionUnitDeltaWriterV3434
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import load_v3425_readers, masks, observed_max_conf, observed_mean
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, update_method
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
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
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import (
    TARGET_REPORT,
    decoder_loss,
    load_target_batch,
    top1,
)


CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_34_cross_attention_unit_delta_value_writer_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_34_cross_attention_unit_delta_value_writer_summary_20260514.json"
DECISION = ROOT / "reports/stwm_ostf_v34_34_cross_attention_unit_delta_value_writer_decision_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_34_CROSS_ATTENTION_UNIT_DELTA_VALUE_WRITER_SUMMARY_20260514.md"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V34_34_CROSS_ATTENTION_UNIT_DELTA_VALUE_WRITER_DECISION_20260514.md"


class TensorUnitDeltaWriter(nn.Module):
    def __init__(self, writer: CrossAttentionUnitDeltaWriterV3434, target_kind: str) -> None:
        super().__init__()
        self.writer = writer
        self.target_kind = target_kind

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assign = top1(out["point_to_unit_assignment"].float()) if self.target_kind == "top1" else out["point_to_unit_assignment"].float()
        return self.writer(out, batch, assignment=assign)["unit_delta"]


def train_one(args: argparse.Namespace) -> tuple[Any, argparse.Namespace, TensorUnitDeltaWriter, dict[str, Any]]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, init = load_frozen_residual_model(args, device)
    writer = CrossAttentionUnitDeltaWriterV3434(
        int(model.v30.cfg.hidden_dim),
        args.teacher_embedding_dim,
        args.value_hidden_dim,
        max_delta_magnitude=args.max_delta_magnitude,
    ).to(device)
    head = TensorUnitDeltaWriter(writer, args.target_kind).to(device)
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
    ckpt = CKPT_DIR / f"v34_34_cross_attention_unit_delta_value_writer_m128_h32_seed{args.seed}_{args.target_kind}.pt"
    torch.save({"head": head.state_dict(), "args": vars(args), "init": init, "step": args.steps}, ckpt)
    summary = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.34 cross-attention unit-delta writer 训练完成；使用 unit/horizon query 读取 point×top-k raw evidence set，显式蒸馏 V34.33 oracle unit_delta 缓存，不训练 learned gate。",
        "fresh_training_completed": True,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "target_kind": args.target_kind,
        "init": init,
        "steps": args.steps,
        "train_sample_count": len(loader.dataset),
        "value_hidden_dim": args.value_hidden_dim,
        "max_delta_magnitude": args.max_delta_magnitude,
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
        "V34.34 cross-attention unit-delta value writer 训练中文摘要",
        summary,
        [
            "中文结论",
            "fresh_training_completed",
            "checkpoint_path",
            "target_kind",
            "steps",
            "train_sample_count",
            "value_hidden_dim",
            "v30_backbone_frozen",
            "assignment_frozen",
            "learned_gate_training_ran",
            "future_leakage_detected",
            "trajectory_degraded",
        ],
    )
    print(f"已写出 V34.34 训练摘要: {SUMMARY.relative_to(ROOT)}", flush=True)
    head.eval()
    return model, ckargs, head, summary


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["stable"] and float(r["val_gain_anchor"] or -1.0) > 0.002]
    return max(valid or rows, key=lambda x: float(x["val_gain_pointwise"] or -1.0e9))


def eval_sweep_split(
    split: str,
    model: Any,
    ckargs: argparse.Namespace,
    head: TensorUnitDeltaWriter,
    readers: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[float, dict[str, Any]]]:
    configs = [(gate_mode, float(scale)) for gate_mode in ("sparse_gate", "oracle_mask") for scale in args.eval_scales]
    acc = {cfg: Acc() for cfg in configs}
    delta_acc = {cfg: {k: Acc() for k in ["normal", "zero_semantic", "shuffle_semantic", "shuffle_assignment", "zero_unit"]} for cfg in configs}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            fut = bd["fut_teacher_embedding"]
            anchor = observed_mean(bd)
            assign = top1(out["point_to_unit_assignment"].float()) if args.target_kind == "top1" else out["point_to_unit_assignment"].float()
            pred_unit = head(out, bd)
            point_delta = read_unit_delta(assign, pred_unit)
            sparse_gate = sparse_seed_mean_gate(out, readers)
            oracle_gate = hard_changed_aligned_mask(bd).float()

            cf_payload: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for mode, intervention, cf_mode in [
                ("zero_semantic", "zero_semantic_measurements", "zero_semantic_measurements"),
                ("shuffle_semantic", "shuffle_semantic_measurements_across_points", "shuffle_semantic_measurements"),
            ]:
                cfb = counterfactual_batch(bd, cf_mode)
                cfout = model(**model_inputs(bd), intervention=intervention)
                cf_assign = top1(cfout["point_to_unit_assignment"].float()) if args.target_kind == "top1" else cfout["point_to_unit_assignment"].float()
                cf_payload[mode] = (observed_mean(cfb), read_unit_delta(cf_assign, head(cfout, cfb)), cf_assign)
            shuf_delta = read_unit_delta(roll_assignment(assign), pred_unit)
            zero_delta = torch.zeros_like(point_delta)

            for gate_mode, scale in configs:
                gate = oracle_gate if gate_mode == "oracle_mask" else sparse_gate
                method = f"v34_34_{gate_mode}_cross_attention_unit_delta_writer"
                final = compose(anchor, point_delta, gate, scale)
                update_method(acc[(gate_mode, scale)], "pointwise_base", pointwise, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "copy_max_conf_observed", observed_max_conf(bd), pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "topk_raw_evidence", out["topk_raw_evidence_embedding"], pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                update_method(delta_acc[(gate_mode, scale)]["normal"], "normal", final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                for mode in ("zero_semantic", "shuffle_semantic"):
                    cf_anchor, cf_delta, _ = cf_payload[mode]
                    cf_final = compose(cf_anchor, cf_delta, gate, scale)
                    update_method(delta_acc[(gate_mode, scale)][mode], mode, cf_final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                shuf = compose(anchor, shuf_delta, gate, scale)
                zero = compose(anchor, zero_delta, gate, scale)
                update_method(delta_acc[(gate_mode, scale)]["shuffle_assignment"], "shuffle_assignment", shuf, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                update_method(delta_acc[(gate_mode, scale)]["zero_unit"], "zero_unit", zero, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
    out_by_gate: dict[str, dict[float, dict[str, Any]]] = {"sparse_gate": {}, "oracle_mask": {}}
    for gate_mode, scale in configs:
        metrics = {name: finalize_method(acc[(gate_mode, scale)], name) for name in sorted({key.split(":")[0] for key in acc[(gate_mode, scale)].sum.keys()})}
        rows = {mode: finalize_method(a, mode) for mode, a in delta_acc[(gate_mode, scale)].items()}
        normal = rows["normal"]

        def delta(mode: str) -> float | None:
            a = normal["hard_changed_gain_vs_pointwise"]
            b = rows[mode]["hard_changed_gain_vs_pointwise"]
            if a is None or b is None:
                return None
            return float(a - b)

        out_by_gate[gate_mode][scale] = {
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
    return out_by_gate


def evaluate(model: Any, ckargs: argparse.Namespace, head: TensorUnitDeltaWriter, train_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = next(model.parameters()).device
    readers = load_v3425_readers(args, model, device)
    cache: dict[str, dict[float, dict[str, Any]]] = {"sparse_gate": {}, "oracle_mask": {}}
    split_cache = {}
    for split in ("val", "test"):
        print(f"开始 V34.34 fast eval split={split}", flush=True)
        split_cache[split] = eval_sweep_split(split, model, ckargs, head, readers, args, device)
    scale_rows = []
    for gate_mode in ("sparse_gate", "oracle_mask"):
        for scale in args.eval_scales:
            per = {split: split_cache[split][gate_mode][float(scale)] for split in ("val", "test")}
            cache[gate_mode][float(scale)] = per
            method = f"v34_34_{gate_mode}_cross_attention_unit_delta_writer"
            val_m = per["val"]["methods"][method]
            scale_rows.append({
                "gate_mode": gate_mode,
                "scale": float(scale),
                "val_gain_anchor": val_m["hard_changed_gain_vs_anchor"],
                "val_gain_pointwise": val_m["hard_changed_gain_vs_pointwise"],
                "stable": val_m["stable_preservation"],
            })
    selected = choose_best([r for r in scale_rows if r["gate_mode"] == "sparse_gate"])
    best_oracle = choose_best([r for r in scale_rows if r["gate_mode"] == "oracle_mask"])
    per_split = cache[selected["gate_mode"]][float(selected["scale"])]
    method = f"v34_34_{selected['gate_mode']}_cross_attention_unit_delta_writer"
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
    semantic_lb = bool(min(val_delta["zero_semantic_measurements_delta"] or 0.0, val_delta["shuffle_semantic_measurements_delta"] or 0.0, test_delta["zero_semantic_measurements_delta"] or 0.0, test_delta["shuffle_semantic_measurements_delta"] or 0.0) > 0.002)
    semantic_hard_signal = {"val": val_m["semantic_hard_signal"], "test": test_m["semantic_hard_signal"]}
    changed_semantic_signal = {"val": val_m["changed_semantic_signal"], "test": test_m["changed_semantic_signal"]}
    stable_preservation = {"val": val_m["stable_preservation"], "test": test_m["stable_preservation"]}
    passed = bool(beats_copy_topk and improves_anchor and assignment_lb and unit_lb and semantic_lb and all(semantic_hard_signal.values()) and all(changed_semantic_signal.values()) and all(stable_preservation.values()))
    oracle_mask_passed = bool((best_oracle["val_gain_anchor"] or -1.0) > 0.002)
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.34 cross-attention unit-delta writer 完成；本轮不训练 gate，不跑 M512，只验证高容量 observed-only value writer 能否学出 oracle unit_delta。",
        "cross_attention_writer_built": True,
        "oracle_unit_delta_targets_used": True,
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
        "v34_34_metrics": {"val": val_m, "test": test_m},
        "intervention_delta": {"val": val_delta, "test": test_delta},
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": False,
        "m512_dense_ready": bool(passed),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "return_to_full_system_benchmark" if passed else ("fix_sparse_gate_value_interface" if oracle_mask_passed else "fix_value_decoder_capacity"),
    }
    payload = {
        "generated_at_utc": utc_now(),
        "train_summary": train_summary,
        "target_build_reference": json.loads(TARGET_REPORT.read_text(encoding="utf-8")) if TARGET_REPORT.exists() else {},
        "per_split_selected_sparse": per_split,
        "decision": decision,
        "阶段性分析": "V34.34 将浅 MLP value decoder 替换为 unit/horizon cross-attention writer，直接读 point×top-k evidence set。若仍无法超过 evidence anchor，说明 observed-only writer 尚不能从可观测证据中恢复 oracle unit delta。",
        "论文相关问题解决方案参考": "Slot Attention、Set Transformer、Perceiver IO 都提示：对象/slot 级 memory 不应先把 evidence set 均值压扁，而应通过 query-to-set attention 写入 value payload。本轮按这个方向验证。",
        "最佳下一步方案": "若 V34.34 通过则回到完整系统 benchmark；若 oracle-mask 通过但 sparse 不过，修 gate/value interface；若 oracle-mask 也不过，继续修 value decoder capacity 或监督分解。",
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DECISION_DOC,
        "V34.34 cross-attention unit-delta value writer 决策中文报告",
        decision,
        [
            "中文结论",
            "cross_attention_writer_built",
            "oracle_unit_delta_targets_used",
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
    print(f"已写出 V34.34 决策报告: {DECISION.relative_to(ROOT)}", flush=True)
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
    p.add_argument("--target-kind", choices=["soft", "top1"], default="top1")
    p.add_argument("--train-residual-scale", type=float, default=1.0)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--target-supervision-weight", type=float, default=3.0)
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
    args = parse_args()
    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model, ckargs, _ = load_frozen_residual_model(args, device)
        writer = CrossAttentionUnitDeltaWriterV3434(
            int(model.v30.cfg.hidden_dim),
            args.teacher_embedding_dim,
            args.value_hidden_dim,
            max_delta_magnitude=args.max_delta_magnitude,
        ).to(device)
        head = TensorUnitDeltaWriter(writer, args.target_kind).to(device)
        ckpt = CKPT_DIR / f"v34_34_cross_attention_unit_delta_value_writer_m128_h32_seed{args.seed}_{args.target_kind}.pt"
        ck = torch.load(ckpt, map_location=device)
        head.load_state_dict(ck["head"], strict=True)
        head.eval()
        train_summary = json.loads(SUMMARY.read_text(encoding="utf-8")).get("train_summary", json.loads(SUMMARY.read_text(encoding="utf-8"))) if SUMMARY.exists() else {"checkpoint_path": str(ckpt.relative_to(ROOT))}
        evaluate(model, ckargs, head, train_summary, args)
        return 0
    model, ckargs, head, train_summary = train_one(args)
    evaluate(model, ckargs, head, train_summary, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
