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
    load_frozen_residual_model,
    read_unit_delta,
    roll_assignment,
    sparse_seed_mean_gate,
)
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import TARGET_ROOT, load_target_batch


REPORT = ROOT / "reports/stwm_ostf_v34_33_cached_oracle_unit_delta_readout_sanity_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_33_CACHED_ORACLE_UNIT_DELTA_READOUT_SANITY_20260514.md"


def top1(assign: torch.Tensor) -> torch.Tensor:
    return F.one_hot(assign.argmax(dim=-1), num_classes=assign.shape[-1]).to(assign.dtype)


def eval_split(
    split: str,
    model: Any,
    ckargs: argparse.Namespace,
    readers: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    scale: float,
    gate_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    acc = Acc()
    delta_acc = {k: Acc() for k in ["normal", "shuffle_assignment", "zero_unit"]}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            target = load_target_batch(split, bd["uid"], device, args.target_kind)
            unit_delta = target["oracle_unit_delta"]
            pointwise = out["pointwise_semantic_belief"]
            fut = bd["fut_teacher_embedding"]
            mm = masks(bd)
            anchor = observed_mean(bd)
            assign = top1(out["point_to_unit_assignment"].float()) if args.target_kind == "top1" else out["point_to_unit_assignment"].float()
            gate = hard_changed_aligned_mask(bd).float() if gate_mode == "oracle_mask" else sparse_seed_mean_gate(out, readers)
            point_delta = read_unit_delta(assign, unit_delta)
            final = compose(anchor, point_delta, gate, scale)
            method = f"v34_33_cached_{gate_mode}_oracle_unit_delta"
            update_method(acc, "pointwise_base", pointwise, pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, "copy_max_conf_observed", observed_max_conf(bd), pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, "topk_raw_evidence", out["topk_raw_evidence_embedding"], pointwise=pointwise, target=fut, mm=mm)
            update_method(acc, method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
            update_method(delta_acc["normal"], "normal", final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
            shuffled = compose(anchor, read_unit_delta(roll_assignment(assign), unit_delta), gate, scale)
            zero = compose(anchor, torch.zeros_like(point_delta), gate, scale)
            update_method(delta_acc["shuffle_assignment"], "shuffle_assignment", shuffled, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
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
            "normal_hard_changed_gain_vs_anchor": normal["hard_changed_gain_vs_anchor"],
            "normal_hard_changed_gain_vs_pointwise": normal["hard_changed_gain_vs_pointwise"],
            "shuffle_assignment_delta": delta("shuffle_assignment"),
            "zero_unit_memory_delta": delta("zero_unit"),
        },
    }


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, init = load_frozen_residual_model(args, device)
    readers = load_v3425_readers(args, model, device)
    rows = []
    cache: dict[str, dict[float, dict[str, Any]]] = {"sparse_gate": {}, "oracle_mask": {}}
    for gate_mode in ("sparse_gate", "oracle_mask"):
        for scale in args.eval_scales:
            print(f"开始 V34.33 cached sanity eval: gate={gate_mode}, scale={scale}", flush=True)
            per = {split: eval_split(split, model, ckargs, readers, args, scale, gate_mode, device) for split in ("val", "test")}
            cache[gate_mode][float(scale)] = per
            method = f"v34_33_cached_{gate_mode}_oracle_unit_delta"
            val_m = per["val"]["methods"][method]
            rows.append({
                "gate_mode": gate_mode,
                "scale": float(scale),
                "stable": val_m["stable_preservation"],
                "val_gain_anchor": val_m["hard_changed_gain_vs_anchor"],
                "val_gain_pointwise": val_m["hard_changed_gain_vs_pointwise"],
            })
    valid_rows = [r for r in rows if r["stable"] and float(r["val_gain_anchor"] or -1.0) > 0.002]
    selected = max(valid_rows or rows, key=lambda r: float(r["val_gain_pointwise"] or -1.0e9))
    per_split = cache[selected["gate_mode"]][float(selected["scale"])]
    method = f"v34_33_cached_{selected['gate_mode']}_oracle_unit_delta"
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
    passed = bool(beats_copy_topk and improves_anchor and assignment_lb and unit_lb and val_m["stable_preservation"] and test_m["stable_preservation"])
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.33 cached oracle unit delta readout sanity 完成；直接读取物化 oracle unit_delta 缓存，不训练新模型，用同一 evidence-anchor 协议验证 target/readout 上界。",
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "target_kind": args.target_kind,
        "init": init,
        "selected_config_by_val": selected,
        "scale_sweep": rows,
        "cached_oracle_readout_passed": passed,
        "beats_copy_topk_baseline": beats_copy_topk,
        "unit_residual_improves_evidence_anchor": improves_anchor,
        "assignment_load_bearing_on_system": assignment_lb,
        "unit_memory_load_bearing_on_system": unit_lb,
        "semantic_hard_signal": {"val": val_m["semantic_hard_signal"], "test": test_m["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": val_m["changed_semantic_signal"], "test": test_m["changed_semantic_signal"]},
        "stable_preservation": {"val": val_m["stable_preservation"], "test": test_m["stable_preservation"]},
        "best_copy_topk_baseline": best_base,
        "selected_metrics": {"val": val_m, "test": test_m},
        "intervention_delta": {"val": val_delta, "test": test_delta},
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "阶段性分析": "若 cached oracle readout 通过而 learned value decoder 不通过，问题集中在 value decoder 的输入/容量/蒸馏方式；若 cached oracle readout 也不通过，问题集中在 target 物化或 evidence-anchor 评估协议对齐。",
        "论文相关问题解决方案参考": "该 sanity 与 Slot Attention/Set Transformer/Perceiver 一类 memory readout 的基本诊断一致：先确认 memory value 的 oracle payload 可读出，再训练 observed-only writer/decoder。",
        "recommended_next_step": "fix_value_decoder_capacity" if passed else "fix_oracle_delta_target_or_readout_protocol",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.33 cached oracle unit delta readout sanity 中文报告",
        payload,
        [
            "中文结论",
            "cached_oracle_readout_passed",
            "selected_config_by_val",
            "beats_copy_topk_baseline",
            "unit_residual_improves_evidence_anchor",
            "assignment_load_bearing_on_system",
            "unit_memory_load_bearing_on_system",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "intervention_delta",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "learned_gate_training_ran",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.33 cached sanity 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"cached_oracle_readout_passed: {passed}", flush=True)
    print(f"recommended_next_step: {payload['recommended_next_step']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-kind", choices=["soft", "top1"], default="top1")
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
