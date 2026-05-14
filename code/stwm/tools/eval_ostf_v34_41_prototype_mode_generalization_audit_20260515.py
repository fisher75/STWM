#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

import stwm.tools.train_eval_ostf_v34_36_predictability_filtered_unit_delta_writer_20260514 as v36
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import masks, observed_mean
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, update_method
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import compose, load_frozen_residual_model, read_unit_delta
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import top1
from stwm.tools.train_eval_ostf_v34_40_prototype_conditioned_mixture_unit_delta_writer_20260515 import (
    TARGET_ROOT,
    build_codebook,
    load_proto_labels,
    load_trained_head,
)


REPORT = ROOT / "reports/stwm_ostf_v34_41_prototype_mode_generalization_audit_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V34_41_PROTOTYPE_MODE_GENERALIZATION_AUDIT_20260515.md"


def zero_inactive(unit_delta: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return unit_delta * active[..., None].float()


def eval_split(split: str, model: Any, ckargs: argparse.Namespace, head: Any, codebook: torch.Tensor, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    acc = {(method, scale): Acc() for method in ("predicted_prototype_codebook", "oracle_label_codebook", "learned_full_mixture") for scale in args.eval_scales}
    proto_correct = 0
    proto_total = 0
    pred_target_cos_sum = 0.0
    pred_target_cos_count = 0
    oracle_target_cos_sum = 0.0
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            target = v36.load_filtered_batch(split, bd["uid"], device)
            proto = load_proto_labels(split, bd["uid"], device)
            full = head.forward_full(out, bd)
            logits = full["prototype_logits"]
            pred_id = logits.argmax(dim=-1)
            active = target["oracle_unit_delta_active"].bool() & proto["prototype_active"].bool()
            if bool(active.any()):
                proto_correct += int((pred_id[active] == proto["prototype_id"][active]).sum().item())
                proto_total += int(active.sum().item())
                pred_delta = codebook.to(device)[pred_id][active]
                oracle_delta = codebook.to(device)[proto["prototype_id"].clamp_min(0)][active]
                target_delta = target["oracle_unit_delta"][active]
                pred_target_cos_sum += float((v36.norm(pred_delta) * v36.norm(target_delta)).sum(dim=-1).sum().item())
                oracle_target_cos_sum += float((v36.norm(oracle_delta) * v36.norm(target_delta)).sum(dim=-1).sum().item())
                pred_target_cos_count += int(active.sum().item())

            assign = top1(out["point_to_unit_assignment"].float())
            pred_proto_unit = zero_inactive(codebook.to(device)[pred_id], active)
            oracle_proto_unit = zero_inactive(codebook.to(device)[proto["prototype_id"].clamp_min(0)], active)
            learned_unit = full["unit_delta"]
            anchor = observed_mean(bd)
            fut = bd["fut_teacher_embedding"]
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            gate = target["point_predictable_mask"].float()
            for scale in args.eval_scales:
                for method, unit in [
                    ("predicted_prototype_codebook", pred_proto_unit),
                    ("oracle_label_codebook", oracle_proto_unit),
                    ("learned_full_mixture", learned_unit),
                ]:
                    final = compose(anchor, read_unit_delta(assign, unit), gate, float(scale))
                    update_method(acc[(method, float(scale))], method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
    rows = []
    for method in ("predicted_prototype_codebook", "oracle_label_codebook", "learned_full_mixture"):
        for scale in args.eval_scales:
            metrics = finalize_method(acc[(method, float(scale))], method)
            rows.append(
                {
                    "method": method,
                    "scale": float(scale),
                    "hard_changed_gain_vs_anchor": metrics["hard_changed_gain_vs_anchor"],
                    "hard_changed_gain_vs_pointwise": metrics["hard_changed_gain_vs_pointwise"],
                    "semantic_hard_signal": metrics["semantic_hard_signal"],
                    "changed_semantic_signal": metrics["changed_semantic_signal"],
                    "stable_preservation": metrics["stable_preservation"],
                }
            )
    best_by_method = {}
    for method in ("predicted_prototype_codebook", "oracle_label_codebook", "learned_full_mixture"):
        method_rows = [r for r in rows if r["method"] == method]
        best_by_method[method] = max(method_rows, key=lambda r: float(r["hard_changed_gain_vs_anchor"] or -1.0e9))
    return {
        "prototype_top1_accuracy": float(proto_correct / max(proto_total, 1)),
        "prototype_active_count": int(proto_total),
        "predicted_prototype_target_direction_cosine": float(pred_target_cos_sum / max(pred_target_cos_count, 1)),
        "oracle_label_prototype_target_direction_cosine": float(oracle_target_cos_sum / max(pred_target_cos_count, 1)),
        "scale_rows": rows,
        "best_by_method": best_by_method,
    }


def main() -> int:
    args = parse_args()
    v36.TARGET_ROOT = TARGET_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    codebook, codebook_report = build_codebook(args)
    model, ckargs, _ = load_frozen_residual_model(args, device)
    head, _summary = load_trained_head(model, codebook, codebook_report, args, device)
    split = {s: eval_split(s, model, ckargs, head, codebook, args, device) for s in ("train", "val", "test")}
    val_pred = split["val"]["best_by_method"]["predicted_prototype_codebook"]
    test_pred = split["test"]["best_by_method"]["predicted_prototype_codebook"]
    val_oracle = split["val"]["best_by_method"]["oracle_label_codebook"]
    test_oracle = split["test"]["best_by_method"]["oracle_label_codebook"]
    oracle_codebook_has_upper_bound = bool(
        (val_oracle["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
        and (test_oracle["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
    )
    predicted_mode_generalizes = bool(
        (val_pred["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
        and (test_pred["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
    )
    if oracle_codebook_has_upper_bound and not predicted_mode_generalizes:
        rec = "fix_observed_only_prototype_mode_selector"
    elif not oracle_codebook_has_upper_bound:
        rec = "fix_shared_prototype_codebook_targets"
    else:
        rec = "train_assignment_bound_prototype_mixture_system"
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.41 prototype mode generalization audit 完成；拆分 codebook 上界与 observed-only prototype mode 选择泛化，避免继续盲训 writer。",
        "codebook_report": codebook_report,
        "split": split,
        "oracle_codebook_has_upper_bound": oracle_codebook_has_upper_bound,
        "predicted_mode_generalizes": predicted_mode_generalizes,
        "recommended_next_step": rec,
        "阶段性分析": "如果 oracle_label_codebook 在 val/test 有正增益而 predicted_prototype_codebook 没有，问题是 prototype mode selector；如果 oracle_label_codebook 自身没上界，问题是 codebook/target。",
        "论文相关问题解决方案参考": "该审计对应 VQ/codebook 与 MoE routing 的分解：先确认共享专家/codebook 本身有上界，再确认 observed-only router 是否能泛化，而不是把两者混成一个失败信号。",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.41 prototype mode generalization audit 中文报告",
        payload,
        ["中文结论", "oracle_codebook_has_upper_bound", "predicted_mode_generalizes", "recommended_next_step", "阶段性分析", "论文相关问题解决方案参考"],
    )
    print(f"已写出 V34.41 prototype mode generalization audit 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"oracle_codebook_has_upper_bound: {oracle_codebook_has_upper_bound}", flush=True)
    print(f"predicted_mode_generalizes: {predicted_mode_generalizes}", flush=True)
    print(f"recommended_next_step: {rec}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--value-hidden-dim", type=int, default=256)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--prototype-count", type=int, default=64)
    p.add_argument("--prototype-temperature", type=float, default=0.7)
    p.add_argument("--max-residual-magnitude", type=float, default=0.35)
    p.add_argument("--target-kind", choices=["top1"], default="top1")
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(learnable_codebook=False)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
