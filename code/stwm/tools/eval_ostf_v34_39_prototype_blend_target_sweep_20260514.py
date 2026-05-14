#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import load_v3425_readers, masks, observed_mean
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, update_method
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import compose, load_frozen_residual_model, read_unit_delta, roll_assignment, sparse_seed_mean_gate
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import top1


BASE_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_targets/pointodyssey"
PROTO_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_38_cluster_regularized_unit_delta_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_39_prototype_blend_target_sweep_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_39_PROTOTYPE_BLEND_TARGET_SWEEP_20260514.md"


def load_blend(split: str, uids: list[str], alpha: float, device: torch.device) -> dict[str, torch.Tensor]:
    deltas, masks_point = [], []
    for uid in uids:
        base = np.load(BASE_ROOT / split / f"{uid}.npz", allow_pickle=True)
        proto = np.load(PROTO_ROOT / split / f"{uid}.npz", allow_pickle=True)
        base_delta = torch.from_numpy(np.asarray(base["predictability_filtered_unit_delta"], dtype=np.float32))
        proto_delta = torch.from_numpy(np.asarray(proto["predictability_filtered_unit_delta"], dtype=np.float32))
        point_mask = torch.from_numpy(np.asarray(base["point_predictable_mask"]).astype(bool))
        deltas.append(float(alpha) * base_delta + (1.0 - float(alpha)) * proto_delta)
        masks_point.append(point_mask)
    return {"unit_delta": torch.stack(deltas, dim=0).to(device), "point_mask": torch.stack(masks_point, dim=0).to(device)}


def eval_split(split: str, model: Any, ckargs: argparse.Namespace, readers: dict[str, dict[str, Any]], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    configs = [(float(alpha), float(scale)) for alpha in args.alphas for scale in args.scales]
    acc = {cfg: Acc() for cfg in configs}
    delta_acc = {cfg: {k: Acc() for k in ["normal", "shuffle_assignment", "zero_unit"]} for cfg in configs}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            assign = top1(out["point_to_unit_assignment"].float())
            anchor = observed_mean(bd)
            fut = bd["fut_teacher_embedding"]
            pointwise = out["pointwise_semantic_belief"]
            mm = masks(bd)
            sparse = sparse_seed_mean_gate(out, readers)
            for alpha in args.alphas:
                target = load_blend(split, bd["uid"], float(alpha), device)
                point_delta = read_unit_delta(assign, target["unit_delta"])
                oracle_gate = target["point_mask"].float()
                for scale in args.scales:
                    cfg = (float(alpha), float(scale))
                    gate = oracle_gate if args.gate_mode == "oracle_mask" else sparse
                    method = f"v34_39_blend_alpha_{float(alpha):.2f}_scale_{float(scale):.2f}"
                    final = compose(anchor, point_delta, gate, float(scale))
                    update_method(acc[cfg], "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
                    update_method(acc[cfg], method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                    update_method(delta_acc[cfg]["normal"], "normal", final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                    shuf = compose(anchor, read_unit_delta(roll_assignment(assign), target["unit_delta"]), gate, float(scale))
                    zero = compose(anchor, torch.zeros_like(point_delta), gate, float(scale))
                    update_method(delta_acc[cfg]["shuffle_assignment"], "shuffle_assignment", shuf, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                    update_method(delta_acc[cfg]["zero_unit"], "zero_unit", zero, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
    rows = []
    by_key = {}
    for alpha, scale in configs:
        method = f"v34_39_blend_alpha_{alpha:.2f}_scale_{scale:.2f}"
        metrics = finalize_method(acc[(alpha, scale)], method)
        normal = finalize_method(delta_acc[(alpha, scale)]["normal"], "normal")
        shuf = finalize_method(delta_acc[(alpha, scale)]["shuffle_assignment"], "shuffle_assignment")
        zero = finalize_method(delta_acc[(alpha, scale)]["zero_unit"], "zero_unit")
        shuf_delta = float(normal["hard_changed_gain_vs_pointwise"] - shuf["hard_changed_gain_vs_pointwise"])
        zero_delta = float(normal["hard_changed_gain_vs_pointwise"] - zero["hard_changed_gain_vs_pointwise"])
        row = {
            "alpha": alpha,
            "scale": scale,
            "hard_changed_gain_vs_anchor": metrics["hard_changed_gain_vs_anchor"],
            "hard_changed_gain_vs_pointwise": metrics["hard_changed_gain_vs_pointwise"],
            "semantic_hard_signal": metrics["semantic_hard_signal"],
            "changed_semantic_signal": metrics["changed_semantic_signal"],
            "stable_preservation": metrics["stable_preservation"],
            "shuffle_assignment_delta": shuf_delta,
            "zero_unit_memory_delta": zero_delta,
        }
        rows.append(row)
        by_key[f"alpha_{alpha:.2f}_scale_{scale:.2f}"] = row
    return {"rows": rows, "by_key": by_key}


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, _ = load_frozen_residual_model(args, device)
    readers = load_v3425_readers(args, model, device)
    splits = {split: eval_split(split, model, ckargs, readers, args, device) for split in ("val", "test")}
    candidates = []
    for val_row in splits["val"]["rows"]:
        key = f"alpha_{val_row['alpha']:.2f}_scale_{val_row['scale']:.2f}"
        test_row = splits["test"]["by_key"][key]
        passed = bool(
            (val_row["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
            and (test_row["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
            and val_row["shuffle_assignment_delta"] > 0.002
            and test_row["shuffle_assignment_delta"] > 0.002
            and val_row["stable_preservation"]
            and test_row["stable_preservation"]
        )
        candidates.append({"key": key, "passed": passed, "val": val_row, "test": test_row})
    viable = [c for c in candidates if c["passed"]]
    selected = max(viable or candidates, key=lambda c: float(c["val"]["hard_changed_gain_vs_anchor"] or -1.0e9))
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.39 prototype blend target sweep 完成；不训练 writer，只评估 crossfit filtered delta 与 prototype-smoothed delta 的 convex blend 是否保留 cached upper bound。",
        "gate_mode": args.gate_mode,
        "alphas": args.alphas,
        "scales": args.scales,
        "selected": selected,
        "blend_cached_target_passed": bool(selected["passed"]),
        "all_candidates": candidates,
        "recommended_next_step": "train_blended_prototype_writer" if selected["passed"] else "fix_prototype_blend_target",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.39 prototype blend target sweep 中文报告", payload, ["中文结论", "gate_mode", "selected", "blend_cached_target_passed", "recommended_next_step"])
    print(f"已写出 V34.39 blend target sweep 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"blend_cached_target_passed: {payload['blend_cached_target_passed']}", flush=True)
    print(f"recommended_next_step: {payload['recommended_next_step']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 0.9, 1.0])
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--gate-mode", choices=["oracle_mask", "sparse_gate"], default="oracle_mask")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
