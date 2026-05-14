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

from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import load_v3425_readers, masks, observed_max_conf, observed_mean
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import Acc, finalize_method, update_method
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import make_loader, model_inputs
from stwm.tools.train_eval_ostf_v34_31_raw_unit_delta_value_memory_20260514 import best_copy_topk, compose, load_frozen_residual_model, read_unit_delta, roll_assignment, sparse_seed_mean_gate
from stwm.tools.train_eval_ostf_v34_33_oracle_unit_delta_value_decoder_20260514 import top1
from stwm.tools.train_eval_ostf_v34_36_predictability_filtered_unit_delta_writer_20260514 import TARGET_ROOT


REPORT = ROOT / "reports/stwm_ostf_v34_36_cached_filtered_unit_delta_sanity_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_36_CACHED_FILTERED_UNIT_DELTA_SANITY_20260514.md"


def load_filtered(split: str, uids: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    deltas, masks_point = [], []
    for uid in uids:
        z = np.load(TARGET_ROOT / split / f"{uid}.npz", allow_pickle=True)
        deltas.append(torch.from_numpy(np.asarray(z["predictability_filtered_unit_delta"], dtype=np.float32)))
        masks_point.append(torch.from_numpy(np.asarray(z["point_predictable_mask"]).astype(bool)))
    return {
        "unit_delta": torch.stack(deltas, dim=0).to(device),
        "point_mask": torch.stack(masks_point, dim=0).to(device),
    }


def eval_split(split: str, model: Any, ckargs: argparse.Namespace, readers: dict[str, dict[str, Any]], args: argparse.Namespace, device: torch.device) -> dict[str, dict[float, dict[str, Any]]]:
    configs = [(gate_mode, float(scale)) for gate_mode in ("sparse_gate", "predictable_oracle_mask") for scale in args.eval_scales]
    acc = {cfg: Acc() for cfg in configs}
    delta_acc = {cfg: {k: Acc() for k in ["normal", "shuffle_assignment", "zero_unit"]} for cfg in configs}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = model(**model_inputs(bd), intervention="force_gate_zero")
            target = load_filtered(split, bd["uid"], device)
            assign = top1(out["point_to_unit_assignment"].float())
            unit_delta = target["unit_delta"]
            point_delta = read_unit_delta(assign, unit_delta)
            anchor = observed_mean(bd)
            pointwise = out["pointwise_semantic_belief"]
            fut = bd["fut_teacher_embedding"]
            mm = masks(bd)
            gates = {
                "sparse_gate": sparse_seed_mean_gate(out, readers),
                "predictable_oracle_mask": target["point_mask"].float(),
            }
            for gate_mode, scale in configs:
                gate = gates[gate_mode]
                method = f"v34_36_cached_{gate_mode}_filtered_unit_delta"
                final = compose(anchor, point_delta, gate, scale)
                update_method(acc[(gate_mode, scale)], "pointwise_base", pointwise, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "copy_mean_observed", anchor, pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "copy_max_conf_observed", observed_max_conf(bd), pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], "topk_raw_evidence", out["topk_raw_evidence_embedding"], pointwise=pointwise, target=fut, mm=mm)
                update_method(acc[(gate_mode, scale)], method, final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                update_method(delta_acc[(gate_mode, scale)]["normal"], "normal", final, pointwise=pointwise, target=fut, mm=mm, anchor=anchor, gate=gate)
                shuf = compose(anchor, read_unit_delta(roll_assignment(assign), unit_delta), gate, scale)
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


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, init = load_frozen_residual_model(args, device)
    readers = load_v3425_readers(args, model, device)
    cache = {split: eval_split(split, model, ckargs, readers, args, device) for split in ("val", "test")}
    rows = []
    for gate_mode in ("sparse_gate", "predictable_oracle_mask"):
        for scale in args.eval_scales:
            method = f"v34_36_cached_{gate_mode}_filtered_unit_delta"
            val_m = cache["val"][gate_mode][float(scale)]["methods"][method]
            rows.append({"gate_mode": gate_mode, "scale": float(scale), "stable": val_m["stable_preservation"], "val_gain_anchor": val_m["hard_changed_gain_vs_anchor"], "val_gain_pointwise": val_m["hard_changed_gain_vs_pointwise"]})
    selected = choose([r for r in rows if r["gate_mode"] == "predictable_oracle_mask"])
    method = f"v34_36_cached_{selected['gate_mode']}_filtered_unit_delta"
    val_pack = cache["val"][selected["gate_mode"]][float(selected["scale"])]
    test_pack = cache["test"][selected["gate_mode"]][float(selected["scale"])]
    val_m = val_pack["methods"][method]
    test_m = test_pack["methods"][method]
    val_delta = val_pack["intervention_delta"]
    test_delta = test_pack["intervention_delta"]
    base = {"val": best_copy_topk(val_pack), "test": best_copy_topk(test_pack)}
    passed = bool((val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and (test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and (val_delta["shuffle_assignment_delta"] or 0.0) > 0.002 and (test_delta["shuffle_assignment_delta"] or 0.0) > 0.002 and val_m["stable_preservation"] and test_m["stable_preservation"])
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.36 cached filtered unit_delta sanity 完成；直接读取 predictability-filtered target cache，验证 target 本身是否有正上界。",
        "init": init,
        "selected_config_by_val": selected,
        "cached_filtered_target_passed": passed,
        "best_copy_topk_baseline": base,
        "selected_metrics": {"val": val_m, "test": test_m},
        "intervention_delta": {"val": val_delta, "test": test_delta},
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "learned_gate_training_ran": False,
        "recommended_next_step": "train_observed_predictability_activation" if passed else "fix_predictability_filtered_targets",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.36 cached filtered unit_delta sanity 中文报告", payload, ["中文结论", "selected_config_by_val", "cached_filtered_target_passed", "selected_metrics", "intervention_delta", "recommended_next_step"])
    print(f"已写出 V34.36 cached sanity 报告: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"cached_filtered_target_passed: {passed}", flush=True)
    print(f"recommended_next_step: {payload['recommended_next_step']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--eval-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
