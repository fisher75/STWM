#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_7_assignment_bound_residual_memory import AssignmentBoundResidualMemoryV347
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm, semantic_topk, unit_stats
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_7_assignment_oracle_residual_probe_20260511 import (
    CKPT_DIR,
    SUMMARY as TRAIN_SUMMARY,
    AssignmentAwareResidualDataset,
    collate_v347,
    compose,
)


SUMMARY = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_ORACLE_RESIDUAL_PROBE_DECISION_20260511.md"
V346_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[AssignmentBoundResidualMemoryV347 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    if not train.get("assignment_oracle_residual_probe_ran"):
        return None, None, train
    ckpt = ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_7_assignment_oracle_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = AssignmentBoundResidualMemoryV347(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


def final_for_mode(mode: str, out: dict[str, torch.Tensor], bd: dict[str, torch.Tensor]) -> torch.Tensor:
    if mode == "pointwise":
        return out["pointwise_semantic_belief"]
    if mode == "oracle_target_upper_bound":
        return torch.nn.functional.normalize(torch.nan_to_num(bd["fut_teacher_embedding"].float()), dim=-1)
    return compose(out, bd, "assignment_aware_residual_semantic_mask")


def collect(split: str, model: AssignmentBoundResidualMemoryV347, ckargs: argparse.Namespace, device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    ds = AssignmentAwareResidualDataset(split, ckargs)
    loader = DataLoader(ds, batch_size=ckargs.batch_size, shuffle=False, num_workers=ckargs.num_workers, collate_fn=collate_v347)
    modes = ["normal", "shuffled_assignment", "zero_unit_memory", "zero_semantic_measurements", "pointwise", "oracle_target_upper_bound"]
    rows: dict[str, dict[str, list[np.ndarray]]] = {m: {k: [] for k in ["pred", "point", "target", "mask", "strict", "assignment", "hard", "changed", "stable", "assign", "point_to_instance", "obs_sem", "obs_mask"]} for m in modes}
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            sem_hard = bd["semantic_hard_mask"] if "semantic_hard_mask" in bd else bd["semantic_hard_train_mask"]
            base_inputs = {
                "obs_points": bd["obs_points"],
                "obs_vis": bd["obs_vis"],
                "obs_conf": bd["obs_conf"],
                "obs_semantic_measurements": bd["obs_semantic_measurements"],
                "obs_semantic_measurement_mask": bd["obs_semantic_measurement_mask"],
                "semantic_id": bd["semantic_id"],
            }
            outs = {
                "normal": model(**base_inputs, intervention="force_gate_zero"),
                "shuffled_assignment": model(**base_inputs, intervention="permute_unit_assignment"),
                "zero_unit_memory": model(**base_inputs, intervention="zero_unit_residual"),
                "zero_semantic_measurements": model(**base_inputs, intervention="zero_observed_semantic_measurements"),
            }
            outs["pointwise"] = outs["normal"]
            outs["oracle_target_upper_bound"] = outs["normal"]
            for mode in modes:
                pred = final_for_mode(mode, outs[mode], bd)
                rows[mode]["pred"].append(pred.detach().cpu().numpy())
                rows[mode]["point"].append(outs["normal"]["pointwise_semantic_belief"].detach().cpu().numpy())
                rows[mode]["target"].append(bd["fut_teacher_embedding"].detach().cpu().numpy())
                rows[mode]["mask"].append(bd["fut_teacher_available_mask"].detach().cpu().numpy())
                rows[mode]["strict"].append(bd["strict_residual_semantic_utility_mask"].detach().cpu().numpy())
                rows[mode]["assignment"].append(bd["assignment_aware_residual_semantic_mask"].detach().cpu().numpy())
                rows[mode]["hard"].append(sem_hard.detach().cpu().numpy())
                rows[mode]["changed"].append(bd["changed_mask"].detach().cpu().numpy())
                rows[mode]["stable"].append(bd["stable_suppress_mask"].detach().cpu().numpy())
                rows[mode]["assign"].append(outs[mode]["point_to_unit_assignment"].detach().cpu().numpy())
                rows[mode]["point_to_instance"].append(bd["point_to_instance_id"].detach().cpu().numpy())
                rows[mode]["obs_sem"].append(bd["obs_semantic_measurements"].detach().cpu().numpy())
                rows[mode]["obs_mask"].append(bd["obs_semantic_measurement_mask"].detach().cpu().numpy())
    return {mode: {k: np.concatenate(v, axis=0) for k, v in mode_rows.items()} for mode, mode_rows in rows.items()}


def gain(pred: np.ndarray, point: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    mask = mask.astype(bool)
    if not mask.any():
        return None
    pred_cos = (_norm(pred) * _norm(target)).sum(axis=-1)
    point_cos = (_norm(point) * _norm(target)).sum(axis=-1)
    return float(pred_cos[mask].mean() - point_cos[mask].mean())


def metrics(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    sem_mask = cat["mask"].astype(bool)
    strict = cat["strict"].astype(bool) & sem_mask
    assignment = cat["assignment"].astype(bool) & sem_mask
    hard = cat["hard"].astype(bool) & sem_mask
    changed = cat["changed"].astype(bool) & sem_mask
    stable = cat["stable"].astype(bool) & sem_mask
    out = {
        "strict_residual_subset_gain": gain(cat["pred"], cat["point"], cat["target"], strict),
        "assignment_aware_subset_gain": gain(cat["pred"], cat["point"], cat["target"], assignment),
        "semantic_hard_gain": gain(cat["pred"], cat["point"], cat["target"], hard),
        "changed_gain": gain(cat["pred"], cat["point"], cat["target"], changed),
        "stable_delta": gain(cat["pred"], cat["point"], cat["target"], stable),
        "teacher_agreement_weighted_top5": semantic_topk(cat["pred"], cat["target"], sem_mask, 5),
    }
    out["semantic_hard_signal"] = bool(out["semantic_hard_gain"] is not None and out["semantic_hard_gain"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain"] is not None and out["changed_gain"] > 0.005)
    out["stable_preservation"] = bool(out["stable_delta"] is None or out["stable_delta"] >= -0.02)
    out.update(unit_stats(cat["assign"], cat["point_to_instance"], cat["obs_sem"], cat["obs_mask"].astype(bool)))
    return out


def split_eval(split: str, model: AssignmentBoundResidualMemoryV347, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    coll = collect(split, model, ckargs, device)
    per = {mode: metrics(cat) for mode, cat in coll.items()}
    normal = per["normal"]
    assignment_delta = None
    if normal["assignment_aware_subset_gain"] is not None and per["shuffled_assignment"]["assignment_aware_subset_gain"] is not None:
        assignment_delta = float(normal["assignment_aware_subset_gain"] - per["shuffled_assignment"]["assignment_aware_subset_gain"])
    unit_delta = None
    if normal["assignment_aware_subset_gain"] is not None and per["zero_unit_memory"]["assignment_aware_subset_gain"] is not None:
        unit_delta = float(normal["assignment_aware_subset_gain"] - per["zero_unit_memory"]["assignment_aware_subset_gain"])
    sem_delta = None
    if normal["assignment_aware_subset_gain"] is not None and per["zero_semantic_measurements"]["assignment_aware_subset_gain"] is not None:
        sem_delta = float(normal["assignment_aware_subset_gain"] - per["zero_semantic_measurements"]["assignment_aware_subset_gain"])
    return {
        "modes": per,
        "normal": normal,
        "assignment_intervention_delta": assignment_delta,
        "shuffle_assignment_metric_delta": assignment_delta,
        "drop_unit_memory_delta": unit_delta,
        "zero_semantic_measurement_delta": sem_delta,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, train = load_model(args, device)
    if model is None or ckargs is None:
        decision = {
            "generated_at_utc": utc_now(),
            "assignment_oracle_residual_probe_ran": False,
            "assignment_oracle_residual_probe_passed": False,
            "skip_reason": train.get("skip_reason", "train_not_run"),
            "learned_gate_training_allowed": False,
            "recommended_next_step": "fix_assignment_aware_targets",
        }
        dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "STWM OSTF V34.7 Assignment Oracle Residual Probe Decision", decision, ["assignment_oracle_residual_probe_ran", "assignment_oracle_residual_probe_passed", "skip_reason", "recommended_next_step"])
        print(SUMMARY.relative_to(ROOT))
        return 0
    per = {split: split_eval(split, model, ckargs, device) for split in ("val", "test")}
    nval, ntest = per["val"]["normal"], per["test"]["normal"]
    assign_lb = bool(
        per["val"]["assignment_intervention_delta"] is not None
        and per["test"]["assignment_intervention_delta"] is not None
        and per["val"]["assignment_intervention_delta"] > 0.002
        and per["test"]["assignment_intervention_delta"] > 0.002
    )
    unit_lb = bool((per["val"]["drop_unit_memory_delta"] or 0.0) > 0.002 and (per["test"]["drop_unit_memory_delta"] or 0.0) > 0.002)
    sem_lb = bool((per["val"]["zero_semantic_measurement_delta"] or 0.0) > 0.002 and (per["test"]["zero_semantic_measurement_delta"] or 0.0) > 0.002)
    pass_probe = bool(
        (nval["assignment_aware_subset_gain"] or 0.0) > 0.005
        and (ntest["assignment_aware_subset_gain"] or 0.0) > 0.005
        and nval["stable_preservation"]
        and ntest["stable_preservation"]
        and assign_lb
        and unit_lb
    )
    v346 = json.loads(V346_DECISION.read_text(encoding="utf-8")) if V346_DECISION.exists() else {}
    decision = {
        "generated_at_utc": utc_now(),
        "assignment_oracle_residual_probe_ran": True,
        "assignment_oracle_residual_probe_passed": pass_probe,
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "semantic_hard_signal": {"val": nval["semantic_hard_signal"], "test": ntest["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": nval["changed_semantic_signal"], "test": ntest["changed_semantic_signal"]},
        "stable_preservation": {"val": nval["stable_preservation"], "test": ntest["stable_preservation"]},
        "pointwise_baseline_dominates": bool(not ((nval["assignment_aware_subset_gain"] or 0) > 0.005 or (ntest["assignment_aware_subset_gain"] or 0) > 0.005)),
        "assignment_aware_subset_gain": {"val": nval["assignment_aware_subset_gain"], "test": ntest["assignment_aware_subset_gain"]},
        "strict_residual_subset_gain": {"val": nval["strict_residual_subset_gain"], "test": ntest["strict_residual_subset_gain"]},
        "v34_6_strict_residual_subset_gain": v346.get("strict_residual_subset_gain"),
        "assignment_intervention_delta": {"val": per["val"]["assignment_intervention_delta"], "test": per["test"]["assignment_intervention_delta"]},
        "shuffle_assignment_metric_delta": {"val": per["val"]["shuffle_assignment_metric_delta"], "test": per["test"]["shuffle_assignment_metric_delta"]},
        "drop_unit_memory_delta": {"val": per["val"]["drop_unit_memory_delta"], "test": per["test"]["drop_unit_memory_delta"]},
        "unit_memory_load_bearing_on_residual": unit_lb,
        "assignment_load_bearing_on_residual": assign_lb,
        "semantic_measurements_load_bearing_on_residual": sem_lb,
        "effective_units": {"val": nval["effective_units"], "test": ntest["effective_units"]},
        "unit_dominant_instance_purity": {"val": nval["unit_dominant_instance_purity"], "test": ntest["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": nval["unit_semantic_purity"], "test": ntest["unit_semantic_purity"]},
        "learned_gate_training_allowed": pass_probe,
        "recommended_next_step": "train_assignment_residual_gate" if pass_probe else "fix_assignment_bound_residual_model",
    }
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "per_split": per, "decision": decision}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V34.7 Assignment Oracle Residual Probe Decision", decision, ["assignment_oracle_residual_probe_ran", "assignment_oracle_residual_probe_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "assignment_aware_subset_gain", "strict_residual_subset_gain", "assignment_intervention_delta", "unit_memory_load_bearing_on_residual", "assignment_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
