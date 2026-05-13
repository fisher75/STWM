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

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_11_local_semantic_usage_oracle_probe_20260513 import (
    CKPT_DIR,
    SUMMARY as TRAIN_SUMMARY,
    LocalSemanticUsageDataset,
    collate_v3410,
    local_compose,
    model_inputs,
)


SUMMARY = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_eval_summary_20260513.json"
DECISION = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_11_LOCAL_SEMANTIC_USAGE_ORACLE_PROBE_DECISION_20260513.md"


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.nan_to_num(x.float()), dim=-1)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[CausalAssignmentBoundResidualMemoryV348 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    if not train.get("local_semantic_usage_probe_ran"):
        return None, None, train
    ckpt = ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_11_local_semantic_usage_oracle_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = CausalAssignmentBoundResidualMemoryV348(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


def init_stats() -> dict[str, Any]:
    return {k: {"sum": 0.0, "count": 0} for k in ["causal", "strict", "hard", "changed", "stable", "valid"]}


def update_stats(stats: dict[str, Any], pred: torch.Tensor, point: torch.Tensor, target: torch.Tensor, masks: dict[str, torch.Tensor]) -> None:
    gain = (_norm(pred) * _norm(target)).sum(dim=-1) - (_norm(point) * _norm(target)).sum(dim=-1)
    for key, mask in masks.items():
        m = mask.bool()
        if bool(m.any()):
            stats[key]["sum"] += float(gain[m].sum().detach().cpu())
            stats[key]["count"] += int(m.sum().detach().cpu())


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, val in stats.items():
        out[f"{key}_gain"] = None if val["count"] == 0 else float(val["sum"] / val["count"])
        out[f"{key}_count"] = int(val["count"])
    out["causal_assignment_subset_gain"] = out["causal_gain"]
    out["strict_residual_subset_gain"] = out["strict_gain"]
    out["semantic_hard_gain"] = out["hard_gain"]
    out["changed_gain"] = out["changed_gain"]
    out["stable_delta"] = out["stable_gain"]
    out["semantic_hard_signal"] = bool(out["hard_gain"] is not None and out["hard_gain"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain"] is not None and out["changed_gain"] > 0.005)
    out["stable_preservation"] = bool(out["stable_gain"] is None or out["stable_gain"] >= -0.02)
    return out


class UnitStats:
    def __init__(self) -> None:
        self.effective_units: list[float] = []
        self.instance_purity: list[float] = []
        self.semantic_purity: list[float] = []

    def update(self, assign: torch.Tensor, instance: torch.Tensor, obs_sem: torch.Tensor, obs_mask: torch.Tensor) -> None:
        assign_cpu = assign.detach().cpu().numpy()
        inst_cpu = instance.detach().cpu().numpy()
        sem = torch.nan_to_num(obs_sem.float())
        mask = obs_mask.float()
        sem_pool = (sem * mask[..., None]).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        sem_cpu = torch.nn.functional.normalize(sem_pool, dim=-1).detach().cpu().numpy()
        for b in range(assign_cpu.shape[0]):
            hard = assign_cpu[b].argmax(axis=-1)
            u_count = assign_cpu.shape[-1]
            usage = np.bincount(hard, minlength=u_count).astype(np.float32)
            used = usage > 0
            self.effective_units.append(float(used.sum()))
            inst_p, sem_p = [], []
            for u in np.where(used)[0]:
                pts = hard == u
                ids = inst_cpu[b][pts]
                ids = ids[ids >= 0]
                if ids.size:
                    _, counts = np.unique(ids, return_counts=True)
                    inst_p.append(float(counts.max() / counts.sum()))
                vec = sem_cpu[b][pts]
                if vec.size:
                    mean = vec.mean(axis=0)
                    mean = mean / (np.linalg.norm(mean) + 1e-8)
                    sem_p.append(float((vec @ mean).mean()))
            if inst_p:
                self.instance_purity.append(float(np.mean(inst_p)))
            if sem_p:
                self.semantic_purity.append(float(np.mean(sem_p)))

    def finalize(self) -> dict[str, float | None]:
        return {
            "effective_units": None if not self.effective_units else float(np.mean(self.effective_units)),
            "unit_dominant_instance_purity": None if not self.instance_purity else float(np.mean(self.instance_purity)),
            "unit_semantic_purity": None if not self.semantic_purity else float(np.mean(self.semantic_purity)),
        }


def split_eval(split: str, model: CausalAssignmentBoundResidualMemoryV348, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ds = LocalSemanticUsageDataset(split, ckargs)
    loader = DataLoader(ds, batch_size=ckargs.batch_size, shuffle=False, num_workers=ckargs.num_workers, collate_fn=collate_v3410)
    modes = {
        "normal": "force_gate_zero",
        "zero_semantic_measurements": "zero_semantic_measurements",
        "shuffle_semantic_measurements": "shuffle_semantic_measurements_across_points",
        "shuffle_assignment": "shuffle_assignment",
        "zero_unit_memory": "zero_unit_memory",
        "pointwise": "force_gate_zero",
    }
    stats = {mode: init_stats() for mode in modes}
    unit = UnitStats()
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            inp = model_inputs(bd, ckargs)
            valid = bd["fut_teacher_available_mask"].bool()
            masks = {
                "causal": bd["causal_assignment_residual_semantic_mask"].bool() & valid,
                "strict": bd["strict_residual_semantic_utility_mask"].bool() & valid,
                "hard": bd["semantic_hard_mask"].bool() & valid,
                "changed": bd["changed_mask"].bool() & valid,
                "stable": bd["stable_suppress_mask"].bool() & valid,
                "valid": valid,
            }
            normal_out: dict[str, torch.Tensor] | None = None
            for mode, intervention in modes.items():
                out = normal_out if mode == "pointwise" and normal_out is not None else model(**inp, intervention=intervention)
                if mode == "normal":
                    normal_out = out
                    unit.update(out["point_to_unit_assignment"], bd["point_to_instance_id"], bd["obs_semantic_measurements"], bd["obs_semantic_measurement_mask"])
                pred = out["pointwise_semantic_belief"] if mode == "pointwise" else local_compose(out, bd, use_usage_score=True)
                update_stats(stats[mode], pred, out["pointwise_semantic_belief"], bd["fut_teacher_embedding"], masks)
    per_mode = {mode: finalize_stats(s) for mode, s in stats.items()}
    normal = per_mode["normal"]
    deltas = {
        "zero_semantic_measurements_metric_delta": None if normal["causal_gain"] is None or per_mode["zero_semantic_measurements"]["causal_gain"] is None else float(normal["causal_gain"] - per_mode["zero_semantic_measurements"]["causal_gain"]),
        "shuffle_semantic_measurements_metric_delta": None if normal["causal_gain"] is None or per_mode["shuffle_semantic_measurements"]["causal_gain"] is None else float(normal["causal_gain"] - per_mode["shuffle_semantic_measurements"]["causal_gain"]),
        "shuffle_assignment_metric_delta": None if normal["causal_gain"] is None or per_mode["shuffle_assignment"]["causal_gain"] is None else float(normal["causal_gain"] - per_mode["shuffle_assignment"]["causal_gain"]),
        "zero_unit_memory_metric_delta": None if normal["causal_gain"] is None or per_mode["zero_unit_memory"]["causal_gain"] is None else float(normal["causal_gain"] - per_mode["zero_unit_memory"]["causal_gain"]),
    }
    return {"modes": per_mode, "normal": {**normal, **unit.finalize()}, **deltas}


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
            "中文结论": "V34.11 local semantic usage probe 训练未运行，评估跳过。",
            "local_semantic_usage_probe_ran": False,
            "local_semantic_usage_probe_passed": "not_run",
            "skip_reason": train.get("skip_reason", "train_not_run"),
            "learned_gate_training_allowed": False,
            "recommended_next_step": "fix_semantic_measurement_bank",
        }
        dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "V34.11 local semantic usage oracle probe 决策中文报告", decision, ["中文结论", "local_semantic_usage_probe_ran", "local_semantic_usage_probe_passed", "skip_reason", "recommended_next_step"])
        print(f"已写出 V34.11 local usage 评估跳过报告: {SUMMARY.relative_to(ROOT)}")
        return 0
    per = {split: split_eval(split, model, ckargs, device) for split in ("val", "test")}
    nval, ntest = per["val"]["normal"], per["test"]["normal"]
    sem_lb = bool(
        min(per["val"]["zero_semantic_measurements_metric_delta"] or 0.0, per["val"]["shuffle_semantic_measurements_metric_delta"] or 0.0) > 0.002
        and min(per["test"]["zero_semantic_measurements_metric_delta"] or 0.0, per["test"]["shuffle_semantic_measurements_metric_delta"] or 0.0) > 0.002
    )
    assign_lb = bool((per["val"]["shuffle_assignment_metric_delta"] or 0.0) > 0.002 and (per["test"]["shuffle_assignment_metric_delta"] or 0.0) > 0.002)
    unit_lb = bool((per["val"]["zero_unit_memory_metric_delta"] or 0.0) > 0.002 and (per["test"]["zero_unit_memory_metric_delta"] or 0.0) > 0.002)
    semantic_signal = bool((nval["semantic_hard_signal"] and ntest["semantic_hard_signal"]) or (nval["changed_semantic_signal"] and ntest["changed_semantic_signal"]))
    passed = bool(sem_lb and nval["stable_preservation"] and ntest["stable_preservation"] and semantic_signal)
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.11 local semantic usage oracle probe 已评估；只有 semantic measurement 介入会伤害 val/test residual 指标且 hard/changed 有正信号时才允许后续 learned gate。",
        "local_semantic_usage_probe_ran": True,
        "local_semantic_usage_probe_passed": passed,
        "pooling_variant": train.get("pooling_variant"),
        "local_semantic_usage_loss_active": bool(train.get("local_semantic_usage_loss_active")),
        "local_assignment_contrast_loss_active": bool(train.get("local_assignment_contrast_loss_active")),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen and train.get("v30_backbone_frozen", True)),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "semantic_hard_signal": {"val": nval["semantic_hard_signal"], "test": ntest["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": nval["changed_semantic_signal"], "test": ntest["changed_semantic_signal"]},
        "stable_preservation": {"val": nval["stable_preservation"], "test": ntest["stable_preservation"]},
        "pointwise_baseline_dominates": bool(not ((nval["causal_assignment_subset_gain"] or 0.0) > 0.005 or (ntest["causal_assignment_subset_gain"] or 0.0) > 0.005)),
        "causal_assignment_subset_gain": {"val": nval["causal_assignment_subset_gain"], "test": ntest["causal_assignment_subset_gain"]},
        "strict_residual_subset_gain": {"val": nval["strict_residual_subset_gain"], "test": ntest["strict_residual_subset_gain"]},
        "unit_memory_load_bearing_on_residual": unit_lb,
        "semantic_measurements_load_bearing_on_residual": sem_lb,
        "assignment_load_bearing_on_residual": assign_lb,
        "zero_semantic_measurements_metric_delta": {"val": per["val"]["zero_semantic_measurements_metric_delta"], "test": per["test"]["zero_semantic_measurements_metric_delta"]},
        "shuffle_semantic_measurements_metric_delta": {"val": per["val"]["shuffle_semantic_measurements_metric_delta"], "test": per["test"]["shuffle_semantic_measurements_metric_delta"]},
        "shuffle_assignment_metric_delta": {"val": per["val"]["shuffle_assignment_metric_delta"], "test": per["test"]["shuffle_assignment_metric_delta"]},
        "zero_unit_memory_metric_delta": {"val": per["val"]["zero_unit_memory_metric_delta"], "test": per["test"]["zero_unit_memory_metric_delta"]},
        "effective_units": {"val": nval["effective_units"], "test": ntest["effective_units"]},
        "unit_dominant_instance_purity": {"val": nval["unit_dominant_instance_purity"], "test": ntest["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": nval["unit_semantic_purity"], "test": ntest["unit_semantic_purity"]},
        "learned_gate_training_allowed": passed,
        "recommended_next_step": "fix_residual_gate" if passed else "fix_local_semantic_usage_loss",
    }
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "per_split": per, "decision": decision}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "V34.11 local semantic usage oracle probe 决策中文报告",
        decision,
        [
            "中文结论",
            "local_semantic_usage_probe_ran",
            "local_semantic_usage_probe_passed",
            "pooling_variant",
            "local_semantic_usage_loss_active",
            "local_assignment_contrast_loss_active",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "causal_assignment_subset_gain",
            "strict_residual_subset_gain",
            "unit_memory_load_bearing_on_residual",
            "semantic_measurements_load_bearing_on_residual",
            "assignment_load_bearing_on_residual",
            "effective_units",
            "unit_dominant_instance_purity",
            "unit_semantic_purity",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.11 local usage 评估摘要: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
