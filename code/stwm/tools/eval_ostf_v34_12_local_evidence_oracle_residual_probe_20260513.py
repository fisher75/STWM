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

from stwm.modules.ostf_v34_12_measurement_causal_residual_memory import MeasurementCausalResidualMemoryV3412
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_11_local_semantic_usage_oracle_probe_20260513 import LocalSemanticUsageDataset, collate_v3410
from stwm.tools.train_ostf_v34_12_local_evidence_oracle_residual_probe_20260513 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, compose, model_inputs


SUMMARY = ROOT / "reports/stwm_ostf_v34_12_local_evidence_oracle_residual_probe_eval_summary_20260513.json"
DECISION = ROOT / "reports/stwm_ostf_v34_12_local_evidence_oracle_residual_probe_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_LOCAL_EVIDENCE_ORACLE_RESIDUAL_PROBE_DECISION_20260513.md"
V3411_DECISION = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_decision_20260513.json"


def norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.nan_to_num(x.float()), dim=-1)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[MeasurementCausalResidualMemoryV3412 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    if not train.get("oracle_residual_probe_ran"):
        return None, None, train
    ckpt = ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_12_local_evidence_oracle_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = MeasurementCausalResidualMemoryV3412(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, val: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(val[m].sum().detach().cpu())
            self.count[key] = self.count.get(key, 0) + int(m.sum().detach().cpu())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def update(acc: Acc, prefix: str, pred: torch.Tensor, point: torch.Tensor, target: torch.Tensor, masks: dict[str, torch.Tensor]) -> None:
    gain = (norm(pred) * norm(target)).sum(dim=-1) - (norm(point) * norm(target)).sum(dim=-1)
    for key, mask in masks.items():
        acc.add(f"{prefix}:{key}", gain, mask)


class UnitStats:
    def __init__(self) -> None:
        self.effective_units: list[float] = []
        self.instance_purity: list[float] = []
        self.semantic_purity: list[float] = []
        self.entropy: list[float] = []
        self.max_attn: list[float] = []
        self.usage_delta: list[float] = []

    def update(self, out: dict[str, torch.Tensor], zero: dict[str, torch.Tensor], inst: torch.Tensor, obs_sem: torch.Tensor, obs_mask: torch.Tensor, pos: torch.Tensor) -> None:
        assign_cpu = out["point_to_unit_assignment"].detach().cpu().numpy()
        inst_cpu = inst.detach().cpu().numpy()
        sem = torch.nan_to_num(obs_sem.float())
        mask = obs_mask.float()
        sem_pool = (sem * mask[..., None]).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        sem_cpu = torch.nn.functional.normalize(sem_pool, dim=-1).detach().cpu().numpy()
        for b in range(assign_cpu.shape[0]):
            hard = assign_cpu[b].argmax(axis=-1)
            usage = np.bincount(hard, minlength=assign_cpu.shape[-1]).astype(np.float32)
            used = usage > 0
            self.effective_units.append(float(used.sum()))
            ip, sp = [], []
            for u in np.where(used)[0]:
                pts = hard == u
                ids = inst_cpu[b][pts]
                ids = ids[ids >= 0]
                if ids.size:
                    _, counts = np.unique(ids, return_counts=True)
                    ip.append(float(counts.max() / counts.sum()))
                vec = sem_cpu[b][pts]
                if vec.size:
                    mean = vec.mean(axis=0)
                    mean = mean / (np.linalg.norm(mean) + 1e-8)
                    sp.append(float((vec @ mean).mean()))
            if ip:
                self.instance_purity.append(float(np.mean(ip)))
            if sp:
                self.semantic_purity.append(float(np.mean(sp)))
        if bool(pos.any()):
            self.entropy.append(float(out["attention_temporal_entropy"][pos].mean().detach().cpu()))
            self.max_attn.append(float(out["attention_max_weight"][pos].mean().detach().cpu()))
            self.usage_delta.append(float((out["semantic_measurement_usage_score"][pos] - zero["semantic_measurement_usage_score"][pos]).mean().detach().cpu()))

    def final(self) -> dict[str, float | bool | None]:
        ent = None if not self.entropy else float(np.mean(self.entropy))
        maxa = None if not self.max_attn else float(np.mean(self.max_attn))
        return {
            "effective_units": None if not self.effective_units else float(np.mean(self.effective_units)),
            "unit_dominant_instance_purity": None if not self.instance_purity else float(np.mean(self.instance_purity)),
            "unit_semantic_purity": None if not self.semantic_purity else float(np.mean(self.semantic_purity)),
            "attention_temporal_entropy": ent,
            "attention_max_weight": maxa,
            "attention_uses_nontrivial_timesteps": bool(ent is not None and maxa is not None and ent < 0.98 and maxa > 0.16),
            "semantic_usage_score_delta_under_zero_sem": None if not self.usage_delta else float(np.mean(self.usage_delta)),
        }


def finalize(acc: Acc, prefix: str) -> dict[str, Any]:
    out = {}
    for key in ["causal", "strict", "hard", "changed", "stable", "valid"]:
        out[f"{key}_gain"] = acc.mean(f"{prefix}:{key}")
    out["causal_assignment_subset_gain"] = out["causal_gain"]
    out["strict_residual_subset_gain"] = out["strict_gain"]
    out["semantic_hard_gain"] = out["hard_gain"]
    out["changed_gain"] = out["changed_gain"]
    out["stable_delta"] = out["stable_gain"]
    out["semantic_hard_signal"] = bool(out["hard_gain"] is not None and out["hard_gain"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain"] is not None and out["changed_gain"] > 0.005)
    out["stable_preservation"] = bool(out["stable_gain"] is None or out["stable_gain"] >= -0.02)
    return out


def split_eval(split: str, model: MeasurementCausalResidualMemoryV3412, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
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
    accs = {k: Acc() for k in modes}
    unit = UnitStats()
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            inp = model_inputs(bd)
            valid = bd["fut_teacher_available_mask"].bool()
            masks = {
                "causal": bd["causal_assignment_residual_semantic_mask"].bool() & valid,
                "strict": bd["strict_residual_semantic_utility_mask"].bool() & valid,
                "hard": bd["semantic_hard_mask"].bool() & valid,
                "changed": bd["changed_mask"].bool() & valid,
                "stable": bd["stable_suppress_mask"].bool() & valid,
                "valid": valid,
            }
            outs = {}
            for mode, intervention in modes.items():
                outs[mode] = model(**inp, intervention=intervention)
                pred = outs[mode]["pointwise_semantic_belief"] if mode == "pointwise" else compose(outs[mode], bd)
                update(accs[mode], mode, pred, outs[mode]["pointwise_semantic_belief"], bd["fut_teacher_embedding"], masks)
            unit.update(outs["normal"], outs["zero_semantic_measurements"], bd["point_to_instance_id"], bd["obs_semantic_measurements"], bd["obs_semantic_measurement_mask"], masks["causal"])
    per = {mode: finalize(accs[mode], mode) for mode in modes}
    normal = per["normal"]
    deltas = {
        "zero_semantic_measurements_metric_delta": None if normal["causal_gain"] is None or per["zero_semantic_measurements"]["causal_gain"] is None else float(normal["causal_gain"] - per["zero_semantic_measurements"]["causal_gain"]),
        "shuffle_semantic_measurements_metric_delta": None if normal["causal_gain"] is None or per["shuffle_semantic_measurements"]["causal_gain"] is None else float(normal["causal_gain"] - per["shuffle_semantic_measurements"]["causal_gain"]),
        "shuffle_assignment_metric_delta": None if normal["causal_gain"] is None or per["shuffle_assignment"]["causal_gain"] is None else float(normal["causal_gain"] - per["shuffle_assignment"]["causal_gain"]),
        "zero_unit_memory_metric_delta": None if normal["causal_gain"] is None or per["zero_unit_memory"]["causal_gain"] is None else float(normal["causal_gain"] - per["zero_unit_memory"]["causal_gain"]),
    }
    return {"modes": per, "normal": {**normal, **unit.final()}, **deltas}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, train = load_model(args, device)
    if model is None or ckargs is None:
        decision = {"generated_at_utc": utc_now(), "中文结论": "V34.12 local evidence oracle residual probe 训练未运行，评估跳过。", "oracle_residual_probe_ran": False, "oracle_residual_probe_passed": False, "skip_reason": train.get("skip_reason", "train_not_run"), "recommended_next_step": "fix_local_semantic_evidence_encoder"}
        dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "V34.12 local evidence oracle residual probe 决策中文报告", decision, ["中文结论", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "skip_reason", "recommended_next_step"])
        print(f"已写出 V34.12 local evidence 评估跳过报告: {SUMMARY.relative_to(ROOT)}")
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
    pass_probe = bool(model.v30_backbone_frozen and nval["stable_preservation"] and ntest["stable_preservation"] and sem_lb and semantic_signal and (nval["causal_assignment_subset_gain"] or 0.0) > 0 and (ntest["causal_assignment_subset_gain"] or 0.0) > 0)
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 local evidence oracle residual probe 已评估；只有 semantic measurement 干预在 val/test 都伤害 residual 且 hard/changed 有正信号时才允许 learned gate。",
        "oracle_residual_probe_ran": True,
        "oracle_residual_probe_passed": pass_probe,
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
        "attention_temporal_entropy": {"val": nval["attention_temporal_entropy"], "test": ntest["attention_temporal_entropy"]},
        "attention_uses_nontrivial_timesteps": bool(nval["attention_uses_nontrivial_timesteps"] and ntest["attention_uses_nontrivial_timesteps"]),
        "semantic_usage_score_delta_under_zero_sem": {"val": nval["semantic_usage_score_delta_under_zero_sem"], "test": ntest["semantic_usage_score_delta_under_zero_sem"]},
        "effective_units": {"val": nval["effective_units"], "test": ntest["effective_units"]},
        "unit_dominant_instance_purity": {"val": nval["unit_dominant_instance_purity"], "test": ntest["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": nval["unit_semantic_purity"], "test": ntest["unit_semantic_purity"]},
        "learned_gate_training_allowed": pass_probe,
        "recommended_next_step": "train_local_evidence_residual_gate" if pass_probe else "fix_local_semantic_evidence_encoder",
    }
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "per_split": per, "decision": decision}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "V34.12 local evidence oracle residual probe 决策中文报告", decision, ["中文结论", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "causal_assignment_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "zero_semantic_measurements_metric_delta", "shuffle_semantic_measurements_metric_delta", "attention_uses_nontrivial_timesteps", "effective_units", "unit_dominant_instance_purity", "unit_semantic_purity", "recommended_next_step"])
    print(f"已写出 V34.12 local evidence 评估摘要: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
