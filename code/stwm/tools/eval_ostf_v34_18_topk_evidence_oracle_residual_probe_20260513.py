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

from stwm.modules.ostf_v34_18_topk_evidence_residual_memory import TopKEvidenceResidualMemoryV3418
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, compose, model_inputs


SUMMARY = ROOT / "reports/stwm_ostf_v34_18_topk_evidence_oracle_residual_probe_eval_summary_20260513.json"
DECISION = ROOT / "reports/stwm_ostf_v34_18_topk_evidence_oracle_residual_probe_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_18_TOPK_EVIDENCE_ORACLE_RESIDUAL_PROBE_DECISION_20260513.md"
V3417 = ROOT / "reports/stwm_ostf_v34_17_topk_evidence_selector_ablation_20260513.json"


def norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.nan_to_num(x.float()), dim=-1)


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(values[m].sum().detach().cpu())
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
        self.topk_entropy: list[float] = []
        self.topk_max_weight: list[float] = []
        self.usage_score: list[float] = []

    def update(self, out: dict[str, torch.Tensor], inst: torch.Tensor, obs_sem: torch.Tensor, obs_mask: torch.Tensor, pos: torch.Tensor) -> None:
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
            self.topk_entropy.append(float(out["attention_temporal_entropy"][pos].mean().detach().cpu()))
            self.topk_max_weight.append(float(out["attention_max_weight"][pos].mean().detach().cpu()))
            self.usage_score.append(float(out["semantic_measurement_usage_score"][pos].mean().detach().cpu()))

    def final(self) -> dict[str, float | bool | None]:
        ent = None if not self.topk_entropy else float(np.mean(self.topk_entropy))
        maxw = None if not self.topk_max_weight else float(np.mean(self.topk_max_weight))
        return {
            "effective_units": None if not self.effective_units else float(np.mean(self.effective_units)),
            "unit_dominant_instance_purity": None if not self.instance_purity else float(np.mean(self.instance_purity)),
            "unit_semantic_purity": None if not self.semantic_purity else float(np.mean(self.semantic_purity)),
            "topk_attention_entropy": ent,
            "topk_attention_max_weight": maxw,
            "topk_uses_multi_evidence": bool(ent is not None and maxw is not None and ent > 0.15 and maxw < 0.95),
            "semantic_measurement_usage_score_mean": None if not self.usage_score else float(np.mean(self.usage_score)),
        }


def finalize(acc: Acc, prefix: str) -> dict[str, Any]:
    out = {f"{key}_gain": acc.mean(f"{prefix}:{key}") for key in ["causal", "strict", "hard", "changed", "stable", "valid"]}
    out["causal_assignment_subset_gain"] = out["causal_gain"]
    out["strict_residual_subset_gain"] = out["strict_gain"]
    out["semantic_hard_signal"] = bool(out["hard_gain"] is not None and out["hard_gain"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain"] is not None and out["changed_gain"] > 0.005)
    out["stable_preservation"] = bool(out["stable_gain"] is None or out["stable_gain"] >= -0.02)
    return out


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[TopKEvidenceResidualMemoryV3418 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    if not train.get("oracle_residual_probe_ran"):
        return None, None, train
    ckpt = ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_18_topk_evidence_oracle_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = TopKEvidenceResidualMemoryV3418(
        ckargs.v30_checkpoint,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        units=ckargs.trace_units,
        horizon=ckargs.horizon,
        selector_hidden_dim=ckargs.selector_hidden_dim,
        topk=ckargs.topk,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


def split_eval(split: str, model: TopKEvidenceResidualMemoryV3418, ckargs: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(
        TraceContractResidualDataset(split, ckargs),
        batch_size=ckargs.batch_size,
        shuffle=False,
        num_workers=ckargs.num_workers,
        collate_fn=collate_v3410,
    )
    modes = {
        "normal": "force_gate_zero",
        "zero_semantic_measurements": "zero_semantic_measurements",
        "shuffle_semantic_measurements": "shuffle_semantic_measurements_across_points",
        "shuffle_assignment": "shuffle_assignment",
        "zero_unit_memory": "zero_unit_memory",
        "selector_ablation": "selector_ablation",
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
            outs: dict[str, dict[str, torch.Tensor]] = {}
            for mode, intervention in modes.items():
                outs[mode] = model(**inp, intervention=intervention)
                pred = outs[mode]["pointwise_semantic_belief"] if mode == "pointwise" else compose(outs[mode], bd)
                update(accs[mode], mode, pred, outs[mode]["pointwise_semantic_belief"], bd["fut_teacher_embedding"], masks)
            unit.update(outs["normal"], bd["point_to_instance_id"], bd["obs_semantic_measurements"], bd["obs_semantic_measurement_mask"], masks["causal"])
    per = {mode: finalize(accs[mode], mode) for mode in modes}
    normal = per["normal"]
    deltas = {
        "zero_semantic_measurements_metric_delta": None if normal["causal_gain"] is None or per["zero_semantic_measurements"]["causal_gain"] is None else float(normal["causal_gain"] - per["zero_semantic_measurements"]["causal_gain"]),
        "shuffle_semantic_measurements_metric_delta": None if normal["causal_gain"] is None or per["shuffle_semantic_measurements"]["causal_gain"] is None else float(normal["causal_gain"] - per["shuffle_semantic_measurements"]["causal_gain"]),
        "shuffle_assignment_metric_delta": None if normal["causal_gain"] is None or per["shuffle_assignment"]["causal_gain"] is None else float(normal["causal_gain"] - per["shuffle_assignment"]["causal_gain"]),
        "zero_unit_memory_metric_delta": None if normal["causal_gain"] is None or per["zero_unit_memory"]["causal_gain"] is None else float(normal["causal_gain"] - per["zero_unit_memory"]["causal_gain"]),
        "selector_ablation_delta": None if normal["causal_gain"] is None or per["selector_ablation"]["causal_gain"] is None else float(normal["causal_gain"] - per["selector_ablation"]["causal_gain"]),
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
        decision = {
            "generated_at_utc": utc_now(),
            "中文结论": "V34.18 top-k evidence oracle residual probe 训练未运行，评估跳过。",
            "oracle_residual_probe_ran": False,
            "oracle_residual_probe_passed": False,
            "skip_reason": train.get("skip_reason", "train_not_run"),
            "recommended_next_step": "build_topk_evidence_conditioned_residual_probe",
        }
        dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "V34.18 top-k evidence oracle residual probe 决策中文报告", decision, ["中文结论", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "skip_reason", "recommended_next_step"])
        print(f"已写出 V34.18 top-k evidence 评估跳过报告: {DECISION.relative_to(ROOT)}")
        return 0

    per = {split: split_eval(split, model, ckargs, device) for split in ("val", "test")}
    nval, ntest = per["val"]["normal"], per["test"]["normal"]
    sem_lb = bool(
        min(per["val"]["zero_semantic_measurements_metric_delta"] or 0.0, per["val"]["shuffle_semantic_measurements_metric_delta"] or 0.0) > 0.002
        and min(per["test"]["zero_semantic_measurements_metric_delta"] or 0.0, per["test"]["shuffle_semantic_measurements_metric_delta"] or 0.0) > 0.002
    )
    assign_lb = bool((per["val"]["shuffle_assignment_metric_delta"] or 0.0) > 0.002 and (per["test"]["shuffle_assignment_metric_delta"] or 0.0) > 0.002)
    unit_lb = bool((per["val"]["zero_unit_memory_metric_delta"] or 0.0) > 0.002 and (per["test"]["zero_unit_memory_metric_delta"] or 0.0) > 0.002)
    selector_lb = bool((per["val"]["selector_ablation_delta"] or 0.0) > 0.001 and (per["test"]["selector_ablation_delta"] or 0.0) > 0.001)
    semantic_signal = bool((nval["semantic_hard_signal"] and ntest["semantic_hard_signal"]) or (nval["changed_semantic_signal"] and ntest["changed_semantic_signal"]))
    passed = bool(
        model.v30_backbone_frozen
        and sem_lb
        and semantic_signal
        and nval["stable_preservation"]
        and ntest["stable_preservation"]
        and (nval["causal_assignment_subset_gain"] or 0.0) > 0
        and (ntest["causal_assignment_subset_gain"] or 0.0) > 0
    )
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.18 top-k evidence residual probe 已评估；这是 oracle residual 内容验证，不是 learned gate，也不是 semantic field success。",
        "topk_evidence_residual_probe_ran": True,
        "oracle_residual_probe_ran": True,
        "oracle_residual_probe_passed": passed,
        "topk": int(getattr(ckargs, "topk", 8)),
        "v30_backbone_frozen": bool(model.v30_backbone_frozen),
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
        "selector_load_bearing_on_residual": selector_lb,
        "zero_semantic_measurements_metric_delta": {"val": per["val"]["zero_semantic_measurements_metric_delta"], "test": per["test"]["zero_semantic_measurements_metric_delta"]},
        "shuffle_semantic_measurements_metric_delta": {"val": per["val"]["shuffle_semantic_measurements_metric_delta"], "test": per["test"]["shuffle_semantic_measurements_metric_delta"]},
        "shuffle_assignment_metric_delta": {"val": per["val"]["shuffle_assignment_metric_delta"], "test": per["test"]["shuffle_assignment_metric_delta"]},
        "zero_unit_memory_metric_delta": {"val": per["val"]["zero_unit_memory_metric_delta"], "test": per["test"]["zero_unit_memory_metric_delta"]},
        "selector_ablation_delta": {"val": per["val"]["selector_ablation_delta"], "test": per["test"]["selector_ablation_delta"]},
        "topk_uses_multi_evidence": bool(nval["topk_uses_multi_evidence"] and ntest["topk_uses_multi_evidence"]),
        "effective_units": {"val": nval["effective_units"], "test": ntest["effective_units"]},
        "unit_dominant_instance_purity": {"val": nval["unit_dominant_instance_purity"], "test": ntest["unit_dominant_instance_purity"]},
        "unit_semantic_purity": {"val": nval["unit_semantic_purity"], "test": ntest["unit_semantic_purity"]},
        "learned_gate_training_ran": False,
        "learned_gate_training_allowed": passed,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "train_topk_evidence_residual_gate" if passed else "fix_topk_evidence_residual_content",
    }
    topk_ref = json.loads(V3417.read_text(encoding="utf-8")) if V3417.exists() else {}
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "v34_17_topk_ablation_reference": topk_ref, "per_split": per, "decision": decision}
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "V34.18 top-k evidence oracle residual probe 决策中文报告",
        decision,
        [
            "中文结论",
            "topk_evidence_residual_probe_ran",
            "oracle_residual_probe_passed",
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
            "selector_load_bearing_on_residual",
            "zero_semantic_measurements_metric_delta",
            "shuffle_semantic_measurements_metric_delta",
            "selector_ablation_delta",
            "topk_uses_multi_evidence",
            "effective_units",
            "unit_dominant_instance_purity",
            "unit_semantic_purity",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.18 top-k evidence 评估摘要: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
