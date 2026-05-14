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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import (
    compose_reader_gate,
    hard_changed_aligned_mask,
    labels,
    load_residual_model,
    make_loader,
    reader_inputs,
)
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
import stwm.tools.train_ostf_v34_23_activation_state_gate_probe_20260513 as v3423


DEFAULT_SEEDS = (123,)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


class ScalarAcc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(value[m].sum().detach().cpu())
            self.count[key] = self.count.get(key, 0) + int(m.sum().detach().cpu())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        if c == 0:
            return None
        return float(self.sum[key] / c)


def gate_tensor(
    out: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    gate_name: str,
    thresholds: dict[str, float],
) -> torch.Tensor:
    target, kind = gate_name.rsplit("_", 1)
    prob = torch.sigmoid(pred[target]).clamp(0.0, 1.0)
    if kind == "threshold":
        gate = (prob >= thresholds[target]).float()
    else:
        gate = prob
    return gate * out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)


def detailed_gate_eval(
    split: str,
    residual_model: Any,
    reader: torch.nn.Module,
    ckargs: argparse.Namespace,
    device: torch.device,
    thresholds: dict[str, float],
    gate_name: str,
    *,
    over_open_threshold: float,
    stable_over_update_margin: float,
) -> dict[str, Any]:
    reader.eval()
    acc = ScalarAcc()
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            pred = reader(**reader_inputs(out))["activation_logits"]
            final = compose_reader_gate(out, pred, gate_name, thresholds)
            pointwise = out["pointwise_semantic_belief"]
            target = bd["fut_teacher_embedding"]
            gain = (norm(final) * norm(target)).sum(dim=-1) - (norm(pointwise) * norm(target)).sum(dim=-1)
            gate = gate_tensor(out, pred, gate_name, thresholds)
            valid = bd["fut_teacher_available_mask"].bool()
            hard = bd["semantic_hard_mask"].bool() & valid
            changed = bd["changed_mask"].bool() & valid
            stable = bd["stable_suppress_mask"].bool() & valid
            aligned = hard_changed_aligned_mask(bd)
            hard_changed = (hard | changed) & valid
            masks = {
                "valid": valid,
                "stable": stable,
                "hard": hard,
                "changed": changed,
                "hard_changed": hard_changed,
                "aligned": aligned,
            }
            over_open = gate > over_open_threshold
            over_update = gain < stable_over_update_margin
            for name, mask in masks.items():
                acc.add(f"{name}_gate_mean", gate, mask)
                acc.add(f"{name}_gain_mean", gain, mask)
                acc.add(f"{name}_over_open_rate", over_open.float(), mask)
                acc.add(f"{name}_over_update_rate", over_update.float(), mask)
            lab = labels(bd, out, 0.005)
            benefit = lab["benefit"] & valid
            acc.add("benefit_gate_mean", gate, benefit)
            acc.add("nonbenefit_gate_mean", gate, (~benefit) & valid)
    out = {
        "gate_over_open_threshold": over_open_threshold,
        "stable_over_update_margin": stable_over_update_margin,
    }
    for key in (
        "stable",
        "hard",
        "changed",
        "hard_changed",
        "aligned",
        "valid",
        "benefit",
        "nonbenefit",
    ):
        out[f"{key}_gate_mean"] = acc.mean(f"{key}_gate_mean")
        out[f"{key}_gain_mean"] = acc.mean(f"{key}_gain_mean")
        out[f"{key}_over_open_rate"] = acc.mean(f"{key}_over_open_rate")
        out[f"{key}_over_update_rate"] = acc.mean(f"{key}_over_update_rate")
    stable_gate = out.get("stable_gate_mean") or 0.0
    hard_changed_gate = out.get("hard_changed_gate_mean") or 0.0
    out["gate_order_ok"] = bool(hard_changed_gate > stable_gate)
    out["stable_over_open_detected"] = bool((out.get("stable_over_open_rate") or 0.0) > 0.25)
    out["stable_over_update_detected"] = bool((out.get("stable_over_update_rate") or 0.0) > 0.05)
    return out


def run_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    out_tag = f"seed{seed}"
    summary = ROOT / f"reports/stwm_ostf_v34_23_{out_tag}_replication_summary_20260514.json"
    doc = ROOT / f"docs/STWM_OSTF_V34_23_{out_tag.upper()}_REPLICATION_SUMMARY_20260514.md"
    ckpt_dir = ROOT / f"outputs/checkpoints/stwm_ostf_v34_23_activation_state_gate_probe_h32_m128_{out_tag}"
    checkpoint = ckpt_dir / f"v34_23_activation_state_gate_probe_m128_h32_{out_tag}.pt"

    old_paths = (v3423.SUMMARY, v3423.DOC, v3423.CKPT_DIR, v3423.CHECKPOINT)
    try:
        v3423.SUMMARY = summary
        v3423.DOC = doc
        v3423.CKPT_DIR = ckpt_dir
        v3423.CHECKPOINT = checkpoint
        train_args = argparse.Namespace(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            reader_hidden_dim=args.reader_hidden_dim,
            utility_margin=args.utility_margin,
            gate_name=args.gate_name,
            semantic_contrast_weight=args.semantic_contrast_weight,
            assignment_contrast_weight=args.assignment_contrast_weight,
            semantic_contrast_margin=args.semantic_contrast_margin,
            assignment_contrast_margin=args.assignment_contrast_margin,
            seed=seed,
            cpu=args.cpu,
        )
        payload = v3423.train_one(train_args)
    finally:
        v3423.SUMMARY, v3423.DOC, v3423.CKPT_DIR, v3423.CHECKPOINT = old_paths

    decision = payload["decision"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, _ = load_residual_model(train_args, device)
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    reader, thresholds, _ = v3423.load_reader(train_args, residual_model, device)
    ck = torch.load(checkpoint, map_location="cpu")
    reader.load_state_dict(ck["reader"], strict=True)
    reader.to(device)
    gate_diagnostics = {
        split: detailed_gate_eval(
            split,
            residual_model,
            reader,
            ckargs,
            device,
            thresholds,
            args.gate_name,
            over_open_threshold=args.gate_over_open_threshold,
            stable_over_update_margin=args.stable_over_update_margin,
        )
        for split in ("val", "test")
    }
    seed42 = json.loads((ROOT / "reports/stwm_ostf_v34_23_activation_state_gate_probe_summary_20260513.json").read_text(encoding="utf-8"))[
        "decision"
    ]
    replicated = bool(
        decision.get("activation_state_gate_probe_passed")
        and decision.get("semantic_hard_signal", {}).get("val")
        and decision.get("semantic_hard_signal", {}).get("test")
        and decision.get("changed_semantic_signal", {}).get("val")
        and decision.get("changed_semantic_signal", {}).get("test")
        and decision.get("stable_preservation", {}).get("val")
        and decision.get("stable_preservation", {}).get("test")
        and decision.get("semantic_measurements_load_bearing_on_residual")
        and decision.get("assignment_load_bearing_on_residual")
        and decision.get("unit_memory_load_bearing_on_residual")
        and gate_diagnostics["val"]["gate_order_ok"]
        and gate_diagnostics["test"]["gate_order_ok"]
        and not gate_diagnostics["val"]["stable_over_update_detected"]
        and not gate_diagnostics["test"]["stable_over_update_detected"]
    )
    analysis = {
        "阶段性分析": (
            "seed123 复现只检验 V34.23 activation-state gate probe 的跨 seed 稳定性，"
            "不是新结构训练，也不允许声明 semantic field success。核心检查包括 hard/changed 正信号、"
            "stable preservation、semantic/assignment/unit intervention delta、gate 是否在 hard/changed 高于 stable，"
            "以及 stable 是否出现过度打开或过度更新。"
        ),
        "论文相关问题解决方案参考": (
            "本轮采用的做法对应 object-centric/slot 类方法中的 intervention 检验原则、"
            "residual-gated 模型中的主路径保护原则，以及因果表征诊断中的 counterfactual ablation 原则："
            "跨 seed 复制时必须报告 gate 行为和干预 delta，而不是只看最终 top-line 指标。"
        ),
        "本轮最佳下一步": (
            "run_v34_23_seed456_replication"
            if seed == 123 and replicated
            else ("run_v34_23_cross_seed_decision" if replicated else "fix_activation_state_gate_training")
        ),
    }
    decision["seed_replication"] = {
        "seed": seed,
        "seed42_reference_passed": bool(seed42.get("activation_state_gate_probe_passed")),
        "seed123_replication_passed": replicated if seed == 123 else None,
        "cross_seed_consistent_with_seed42": replicated,
        "gate_diagnostics": gate_diagnostics,
        "analysis": analysis,
        "recommended_next_step": analysis["本轮最佳下一步"],
    }
    payload["decision"] = decision
    dump_json(summary, payload)
    write_doc(
        doc,
        f"V34.23 {out_tag} 复现实验中文报告",
        decision,
        [
            "中文结论",
            "activation_state_gate_probe_ran",
            "activation_state_gate_probe_passed",
            "seed_replication",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "hard_changed_gain",
            "semantic_measurements_load_bearing_on_residual",
            "assignment_load_bearing_on_residual",
            "unit_memory_load_bearing_on_residual",
            "integrated_identity_field_claim_allowed",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    return payload


def load_seed_summary(seed: int) -> dict[str, Any] | None:
    path = ROOT / f"reports/stwm_ostf_v34_23_seed{seed}_replication_summary_20260514.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cross_seed_decision() -> dict[str, Any]:
    seed42_path = ROOT / "reports/stwm_ostf_v34_23_activation_state_gate_probe_summary_20260513.json"
    seed42 = json.loads(seed42_path.read_text(encoding="utf-8"))["decision"] if seed42_path.exists() else {}
    seed_runs: dict[str, Any] = {}
    for seed in (123, 456):
        payload = load_seed_summary(seed)
        if payload is not None:
            seed_runs[f"seed{seed}"] = payload["decision"]
    seed_pass = {
        name: bool(dec.get("seed_replication", {}).get("cross_seed_consistent_with_seed42"))
        for name, dec in seed_runs.items()
    }
    required_done = "seed123" in seed_pass and "seed456" in seed_pass
    all_required_passed = bool(required_done and all(seed_pass.values()) and seed42.get("activation_state_gate_probe_passed"))
    gate_risks = {}
    for name, dec in seed_runs.items():
        gd = dec.get("seed_replication", {}).get("gate_diagnostics", {})
        gate_risks[name] = {
            "val_stable_gate_mean": gd.get("val", {}).get("stable_gate_mean"),
            "test_stable_gate_mean": gd.get("test", {}).get("stable_gate_mean"),
            "val_stable_over_open_rate": gd.get("val", {}).get("stable_over_open_rate"),
            "test_stable_over_open_rate": gd.get("test", {}).get("stable_over_open_rate"),
            "val_stable_over_update_rate": gd.get("val", {}).get("stable_over_update_rate"),
            "test_stable_over_update_rate": gd.get("test", {}).get("stable_over_update_rate"),
            "val_gate_order_ok": gd.get("val", {}).get("gate_order_ok"),
            "test_gate_order_ok": gd.get("test", {}).get("gate_order_ok"),
        }
    recommended = "stop_and_analyze_claim_boundary"
    if not seed42.get("activation_state_gate_probe_passed"):
        recommended = "fix_activation_state_gate_training"
    elif "seed123" not in seed_pass:
        recommended = "run_v34_23_seed123_replication"
    elif "seed456" not in seed_pass:
        recommended = "run_v34_23_seed456_replication"
    elif not all_required_passed:
        recommended = "fix_activation_state_gate_training"
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.23 seed42/123/456 复现决策已汇总；即使多 seed 通过，也仍不能声明 semantic field success，因为 stable gate over-open 风险仍需单独处理。",
        "seed42_reference_passed": bool(seed42.get("activation_state_gate_probe_passed")),
        "seed_replication_pass": seed_pass,
        "required_seeds_done": required_done,
        "all_required_seeds_passed": all_required_passed,
        "semantic_hard_signal": {name: dec.get("semantic_hard_signal") for name, dec in seed_runs.items()},
        "changed_semantic_signal": {name: dec.get("changed_semantic_signal") for name, dec in seed_runs.items()},
        "stable_preservation": {name: dec.get("stable_preservation") for name, dec in seed_runs.items()},
        "intervention_delta": {
            name: {
                split: {
                    "zero_semantic_measurements_delta": dec.get("intervention_eval", {}).get(split, {}).get("zero_semantic_measurements_delta"),
                    "shuffle_semantic_measurements_delta": dec.get("intervention_eval", {}).get(split, {}).get("shuffle_semantic_measurements_delta"),
                    "shuffle_assignment_delta": dec.get("intervention_eval", {}).get(split, {}).get("shuffle_assignment_delta"),
                    "zero_unit_memory_delta": dec.get("intervention_eval", {}).get(split, {}).get("zero_unit_memory_delta"),
                }
                for split in ("val", "test")
            }
            for name, dec in seed_runs.items()
        },
        "gate_over_open_and_stable_update": gate_risks,
        "阶段性分析": "seed123 与 seed456 是复现实验，不是继续修 bug；当前复现重点是验证 activation-state gate probe 的跨 seed 稳定性和干预因果性，同时明确 gate 过开风险。",
        "论文相关问题解决方案参考": "建议继续沿用 residual main-path preservation、counterfactual intervention、slot/object-memory assignment ablation 的评估框架；若进入下一轮修复，应优先做 gate calibration / sparse gate regularization，而不是扩大模型或改 trajectory backbone。",
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": recommended,
    }
    out = ROOT / "reports/stwm_ostf_v34_23_cross_seed_replication_decision_20260514.json"
    doc = ROOT / "docs/STWM_OSTF_V34_23_CROSS_SEED_REPLICATION_DECISION_20260514.md"
    dump_json(out, decision)
    write_doc(
        doc,
        "V34.23 跨 seed 复现决策中文总结",
        decision,
        [
            "中文结论",
            "seed42_reference_passed",
            "seed_replication_pass",
            "required_seeds_done",
            "all_required_seeds_passed",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "intervention_delta",
            "gate_over_open_and_stable_update",
            "阶段性分析",
            "论文相关问题解决方案参考",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    return decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5.0e-5)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--utility-margin", type=float, default=0.005)
    p.add_argument("--gate-name", default="benefit_soft")
    p.add_argument("--semantic-contrast-weight", type=float, default=0.35)
    p.add_argument("--assignment-contrast-weight", type=float, default=0.25)
    p.add_argument("--semantic-contrast-margin", type=float, default=0.004)
    p.add_argument("--assignment-contrast-margin", type=float, default=0.004)
    p.add_argument("--gate-over-open-threshold", type=float, default=0.05)
    p.add_argument("--stable-over-update-margin", type=float, default=-0.02)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results: dict[str, Any] = {"generated_at_utc": utc_now(), "runs": {}}
    for seed in args.seeds:
        print(f"开始 V34.23 seed{seed} 复现实验。", flush=True)
        results["runs"][f"seed{seed}"] = run_seed(seed, args)
    seed_tag = "_".join(f"seed{s}" for s in args.seeds)
    summary = ROOT / f"reports/stwm_ostf_v34_23_seed_replication_round_summary_{seed_tag}_20260514.json"
    doc = ROOT / f"docs/STWM_OSTF_V34_23_SEED_REPLICATION_ROUND_SUMMARY_{seed_tag.upper()}_20260514.md"
    seed_decisions = {k: v["decision"]["seed_replication"] for k, v in results["runs"].items()}
    all_passed = bool(seed_decisions and all(bool(v.get("cross_seed_consistent_with_seed42")) for v in seed_decisions.values()))
    round_decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.23 跨 seed 复现实验轮完成；这是复现，不是 bug 修复，也不声明 semantic field success。",
        "seeds": list(seed_decisions),
        "all_requested_seeds_passed": all_passed,
        "seed_decisions": seed_decisions,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_23_seed456_replication" if ("seed123" in seed_decisions and all_passed and len(seed_decisions) == 1) else ("run_v34_23_cross_seed_decision" if all_passed else "fix_activation_state_gate_training"),
    }
    results["round_decision"] = round_decision
    results["cross_seed_decision"] = write_cross_seed_decision()
    dump_json(summary, results)
    write_doc(
        doc,
        "V34.23 seed 复现实验轮中文总结",
        round_decision,
        [
            "中文结论",
            "seeds",
            "all_requested_seeds_passed",
            "seed_decisions",
            "integrated_identity_field_claim_allowed",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.23 seed 复现实验轮总结: {summary.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
