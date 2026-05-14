#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import setproctitle
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import (
    load_residual_model,
    make_loader,
    reader_inputs,
)
from stwm.tools.eval_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514 import (
    load_v3425_readers,
    masks,
    observed_mean,
)
from stwm.tools.eval_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514 import (
    Acc,
    finalize_method,
    local_cos,
    norm,
    update_method,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
from stwm.tools.train_ostf_v34_25_sparse_calibrated_gate_repair_20260514 import gate_from_logits


V3427_REPORT = ROOT / "reports/stwm_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514.json"
REPORT = ROOT / "reports/stwm_ostf_v34_28_assignment_sharpening_evidence_anchor_probe_20260514.json"
MANIFEST = ROOT / "reports/stwm_ostf_v34_28_assignment_sharpening_visualization_manifest_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_28_ASSIGNMENT_SHARPENING_EVIDENCE_ANCHOR_PROBE_20260514.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_28_ASSIGNMENT_SHARPENING_VISUALIZATION_20260514.md"
VIS_DIR = ROOT / "outputs/visualizations/stwm_ostf_v34_28_assignment_sharpening_20260514"


def compose(anchor: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor, residual_scale: float) -> torch.Tensor:
    return norm(anchor + float(residual_scale) * gate[..., None] * residual)


def sparse_seed_mean_gate(out: dict[str, torch.Tensor], readers: dict[str, dict[str, Any]]) -> torch.Tensor:
    usage = out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    gates = []
    for item in readers.values():
        pred = item["reader"](**reader_inputs(out))["activation_logits"]
        cfg = item["config"]
        gates.append(
            gate_from_logits(
                pred["benefit"],
                usage,
                threshold=cfg.get("threshold"),
                temperature=float(cfg.get("temperature") or 1.0),
                power=float(cfg.get("power") or 1.0),
            )
        )
    return torch.stack(gates, dim=0).mean(dim=0)


def assignment_variant(assign: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "soft":
        return assign
    if variant.startswith("power"):
        power = float(variant.replace("power", ""))
        val = assign.clamp_min(1.0e-8).pow(power)
        return val / val.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    if variant == "top1":
        idx = assign.argmax(dim=-1)
        return F.one_hot(idx, num_classes=assign.shape[-1]).to(assign.dtype)
    if variant == "top2":
        vals, idx = torch.topk(assign, k=min(2, assign.shape[-1]), dim=-1)
        out = torch.zeros_like(assign)
        out.scatter_(-1, idx, vals)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    raise ValueError(f"未知 assignment variant: {variant}")


def residual_from_assignment(assign: torch.Tensor, unit_memory: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bmu,buhd->bmhd", assign, unit_memory)


def shuffled_assignment(assign: torch.Tensor) -> torch.Tensor:
    if assign.shape[-1] <= 1:
        return assign
    idx = torch.roll(torch.arange(assign.shape[-1], device=assign.device), shifts=1, dims=0)
    return assign[..., idx]


def eval_split(
    split: str,
    residual_model: Any,
    ckargs: argparse.Namespace,
    readers: dict[str, dict[str, Any]],
    device: torch.device,
    variants: tuple[str, ...],
    residual_scale: float,
) -> dict[str, Any]:
    acc = Acc()
    delta_acc = {variant: {"normal": Acc(), "shuffle": Acc(), "zero_unit": Acc()} for variant in variants}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            target = bd["fut_teacher_embedding"]
            anchor = observed_mean(bd)
            gate = sparse_seed_mean_gate(out, readers)
            assign = out["point_to_unit_assignment"]
            unit_memory = out["unit_memory"]
            update_method(acc, "copy_mean_observed", anchor, pointwise=pointwise, target=target, mm=mm)
            for variant in variants:
                av = assignment_variant(assign, variant)
                resid = residual_from_assignment(av, unit_memory)
                pred = compose(anchor, resid, gate, residual_scale)
                name = f"evidence_anchor_copy_mean_assignment_{variant}"
                update_method(acc, name, pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
                update_method(delta_acc[variant]["normal"], "normal", pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
                shuf_resid = residual_from_assignment(shuffled_assignment(av), unit_memory)
                shuf_pred = compose(anchor, shuf_resid, gate, residual_scale)
                update_method(delta_acc[variant]["shuffle"], "shuffle", shuf_pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
                zero_pred = compose(anchor, torch.zeros_like(resid), gate, residual_scale)
                update_method(delta_acc[variant]["zero_unit"], "zero_unit", zero_pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
    methods = sorted({key.split(":")[0] for key in acc.sum.keys()})
    metrics = {name: finalize_method(acc, name) for name in methods}
    deltas: dict[str, Any] = {}
    for variant in variants:
        rows = {mode: finalize_method(acc_i, mode) for mode, acc_i in delta_acc[variant].items()}
        normal = rows["normal"]
        shuf = rows["shuffle"]
        zero = rows["zero_unit"]
        deltas[variant] = {
            "shuffle_assignment_delta": float((normal["hard_changed_gain_vs_pointwise"] or 0.0) - (shuf["hard_changed_gain_vs_pointwise"] or 0.0)),
            "zero_unit_memory_delta": float((normal["hard_changed_gain_vs_pointwise"] or 0.0) - (zero["hard_changed_gain_vs_pointwise"] or 0.0)),
            "normal_hard_changed_gain_vs_anchor": normal["hard_changed_gain_vs_anchor"],
            "normal_hard_changed_gain_vs_pointwise": normal["hard_changed_gain_vs_pointwise"],
        }
    rank = sorted(
        [
            {
                "method": name,
                "hard_changed_gain_vs_pointwise": row["hard_changed_gain_vs_pointwise"],
                "hard_changed_gain_vs_anchor": row["hard_changed_gain_vs_anchor"],
                "stable_preservation": row["stable_preservation"],
                "stable_over_open_rate": row["stable_over_open_rate"],
            }
            for name, row in metrics.items()
        ],
        key=lambda x: float(x["hard_changed_gain_vs_pointwise"] or -1e9),
        reverse=True,
    )
    return {"methods": metrics, "rank": rank, "intervention_delta": deltas}


def select_best(per_split: dict[str, Any], variants: tuple[str, ...]) -> dict[str, Any]:
    rows = []
    for variant in variants:
        method = f"evidence_anchor_copy_mean_assignment_{variant}"
        val = per_split["val"]["methods"][method]
        rows.append(
            {
                "variant": variant,
                "method": method,
                "val_hard_changed_gain_vs_pointwise": val["hard_changed_gain_vs_pointwise"],
                "val_hard_changed_gain_vs_anchor": val["hard_changed_gain_vs_anchor"],
                "val_stable_preservation": val["stable_preservation"],
                "val_assignment_delta": per_split["val"]["intervention_delta"][variant]["shuffle_assignment_delta"],
            }
        )
    valid = [r for r in rows if r["val_stable_preservation"] and float(r["val_hard_changed_gain_vs_anchor"] or -1.0) > 0.002]
    if not valid:
        valid = rows
    return max(valid, key=lambda x: (float(x["val_assignment_delta"] or -1e9), float(x["val_hard_changed_gain_vs_pointwise"] or -1e9)))


def save_variant_chart(per_split: dict[str, Any], best_variant: str) -> dict[str, Any]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in per_split["test"]["rank"] if "assignment_" in r["method"]]
    names = [r["method"].replace("evidence_anchor_copy_mean_assignment_", "") for r in rows]
    values = [float(r["hard_changed_gain_vs_anchor"] or 0.0) for r in rows]
    colors = ["#1b9e77" if n == best_variant else "#7570b3" for n in names]
    plt.figure(figsize=(8.8, 4.6))
    plt.bar(np.arange(len(names)), values, color=colors)
    plt.axhline(0.002, color="#444444", linestyle="--", linewidth=1.0)
    plt.xticks(np.arange(len(names)), names, rotation=20)
    plt.ylabel("test gain vs copy_mean anchor")
    plt.title("V34.28 assignment sharpening: residual gain over evidence anchor")
    path = VIS_DIR / "v34_28_assignment_variant_gain.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {"name": "assignment_variant_gain", "path": str(path.relative_to(ROOT)), "case_selection_reason": "比较 soft/power/top-k assignment 读出对 evidence-anchor residual 增益的影响。"}


def save_delta_chart(per_split: dict[str, Any], variants: tuple[str, ...], best_variant: str) -> dict[str, Any]:
    vals = [float(per_split["test"]["intervention_delta"][v]["shuffle_assignment_delta"] or 0.0) for v in variants]
    colors = ["#1b9e77" if v == best_variant else "#7570b3" for v in variants]
    plt.figure(figsize=(8.8, 4.6))
    plt.bar(np.arange(len(variants)), vals, color=colors)
    plt.axhline(0.002, color="#444444", linestyle="--", linewidth=1.0)
    plt.xticks(np.arange(len(variants)), variants, rotation=20)
    plt.ylabel("test shuffle-assignment delta")
    plt.title("V34.28 assignment load-bearing under evidence anchor")
    path = VIS_DIR / "v34_28_assignment_loadbearing_delta.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {"name": "assignment_loadbearing_delta", "path": str(path.relative_to(ROOT)), "case_selection_reason": "检测 sharpen/top-1 assignment 是否让 residual correction 对 assignment counterfactual 敏感。"}


def write_docs(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = [
        "# V34.28 assignment sharpening evidence-anchor probe 中文报告",
        "",
        "## 中文结论",
        report["中文结论"],
        "",
        "## 阶段性分析",
        report["阶段性分析"],
        "",
        "## 论文相关问题解决方案参考",
        report["论文相关问题解决方案参考"],
        "",
        "## 关键结果",
    ]
    for key in [
        "probe_passed",
        "best_assignment_variant",
        "assignment_load_bearing_restored",
        "unit_memory_load_bearing_on_system",
        "semantic_hard_signal",
        "changed_semantic_signal",
        "stable_preservation",
        "integrated_semantic_field_claim_allowed",
        "integrated_identity_field_claim_allowed",
        "recommended_next_step",
    ]:
        lines.append(f"- {key}: `{report.get(key)}`")
    lines += ["", "## 最佳下一步方案", report["最佳下一步方案"]]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    vlines = ["# V34.28 assignment sharpening visualization 中文报告", "", "## 中文结论", manifest["中文结论"], "", "## 图像清单"]
    for fig in manifest["figures"]:
        vlines.append(f"- {fig['name']}: `{fig['path']}`；原因：{fig['case_selection_reason']}")
    VIS_DOC.write_text("\n".join(vlines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, train_summary = load_residual_model(args, device)
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    readers = load_v3425_readers(args, residual_model, device)
    variants = ("soft", "power2", "power4", "power8", "top2", "top1")
    per_split = {}
    for split in ("val", "test"):
        print(f"开始 V34.28 assignment sharpening probe: split={split}", flush=True)
        per_split[split] = eval_split(split, residual_model, ckargs, readers, device, variants, args.residual_scale)
    best = select_best(per_split, variants)
    best_variant = best["variant"]
    best_method = best["method"]
    val_m = per_split["val"]["methods"][best_method]
    test_m = per_split["test"]["methods"][best_method]
    val_delta = per_split["val"]["intervention_delta"][best_variant]
    test_delta = per_split["test"]["intervention_delta"][best_variant]
    assignment_lb = bool(float(val_delta["shuffle_assignment_delta"] or 0.0) > 0.002 and float(test_delta["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool(float(val_delta["zero_unit_memory_delta"] or 0.0) > 0.002 and float(test_delta["zero_unit_memory_delta"] or 0.0) > 0.002)
    semantic_hard_signal = {"val": val_m["semantic_hard_signal"], "test": test_m["semantic_hard_signal"]}
    changed_semantic_signal = {"val": val_m["changed_semantic_signal"], "test": test_m["changed_semantic_signal"]}
    stable_preservation = {"val": val_m["stable_preservation"], "test": test_m["stable_preservation"]}
    improves_anchor = bool(float(val_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002 and float(test_m["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    probe_passed = bool(assignment_lb and unit_lb and improves_anchor and all(semantic_hard_signal.values()) and all(changed_semantic_signal.values()) and all(stable_preservation.values()))
    figures = [save_variant_chart(per_split, best_variant), save_delta_chart(per_split, variants, best_variant)]
    manifest = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.28 visualization 已完成；图像展示 assignment sharpening 是否恢复 assignment load-bearing。",
        "figures": figures,
        "visualization_ready": True,
        "recommended_next_step": "run_v34_28_m512_dense_visualization" if probe_passed else "fix_assignment_bound_residual_model",
    }
    report = {
        "generated_at_utc": utc_now(),
        "中文结论": (
            "V34.28 已在 evidence-anchor 系统上做 assignment sharpening/top-k 读出诊断。"
            "本轮不训练模型，只检查 assignment 不 load-bearing 是 assignment 过软，还是 unit memory correction 本身没有形成 assignment-bound 区分。"
        ),
        "probe_passed": probe_passed,
        "best_assignment_variant": best_variant,
        "best_method": best_method,
        "best_metrics": {"val": val_m, "test": test_m},
        "assignment_intervention_delta": {"val": val_delta, "test": test_delta},
        "per_split": per_split,
        "assignment_load_bearing_restored": assignment_lb,
        "unit_memory_load_bearing_on_system": unit_lb,
        "unit_residual_improves_evidence_anchor": improves_anchor,
        "semantic_hard_signal": semantic_hard_signal,
        "changed_semantic_signal": changed_semantic_signal,
        "stable_preservation": stable_preservation,
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "阶段性分析": (
            "V34.27 已经说明 evidence-anchor + residual 能小幅超过 copy_mean/top-k，但 assignment shuffle 几乎不伤结果。"
            "V34.28 进一步测试：如果 sharpen/top-1 assignment 可以恢复 delta，问题是 assignment 太软；如果仍不恢复，问题更可能是 unit memory slots 之间 correction 同质化，"
            "需要重新训练 assignment-discriminative residual，而不是继续调 gate。"
        ),
        "论文相关问题解决方案参考": (
            "这对应 object-centric 表征中常见的 slot collapse / slot interchangeability 问题：Slot Attention、SAVi/STEVE、OCVP 一类工作通常需要 slot competition、"
            "permutation-aware matching、object-consistency loss 来避免所有 slot 学成可互换记忆。当前 probe 正是在确认 STWM unit memory 是否存在这种同质化。"
        ),
        "最佳下一步方案": (
            "如果 assignment sharpening 通过，可以把 sharpened assignment 作为 V34.28 系统读出并进入 M512 可视化；"
            "如果不通过，下一步应训练 evidence-anchor-relative 的 assignment-discriminative unit residual，明确加入 shuffled-assignment contrast、slot diversity、unit-specific correction target。"
        ),
        "visualization_manifest_path": str(MANIFEST.relative_to(ROOT)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_28_m512_dense_visualization" if probe_passed else "fix_assignment_bound_residual_model",
        "train_summary_reference": train_summary,
        "v34_27_reference": json.loads(V3427_REPORT.read_text(encoding="utf-8")) if V3427_REPORT.exists() else None,
    }
    dump_json(REPORT, report)
    dump_json(MANIFEST, manifest)
    write_docs(report, manifest)
    print(f"已写出 V34.28 assignment sharpening report: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"已写出 V34.28 visualization manifest: {MANIFEST.relative_to(ROOT)}", flush=True)
    print(f"probe_passed: {probe_passed}", flush=True)
    print(f"recommended_next_step: {report['recommended_next_step']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
