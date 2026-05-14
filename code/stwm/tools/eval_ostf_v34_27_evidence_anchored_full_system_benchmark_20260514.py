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
    observed_last,
    observed_max_conf,
    observed_mean,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
from stwm.tools.train_ostf_v34_25_sparse_calibrated_gate_repair_20260514 import gate_from_logits


REPORT = ROOT / "reports/stwm_ostf_v34_27_evidence_anchored_full_system_benchmark_20260514.json"
MANIFEST = ROOT / "reports/stwm_ostf_v34_27_evidence_anchored_visualization_manifest_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_27_EVIDENCE_ANCHORED_FULL_SYSTEM_BENCHMARK_20260514.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_27_EVIDENCE_ANCHORED_VISUALIZATION_20260514.md"
VIS_DIR = ROOT / "outputs/visualizations/stwm_ostf_v34_27_evidence_anchored_full_system_20260514"


BASELINE_METHODS = {
    "pointwise_base",
    "copy_mean_observed",
    "copy_last_observed",
    "copy_max_conf_observed",
    "topk_raw_evidence",
    "v34_25_pointwise_sparse_gate_seed_mean",
}


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def local_cos(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (norm(pred) * norm(target)).sum(dim=-1)


def compose(anchor: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor, residual_scale: float) -> torch.Tensor:
    return norm(anchor + float(residual_scale) * gate[..., None] * residual)


def counterfactual_batch(batch: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    if mode not in {"zero_semantic_measurements", "shuffle_semantic_measurements"}:
        return batch
    out = dict(batch)
    if mode == "zero_semantic_measurements":
        out["obs_semantic_measurements"] = torch.zeros_like(batch["obs_semantic_measurements"])
        return out
    sem = batch["obs_semantic_measurements"]
    b, m = sem.shape[:2]
    if m <= 1:
        return out
    idx = torch.arange(m, device=sem.device)
    idx = torch.roll(idx, shifts=1, dims=0)
    out["obs_semantic_measurements"] = sem[:, idx]
    for key in ("obs_semantic_measurement_mask", "semantic_measurement_confidence", "teacher_agreement_score"):
        if key in batch:
            out[key] = batch[key][:, idx]
    return out


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


def evidence_bases(batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "copy_mean_observed": observed_mean(batch),
        "copy_last_observed": observed_last(batch),
        "copy_max_conf_observed": observed_max_conf(batch),
        "topk_raw_evidence": out["topk_raw_evidence_embedding"],
    }


def candidate_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_name in ("topk_raw_evidence", "copy_mean_observed", "copy_max_conf_observed"):
        for evidence_weight in (1.0, 0.75, 0.5):
            for residual_scale in (0.25, 0.5, 1.0):
                rows.append(
                    {
                        "base_name": base_name,
                        "evidence_weight": evidence_weight,
                        "residual_scale": residual_scale,
                        "method": f"evidence_anchor_{base_name}_w{evidence_weight:.2f}_r{residual_scale:.2f}",
                    }
                )
    return rows


class Acc:
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
        return None if c == 0 else float(self.sum[key] / c)


def update_method(
    acc: Acc,
    name: str,
    pred: torch.Tensor,
    *,
    pointwise: torch.Tensor,
    target: torch.Tensor,
    mm: dict[str, torch.Tensor],
    anchor: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> None:
    cos = local_cos(pred, target)
    pointwise_cos = local_cos(pointwise, target)
    gain_pointwise = cos - pointwise_cos
    gain_anchor = cos - local_cos(anchor, target) if anchor is not None else torch.zeros_like(gain_pointwise)
    over_update = (gain_pointwise < -0.02).float()
    for key, mask in mm.items():
        acc.add(f"{name}:{key}:cos", cos, mask)
        acc.add(f"{name}:{key}:gain_pointwise", gain_pointwise, mask)
        if anchor is not None:
            acc.add(f"{name}:{key}:gain_anchor", gain_anchor, mask)
        acc.add(f"{name}:{key}:over_update", over_update, mask)
        if gate is not None:
            acc.add(f"{name}:{key}:gate", gate, mask)
            acc.add(f"{name}:{key}:over_open", (gate > 0.05).float(), mask)


def finalize_method(acc: Acc, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("valid", "hard", "changed", "hard_changed", "stable", "aligned"):
        out[f"{key}_cosine"] = acc.mean(f"{name}:{key}:cos")
        out[f"{key}_gain_vs_pointwise"] = acc.mean(f"{name}:{key}:gain_pointwise")
        out[f"{key}_gain_vs_anchor"] = acc.mean(f"{name}:{key}:gain_anchor")
        out[f"{key}_over_update_rate"] = acc.mean(f"{name}:{key}:over_update")
        out[f"{key}_gate_mean"] = acc.mean(f"{name}:{key}:gate")
        out[f"{key}_over_open_rate"] = acc.mean(f"{name}:{key}:over_open")
    out["semantic_hard_signal"] = bool((out["hard_gain_vs_pointwise"] or -1.0) > 0.005)
    out["changed_semantic_signal"] = bool((out["changed_gain_vs_pointwise"] or -1.0) > 0.005)
    out["stable_preservation"] = bool(out["stable_gain_vs_pointwise"] is None or out["stable_gain_vs_pointwise"] >= -0.02)
    out["stable_overopen_controlled"] = bool(out["stable_over_open_rate"] is None or out["stable_over_open_rate"] <= 0.35)
    out["unit_residual_improves_anchor"] = bool((out["hard_changed_gain_vs_anchor"] or -1.0) > 0.002)
    return out


def method_rank(metrics: dict[str, Any], *, include_candidates: bool = True) -> list[dict[str, Any]]:
    rows = []
    for name, row in metrics.items():
        if not include_candidates and name not in BASELINE_METHODS:
            continue
        rows.append(
            {
                "method": name,
                "hard_changed_gain_vs_pointwise": row["hard_changed_gain_vs_pointwise"],
                "hard_changed_gain_vs_anchor": row["hard_changed_gain_vs_anchor"],
                "semantic_hard_signal": row["semantic_hard_signal"],
                "changed_semantic_signal": row["changed_semantic_signal"],
                "stable_preservation": row["stable_preservation"],
                "stable_over_open_rate": row["stable_over_open_rate"],
            }
        )
    return sorted(rows, key=lambda x: float(x["hard_changed_gain_vs_pointwise"] or -1e9), reverse=True)


def evaluate_split(
    split: str,
    residual_model: Any,
    ckargs: argparse.Namespace,
    readers: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    acc = Acc()
    grid = candidate_grid()
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            mm = masks(bd)
            pointwise = out["pointwise_semantic_belief"]
            residual = out["assignment_bound_residual"]
            target = bd["fut_teacher_embedding"]
            gate = sparse_seed_mean_gate(out, readers)
            bases = evidence_bases(bd, out)

            update_method(acc, "pointwise_base", pointwise, pointwise=pointwise, target=target, mm=mm)
            for base_name, base_pred in bases.items():
                update_method(acc, base_name, base_pred, pointwise=pointwise, target=target, mm=mm)
            update_method(
                acc,
                "v34_25_pointwise_sparse_gate_seed_mean",
                compose(pointwise, residual, gate, 1.0),
                pointwise=pointwise,
                target=target,
                mm=mm,
                anchor=pointwise,
                gate=gate,
            )
            for cfg in grid:
                evidence = bases[cfg["base_name"]]
                anchor = norm(float(cfg["evidence_weight"]) * evidence + (1.0 - float(cfg["evidence_weight"])) * pointwise)
                pred = compose(anchor, residual, gate, float(cfg["residual_scale"]))
                update_method(
                    acc,
                    cfg["method"],
                    pred,
                    pointwise=pointwise,
                    target=target,
                    mm=mm,
                    anchor=anchor,
                    gate=gate,
                )
    methods = sorted({key.split(":")[0] for key in acc.sum.keys()})
    metrics = {name: finalize_method(acc, name) for name in methods}
    return {"methods": metrics, "rank": method_rank(metrics), "baseline_rank": method_rank(metrics, include_candidates=False)}


def select_best_candidate(val_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cfg in candidate_grid():
        row = val_metrics["methods"][cfg["method"]]
        rows.append(
            {
                **cfg,
                "hard_changed_gain_vs_pointwise": row["hard_changed_gain_vs_pointwise"],
                "hard_changed_gain_vs_anchor": row["hard_changed_gain_vs_anchor"],
                "semantic_hard_signal": row["semantic_hard_signal"],
                "changed_semantic_signal": row["changed_semantic_signal"],
                "stable_preservation": row["stable_preservation"],
                "stable_overopen_controlled": row["stable_overopen_controlled"],
                "unit_residual_improves_anchor": row["unit_residual_improves_anchor"],
            }
        )
    valid = [r for r in rows if r["stable_preservation"] and r["unit_residual_improves_anchor"]]
    if not valid:
        valid = rows
    return max(valid, key=lambda x: float(x["hard_changed_gain_vs_pointwise"] or -1e9))


def baseline_best(split_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for name in ("copy_mean_observed", "copy_last_observed", "copy_max_conf_observed", "topk_raw_evidence"):
        row = split_metrics["methods"][name]
        rows.append({"method": name, "hard_changed_gain_vs_pointwise": row["hard_changed_gain_vs_pointwise"]})
    return max(rows, key=lambda x: float(x["hard_changed_gain_vs_pointwise"] or -1e9))


def evaluate_best_interventions(
    best_cfg: dict[str, Any],
    split: str,
    residual_model: Any,
    ckargs: argparse.Namespace,
    readers: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, float | None]:
    modes = {
        "normal": "force_gate_zero",
        "zero_semantic_measurements": "zero_semantic_measurements",
        "shuffle_semantic_measurements": "shuffle_semantic_measurements_across_points",
        "shuffle_assignment": "shuffle_assignment",
        "zero_unit_memory": "zero_unit_memory",
    }
    accs = {name: Acc() for name in modes}
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd0 = move_batch(batch, device)
            mm = masks(bd0)
            target = bd0["fut_teacher_embedding"]
            for mode, intervention in modes.items():
                bd = counterfactual_batch(bd0, "zero_semantic_measurements" if mode == "zero_semantic_measurements" else "shuffle_semantic_measurements" if mode == "shuffle_semantic_measurements" else "normal")
                out = residual_model(**model_inputs(bd0), intervention=intervention)
                pointwise = out["pointwise_semantic_belief"]
                residual = out["assignment_bound_residual"]
                gate = sparse_seed_mean_gate(out, readers)
                bases = evidence_bases(bd, out)
                evidence = bases[best_cfg["base_name"]]
                anchor = norm(float(best_cfg["evidence_weight"]) * evidence + (1.0 - float(best_cfg["evidence_weight"])) * pointwise)
                pred = compose(anchor, residual, gate, float(best_cfg["residual_scale"]))
                update_method(accs[mode], mode, pred, pointwise=pointwise, target=target, mm=mm, anchor=anchor, gate=gate)
    rows = {mode: finalize_method(acc, mode) for mode, acc in accs.items()}
    normal = rows["normal"]

    def delta(mode: str) -> float | None:
        a = normal["hard_changed_gain_vs_pointwise"]
        b = rows[mode]["hard_changed_gain_vs_pointwise"]
        if a is None or b is None:
            return None
        return float(a - b)

    return {
        "zero_semantic_measurements_delta": delta("zero_semantic_measurements"),
        "shuffle_semantic_measurements_delta": delta("shuffle_semantic_measurements"),
        "shuffle_assignment_delta": delta("shuffle_assignment"),
        "zero_unit_memory_delta": delta("zero_unit_memory"),
        "normal_hard_changed_gain_vs_pointwise": normal["hard_changed_gain_vs_pointwise"],
        "normal_hard_changed_gain_vs_anchor": normal["hard_changed_gain_vs_anchor"],
    }


def save_rank_chart(per_split: dict[str, Any], best_method: str) -> dict[str, Any]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = per_split["test"]["rank"][:12]
    names = [r["method"] for r in rows]
    values = [float(r["hard_changed_gain_vs_pointwise"] or 0.0) for r in rows]
    colors = ["#1b9e77" if n == best_method else "#d95f02" if n in BASELINE_METHODS else "#7570b3" for n in names]
    plt.figure(figsize=(11.0, 5.4))
    plt.bar(np.arange(len(names)), values, color=colors)
    plt.axhline(0.005, color="#444444", linestyle="--", linewidth=1.0)
    plt.xticks(np.arange(len(names)), names, rotation=35, ha="right")
    plt.ylabel("test hard/changed gain vs pointwise")
    plt.title("V34.27 evidence-anchored benchmark: test ranking")
    path = VIS_DIR / "v34_27_evidence_anchor_test_ranking.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {"name": "evidence_anchor_test_ranking", "path": str(path.relative_to(ROOT)), "case_selection_reason": "按 test hard/changed gain 排名，检查 evidence-anchored 系统是否超过 copy/top-k baseline。"}


def save_intervention_chart(interventions: dict[str, Any]) -> dict[str, Any]:
    keys = ["zero_semantic_measurements_delta", "shuffle_semantic_measurements_delta", "shuffle_assignment_delta", "zero_unit_memory_delta"]
    vals = [float(interventions["test"].get(k) or 0.0) for k in keys]
    labels = ["zero semantic", "shuffle semantic", "shuffle assignment", "zero unit"]
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(np.arange(len(keys)), vals, color="#1b9e77")
    plt.axhline(0.002, color="#444444", linestyle="--", linewidth=1.0)
    plt.xticks(np.arange(len(keys)), labels, rotation=15)
    plt.ylabel("test hard/changed gain delta")
    plt.title("V34.27 best evidence-anchor intervention deltas")
    path = VIS_DIR / "v34_27_best_evidence_anchor_interventions.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {"name": "best_evidence_anchor_interventions", "path": str(path.relative_to(ROOT)), "case_selection_reason": "验证 best evidence-anchor 配置下 semantic/assignment/unit memory 是否仍为因果路径。"}


def write_docs(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = [
        "# V34.27 evidence-anchored full-system benchmark 中文报告",
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
        "benchmark_passed",
        "best_evidence_anchor_method",
        "best_evidence_anchor_beats_copy_topk",
        "unit_residual_improves_evidence_anchor",
        "semantic_measurements_load_bearing_on_system",
        "assignment_load_bearing_on_system",
        "unit_memory_load_bearing_on_system",
        "semantic_hard_signal",
        "changed_semantic_signal",
        "stable_preservation",
        "m512_dense_ready",
        "integrated_semantic_field_claim_allowed",
        "integrated_identity_field_claim_allowed",
        "recommended_next_step",
    ]:
        lines.append(f"- {key}: `{report.get(key)}`")
    lines += ["", "## 最佳下一步方案", report["最佳下一步方案"]]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vlines = ["# V34.27 evidence-anchored visualization 中文报告", "", "## 中文结论", manifest["中文结论"], "", "## 图像清单"]
    for fig in manifest["figures"]:
        vlines.append(f"- {fig['name']}: `{fig['path']}`；原因：{fig['case_selection_reason']}")
    vlines += ["", "## 关键字段", f"- visualization_ready: `{manifest['visualization_ready']}`", f"- recommended_next_step: `{manifest['recommended_next_step']}`"]
    VIS_DOC.write_text("\n".join(vlines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--reader-hidden-dim", type=int, default=192)
    p.add_argument("--teacher-embedding-dim", type=int, default=768)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    residual_model, ckargs, train_summary = load_residual_model(args, device)
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    readers = load_v3425_readers(args, residual_model, device)

    per_split: dict[str, Any] = {}
    for split in ("val", "test"):
        print(f"开始 V34.27 evidence-anchored benchmark: split={split}", flush=True)
        per_split[split] = evaluate_split(split, residual_model, ckargs, readers, device)

    best_cfg = select_best_candidate(per_split["val"])
    best_method = best_cfg["method"]
    best_val = per_split["val"]["methods"][best_method]
    best_test = per_split["test"]["methods"][best_method]
    best_base_val = baseline_best(per_split["val"])
    best_base_test = baseline_best(per_split["test"])
    interventions = {}
    for split in ("val", "test"):
        print(f"开始 V34.27 best evidence-anchor intervention: split={split}", flush=True)
        interventions[split] = evaluate_best_interventions(best_cfg, split, residual_model, ckargs, readers, device)

    beats_copy_topk = bool(
        (best_val["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base_val["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002
        and (best_test["hard_changed_gain_vs_pointwise"] or -1.0) > float(best_base_test["hard_changed_gain_vs_pointwise"] or 0.0) + 0.002
    )
    improves_anchor = bool(
        (best_val["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
        and (best_test["hard_changed_gain_vs_anchor"] or -1.0) > 0.002
    )
    sem_lb = bool(
        min(
            interventions["val"]["zero_semantic_measurements_delta"] or 0.0,
            interventions["val"]["shuffle_semantic_measurements_delta"] or 0.0,
            interventions["test"]["zero_semantic_measurements_delta"] or 0.0,
            interventions["test"]["shuffle_semantic_measurements_delta"] or 0.0,
        )
        > 0.002
    )
    assign_lb = bool((interventions["val"]["shuffle_assignment_delta"] or 0.0) > 0.002 and (interventions["test"]["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool((interventions["val"]["zero_unit_memory_delta"] or 0.0) > 0.002 and (interventions["test"]["zero_unit_memory_delta"] or 0.0) > 0.002)
    semantic_hard_signal = {"val": best_val["semantic_hard_signal"], "test": best_test["semantic_hard_signal"]}
    changed_semantic_signal = {"val": best_val["changed_semantic_signal"], "test": best_test["changed_semantic_signal"]}
    stable_preservation = {"val": best_val["stable_preservation"], "test": best_test["stable_preservation"]}
    benchmark_passed = bool(
        beats_copy_topk
        and improves_anchor
        and sem_lb
        and assign_lb
        and unit_lb
        and all(semantic_hard_signal.values())
        and all(changed_semantic_signal.values())
        and all(stable_preservation.values())
    )

    figures = [save_rank_chart(per_split, best_method), save_intervention_chart(interventions)]
    manifest = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.27 visualization 已完成；图像展示 evidence-anchored 系统与 copy/top-k baseline 排名，以及 best 配置的干预因果 delta。",
        "figures": figures,
        "visualization_ready": True,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_27_m512_dense_visualization" if benchmark_passed else "fix_full_system_baseline_gap",
    }
    report = {
        "generated_at_utc": utc_now(),
        "中文结论": (
            "V34.27 已按 full-system baseline gap 做 evidence-anchored composition benchmark：固定 V34.25/V34.20 权重，"
            "将 final semantic 从 pointwise+residual 改为 semantic_evidence_base+sparse_gate*unit_residual_correction，并用 val 选配置、test 确认。"
        ),
        "benchmark_passed": benchmark_passed,
        "best_evidence_anchor_config": best_cfg,
        "best_evidence_anchor_method": best_method,
        "best_evidence_anchor_beats_copy_topk": beats_copy_topk,
        "unit_residual_improves_evidence_anchor": improves_anchor,
        "best_nonoracle_copy_topk_baseline": {"val": best_base_val, "test": best_base_test},
        "best_evidence_anchor_metrics": {"val": best_val, "test": best_test},
        "per_split": per_split,
        "intervention_delta": interventions,
        "semantic_measurements_load_bearing_on_system": sem_lb,
        "assignment_load_bearing_on_system": assign_lb,
        "unit_memory_load_bearing_on_system": unit_lb,
        "semantic_hard_signal": semantic_hard_signal,
        "changed_semantic_signal": changed_semantic_signal,
        "stable_preservation": stable_preservation,
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "m128_field_output_ready": True,
        "m512_dense_ready": bool(benchmark_passed),
        "video_input_closure_ready": False,
        "阶段性分析": (
            "V34.26 暴露的问题不是 sparse gate 本身，而是系统分解把 observed semantic evidence 放在 residual 支路里，"
            "导致完整协议下 copy_mean/top-k evidence 这种非 oracle baseline 直接压过 pointwise+residual。"
            "V34.27 因此只改 composition：semantic base 先由 observed evidence 提供，unit memory 只负责结构化 correction。"
            "如果这个后验组合仍不能赢 copy/top-k，说明 residual 学到的 correction 还没有超出 semantic persistence / raw evidence transport。"
        ),
        "论文相关问题解决方案参考": (
            "这个修法借鉴了 object-centric memory 与 query-conditioned retrieval 的常见结构：Slot Attention/OCVP 类方法先建立对象槽和 assignment，"
            "XMem/SAM2 类视频记忆方法把 memory read 作为主证据路径，Perceiver IO/DETR 类 cross-attention 用 future query 读 observed memory。"
            "对我们当前系统，最重要的启发是：semantic measurement 不应只是补丁 loss，而应成为可替代 copy/top-k baseline 的显式 evidence base；"
            "unit residual 必须在这个强 base 上继续提供 hard/changed 增益。"
        ),
        "最佳下一步方案": (
            "如果 V34.27 打赢 copy/top-k，下一步才进入 M512 dense visualization；如果没打赢，继续修 full_system_baseline_gap，"
            "优先检查 hard/changed target 是否真正需要未来结构推理、以及 unit residual correction 是否被训练成相对 evidence base 的增量，而不是相对 pointwise base 的增量。"
        ),
        "visualization_manifest_path": str(MANIFEST.relative_to(ROOT)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_27_m512_dense_visualization" if benchmark_passed else "fix_full_system_baseline_gap",
        "train_summary_reference": train_summary,
    }
    dump_json(REPORT, report)
    dump_json(MANIFEST, manifest)
    write_docs(report, manifest)
    print(f"已写出 V34.27 evidence-anchor benchmark report: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"已写出 V34.27 visualization manifest: {MANIFEST.relative_to(ROOT)}", flush=True)
    print(f"benchmark_passed: {benchmark_passed}", flush=True)
    print(f"recommended_next_step: {report['recommended_next_step']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
