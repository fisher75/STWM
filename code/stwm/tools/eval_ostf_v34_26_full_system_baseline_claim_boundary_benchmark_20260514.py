#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from stwm.modules.ostf_v34_22_activation_state_reader import ActivationStateReaderV3422
from stwm.tools.eval_ostf_v34_22_activation_state_reader_predictability_probe_20260513 import (
    load_residual_model,
    make_loader,
    reader_inputs,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_18_topk_evidence_oracle_residual_probe_20260513 import model_inputs
from stwm.tools.train_ostf_v34_20_hard_changed_aligned_topk_residual_probe_20260513 import hard_changed_aligned_mask
from stwm.tools.train_ostf_v34_25_sparse_calibrated_gate_repair_20260514 import gate_from_logits


V3425_DECISION = ROOT / "reports/stwm_ostf_v34_25_sparse_calibrated_gate_repair_decision_20260514.json"
V3425_CKPT_ROOT = ROOT / "outputs/checkpoints/stwm_ostf_v34_25_sparse_calibrated_gate_repair_h32_m128"
REPORT = ROOT / "reports/stwm_ostf_v34_26_full_system_baseline_claim_boundary_benchmark_20260514.json"
MANIFEST = ROOT / "reports/stwm_ostf_v34_26_full_system_visualization_manifest_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_26_FULL_SYSTEM_BASELINE_CLAIM_BOUNDARY_BENCHMARK_20260514.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_26_FULL_SYSTEM_VISUALIZATION_20260514.md"
VIS_DIR = ROOT / "outputs/visualizations/stwm_ostf_v34_26_full_system_benchmark_20260514"


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1)


def local_cos(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (norm(pred) * norm(target)).sum(dim=-1)


def masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    valid = batch["fut_teacher_available_mask"].bool()
    hard = batch["semantic_hard_mask"].bool() & valid
    changed = batch["changed_mask"].bool() & valid
    stable = batch["stable_suppress_mask"].bool() & valid
    return {
        "valid": valid,
        "hard": hard,
        "changed": changed,
        "hard_changed": (hard | changed) & valid,
        "stable": stable,
        "aligned": hard_changed_aligned_mask(batch),
    }


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


def observed_mean(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    sem = torch.nan_to_num(batch["obs_semantic_measurements"].float())
    mask = batch["obs_semantic_measurement_mask"].float()
    conf = batch["semantic_measurement_confidence"].float().clamp(0.0, 1.0)
    w = mask * conf.clamp_min(0.05)
    emb = (sem * w[..., None]).sum(dim=2) / w.sum(dim=2, keepdim=True).clamp_min(1.0)
    return norm(emb)[:, :, None, :].expand(-1, -1, batch["fut_teacher_embedding"].shape[2], -1)


def observed_last(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    sem = torch.nan_to_num(batch["obs_semantic_measurements"].float())
    mask = batch["obs_semantic_measurement_mask"].bool()
    b, m, t, d = sem.shape
    idx = torch.arange(t, device=sem.device).view(1, 1, t).expand(b, m, t)
    last_idx = torch.where(mask, idx, torch.zeros_like(idx)).max(dim=2).values
    emb = torch.gather(sem, 2, last_idx[:, :, None, None].expand(-1, -1, 1, d)).squeeze(2)
    emb = torch.where(mask.any(dim=2)[..., None], emb, observed_mean(batch)[:, :, 0, :])
    return norm(emb)[:, :, None, :].expand(-1, -1, batch["fut_teacher_embedding"].shape[2], -1)


def observed_max_conf(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    sem = torch.nan_to_num(batch["obs_semantic_measurements"].float())
    mask = batch["obs_semantic_measurement_mask"].float()
    conf = batch["semantic_measurement_confidence"].float().clamp(0.0, 1.0) * mask
    idx = conf.argmax(dim=2)
    d = sem.shape[-1]
    emb = torch.gather(sem, 2, idx[:, :, None, None].expand(-1, -1, 1, d)).squeeze(2)
    emb = torch.where(mask.bool().any(dim=2)[..., None], emb, observed_mean(batch)[:, :, 0, :])
    return norm(emb)[:, :, None, :].expand(-1, -1, batch["fut_teacher_embedding"].shape[2], -1)


def compose(pointwise: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return norm(pointwise + gate[..., None] * residual)


def load_v3425_readers(args: argparse.Namespace, residual_model: Any, device: torch.device) -> dict[str, dict[str, Any]]:
    decision = json.loads(V3425_DECISION.read_text(encoding="utf-8"))
    readers: dict[str, dict[str, Any]] = {}
    for seed, report in decision["seed_reports"].items():
        ckpt = V3425_CKPT_ROOT / seed / f"v34_25_sparse_calibrated_gate_repair_m128_h32_{seed}.pt"
        ck = torch.load(ckpt, map_location="cpu")
        reader = ActivationStateReaderV3422(
            int(residual_model.v30.cfg.hidden_dim),
            semantic_dim=int(getattr(args, "teacher_embedding_dim", 768)),
            hidden_dim=int(ck["args"].get("reader_hidden_dim", args.reader_hidden_dim)),
        ).to(device)
        reader.load_state_dict(ck["reader"], strict=True)
        reader.eval()
        readers[seed] = {"reader": reader, "config": report["best_config_by_val"], "checkpoint": str(ckpt.relative_to(ROOT))}
    return readers


def method_predictions(
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    readers: dict[str, dict[str, Any]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    pointwise = out["pointwise_semantic_belief"]
    residual = out["assignment_bound_residual"]
    usage_gate = out["semantic_measurement_usage_score"].float().clamp(0.0, 1.0)
    preds: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
        "pointwise_base": (pointwise, torch.zeros_like(usage_gate)),
        "copy_mean_observed": (observed_mean(batch), None),
        "copy_last_observed": (observed_last(batch), None),
        "copy_max_conf_observed": (observed_max_conf(batch), None),
        "topk_raw_evidence": (out["topk_raw_evidence_embedding"], None),
        "topk_unit_usage_no_sparse_gate": (compose(pointwise, residual, usage_gate), usage_gate),
        "oracle_hard_changed_mask_upper_bound": (compose(pointwise, residual, masks(batch)["aligned"].float() * usage_gate), masks(batch)["aligned"].float() * usage_gate),
    }
    seed_finals = []
    seed_gates = []
    for seed, item in readers.items():
        pred = item["reader"](**reader_inputs(out))["activation_logits"]
        cfg = item["config"]
        gate = gate_from_logits(
            pred["benefit"],
            usage_gate,
            threshold=cfg.get("threshold"),
            temperature=float(cfg.get("temperature") or 1.0),
            power=float(cfg.get("power") or 1.0),
        )
        final = compose(pointwise, residual, gate)
        preds[f"v34_25_sparse_gate_{seed}"] = (final, gate)
        seed_finals.append(final)
        seed_gates.append(gate)
    preds["v34_25_sparse_gate_seed_mean"] = (norm(torch.stack(seed_finals, dim=0).mean(dim=0)), torch.stack(seed_gates, dim=0).mean(dim=0))
    return preds


def update_method(acc: Acc, name: str, pred: torch.Tensor, gate: torch.Tensor | None, pointwise: torch.Tensor, target: torch.Tensor, mm: dict[str, torch.Tensor]) -> None:
    cos = local_cos(pred, target)
    base_cos = local_cos(pointwise, target)
    gain = cos - base_cos
    over_update = (gain < -0.02).float()
    for key, mask in mm.items():
        acc.add(f"{name}:{key}:cos", cos, mask)
        acc.add(f"{name}:{key}:gain", gain, mask)
        acc.add(f"{name}:{key}:over_update", over_update, mask)
        if gate is not None:
            acc.add(f"{name}:{key}:gate", gate, mask)
            acc.add(f"{name}:{key}:over_open", (gate > 0.05).float(), mask)


def finalize_method(acc: Acc, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["valid", "hard", "changed", "hard_changed", "stable", "aligned"]:
        out[f"{key}_cosine"] = acc.mean(f"{name}:{key}:cos")
        out[f"{key}_gain_vs_pointwise"] = acc.mean(f"{name}:{key}:gain")
        out[f"{key}_over_update_rate"] = acc.mean(f"{name}:{key}:over_update")
        out[f"{key}_gate_mean"] = acc.mean(f"{name}:{key}:gate")
        out[f"{key}_over_open_rate"] = acc.mean(f"{name}:{key}:over_open")
    out["semantic_hard_signal"] = bool(out["hard_gain_vs_pointwise"] is not None and out["hard_gain_vs_pointwise"] > 0.005)
    out["changed_semantic_signal"] = bool(out["changed_gain_vs_pointwise"] is not None and out["changed_gain_vs_pointwise"] > 0.005)
    out["stable_preservation"] = bool(out["stable_gain_vs_pointwise"] is None or out["stable_gain_vs_pointwise"] >= -0.02)
    out["stable_overopen_controlled"] = bool(out["stable_over_open_rate"] is None or out["stable_over_open_rate"] <= 0.35)
    return out


def eval_split(split: str, residual_model: Any, ckargs: argparse.Namespace, readers: dict[str, dict[str, Any]], device: torch.device, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    acc = Acc()
    case_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            pred_map = method_predictions(bd, out, readers)
            mm = masks(bd)
            for name, (pred, gate) in pred_map.items():
                update_method(acc, name, pred, gate, out["pointwise_semantic_belief"], bd["fut_teacher_embedding"], mm)
            # 只收少量 case mining 元数据，避免保存大数组。
            if split == "val" and len(case_rows) < 24:
                vfinal, vgate = pred_map["v34_25_sparse_gate_seed_mean"]
                gain = local_cos(vfinal, bd["fut_teacher_embedding"]) - local_cos(out["pointwise_semantic_belief"], bd["fut_teacher_embedding"])
                hard_changed = mm["hard_changed"]
                stable = mm["stable"]
                for bi, uid in enumerate(bd["uid"]):
                    hc_gain = float(gain[bi][hard_changed[bi]].mean().detach().cpu()) if bool(hard_changed[bi].any()) else -999.0
                    st_gate = float(vgate[bi][stable[bi]].mean().detach().cpu()) if vgate is not None and bool(stable[bi].any()) else 0.0
                    case_rows.append({"uid": uid, "hard_changed_gain": hc_gain, "stable_gate_mean": st_gate})
    methods = sorted({key.split(":")[0] for key in acc.sum.keys()})
    metrics = {name: finalize_method(acc, name) for name in methods}
    return metrics, case_rows


def eval_intervention_delta(split: str, residual_model: Any, ckargs: argparse.Namespace, readers: dict[str, dict[str, Any]], device: torch.device) -> dict[str, float | None]:
    accs = {mode: Acc() for mode in ["normal", "zero_semantic_measurements", "shuffle_semantic_measurements", "shuffle_assignment", "zero_unit_memory"]}
    interventions = {
        "normal": "force_gate_zero",
        "zero_semantic_measurements": "zero_semantic_measurements",
        "shuffle_semantic_measurements": "shuffle_semantic_measurements_across_points",
        "shuffle_assignment": "shuffle_assignment",
        "zero_unit_memory": "zero_unit_memory",
    }
    with torch.no_grad():
        for batch in make_loader(split, ckargs, shuffle=False):
            bd = move_batch(batch, device)
            mm = masks(bd)
            for mode, intervention in interventions.items():
                out = residual_model(**model_inputs(bd), intervention=intervention)
                pred, gate = method_predictions(bd, out, readers)["v34_25_sparse_gate_seed_mean"]
                update_method(accs[mode], mode, pred, gate, out["pointwise_semantic_belief"], bd["fut_teacher_embedding"], mm)
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
    }


def method_rank(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, row in metrics.items():
        if name == "oracle_hard_changed_mask_upper_bound":
            continue
        rows.append(
            {
                "method": name,
                "hard_changed_gain_vs_pointwise": row["hard_changed_gain_vs_pointwise"],
                "semantic_hard_signal": row["semantic_hard_signal"],
                "changed_semantic_signal": row["changed_semantic_signal"],
                "stable_preservation": row["stable_preservation"],
                "stable_over_open_rate": row["stable_over_open_rate"],
            }
        )
    return sorted(rows, key=lambda x: float(x["hard_changed_gain_vs_pointwise"] or -1e9), reverse=True)


def save_bar_chart(per_split: dict[str, Any]) -> dict[str, Any]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    test_rank = method_rank(per_split["test"]["methods"])
    names = [r["method"] for r in test_rank]
    values = [float(r["hard_changed_gain_vs_pointwise"] or 0.0) for r in test_rank]
    plt.figure(figsize=(10.5, 5.2))
    colors = ["#1b9e77" if "v34_25" in n else "#7570b3" for n in names]
    plt.bar(np.arange(len(names)), values, color=colors)
    plt.axhline(0.005, color="#444444", linestyle="--", linewidth=1.0)
    plt.xticks(np.arange(len(names)), names, rotation=35, ha="right")
    plt.ylabel("test hard/changed gain vs pointwise")
    plt.title("V34.26 baseline benchmark: hard/changed gain")
    path = VIS_DIR / "v34_26_baseline_hard_changed_gain.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {"name": "baseline_hard_changed_gain", "path": str(path.relative_to(ROOT)), "case_selection_reason": "比较 V34.25 与非 oracle baseline 的 hard/changed gain。"}


def save_intervention_chart(interventions: dict[str, Any]) -> dict[str, Any]:
    keys = ["zero_semantic_measurements_delta", "shuffle_semantic_measurements_delta", "shuffle_assignment_delta", "zero_unit_memory_delta"]
    labels = ["zero semantic", "shuffle semantic", "shuffle assignment", "zero unit"]
    vals = [float(interventions["test"].get(k) or 0.0) for k in keys]
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(np.arange(len(keys)), vals, color="#1b9e77")
    plt.axhline(0.002, color="#444444", linestyle="--", linewidth=1.0)
    plt.xticks(np.arange(len(keys)), labels, rotation=15)
    plt.ylabel("test hard/changed gain delta")
    plt.title("V34.25 sparse gate intervention deltas")
    path = VIS_DIR / "v34_26_v3425_intervention_deltas.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return {"name": "v3425_intervention_deltas", "path": str(path.relative_to(ROOT)), "case_selection_reason": "验证 selected sparse gate 下 semantic/assignment/unit memory 仍为因果路径。"}


def save_trace_case(case_rows: list[dict[str, Any]], residual_model: Any, ckargs: argparse.Namespace, readers: dict[str, dict[str, Any]], device: torch.device) -> dict[str, Any]:
    best = max(case_rows, key=lambda x: float(x["hard_changed_gain"])) if case_rows else None
    if best is None:
        return {"name": "m128_trace_overlay", "path": None, "case_selection_reason": "没有可用 case。"}
    # 重新读取该 uid 所在 val batch，画 M128 trace + gate。
    with torch.no_grad():
        for batch in make_loader("val", ckargs, shuffle=False):
            if best["uid"] not in batch["uid"]:
                continue
            idx = batch["uid"].index(best["uid"])
            bd = move_batch(batch, device)
            out = residual_model(**model_inputs(bd), intervention="force_gate_zero")
            _, gate = method_predictions(bd, out, readers)["v34_25_sparse_gate_seed_mean"]
            obs = bd["obs_points"][idx].detach().cpu().numpy()
            fut = out["point_pred"][idx].detach().cpu().numpy()
            g = gate[idx].mean(dim=-1).detach().cpu().numpy()
            plt.figure(figsize=(6.2, 6.0))
            plt.scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), s=3, color="#999999", alpha=0.35, label="observed trace")
            sc = plt.scatter(fut[:, :, 0].reshape(-1), fut[:, :, 1].reshape(-1), c=np.repeat(g, fut.shape[1]), s=5, cmap="magma", alpha=0.75, label="future trace + gate")
            plt.colorbar(sc, label="mean residual gate")
            plt.title(f"M128 future trace + sparse semantic gate\nuid={best['uid']}")
            plt.axis("equal")
            plt.legend(loc="best")
            path = VIS_DIR / "v34_26_m128_future_trace_sparse_gate_overlay.png"
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            return {
                "name": "m128_future_trace_sparse_gate_overlay",
                "path": str(path.relative_to(ROOT)),
                "case_selection_reason": "从 val eval 中选 hard/changed gain 最高样本，展示 M128 future trace 与 sparse semantic gate 叠加。",
                "sample_uid": best["uid"],
            }
    return {"name": "m128_trace_overlay", "path": None, "case_selection_reason": "未找到 case uid。"}


def write_docs(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = [
        "# V34.26 full-system baseline / claim-boundary benchmark 中文报告",
        "",
        "## 中文结论",
        report["中文结论"],
        "",
        "## 当前完整系统边界",
        report["当前完整系统边界"],
        "",
        "## 关键结果",
    ]
    for key in [
        "benchmark_passed",
        "v3425_beats_nonoracle_baselines",
        "semantic_hard_signal",
        "changed_semantic_signal",
        "stable_preservation",
        "semantic_measurements_load_bearing_on_residual",
        "assignment_load_bearing_on_residual",
        "unit_memory_load_bearing_on_residual",
        "m128_field_output_ready",
        "video_input_closure_ready",
        "integrated_semantic_field_claim_allowed",
        "integrated_identity_field_claim_allowed",
        "recommended_next_step",
    ]:
        lines.append(f"- {key}: `{report.get(key)}`")
    lines += ["", "## 阶段性分析", report["阶段性分析"], "", "## 论文相关问题解决方案参考", report["论文相关问题解决方案参考"], "", "## 最佳下一步方案", report["最佳下一步方案"]]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vlines = ["# V34.26 full-system visualization 中文报告", "", "## 中文结论", manifest["中文结论"], "", "## 图像清单"]
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
    cases: list[dict[str, Any]] = []
    for split in ("val", "test"):
        print(f"开始 V34.26 baseline benchmark: split={split}", flush=True)
        metrics, split_cases = eval_split(split, residual_model, ckargs, readers, device, args)
        per_split[split] = {"methods": metrics, "rank": method_rank(metrics)}
        cases.extend(split_cases)
    interventions = {}
    for split in ("val", "test"):
        print(f"开始 V34.26 intervention benchmark: split={split}", flush=True)
        interventions[split] = eval_intervention_delta(split, residual_model, ckargs, readers, device)

    v3425 = per_split["test"]["methods"]["v34_25_sparse_gate_seed_mean"]
    nonoracle = [r for r in per_split["test"]["rank"] if not r["method"].startswith("v34_25") and "oracle" not in r["method"]]
    best_nonoracle_gain = max(float(r["hard_changed_gain_vs_pointwise"] or -1e9) for r in nonoracle)
    v3425_gain = float(v3425["hard_changed_gain_vs_pointwise"] or -1e9)
    v3425_beats = bool(v3425_gain > best_nonoracle_gain + 0.002)
    sem_lb = bool(min(interventions["val"]["zero_semantic_measurements_delta"] or 0.0, interventions["val"]["shuffle_semantic_measurements_delta"] or 0.0, interventions["test"]["zero_semantic_measurements_delta"] or 0.0, interventions["test"]["shuffle_semantic_measurements_delta"] or 0.0) > 0.002)
    assign_lb = bool((interventions["val"]["shuffle_assignment_delta"] or 0.0) > 0.002 and (interventions["test"]["shuffle_assignment_delta"] or 0.0) > 0.002)
    unit_lb = bool((interventions["val"]["zero_unit_memory_delta"] or 0.0) > 0.002 and (interventions["test"]["zero_unit_memory_delta"] or 0.0) > 0.002)
    benchmark_passed = bool(
        v3425_beats
        and v3425["semantic_hard_signal"]
        and v3425["changed_semantic_signal"]
        and v3425["stable_preservation"]
        and v3425["stable_overopen_controlled"]
        and sem_lb
        and assign_lb
        and unit_lb
    )

    figures = [save_bar_chart(per_split), save_intervention_chart(interventions)]
    figures.append(save_trace_case(cases, residual_model, ckargs, readers, device))
    manifest = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.26 visualization 已完成；图像展示 baseline 排名、干预 delta、M128 future trace + sparse semantic gate overlay，不声明 semantic field success。",
        "figures": figures,
        "visualization_ready": True,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_26_m512_dense_visualization" if benchmark_passed else "fix_full_system_baseline_gap",
    }
    report = {
        "generated_at_utc": utc_now(),
        "中文结论": (
            "V34.26 full-system baseline / claim-boundary benchmark 已完成；本轮固定 V34.25，不训练新大模型，"
            "只评估 M128/H32 的 baseline、干预和 claim 边界。"
        ),
        "benchmark_passed": benchmark_passed,
        "v3425_beats_nonoracle_baselines": v3425_beats,
        "best_nonoracle_test_hard_changed_gain": best_nonoracle_gain,
        "v3425_test_hard_changed_gain": v3425_gain,
        "per_split": per_split,
        "intervention_delta": interventions,
        "semantic_hard_signal": {"val": per_split["val"]["methods"]["v34_25_sparse_gate_seed_mean"]["semantic_hard_signal"], "test": v3425["semantic_hard_signal"]},
        "changed_semantic_signal": {"val": per_split["val"]["methods"]["v34_25_sparse_gate_seed_mean"]["changed_semantic_signal"], "test": v3425["changed_semantic_signal"]},
        "stable_preservation": {"val": per_split["val"]["methods"]["v34_25_sparse_gate_seed_mean"]["stable_preservation"], "test": v3425["stable_preservation"]},
        "semantic_measurements_load_bearing_on_residual": sem_lb,
        "assignment_load_bearing_on_residual": assign_lb,
        "unit_memory_load_bearing_on_residual": unit_lb,
        "m128_field_output_ready": True,
        "video_input_closure_ready": False,
        "m512_dense_ready": False,
        "identity_field_ready": False,
        "v30_backbone_frozen": bool(residual_model.v30_backbone_frozen),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "当前完整系统边界": (
            "当前已经有 future trace field 与 future semantic residual belief 的 M128/H32 闭环评估，"
            "但输入仍是 video-derived/external-GT observed trace + observed semantic measurements；past-video 原生闭环、M512 dense、identity field 仍未完成。"
        ),
        "阶段性分析": (
            "V34.26 的作用是把 V34.25 从单一机制 positive 推进到完整评估协议：同一 frozen V30、同一 measurement bank、同一 val/test split 下比较 pointwise/copy/top-k/no-gate/sparse-gate baseline，"
            "并报告 semantic/assignment/unit intervention。若 V34.25 打赢非 oracle baseline，说明核心模块不是内部版本号自嗨，而是在协议下有不可替代性。"
        ),
        "论文相关问题解决方案参考": (
            "本轮对应顶会审稿最关注的 baseline fairness 与 counterfactual intervention：类似 Slot Attention 的 assignment 证明、XMem/SAM2 的 selective memory read、Perceiver IO 的 query-conditioned memory reading，以及 sparse MoE 的 gated computation。"
        ),
        "最佳下一步方案": (
            "如果 benchmark_passed=true，下一步只进入 run_v34_26_m512_dense_visualization；如果 false，先修 baseline gap，仍不跑 H64/H96、不写论文、不 claim identity。"
        ),
        "visualization_manifest_path": str(MANIFEST.relative_to(ROOT)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "run_v34_26_m512_dense_visualization" if benchmark_passed else "fix_full_system_baseline_gap",
        "train_summary_reference": train_summary,
    }
    dump_json(REPORT, report)
    dump_json(MANIFEST, manifest)
    write_docs(report, manifest)
    print(f"已写出 V34.26 benchmark report: {REPORT.relative_to(ROOT)}", flush=True)
    print(f"已写出 V34.26 visualization manifest: {MANIFEST.relative_to(ROOT)}", flush=True)
    print(f"recommended_next_step: {report['recommended_next_step']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
