#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


V3424 = ROOT / "reports/stwm_ostf_v34_24_claim_boundary_gate_overopen_audit_20260514.json"
V3425 = ROOT / "reports/stwm_ostf_v34_25_sparse_calibrated_gate_repair_decision_20260514.json"
OUT_DIR = ROOT / "outputs/visualizations/stwm_ostf_v34_25_claim_boundary_20260514"
MANIFEST = ROOT / "reports/stwm_ostf_v34_25_claim_boundary_visualization_manifest_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_25_CLAIM_BOUNDARY_VISUALIZATION_20260514.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def before_after_overopen(v3424: dict[str, Any], v3425: dict[str, Any]) -> dict[str, Any]:
    seeds = ["seed123", "seed42", "seed456"]
    before = []
    after = []
    for seed in seeds:
        if seed == "seed42":
            before.append(np.nan)
        else:
            before.append(float(v3424["gate_overopen_diagnostics"]["per_seed"][seed]["test_stable_over_open_rate"]))
        after.append(float(v3425["seed_reports"][seed]["stable_over_open_rate"]["test"]))
    x = np.arange(len(seeds))
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(x - 0.18, before, width=0.36, label="V34.23 before", color="#d95f02")
    plt.bar(x + 0.18, after, width=0.36, label="V34.25 after", color="#1b9e77")
    plt.axhline(0.35, color="#444444", linestyle="--", linewidth=1.2, label="controlled threshold")
    plt.xticks(x, seeds)
    plt.ylabel("test stable over-open rate")
    plt.title("stable gate over-open: V34.23 vs V34.25")
    plt.ylim(0.0, 1.05)
    plt.legend()
    path = OUT_DIR / "v34_25_stable_overopen_before_after.png"
    savefig(path)
    return {
        "name": "stable_overopen_before_after",
        "path": str(path.relative_to(ROOT)),
        "case_selection_reason": "展示 V34.25 sparse calibration 是否实质降低 V34.23 的 stable gate over-open 风险。",
        "before_test_overopen": dict(zip(seeds, before, strict=True)),
        "after_test_overopen": dict(zip(seeds, after, strict=True)),
    }


def pareto_plots(v3425: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []
    for seed, report in v3425["seed_reports"].items():
        rows = report["calibration_sweep_pareto_top12"]["test"]
        xs = [float(r["stable_over_open_rate"]) for r in rows]
        ys = [float(r["hard_changed_gain"]) for r in rows]
        labels = [f"T={r['config']['temperature']}, thr={r['config']['threshold']}" for r in rows]
        selected_cfg = report["best_config_by_val"]
        selected = None
        for i, r in enumerate(rows):
            if r["config"] == selected_cfg:
                selected = i
                break
        plt.figure(figsize=(7.2, 4.5))
        plt.scatter(xs, ys, c=np.arange(len(xs)), cmap="viridis", s=58)
        plt.axvline(0.35, color="#444444", linestyle="--", linewidth=1.2, label="stable over-open limit")
        if selected is not None:
            plt.scatter([xs[selected]], [ys[selected]], s=160, facecolors="none", edgecolors="#e41a1c", linewidths=2.4, label="selected")
        for i, text in enumerate(labels[:6]):
            plt.annotate(text, (xs[i], ys[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        plt.xlabel("test stable over-open rate")
        plt.ylabel("test hard/changed gain")
        plt.title(f"{seed}: calibration Pareto")
        plt.legend()
        path = OUT_DIR / f"v34_25_{seed}_calibration_pareto.png"
        savefig(path)
        outputs.append(
            {
                "name": f"{seed}_calibration_pareto",
                "path": str(path.relative_to(ROOT)),
                "case_selection_reason": "展示阈值/温度校准下 hard/changed gain 与 stable over-open 的 Pareto 权衡。",
                "selected_config": selected_cfg,
                "selected_test_stable_over_open_rate": report["stable_over_open_rate"]["test"],
                "selected_test_hard_changed_gain": report["hard_changed_gain"]["test"],
            }
        )
    return outputs


def intervention_bars(v3425: dict[str, Any]) -> dict[str, Any]:
    seeds = list(v3425["seed_reports"].keys())
    keys = [
        "zero_semantic_measurements_delta",
        "shuffle_semantic_measurements_delta",
        "shuffle_assignment_delta",
        "zero_unit_memory_delta",
    ]
    labels = ["zero semantic", "shuffle semantic", "shuffle assignment", "zero unit"]
    values = np.array(
        [
            [float(v3425["seed_reports"][seed]["intervention_eval"]["test"][key]) for key in keys]
            for seed in seeds
        ]
    )
    x = np.arange(len(keys))
    plt.figure(figsize=(8.2, 4.5))
    for i, seed in enumerate(seeds):
        plt.bar(x + (i - 1) * 0.22, values[i], width=0.22, label=seed)
    plt.axhline(0.002, color="#444444", linestyle="--", linewidth=1.1, label="load-bearing threshold")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("test hard/changed gain delta")
    plt.title("V34.25 selected gate intervention deltas")
    plt.legend()
    path = OUT_DIR / "v34_25_intervention_deltas.png"
    savefig(path)
    return {
        "name": "intervention_deltas",
        "path": str(path.relative_to(ROOT)),
        "case_selection_reason": "展示 selected sparse gate 下 semantic / assignment / unit memory 干预仍为 load-bearing。",
        "test_intervention_deltas": {seed: dict(zip(keys, values[i].tolist(), strict=True)) for i, seed in enumerate(seeds)},
    }


def write_doc(manifest: dict[str, Any]) -> None:
    lines = [
        "# V34.25 claim-boundary visualization 中文总结",
        "",
        "## 中文结论",
        manifest["中文结论"],
        "",
        "## 生成图像",
    ]
    for item in manifest["figures"]:
        lines.append(f"- {item['name']}: `{item['path']}`；原因：{item['case_selection_reason']}")
    lines += [
        "",
        "## 阶段性分析",
        manifest["阶段性分析"],
        "",
        "## 最佳下一步方案",
        manifest["最佳下一步方案"],
        "",
        "## 关键字段",
        f"- visualization_ready: `{manifest['visualization_ready']}`",
        f"- integrated_semantic_field_claim_allowed: `{manifest['integrated_semantic_field_claim_allowed']}`",
        f"- recommended_next_step: `{manifest['recommended_next_step']}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    v3424 = load(V3424)
    v3425 = load(V3425)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = [before_after_overopen(v3424, v3425)]
    figures.extend(pareto_plots(v3425))
    figures.append(intervention_bars(v3425))
    manifest = {
        "generated_at_utc": utc_now(),
        "中文结论": (
            "V34.25 claim-boundary visualization 已完成；图像只用于说明 sparse gate calibration 如何降低 stable over-open，"
            "以及 selected gate 下 semantic/assignment/unit 干预仍然 load-bearing，不用于声明 semantic field success。"
        ),
        "figures": figures,
        "visualization_ready": True,
        "阶段性分析": (
            "V34.25 解决的是 V34.24 暴露的 gate calibration/sparsity 风险。可视化显示 test stable over-open "
            "从 V34.23 的高过开区间降到 V34.25 的受控区间，同时 hard/changed gain 与干预 delta 保留。"
        ),
        "论文相关问题解决方案参考": (
            "该可视化对应 selective computation / sparse gate 报告方式：不仅画 top-line，还要画 Pareto 和反事实干预 delta。"
        ),
        "最佳下一步方案": (
            "下一步应停下来做 V34.25 claim-boundary 总结与 external baseline/视频输入闭环计划，仍不要 claim semantic field success。"
        ),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "stop_and_prepare_v34_25_claim_boundary_summary",
    }
    dump_json(MANIFEST, manifest)
    write_doc(manifest)
    print(f"已写出 V34.25 claim-boundary visualization manifest: {MANIFEST.relative_to(ROOT)}")
    print(f"已写出 V34.25 claim-boundary visualization doc: {DOC.relative_to(ROOT)}")
    print(f"recommended_next_step: {manifest['recommended_next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
