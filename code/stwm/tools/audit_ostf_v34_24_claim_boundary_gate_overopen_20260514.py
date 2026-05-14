#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SEED42 = ROOT / "reports/stwm_ostf_v34_23_activation_state_gate_probe_summary_20260513.json"
SEED123 = ROOT / "reports/stwm_ostf_v34_23_seed123_replication_summary_20260514.json"
SEED456 = ROOT / "reports/stwm_ostf_v34_23_seed456_replication_summary_20260514.json"
CROSS_SEED = ROOT / "reports/stwm_ostf_v34_23_cross_seed_replication_decision_20260514.json"

REPORT = ROOT / "reports/stwm_ostf_v34_24_claim_boundary_gate_overopen_audit_20260514.json"
DOC = ROOT / "docs/STWM_OSTF_V34_24_CLAIM_BOUNDARY_GATE_OVEROPEN_AUDIT_20260514.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def decision(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("decision", payload)


def both_true(x: dict[str, Any] | None) -> bool:
    if not isinstance(x, dict):
        return False
    return bool(x.get("val") and x.get("test"))


def get_seed_gate(dec: dict[str, Any]) -> dict[str, Any]:
    return dec.get("seed_replication", {}).get("gate_diagnostics", {})


def get_gate_metric(gate_diag: dict[str, Any], split: str, key: str) -> float | None:
    value = gate_diag.get(split, {}).get(key)
    if value is None:
        return None
    return float(value)


def safe_mean(values: list[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    if not real:
        return None
    return float(mean(real))


def gate_summary(seed_decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for name, dec in seed_decisions.items():
        gd = get_seed_gate(dec)
        if not gd:
            per_seed[name] = {"gate_diagnostics_available": False}
            continue
        per_seed[name] = {
            "gate_diagnostics_available": True,
            "val_stable_gate_mean": get_gate_metric(gd, "val", "stable_gate_mean"),
            "test_stable_gate_mean": get_gate_metric(gd, "test", "stable_gate_mean"),
            "val_hard_changed_gate_mean": get_gate_metric(gd, "val", "hard_changed_gate_mean"),
            "test_hard_changed_gate_mean": get_gate_metric(gd, "test", "hard_changed_gate_mean"),
            "val_stable_over_open_rate": get_gate_metric(gd, "val", "stable_over_open_rate"),
            "test_stable_over_open_rate": get_gate_metric(gd, "test", "stable_over_open_rate"),
            "val_stable_over_update_rate": get_gate_metric(gd, "val", "stable_over_update_rate"),
            "test_stable_over_update_rate": get_gate_metric(gd, "test", "stable_over_update_rate"),
            "val_gate_order_ok": bool(gd.get("val", {}).get("gate_order_ok")),
            "test_gate_order_ok": bool(gd.get("test", {}).get("gate_order_ok")),
            "val_stable_over_open_detected": bool(gd.get("val", {}).get("stable_over_open_detected")),
            "test_stable_over_open_detected": bool(gd.get("test", {}).get("stable_over_open_detected")),
            "val_stable_over_update_detected": bool(gd.get("val", {}).get("stable_over_update_detected")),
            "test_stable_over_update_detected": bool(gd.get("test", {}).get("stable_over_update_detected")),
        }
    aggregate = {
        "mean_val_stable_gate_mean": safe_mean([x.get("val_stable_gate_mean") for x in per_seed.values()]),
        "mean_test_stable_gate_mean": safe_mean([x.get("test_stable_gate_mean") for x in per_seed.values()]),
        "mean_val_stable_over_open_rate": safe_mean([x.get("val_stable_over_open_rate") for x in per_seed.values()]),
        "mean_test_stable_over_open_rate": safe_mean([x.get("test_stable_over_open_rate") for x in per_seed.values()]),
        "mean_val_stable_over_update_rate": safe_mean([x.get("val_stable_over_update_rate") for x in per_seed.values()]),
        "mean_test_stable_over_update_rate": safe_mean([x.get("test_stable_over_update_rate") for x in per_seed.values()]),
    }
    stable_overopen = any(
        bool(v.get("val_stable_over_open_detected") or v.get("test_stable_over_open_detected"))
        for v in per_seed.values()
    )
    stable_overupdate = any(
        bool(v.get("val_stable_over_update_detected") or v.get("test_stable_over_update_detected"))
        for v in per_seed.values()
    )
    gate_order = all(
        bool(v.get("val_gate_order_ok") and v.get("test_gate_order_ok"))
        for v in per_seed.values()
        if v.get("gate_diagnostics_available")
    )
    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gate_order_replicated": gate_order,
        "stable_gate_overopen_detected": stable_overopen,
        "stable_overupdate_detected": stable_overupdate,
        "gate_overopen_is_calibration_risk": bool(stable_overopen and not stable_overupdate),
    }


def intervention_summary(seed_decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, dec in seed_decisions.items():
        out[name] = {}
        for split in ("val", "test"):
            ev = dec.get("intervention_eval", {}).get(split, {})
            out[name][split] = {
                "zero_semantic_measurements_delta": ev.get("zero_semantic_measurements_delta"),
                "shuffle_semantic_measurements_delta": ev.get("shuffle_semantic_measurements_delta"),
                "shuffle_assignment_delta": ev.get("shuffle_assignment_delta"),
                "zero_unit_memory_delta": ev.get("zero_unit_memory_delta"),
            }
    return out


def all_decisions_pass(decisions: dict[str, dict[str, Any]], key: str) -> bool:
    return all(bool(dec.get(key)) for dec in decisions.values())


def write_markdown(payload: dict[str, Any]) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V34.24 claim boundary / gate over-open 风险审计",
        "",
        "## 中文结论",
        payload["中文结论"],
        "",
        "## 允许的阶段性 claim",
    ]
    lines.extend(f"- {x}" for x in payload["allowed_claims"])
    lines += ["", "## 禁止的 claim"]
    lines.extend(f"- {x}" for x in payload["forbidden_claims"])
    lines += ["", "## Gate 风险核心证据"]
    gate = payload["gate_overopen_diagnostics"]
    for seed, item in gate["per_seed"].items():
        if not item.get("gate_diagnostics_available"):
            lines.append(f"- {seed}: gate diagnostics 不完整。")
            continue
        lines.append(
            "- "
            f"{seed}: val stable gate mean={item['val_stable_gate_mean']:.6f}, "
            f"test stable gate mean={item['test_stable_gate_mean']:.6f}, "
            f"val stable over-open={item['val_stable_over_open_rate']:.6f}, "
            f"test stable over-open={item['test_stable_over_open_rate']:.6f}, "
            f"val/test stable over-update="
            f"{item['val_stable_over_update_rate']:.6f}/{item['test_stable_over_update_rate']:.6f}。"
        )
    lines += [
        "",
        "## 阶段性分析",
        payload["阶段性分析"],
        "",
        "## 论文相关问题解决方案参考",
    ]
    for ref in payload["paper_inspired_solution_references"]:
        lines.append(f"- {ref['name']}: {ref['启发']} {ref['url']}")
    lines += [
        "",
        "## 精确阻塞点",
    ]
    lines.extend(f"- {x}" for x in payload["exact_blockers"])
    lines += [
        "",
        "## 最佳下一步方案",
        payload["最佳下一步方案"],
        "",
        "## 关键字段",
    ]
    for key in [
        "claim_boundary_audit_done",
        "cross_seed_semantic_hard_positive",
        "cross_seed_changed_positive",
        "cross_seed_stable_preserved",
        "semantic_measurement_load_bearing_replicated",
        "assignment_load_bearing_replicated",
        "unit_memory_load_bearing_replicated",
        "gate_order_replicated",
        "stable_gate_overopen_detected",
        "stable_overupdate_detected",
        "semantic_field_success_claim_allowed",
        "integrated_semantic_field_claim_allowed",
        "integrated_identity_field_claim_allowed",
        "recommended_next_step",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    seed42 = decision(load_json(SEED42))
    seed123 = decision(load_json(SEED123))
    seed456 = decision(load_json(SEED456))
    cross_seed = load_json(CROSS_SEED)
    seed_decisions = {"seed123": seed123, "seed456": seed456}

    missing_inputs = [
        str(p.relative_to(ROOT))
        for p in (SEED42, SEED123, SEED456, CROSS_SEED)
        if not p.exists()
    ]
    seed42_passed = bool(seed42.get("activation_state_gate_probe_passed"))
    seed123_passed = bool(seed123.get("seed_replication", {}).get("cross_seed_consistent_with_seed42"))
    seed456_passed = bool(seed456.get("seed_replication", {}).get("cross_seed_consistent_with_seed42"))
    cross_seed_semantic_hard = bool(seed42_passed and all(both_true(d.get("semantic_hard_signal")) for d in seed_decisions.values()))
    cross_seed_changed = bool(seed42_passed and all(both_true(d.get("changed_semantic_signal")) for d in seed_decisions.values()))
    cross_seed_stable = bool(seed42_passed and all(both_true(d.get("stable_preservation")) for d in seed_decisions.values()))
    sem_lb = bool(seed42.get("semantic_measurements_load_bearing_on_residual") and all_decisions_pass(seed_decisions, "semantic_measurements_load_bearing_on_residual"))
    assign_lb = bool(seed42.get("assignment_load_bearing_on_residual") and all_decisions_pass(seed_decisions, "assignment_load_bearing_on_residual"))
    unit_lb = bool(seed42.get("unit_memory_load_bearing_on_residual") and all_decisions_pass(seed_decisions, "unit_memory_load_bearing_on_residual"))
    gate = gate_summary(seed_decisions)
    stable_overopen = bool(gate["stable_gate_overopen_detected"])
    stable_overupdate = bool(gate["stable_overupdate_detected"])
    all_seed_pass = bool(seed42_passed and seed123_passed and seed456_passed)

    allowed_claims = [
        "V34.23 在 V30 frozen、H32/M128、seed42/123/456 上复现了 hard/changed residual 正信号。",
        "V34.23 的 top-k evidence residual、activation-state gate probe、semantic/assignment/unit intervention 在复现实验中显示 load-bearing。",
        "V34.23 的 stable preservation 在 seed123/456 val/test 仍为 true，且 stable over-update 率很低。",
        "V34.23 可以被描述为“受控 residual probe 的多 seed positive”，不是完整 semantic field 成功。",
    ]
    forbidden_claims = [
        "不允许 claim integrated semantic field success。",
        "不允许 claim integrated identity field success。",
        "不允许 claim learned gate 已经 sparse/calibrated/production-ready。",
        "不允许 claim video-to-semantic world model 已闭环；当前仍是 observed trace + observed semantic measurements 输入。",
        "不允许把结果外推到 H64/H96、M512 dense 或 1B 规模。",
        "不允许推荐写论文或 Overleaf。",
    ]
    exact_blockers = [
        "stable gate over-open 在 seed123/456 的 val/test 上重复出现；这说明 gate 虽然保持 stable 不退化，但没有学到足够稀疏的 hard/changed 选择边界。",
        "stable over-update 率很低，说明当前主要风险是 calibration / sparsity，而不是直接破坏 stable 输出。",
        "seed42 原始 summary 没有同等详细的 stable over-open audit；严格 claim 只能依赖 seed123/456 的详细 gate 风险诊断和 seed42 的 top-line 复现。",
        "当前没有 identity field 复现主张，identity 仍不能被纳入 integrated claim。",
        "当前没有 H64/H96/M512/视频输入闭环复现，仍需留在 frozen V30 M128 范围内。",
    ]
    best_next = (
        "下一轮如果继续，应只做 sparse/calibrated gate 风险处理：引入 stable-negative focal/hinge、预算约束或 temperature/threshold 校准，"
        "并强制报告 gate over-open、stable over-update、semantic/assignment/unit intervention delta。不要扩大模型，不要跑 H64/H96，不要 claim semantic field success。"
    )
    recommended_next = "fix_gate_calibration_sparse_gate"
    if missing_inputs:
        recommended_next = "fix_artifact_packaging"
    elif not all_seed_pass:
        recommended_next = "rerun_v34_23_seed_replication"
    elif stable_overupdate:
        recommended_next = "fix_stable_over_update"
    elif stable_overopen:
        recommended_next = "fix_gate_calibration_sparse_gate"
    else:
        recommended_next = "stop_and_return_to_target_mapping"

    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": (
            "V34.23 seed42/123/456 已经形成稳定的 residual probe positive，但 claim 边界必须收紧："
            "可以说 hard/changed residual 与 semantic/assignment/unit memory 路径在 M128/H32 多 seed 上有正信号；"
            "不能说 integrated semantic field success。主要 blocker 是 stable gate over-open：stable 区域 gate 大量打开，"
            "虽然 stable over-update 很低，但这属于 calibration/sparsity 风险，必须单独修。"
        ),
        "claim_boundary_audit_done": True,
        "missing_inputs": missing_inputs,
        "seed42_passed": seed42_passed,
        "seed123_passed": seed123_passed,
        "seed456_passed": seed456_passed,
        "all_required_seeds_passed": all_seed_pass,
        "cross_seed_report_recommended_next_step": cross_seed.get("recommended_next_step"),
        "cross_seed_semantic_hard_positive": cross_seed_semantic_hard,
        "cross_seed_changed_positive": cross_seed_changed,
        "cross_seed_stable_preserved": cross_seed_stable,
        "semantic_measurement_load_bearing_replicated": sem_lb,
        "assignment_load_bearing_replicated": assign_lb,
        "unit_memory_load_bearing_replicated": unit_lb,
        "gate_order_replicated": bool(gate["gate_order_replicated"]),
        "stable_gate_overopen_detected": stable_overopen,
        "stable_overupdate_detected": stable_overupdate,
        "gate_overopen_is_calibration_risk": bool(gate["gate_overopen_is_calibration_risk"]),
        "gate_overopen_diagnostics": gate,
        "intervention_delta_by_seed": intervention_summary(seed_decisions),
        "allowed_claims": allowed_claims,
        "forbidden_claims": forbidden_claims,
        "semantic_field_success_claim_allowed": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "exact_blockers": exact_blockers,
        "exact_code_locations": {
            "v34_23_train_gate_loss": "code/stwm/tools/train_ostf_v34_23_activation_state_gate_probe_20260513.py:102",
            "v34_23_gate_contrast_loss": "code/stwm/tools/train_ostf_v34_23_activation_state_gate_probe_20260513.py:125",
            "v34_23_seed_gate_diagnostics": "code/stwm/tools/run_ostf_v34_23_seed_replication_20260514.py:74",
            "v34_23_stable_overopen_definition": "code/stwm/tools/run_ostf_v34_23_seed_replication_20260514.py:141",
            "v34_23_cross_seed_claim_boundary": "code/stwm/tools/run_ostf_v34_23_seed_replication_20260514.py:283",
        },
        "阶段性分析": (
            "这轮不是继续修 bug，也不是进入 H64/H96，而是把跨 seed 复现后的 claim boundary 切干净。"
            "最新状态支持“residual memory 在 hard/changed 上有可复现的因果正信号”，但 gate 还没有成为可靠稀疏选择器。"
            "stable 区域 gate 大量打开却几乎不造成 over-update，说明主路径保护仍有效，但 gate 本身的选择性不足；"
            "如果现在 claim semantic field success，会把 probe positive 和 integrated calibrated field 混为一谈，这是不安全的。"
        ),
        "paper_inspired_solution_references": [
            {
                "name": "Slot Attention / object-centric slots",
                "url": "https://arxiv.org/abs/2006.15055",
                "启发": "slot/unit memory 必须用分配和干预证明 load-bearing，不能只看最终指标。",
            },
            {
                "name": "XMem video memory",
                "url": "https://arxiv.org/abs/2207.07115",
                "启发": "视频 memory 需要明确的读写边界和选择策略；当前 gate 过开说明选择策略还不够稀疏。",
            },
            {
                "name": "Perceiver IO",
                "url": "https://arxiv.org/abs/2107.14795",
                "启发": "query-conditioned memory reading 是合理方向，但输出 query 的 gate 仍要做 calibration，而不是默认全读。",
            },
            {
                "name": "FiLM / conditional modulation",
                "url": "https://arxiv.org/abs/1709.07871",
                "启发": "条件调制要约束作用强度；否则 residual gate 容易变成泛化的全局调制而非 hard-case correction。",
            },
        ],
        "最佳下一步方案": best_next,
        "recommended_next_step": recommended_next,
    }
    dump_json(REPORT, payload)
    write_markdown(payload)
    print("已完成 V34.24 claim boundary / gate over-open 风险审计。")
    print(f"报告 JSON: {REPORT.relative_to(ROOT)}")
    print(f"中文文档: {DOC.relative_to(ROOT)}")
    print(f"recommended_next_step: {recommended_next}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
