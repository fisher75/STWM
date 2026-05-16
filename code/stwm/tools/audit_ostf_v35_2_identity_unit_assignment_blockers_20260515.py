#!/usr/bin/env python3
"""审计 V35 三 seed 后的 identity / unit / assignment blocker。"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPLICATION = ROOT / "reports/stwm_ostf_v35_seed42_123_456_replication_decision_20260515.json"
TARGET_BUILD = ROOT / "reports/stwm_ostf_v35_1_fixed_semantic_state_target_build_20260515.json"
MODEL = ROOT / "code/stwm/modules/ostf_v35_semantic_state_world_model.py"
TRAIN = ROOT / "code/stwm/tools/train_ostf_v35_semantic_state_head_20260515.py"
EVAL = ROOT / "code/stwm/tools/eval_ostf_v35_semantic_state_head_20260515.py"
REPORT = ROOT / "reports/stwm_ostf_v35_2_identity_unit_assignment_blocker_audit_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_2_IDENTITY_UNIT_ASSIGNMENT_BLOCKER_AUDIT_20260515.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def find_lines(path: Path, patterns: list[str]) -> dict[str, list[int]]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    out: dict[str, list[int]] = {}
    for pat in patterns:
        rgx = re.compile(pat)
        out[pat] = [i + 1 for i, line in enumerate(lines) if rgx.search(line)]
    return out


def main() -> None:
    print("V35.2: 审计 identity / unit / assignment blocker。", flush=True)
    rep = load(REPLICATION)
    target = load(TARGET_BUILD)
    per_seed = rep.get("per_seed", {})
    identity_aucs = {
        str(seed): {
            "val": d.get("identity_consistency", {}).get("val_auc"),
            "test": d.get("identity_consistency", {}).get("test_auc"),
        }
        for seed, d in per_seed.items()
    }
    uncertainty_aucs = {
        str(seed): {
            "val": d.get("uncertainty_quality", {}).get("val_auc"),
            "test": d.get("uncertainty_quality", {}).get("test_auc"),
        }
        for seed, d in per_seed.items()
    }
    code_locations = {
        "copy_prior_and_unit_gate": find_lines(MODEL, ["copy_prior_strength", "unit_gate", "assignment_head", "zero_unit_memory", "shuffle_assignment"]),
        "identity_loss": find_lines(TRAIN, ["same_instance_bce", "balanced_bce\\(out\\[\"same_instance_logits\"\\]", "same_mask"]),
        "intervention_eval": find_lines(EVAL, ["shuffle_assignment", "zero_unit_memory", "semantic_measurement_load_bearing", "assignment_load_bearing"]),
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "three_seed_semantic_state_head_passed": bool(rep.get("seed42_123_456_semantic_state_head_passed")),
        "stable_changed_hard_replicated": bool(rep.get("stable_preservation_replicated") and rep.get("changed_semantic_signal_replicated") and rep.get("semantic_hard_signal_replicated")),
        "semantic_measurement_load_bearing_replicated": bool(rep.get("semantic_measurement_load_bearing_replicated")),
        "identity_consistency_weak": True,
        "identity_auc_by_seed": identity_aucs,
        "identity_target_positive_ratio_by_split": target.get("identity_positive_ratio_by_split"),
        "identity_blocker": "identity target 极度正类偏置，当前 same-instance head 只学到 continuation prior，缺少 identity confuser / hard negative / reappear 对比监督。",
        "uncertainty_head_weak_after_neural_head": True,
        "uncertainty_auc_by_seed": uncertainty_aucs,
        "unit_memory_not_load_bearing": not bool(rep.get("unit_memory_load_bearing_replicated")),
        "assignment_not_load_bearing": not bool(rep.get("assignment_load_bearing_replicated")),
        "unit_assignment_blocker": "当前 copy-preserving head 的语义 cluster 主要由 copy prior 与 point-level semantic features 决定，unit memory 和 assignment 只作为弱隐藏残差，没有直接支撑 state transition 或 identity consistency。",
        "evidence_anchor_family_still_weak": True,
        "recommended_fix": "build_v35_2_identity_confuser_unit_assignment_targets_and_train_loadbearing_head",
        "exact_code_locations": code_locations,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "中文结论": "V35 三 seed 证明低维 semantic state head 有稳定语义信号，但 identity 与 unit/assignment 机制仍未成立。下一步应构建 identity confuser/hard-negative/reappear targets，并给 unit/assignment 添加直接 load-bearing supervision；不应扩大到 H64/H96/M512 或 claim 完整 semantic field。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.2 Identity / Unit / Assignment Blocker Audit\n\n"
        f"- three_seed_semantic_state_head_passed: {report['three_seed_semantic_state_head_passed']}\n"
        f"- stable_changed_hard_replicated: {report['stable_changed_hard_replicated']}\n"
        f"- semantic_measurement_load_bearing_replicated: {report['semantic_measurement_load_bearing_replicated']}\n"
        f"- identity_consistency_weak: {report['identity_consistency_weak']}\n"
        f"- unit_memory_not_load_bearing: {report['unit_memory_not_load_bearing']}\n"
        f"- assignment_not_load_bearing: {report['assignment_not_load_bearing']}\n"
        f"- recommended_fix: {report['recommended_fix']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"recommended_fix": report["recommended_fix"], "identity_consistency_weak": True, "assignment_not_load_bearing": report["assignment_not_load_bearing"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
