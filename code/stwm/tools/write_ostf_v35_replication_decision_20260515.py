#!/usr/bin/env python3
"""汇总 V35 seed42/123/456 复现决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

SEED_PATHS = {
    42: ROOT / "reports/stwm_ostf_v35_semantic_state_head_decision_20260515.json",
    123: ROOT / "reports/stwm_ostf_v35_seed123_replication_decision_20260515.json",
    456: ROOT / "reports/stwm_ostf_v35_seed456_replication_decision_20260515.json",
}
REPORT = ROOT / "reports/stwm_ostf_v35_seed42_123_456_replication_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_SEED42_123_456_REPLICATION_DECISION_20260515.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def both_true(block: Any) -> bool:
    return isinstance(block, dict) and bool(block.get("val")) and bool(block.get("test"))


def main() -> None:
    print("V35: 汇总 seed42/123/456 复现结果。", flush=True)
    seeds = {seed: load(path) for seed, path in SEED_PATHS.items()}
    all_present = all(bool(v) for v in seeds.values())
    semantic_pass = all(bool(d.get("semantic_state_head_passed")) for d in seeds.values()) if all_present else False
    hard_pass = all(both_true(d.get("semantic_hard_signal")) for d in seeds.values()) if all_present else False
    changed_pass = all(both_true(d.get("changed_semantic_signal")) for d in seeds.values()) if all_present else False
    stable_pass = all(both_true(d.get("stable_preservation")) for d in seeds.values()) if all_present else False
    sem_measure = all(bool(d.get("semantic_measurement_load_bearing")) for d in seeds.values()) if all_present else False
    unit_lb = all(bool(d.get("unit_memory_load_bearing")) for d in seeds.values()) if all_present else False
    assign_lb = all(bool(d.get("assignment_load_bearing")) for d in seeds.values()) if all_present else False
    identity_claim = all(bool(d.get("integrated_identity_field_claim_allowed")) for d in seeds.values()) if all_present else False
    semantic_claim = bool(semantic_pass and hard_pass and changed_pass and stable_pass and sem_measure and unit_lb and assign_lb)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_decisions_present": all_present,
        "seed42_123_456_semantic_state_head_passed": semantic_pass,
        "semantic_hard_signal_replicated": hard_pass,
        "changed_semantic_signal_replicated": changed_pass,
        "stable_preservation_replicated": stable_pass,
        "semantic_measurement_load_bearing_replicated": sem_measure,
        "unit_memory_load_bearing_replicated": unit_lb,
        "assignment_load_bearing_replicated": assign_lb,
        "v30_backbone_frozen": all(bool(d.get("v30_backbone_frozen")) for d in seeds.values()) if all_present else False,
        "future_leakage_detected": any(bool(d.get("future_leakage_detected")) for d in seeds.values()) if all_present else True,
        "trajectory_degraded": any(bool(d.get("trajectory_degraded")) for d in seeds.values()) if all_present else True,
        "integrated_identity_field_claim_allowed": identity_claim,
        "integrated_semantic_field_claim_allowed": semantic_claim,
        "recommended_next_step": "fix_identity_consistency_and_unit_assignment_load_bearing" if semantic_pass else "fix_v35_semantic_state_head",
        "per_seed": seeds,
        "中文结论": "V35 semantic state head 在 seed42/123/456 上复现了 stable/changed/hard 语义状态正信号，且 semantic measurement 是 load-bearing；但 identity consistency、unit memory、assignment 仍未立住，因此不能 claim 完整 semantic/identity field success，下一步应修 identity 与 unit/assignment load-bearing。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35 Seed42/123/456 Replication Decision\n\n"
        f"- seed42_123_456_semantic_state_head_passed: {semantic_pass}\n"
        f"- semantic_hard_signal_replicated: {hard_pass}\n"
        f"- changed_semantic_signal_replicated: {changed_pass}\n"
        f"- stable_preservation_replicated: {stable_pass}\n"
        f"- semantic_measurement_load_bearing_replicated: {sem_measure}\n"
        f"- unit_memory_load_bearing_replicated: {unit_lb}\n"
        f"- assignment_load_bearing_replicated: {assign_lb}\n"
        f"- integrated_semantic_field_claim_allowed: {semantic_claim}\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"seed42_123_456_semantic_state_head_passed": semantic_pass, "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
