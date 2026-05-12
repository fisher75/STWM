#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools import eval_ostf_v34_8_causal_assignment_oracle_residual_probe_20260512 as e348
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_train_summary_20260512.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_eval_summary_20260512.json"
DECISION = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_decision_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_9_TRACE_FIXED_ORACLE_RESIDUAL_PROBE_DECISION_20260512.md"
BANK_REPORT = ROOT / "reports/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.json"


def main() -> int:
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=32)
    args.add_argument("--num-workers", type=int, default=0)
    args.add_argument("--cpu", action="store_true")
    ns = args.parse_args()
    e348.TRAIN_SUMMARY = TRAIN_SUMMARY
    e348.SUMMARY = SUMMARY
    e348.DECISION = DECISION
    e348.DOC = DOC
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--batch-size", str(ns.batch_size), "--num-workers", str(ns.num_workers)] + (["--cpu"] if ns.cpu else [])
    try:
        e348.main()
    finally:
        sys.argv = old_argv
    bank = json.loads(BANK_REPORT.read_text(encoding="utf-8")) if BANK_REPORT.exists() else {}
    if DECISION.exists():
        d = json.loads(DECISION.read_text(encoding="utf-8"))
        d["中文结论"] = "V34.9 trace-fixed oracle residual probe 已评估；只有 trace contract、semantic measurement 与 assignment 均通过才允许 learned gate。"
        d["trace_state_contract_passed"] = bool(bank.get("trace_state_contract_passed"))
        d["oracle_residual_probe_passed"] = bool(d.get("oracle_residual_probe_passed") and d["trace_state_contract_passed"])
        d["recommended_next_step"] = "train_trace_fixed_residual_gate" if d["oracle_residual_probe_passed"] else d.get("recommended_next_step", "fix_semantic_measurement_bank")
        dump_json(DECISION, d)
        s = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
        s["decision"] = d
        dump_json(SUMMARY, s)
        write_doc(DOC, "V34.9 trace-fixed oracle residual probe 决策中文报告", d, ["中文结论", "trace_state_contract_passed", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "causal_assignment_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "effective_units", "unit_dominant_instance_purity", "unit_semantic_purity", "recommended_next_step"])
    print(f"已写出 V34.9 trace-fixed 评估摘要: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
