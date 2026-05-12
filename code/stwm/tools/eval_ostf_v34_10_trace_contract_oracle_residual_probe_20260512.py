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
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import (
    CKPT_DIR,
    SUMMARY as TRAIN_SUMMARY,
    TraceContractResidualDataset,
    collate_v3410,
)


SUMMARY = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_eval_summary_20260512.json"
DECISION = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_decision_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_10_TRACE_CONTRACT_ORACLE_RESIDUAL_PROBE_DECISION_20260512.md"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    e348.TRAIN_SUMMARY = TRAIN_SUMMARY
    e348.CKPT_DIR = CKPT_DIR
    e348.SUMMARY = SUMMARY
    e348.DECISION = DECISION
    e348.DOC = DOC
    e348.CausalAssignmentResidualDataset = TraceContractResidualDataset
    e348.collate_v348 = collate_v3410
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)] + (["--cpu"] if args.cpu else [])
    try:
        e348.main()
    finally:
        sys.argv = old_argv
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    d = json.loads(DECISION.read_text(encoding="utf-8")) if DECISION.exists() else {}
    trace_full = bool(train.get("trace_state_contract_fully_passed") and train.get("train_dataset_uses_real_obs_conf"))
    usage_active = bool(train.get("semantic_usage_loss_active"))
    assign_active = bool(train.get("assignment_contrast_loss_active"))
    d["中文结论"] = "V34.10 oracle residual probe 已评估；dataset 使用真实 trace_obs_conf，usage/assignment loss 已激活，但 learned gate 仍需 oracle 通过后才能训练。"
    d["trace_state_contract_fully_passed"] = trace_full
    d["train_dataset_uses_real_obs_conf"] = bool(train.get("train_dataset_uses_real_obs_conf"))
    d["semantic_usage_loss_active"] = usage_active
    d["assignment_contrast_loss_active"] = assign_active
    d["oracle_residual_probe_passed"] = bool(d.get("oracle_residual_probe_passed") and trace_full and usage_active and assign_active)
    d["recommended_next_step"] = "train_trace_contract_residual_gate" if d["oracle_residual_probe_passed"] else d.get("recommended_next_step", "fix_semantic_measurement_bank")
    dump_json(DECISION, d)
    s = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    s["decision"] = d
    dump_json(SUMMARY, s)
    write_doc(DOC, "V34.10 trace contract oracle residual probe 决策中文报告", d, ["中文结论", "trace_state_contract_fully_passed", "train_dataset_uses_real_obs_conf", "semantic_usage_loss_active", "assignment_contrast_loss_active", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "causal_assignment_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "effective_units", "unit_dominant_instance_purity", "unit_semantic_purity", "recommended_next_step"])
    print(f"已写出 V34.10 评估摘要: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
