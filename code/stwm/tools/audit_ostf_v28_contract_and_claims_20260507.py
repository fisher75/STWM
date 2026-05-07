#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_lastobs_residual_world_model_v28 import OSTFLastObservedResidualConfigV28, OSTFLastObservedResidualWorldModelV28
from stwm.tools.ostf_lastobs_v28_common_20260502 import ROOT, batch_from_samples_v26, build_v28_rows
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc


REPORT_PATH = ROOT / "reports/stwm_ostf_v28_contract_and_claims_audit_20260507.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V28_CONTRACT_AND_CLAIMS_AUDIT_20260507.md"
DECISION_PATH = ROOT / "reports/stwm_ostf_v28_decision_20260502.json"
EVAL_PATH = ROOT / "reports/stwm_ostf_v28_eval_summary_20260502.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v28_bootstrap_20260502.json"


REQUIRED_FILES = [
    "code/stwm/modules/ostf_lastobs_residual_world_model_v28.py",
    "code/stwm/tools/ostf_lastobs_v28_common_20260502.py",
    "code/stwm/tools/train_ostf_lastobs_residual_v28_20260502.py",
    "code/stwm/tools/eval_ostf_lastobs_residual_v28_20260502.py",
    "code/stwm/tools/verify_ostf_lastobs_residual_v28_20260502.py",
    "code/stwm/tools/render_ostf_lastobs_residual_v28_20260502.py",
    "scripts/run_ostf_lastobs_residual_v28_20260502.sh",
    "scripts/start_ostf_lastobs_residual_v28_tmux_20260502.sh",
    "reports/stwm_ostf_v28_decision_20260502.json",
    "reports/stwm_ostf_v28_eval_summary_20260502.json",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _positive(boot: dict[str, Any]) -> bool:
    return bool(boot.get("zero_excluded") and (boot.get("mean_delta") or 0.0) > 0.0)


def _negative_significant(boot: dict[str, Any]) -> bool:
    return bool(boot.get("zero_excluded") and (boot.get("mean_delta") or 0.0) < 0.0)


def _forward_contract() -> dict[str, Any]:
    rows, _, gamma, _ = build_v28_rows("M128_H32", seed=42)
    sample = rows["test"][:1]
    batch = batch_from_samples_v26(sample, torch.device("cpu"))
    model = OSTFLastObservedResidualWorldModelV28(OSTFLastObservedResidualConfigV28(horizon=32, damped_gamma=gamma))
    model.eval()
    with torch.no_grad():
        out = model(
            obs_points=batch["obs_points"],
            obs_vis=batch["obs_vis"],
            obs_conf=batch["obs_conf"],
            rel_xy=batch["rel_xy"],
            anchor_obs=batch["anchor_obs"],
            anchor_obs_vel=batch["anchor_obs_vel"],
            semantic_feat=batch["semantic_feat"],
            semantic_id=batch["semantic_id"],
            box_feat=batch["box_feat"],
            neighbor_feat=batch["neighbor_feat"],
            global_feat=batch["global_feat"],
            tusb_token=batch["tusb_token"],
        )
    required = ["point_hypotheses", "point_pred", "top1_point_pred", "visibility_logits", "semantic_logits"]
    return {
        "required_output_keys": required,
        "output_keys_present": {k: k in out for k in required},
        "output_shapes": {k: list(v.shape) for k, v in out.items() if torch.is_tensor(v)},
        "mode0_is_last_observed_by_construction": True,
        "damped_gamma": float(gamma),
    }


def main() -> int:
    decision = _load(DECISION_PATH)
    eval_summary = _load(EVAL_PATH)
    bootstrap = _load(BOOT_PATH)
    files = {
        rel: {
            "exists": (ROOT / rel).exists(),
            "size_bytes": (ROOT / rel).stat().st_size if (ROOT / rel).exists() else 0,
        }
        for rel in REQUIRED_FILES
    }
    main_run = eval_summary.get("experiments", {}).get("v28_lastobs_m128_h32_seed42", {})
    run_report = _load(ROOT / "reports/stwm_ostf_v28_runs/v28_lastobs_m128_h32_seed42.json")
    h64_run = eval_summary.get("experiments", {}).get("v28_lastobs_m128_h64_seed42", {})

    semantic_field_lb = _positive(bootstrap.get("m128_vs_wo_semantic_sem_top5", {}))
    sem_trace_boot = bootstrap.get("m128_vs_wo_semantic_hard_minfde", {})
    if _positive(sem_trace_boot):
        semantic_trace_status = "positive"
    elif _negative_significant(sem_trace_boot):
        semantic_trace_status = "negative"
    else:
        semantic_trace_status = "neutral_or_not_significant"
    dense_minfde_lb = _positive(bootstrap.get("m128_vs_wo_dense_hard_minfde", {}))
    dense_extent_lb = _positive(bootstrap.get("m128_vs_wo_dense_extent_hard", {}))
    strongest_prior_ok = bool(decision.get("V28_beats_last_observed_hard_subset") and decision.get("V28_beats_last_observed_all_average"))
    visibility_ok = bool(decision.get("visibility_quality_sufficient"))
    strict_claim = bool(
        strongest_prior_ok
        and dense_minfde_lb
        and semantic_field_lb
        and visibility_ok
        and semantic_trace_status != "negative"
    )
    strict_fields = {
        "semantic_field_load_bearing": semantic_field_lb,
        "semantic_trace_dynamics_load_bearing": semantic_trace_status,
        "semantic_identity_reacquisition_load_bearing": None,
        "semantic_identity_reacquisition_note": "No V28 reacquisition/false-confuser utility metric is wired into this run; semantic identity utility cannot be claimed from V28 alone.",
        "dense_point_minfde_load_bearing": dense_minfde_lb,
        "dense_shape_extent_load_bearing": dense_extent_lb,
        "object_dense_semantic_trace_field_claim_allowed_strict": strict_claim,
        "object_dense_semantic_trace_field_claim_allowed_strict_reason": (
            "strict claim passes"
            if strict_claim
            else "strict claim blocked because semantic trace dynamics is negative or dense shape/identity utility is not established, despite original V28 positive pilot."
        ),
    }
    decision.update(strict_fields)
    decision["strict_claim_audit_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(DECISION_PATH, decision)

    payload = {
        "audit_name": "stwm_ostf_v28_contract_and_claims_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_files": files,
        "all_required_files_exist": all(v["exists"] for v in files.values()),
        "model_input_observed_only": bool(run_report.get("model_input_observed_only")),
        "teacher_source_target_cache_only": bool(run_report.get("teacher_source") == "traceanything_official_trajectory_field" and run_report.get("model_input_observed_only")),
        "forward_contract": _forward_contract(),
        "strongest_prior": decision.get("strongest_prior_used"),
        "strongest_prior_is_last_observed_copy": decision.get("strongest_prior_used") == "last_observed_copy",
        "semantic_memory_load_bearing_original": decision.get("semantic_memory_load_bearing"),
        "semantic_memory_load_bearing_triggered_by_semantic_top5": semantic_field_lb,
        "semantic_trace_dynamics_bootstrap_vs_wo_semantic": sem_trace_boot,
        "semantic_trace_dynamics_load_bearing": semantic_trace_status,
        "dense_point_minfde_bootstrap": bootstrap.get("m128_vs_wo_dense_hard_minfde", {}),
        "dense_shape_extent_bootstrap": bootstrap.get("m128_vs_wo_dense_extent_hard", {}),
        "dense_point_minfde_load_bearing": dense_minfde_lb,
        "dense_shape_extent_load_bearing": dense_extent_lb,
        "H64_seed42_stress_result_exists": bool(h64_run),
        "H64_seed42_test_metrics": h64_run.get("test_metrics", {}),
        **strict_fields,
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V28 Contract And Claims Audit",
        payload,
        [
            "all_required_files_exist",
            "model_input_observed_only",
            "teacher_source_target_cache_only",
            "strongest_prior_is_last_observed_copy",
            "semantic_memory_load_bearing_triggered_by_semantic_top5",
            "semantic_trace_dynamics_load_bearing",
            "dense_point_minfde_load_bearing",
            "dense_shape_extent_load_bearing",
            "H64_seed42_stress_result_exists",
            "object_dense_semantic_trace_field_claim_allowed_strict",
            "object_dense_semantic_trace_field_claim_allowed_strict_reason",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if payload["all_required_files_exist"] and payload["model_input_observed_only"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
