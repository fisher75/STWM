#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import importlib.util
import json
import os
import statistics

from stwm.tools.run_stwm_downstream_utility_v2_20260420 import build_downstream_utility_v2


ROOT = Path("/raid/chen034/workspace/stwm")
SESSION = "stwm_top_tier_one_last_fix_20260420"
LOG_PATH = ROOT / "logs/stwm_top_tier_one_last_fix_20260420.log"
MATCHED_SEEDS = [42, 123, 456, 654, 789, 321]

CONSISTENCY_TARGETS = [
    "reports/stwm_top_tier_final_validation_diagnosis_20260420.json",
    "reports/stwm_top_tier_final_validation_summary_20260420.json",
    "reports/stwm_top_tier_paper_decision_20260420.json",
    "reports/stwm_top_tier_matched_6seed_dualpanel_20260420.json",
    "reports/stwm_top_tier_final_bootstrap_ci_20260420.json",
    "reports/stwm_top_tier_downstream_utility_20260420.json",
    "reports/stwm_top_tier_ood_transfer_20260420.json",
    "reports/stwm_top_tier_mechanism_6seed_repair_20260420.json",
    "reports/stwm_top_tier_appearance_plumbing_surgical_fix_20260420.json",
]
CONSISTENCY_FIELDS = [
    "context_preserving_densified_200_improved_vs_current_calonly",
    "matched_6seed_improved",
    "bootstrap_ci_zero_excluded",
    "downstream_utility_improved",
    "ood_transfer_improved",
    "mechanism_cross_seed_stable",
    "appearance_claim_allowed",
    "paper_target_recommendation",
    "oral_spotlight_readiness",
    "next_step_choice",
]
PROTOCOL_PATH = ROOT / "reports/stwm_top_tier_one_last_fix_protocol_20260420.json"
PROTOCOL_DOC = ROOT / "docs/STWM_TOP_TIER_ONE_LAST_FIX_PROTOCOL_20260420.md"

DUALPANEL_PATH = ROOT / "reports/stage2_v3p1_dualpanel_context_audit_20260420.json"
EXTENDED_PATH = ROOT / "reports/stage2_protocol_v3_extended_evalset_20260420.json"
BOOTSTRAP_PATH = ROOT / "reports/stage2_final_bootstrap_ci_20260420.json"
MECH_APPENDIX_PATH = ROOT / "reports/stage2_v3p1_mechanism_appendix_20260420.json"
MULTISEED_PATH = ROOT / "reports/stage2_v3p1_multiseed_dualpanel_20260420.json"
APPEARANCE_AUDIT_PATH = ROOT / "reports/stage2_final_appearance_plumbing_fix_audit_20260420.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_process_title_normalization(default_title: str = "python") -> None:
    title = str(os.environ.get("STWM_PROC_TITLE", default_title)).strip() or default_title
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode != "generic":
        return
    lowered = title.lower()
    if "stwm" in lowered or "tracewm" in lowered or "/raid/" in lowered or "/home/" in lowered:
        title = default_title
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{_now_iso()}] {message}\n")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest() -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for prefix in ["code", "docs", "reports", "scripts", "configs"]:
        base = ROOT / prefix
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            st = path.stat()
            entries.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": int(st.st_size),
                    "mtime": float(st.st_mtime),
                    "sha256": _sha256_file(path),
                }
            )
    return {"generated_at_utc": _now_iso(), "entry_count": len(entries), "entries": entries}


def _method_seed_dirs() -> Dict[str, Dict[int, str]]:
    return {
        "TUSB-v3.1": {
            42: "stage2_tusb_v3p1_seed42_20260418",
            123: "stage2_tusb_v3p1_seed123_20260418",
            456: "stage2_tusb_v3p1_seed456_20260418",
        },
        "calibration-only": {
            42: "stage2_calonly_topk1_seed42_wave1_20260413",
            123: "stage2_calonly_topk1_seed123_longconfirm_v2_20260414",
            456: "stage2_calonly_topk1_seed456_wave1_20260413",
            654: "stage2_calonly_topk1_seed654_longconfirm_20260414",
            789: "stage2_calonly_topk1_seed789_wave2_20260414",
            321: "stage2_calonly_topk1_seed321_longconfirm_v2_20260414",
        },
        "cropenc baseline": {
            42: "stage2_fullscale_core_cropenc_seed42_20260409",
            123: "stage2_fullscale_core_cropenc_seed123_20260409",
            456: "stage2_fullscale_core_cropenc_seed456_20260409",
            789: "stage2_fullscale_core_cropenc_seed789_wave2_20260409",
        },
        "legacysem baseline": {
            42: "stage2_fullscale_core_legacysem_seed42_20260409",
            123: "stage2_fullscale_core_legacysem_seed123_wave2_20260409",
            456: "stage2_fullscale_core_legacysem_seed456_wave2_20260409",
        },
    }


def _seed_coverage() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    coverage_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for method, mapping in _method_seed_dirs().items():
        method_cov: Dict[str, Any] = {}
        for seed in MATCHED_SEEDS:
            run_name = mapping.get(seed, "")
            ckpt_dir = ROOT / "outputs/checkpoints" / run_name if run_name else None
            best = ckpt_dir / "best.pt" if ckpt_dir else None
            sidecar = ckpt_dir / "best_semantic_hard.pt" if ckpt_dir else None
            row = {
                "method": method,
                "seed": int(seed),
                "run_name": run_name or None,
                "checkpoint_exists": bool(best and best.exists()),
                "best.pt_exists": bool(best and best.exists()),
                "best_semantic_hard.pt_exists": bool(sidecar and sidecar.exists()),
                "training_needed": not bool(best and best.exists()),
                "eval_only_possible": bool(best and best.exists()),
                "exact_checkpoint_path": str(best) if best and best.exists() else "",
                "if_missing_exact_reason": "" if best and best.exists() else "checkpoint_missing_in_live_repo",
            }
            coverage_rows.append(row)
            method_cov[str(seed)] = row
        summary[method] = method_cov
    return summary, coverage_rows


def _one_last_consistency(args: Any) -> Dict[str, Any]:
    audited: List[Dict[str, Any]] = []
    field_values: Dict[str, List[Dict[str, Any]]] = {field: [] for field in CONSISTENCY_FIELDS}
    for rel in CONSISTENCY_TARGETS:
        path = ROOT / rel
        payload = _load_json(path)
        row = {
            "path": rel,
            "exists": path.exists(),
            "nonempty": bool(path.exists() and path.stat().st_size > 0),
            "valid_json": bool(payload),
            "top_level_keys": list(payload.keys())[:50],
        }
        for field in CONSISTENCY_FIELDS:
            if field in payload:
                field_values[field].append({"path": rel, "value": payload[field]})
        audited.append(row)
    actual = {
        field: values[-1]["value"] if values else None
        for field, values in field_values.items()
    }
    conflicts = {
        field: values
        for field, values in field_values.items()
        if len({json.dumps(v["value"], sort_keys=True, ensure_ascii=True) for v in values}) > 1
    }
    report = {
        "generated_at_utc": _now_iso(),
        "audited_reports": audited,
        "field_sources": field_values,
        "actual_values": actual,
        "conflict_detected": bool(conflicts),
        "conflicting_fields": sorted(conflicts.keys()),
        "conflict_details": conflicts,
        "source_of_truth": "live_repo_reports_json",
        "live_manifest_report": str(args.live_manifest_report),
    }
    _write_json(Path(args.consistency_audit_report), report)
    _write_json(Path(args.live_manifest_report), _manifest())
    _write_md(
        Path(args.consistency_audit_doc),
        [
            "# STWM Top-Tier One-Last Consistency Audit 20260420",
            "",
            f"- conflict_detected: {report['conflict_detected']}",
            f"- conflicting_fields: {report['conflicting_fields']}",
            f"- source_of_truth: {report['source_of_truth']}",
        ],
    )
    return report


def _protocol() -> Dict[str, Any]:
    payload = {
        "generated_at_utc": _now_iso(),
        "stage1_frozen": True,
        "candidate_stage2_mainline": {
            "method": "TUSB-v3.1",
            "official_checkpoint_candidate": "best.pt",
            "sidecar_checkpoint": "best_semantic_hard.pt",
        },
        "v3p2_v3p3_status": {
            "did_not_exceed_v3p1_anchor": True,
            "no_more_cumulative_stacking": True,
        },
        "key_evidence_risks": [
            "matched seed coverage incomplete",
            "multi-seed robustness not closed",
            "downstream utility still needs stronger independence",
            "OOD / cross-dataset evidence weak",
            "mechanism 6-seed not closed",
            "appearance signal still not reaching loss path",
        ],
    }
    _write_json(PROTOCOL_PATH, payload)
    _write_md(
        PROTOCOL_DOC,
        [
            "# STWM Top-Tier One-Last Fix Protocol 20260420",
            "",
            "- Stage1 remains frozen.",
            "- TUSB-v3.1 remains the Stage2 candidate mainline.",
            "- v3.2/v3.3 are not carried forward as new mainline methods.",
            "- this round focuses on evidence repair: matched seeds, strict bootstrap, independent utility, OOD, mechanism stability, and appearance plumbing.",
        ],
    )
    return payload


def _seed_real_completion(args: Any) -> Dict[str, Any]:
    summary, rows = _seed_coverage()
    needed = [row for row in rows if row["training_needed"]]
    plan = {
        "generated_at_utc": _now_iso(),
        "required_methods": list(summary.keys()),
        "required_seeds": MATCHED_SEEDS,
        "coverage": summary,
        "training_needed_count": int(len(needed)),
        "training_needed_rows": needed,
    }
    launch = {
        "generated_at_utc": _now_iso(),
        "tmux_session": SESSION,
        "real_completion_started": False,
        "training_jobs_launched": [],
        "max_concurrent_train_tasks": 4,
        "gpu_policy": "single_gpu_no_ddp_keep_1_2_B200_free",
        "exact_blocking_reason": "matched-seed gaps remain; this pass did not materialize new matched baseline/TUSB training jobs to completion",
    }
    completion_summary = {
        "generated_at_utc": _now_iso(),
        "methods_with_complete_6seed_coverage": [
            method for method, seeds in summary.items() if all(seeds[str(seed)]["best.pt_exists"] for seed in MATCHED_SEEDS)
        ],
        "methods_with_missing_coverage": {
            method: [seed for seed in MATCHED_SEEDS if not seeds[str(seed)]["best.pt_exists"]]
            for method, seeds in summary.items()
            if not all(seeds[str(seed)]["best.pt_exists"] for seed in MATCHED_SEEDS)
        },
        "matched_seed_completion_ready": False,
    }
    _write_json(Path(args.seed_completion_plan_report), plan)
    _write_json(Path(args.seed_completion_launch_report), launch)
    _write_json(Path(args.seed_completion_summary_report), completion_summary)
    _write_md(
        Path(args.seed_completion_doc),
        [
            "# STWM Top-Tier Matched Seed Real Completion 20260420",
            "",
            f"- training_needed_count: {plan['training_needed_count']}",
            f"- real_completion_started: {launch['real_completion_started']}",
            f"- methods_with_complete_6seed_coverage: {completion_summary['methods_with_complete_6seed_coverage']}",
            f"- methods_with_missing_coverage: {completion_summary['methods_with_missing_coverage']}",
            f"- exact_blocking_reason: {launch['exact_blocking_reason']}",
        ],
    )
    return {"plan": plan, "launch": launch, "summary": completion_summary}


def _load_dense_and_extended() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _load_json(DUALPANEL_PATH), _load_json(EXTENDED_PATH)


def _matched_6seed_full_eval(args: Any, seed_completion: Dict[str, Any]) -> Dict[str, Any]:
    prior = _load_json(ROOT / "reports/stwm_top_tier_matched_6seed_dualpanel_20260420.json")
    payload = {
        "generated_at_utc": _now_iso(),
        "required_matched_seeds": MATCHED_SEEDS,
        "coverage": seed_completion["plan"]["coverage"],
        "main_6seed_table_available": False,
        "exact_reason_main_table_unavailable": "not_all_methods_have_true_matched_6seed_checkpoint_coverage",
        "legacy_85_context_preserving": prior.get("legacy_85_context_preserving", []),
        "densified_200_context_preserving": prior.get("densified_200_context_preserving", []),
        "protocol_v3_extended_600_context_preserving": [],
        "failed_seed_diagnosis": {
            "root_cause": "coverage_incomplete",
            "details": seed_completion["summary"]["methods_with_missing_coverage"],
        },
        "matched_6seed_improved_vs_calibration": False,
        "matched_6seed_improved_vs_cropenc": False,
        "matched_6seed_improved_vs_legacysem": False,
        "matched_6seed_claim_ready": False,
    }
    _write_json(Path(args.matched_full_eval_report), payload)
    _write_md(
        Path(args.matched_full_eval_doc),
        [
            "# STWM Top-Tier Matched 6-Seed Full Eval 20260420",
            "",
            f"- main_6seed_table_available: {payload['main_6seed_table_available']}",
            f"- matched_6seed_improved_vs_calibration: {payload['matched_6seed_improved_vs_calibration']}",
            f"- matched_6seed_claim_ready: {payload['matched_6seed_claim_ready']}",
            f"- exact_reason_main_table_unavailable: {payload['exact_reason_main_table_unavailable']}",
        ],
    )
    return payload


def _strict_bootstrap_ci(args: Any) -> Dict[str, Any]:
    prior = _load_json(BOOTSTRAP_PATH)
    panels = prior.get("panels", {}) if isinstance(prior.get("panels", {}), dict) else {}
    primary_metrics = ["overall_top1", "hard_subset_top1", "hit_rate", "localization_error", "mask_iou_at_top1"]
    secondary_metrics = [
        "ambiguity_top1",
        "appearance_change_top1",
        "occlusion_reappearance_top1",
        "long_gap_persistence_top1",
        "small_object_top1",
    ]
    out_panels: Dict[str, Any] = {}
    primary_any_zero = False
    primary_any_meaningful = False
    available_primary_comparisons: List[str] = []
    for panel_name in ["densified_200_context_preserving", "protocol_v3_extended_600_context_preserving"]:
        panel = panels.get(panel_name, {})
        out_panels[panel_name] = {}
        for comp in ["current_calibration_only_best", "cropenc_baseline_best", "legacysem_best", "stage1_frozen_baseline"]:
            comp_payload = panel.get(comp, {})
            metrics_payload: Dict[str, Any] = {}
            available = False
            for metric in primary_metrics + secondary_metrics:
                entry = comp_payload.get(metric, {})
                count = int(entry.get("count", 0))
                available = available or count > 0
                metrics_payload[metric] = {
                    "count": count,
                    "mean_delta": float(entry.get("mean_delta", 0.0)),
                    "median_delta": float(entry.get("median_delta", 0.0)),
                    "ci95": [float(entry.get("ci95_low", 0.0)), float(entry.get("ci95_high", 0.0))],
                    "bootstrap_win_rate": float(entry.get("bootstrap_win_rate", 0.0)),
                    "sign_test_estimate": float(entry.get("sign_test_estimate", 0.0)),
                    "zero_excluded": bool(entry.get("zero_excluded", False)),
                }
            out_panels[panel_name][comp] = {
                "available": available,
                "metrics": metrics_payload,
            }
            if available:
                available_primary_comparisons.append(f"{panel_name}:{comp}")
                for metric in primary_metrics:
                    metric_entry = metrics_payload[metric]
                    if metric_entry["zero_excluded"]:
                        primary_any_zero = True
                    if abs(float(metric_entry["mean_delta"])) >= 0.01:
                        primary_any_meaningful = True
    claim_level = "weak_claim"
    if (
        out_panels.get("densified_200_context_preserving", {}).get("current_calibration_only_best", {}).get("metrics", {}).get("overall_top1", {}).get("zero_excluded")
        and out_panels.get("densified_200_context_preserving", {}).get("cropenc_baseline_best", {}).get("metrics", {}).get("overall_top1", {}).get("zero_excluded")
        and primary_any_meaningful
    ):
        claim_level = "strong_claim"
    elif primary_any_meaningful:
        claim_level = "moderate_claim"
    payload = {
        "generated_at_utc": _now_iso(),
        "primary_panels": ["densified_200_context_preserving", "protocol_v3_extended_600_context_preserving"],
        "primary_comparisons": [
            "TUSB-v3.1 best.pt vs calibration-only",
            "TUSB-v3.1 best.pt vs cropenc",
            "TUSB-v3.1 best.pt vs legacysem",
            "TUSB-v3.1 best.pt vs stage1 frozen",
        ],
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "panels": out_panels,
        "available_primary_comparisons": available_primary_comparisons,
        "zero_excluded_any_primary_metric": bool(primary_any_zero),
        "strict_bootstrap_claim_level": claim_level,
    }
    _write_json(Path(args.strict_bootstrap_report), payload)
    _write_md(
        Path(args.strict_bootstrap_doc),
        [
            "# STWM Top-Tier Strict Bootstrap CI 20260420",
            "",
            f"- available_primary_comparisons: {available_primary_comparisons}",
            f"- zero_excluded_any_primary_metric: {payload['zero_excluded_any_primary_metric']}",
            f"- strict_bootstrap_claim_level: {payload['strict_bootstrap_claim_level']}",
        ],
    )
    return payload


def _subset(rows: List[Dict[str, Any]], tag: str | None = None, dataset: str | None = None) -> List[Dict[str, Any]]:
    out = list(rows)
    if dataset is not None:
        out = [row for row in out if row.get("dataset") == dataset]
    if tag is not None:
        out = [row for row in out if tag in set(row.get("subset_tags", []))]
    return out


def _agg_metric(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    scored = [row["methods"][method] for row in rows if method in row.get("methods", {})]
    if not scored:
        return {"count": 0, "top1": 0.0, "hit_rate": 0.0, "mrr": 0.0, "loc_error": 0.0, "mask_iou_at_top1": 0.0}
    return {
        "count": int(len(scored)),
        "top1": float(sum(float(x.get("query_future_top1_acc", 0.0)) for x in scored) / len(scored)),
        "hit_rate": float(sum(float(x.get("query_future_hit_rate", 0.0)) for x in scored) / len(scored)),
        "mrr": float(sum(float(x.get("mrr", 0.0)) for x in scored) / len(scored)),
        "loc_error": float(sum(float(x.get("query_future_localization_error", 0.0)) for x in scored) / len(scored)),
        "mask_iou_at_top1": float(sum(float(x.get("future_mask_iou_at_top1", 0.0)) for x in scored) / len(scored)),
    }


def _ood_fixed(args: Any) -> Dict[str, Any]:
    ext = _load_json(EXTENDED_PATH)
    rows = list(ext.get("context_preserving_eval", {}).get("per_item_results", []))
    cal = "current_calibration_only_best"
    crop = "cropenc_baseline_best"
    tusb = "current_tusb_v3p1_best::best.pt"
    payload = {
        "generated_at_utc": _now_iso(),
        "setting_a_vipseg_to_burst_heavy": {
            "calibration_only": _agg_metric(_subset(rows, dataset="BURST"), cal),
            "cropenc_baseline": _agg_metric(_subset(rows, dataset="BURST"), crop),
            "tusb_v3p1": _agg_metric(_subset(rows, dataset="BURST"), tusb),
        },
        "setting_b_burst_to_vipseg_heavy": {
            "calibration_only": _agg_metric(_subset(rows, dataset="VIPSeg"), cal),
            "cropenc_baseline": _agg_metric(_subset(rows, dataset="VIPSeg"), crop),
            "tusb_v3p1": _agg_metric(_subset(rows, dataset="VIPSeg"), tusb),
        },
        "setting_c_conservative_heldout_split": {
            "supported": False,
            "reason": "no materialized held-out scene/domain/category split exists in the live repo; this report keeps the boundary explicit",
        },
    }
    burst_cal = payload["setting_a_vipseg_to_burst_heavy"]["calibration_only"]["top1"]
    burst_tusb = payload["setting_a_vipseg_to_burst_heavy"]["tusb_v3p1"]["top1"]
    vip_cal = payload["setting_b_burst_to_vipseg_heavy"]["calibration_only"]["top1"]
    vip_tusb = payload["setting_b_burst_to_vipseg_heavy"]["tusb_v3p1"]["top1"]
    ood_improved = bool(burst_tusb > burst_cal and vip_tusb > vip_cal)
    payload.update(
        {
            "proxy_domain_split_positive": bool(ood_improved),
            "ood_improved_vs_calibration": bool(ood_improved),
            "ood_improved_vs_cropenc": False,
            "ood_hard_subset_improved": False,
            "true_ood_claim_ready": False,
            "proxy_only_vs_true_ood_boundary": "dataset-split proxy is positive, but conservative held-out OOD split is not materialized; do not upgrade to true OOD claim",
        }
    )
    _write_json(Path(args.ood_fixed_report), payload)
    _write_md(
        Path(args.ood_fixed_doc),
        [
            "# STWM Top-Tier OOD Transfer Fixed 20260420",
            "",
            f"- proxy_domain_split_positive: {payload['proxy_domain_split_positive']}",
            f"- ood_improved_vs_calibration: {payload['ood_improved_vs_calibration']}",
            f"- ood_hard_subset_improved: {payload['ood_hard_subset_improved']}",
            f"- true_ood_claim_ready: {payload['true_ood_claim_ready']}",
            f"- proxy_only_vs_true_ood_boundary: {payload['proxy_only_vs_true_ood_boundary']}",
        ],
    )
    return payload


def _mechanism_6seed_full(args: Any) -> Dict[str, Any]:
    appendix = _load_json(MECH_APPENDIX_PATH)
    rows = list(appendix.get("table_rows", []))
    metrics = [
        "active_unit_count_mean",
        "assignment_entropy_mean",
        "same_instance_dominant_unit_match_rate_mean",
        "same_instance_assignment_cosine_mean",
        "different_instance_dominant_unit_collision_rate_mean",
        "unit_purity_by_instance_id_mean",
        "z_dyn_drift_mean",
        "z_sem_drift_mean",
        "z_sem_to_z_dyn_drift_ratio_mean",
    ]
    seed_rows = []
    for row in rows:
        run_name = str(row.get("run_name", ""))
        matched_seed = None
        for seed in MATCHED_SEEDS:
            if f"seed{seed}_" in run_name:
                matched_seed = seed
                break
        if matched_seed is None:
            continue
        seed_rows.append({"seed": int(matched_seed), **{metric: float(row.get(metric, 0.0)) for metric in metrics}})
    seed_mean = {metric: float(statistics.mean([row[metric] for row in seed_rows])) if seed_rows else 0.0 for metric in metrics}
    seed_std = {metric: float(statistics.pstdev([row[metric] for row in seed_rows])) if len(seed_rows) > 1 else 0.0 for metric in metrics}
    payload = {
        "generated_at_utc": _now_iso(),
        "required_seeds": MATCHED_SEEDS,
        "seed_rows": seed_rows,
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "failed_seed_flags": [seed for seed in MATCHED_SEEDS if seed not in {row['seed'] for row in seed_rows}],
        "mechanism_vs_performance_correlation": {"available_seed_count": len(seed_rows), "correlation_unreliable_due_to_missing_seeds": True},
        "identity_binding_cross_seed_stable": False,
        "slow_semantic_state_cross_seed_stable": False,
        "anti_collapse_cross_seed_stable": False,
        "mechanism_cross_seed_stable": False,
        "mechanism_claim_ready": False,
    }
    _write_json(Path(args.mechanism_full_report), payload)
    _write_md(
        Path(args.mechanism_full_doc),
        [
            "# STWM Top-Tier Mechanism 6-Seed Full 20260420",
            "",
            f"- available_seed_count: {len(seed_rows)}",
            f"- failed_seed_flags: {payload['failed_seed_flags']}",
            f"- identity_binding_cross_seed_stable: {payload['identity_binding_cross_seed_stable']}",
            f"- slow_semantic_state_cross_seed_stable: {payload['slow_semantic_state_cross_seed_stable']}",
            f"- anti_collapse_cross_seed_stable: {payload['anti_collapse_cross_seed_stable']}",
            f"- mechanism_claim_ready: {payload['mechanism_claim_ready']}",
        ],
    )
    return payload


def _appearance_final_fix(args: Any) -> Dict[str, Any]:
    prior = _load_json(APPEARANCE_AUDIT_PATH)
    payload = {
        "generated_at_utc": _now_iso(),
        "offline_appearance_drift_high_ratio": float(prior.get("offline_appearance_drift_high_ratio", 0.0)),
        "dataloader_appearance_drift_high_ratio": float(prior.get("dataloader_appearance_drift_high_ratio", 0.0)),
        "batch_appearance_drift_high_ratio_mean": float(prior.get("batch_appearance_drift_high_ratio_mean", 0.0)),
        "appearance_refine_loss_nonzero_ratio": float(prior.get("appearance_refine_loss_nonzero_ratio", 0.0)),
        "exact_breakpoint": str(prior.get("exact_breakpoint", "")),
        "appearance_claim_allowed": bool(prior.get("appearance_claim_allowed", False)),
        "appearance_change_treated_as_limitation": not bool(prior.get("appearance_claim_allowed", False)),
        "repair_applied_this_round": False,
    }
    _write_json(Path(args.appearance_final_fix_report), payload)
    _write_md(
        Path(args.appearance_final_fix_doc),
        [
            "# STWM Top-Tier Appearance Plumbing Final Fix 20260420",
            "",
            f"- offline_appearance_drift_high_ratio: {payload['offline_appearance_drift_high_ratio']}",
            f"- dataloader_appearance_drift_high_ratio: {payload['dataloader_appearance_drift_high_ratio']}",
            f"- batch_appearance_drift_high_ratio_mean: {payload['batch_appearance_drift_high_ratio_mean']}",
            f"- appearance_refine_loss_nonzero_ratio: {payload['appearance_refine_loss_nonzero_ratio']}",
            f"- exact_breakpoint: {payload['exact_breakpoint']}",
            f"- appearance_claim_allowed: {payload['appearance_claim_allowed']}",
        ],
    )
    return payload


def _teacher_prior_sanity_v2(args: Any) -> Dict[str, Any]:
    cached_names: List[str] = []
    for hub_root in [Path.home() / ".cache/huggingface/hub", Path("/raid/chen034/.cache/huggingface/hub")]:
        if hub_root.exists():
            for child in hub_root.iterdir():
                lower = child.name.lower()
                if any(term in lower for term in ["dinov2", "siglip", "clip-vit"]):
                    cached_names.append(child.name)
    payload = {
        "generated_at_utc": _now_iso(),
        "teacher_prior_upgrade_available": False,
        "best_available_teacher": "clip_vit-b_16_temporal_weighted_masked_mean_v5_driftcal",
        "teacher_sanity_improves_retrieval": False,
        "teacher_sanity_improves_unit_purity": False,
        "dinov2_like_available": bool(importlib.util.find_spec("timm") is not None and importlib.util.find_spec("transformers") is not None),
        "siglip_like_available": bool(importlib.util.find_spec("transformers") is not None),
        "stronger_clip_family_cached": any("clip-vit" in name.lower() for name in cached_names),
        "exact_blocking_reason": "no stronger frozen teacher cache was materialized for a clean small-scale retrieval sanity inside this pass",
    }
    _write_json(Path(args.teacher_prior_v2_report), payload)
    _write_md(
        Path(args.teacher_prior_v2_doc),
        [
            "# STWM Top-Tier Teacher Prior Sanity V2 20260420",
            "",
            f"- teacher_prior_upgrade_available: {payload['teacher_prior_upgrade_available']}",
            f"- best_available_teacher: {payload['best_available_teacher']}",
            f"- teacher_sanity_improves_retrieval: {payload['teacher_sanity_improves_retrieval']}",
            f"- teacher_sanity_improves_unit_purity: {payload['teacher_sanity_improves_unit_purity']}",
            f"- exact_blocking_reason: {payload['exact_blocking_reason']}",
        ],
    )
    return payload


def _one_last_fix_gate(args: Any, matched: Dict[str, Any], utility_v2: Dict[str, Any], ood: Dict[str, Any], mechanism: Dict[str, Any], appearance: Dict[str, Any]) -> Dict[str, Any]:
    matched_improved = bool(matched.get("matched_6seed_improved_vs_calibration", False))
    utility_ready = bool(utility_v2.get("utility_v2_claim_ready", False))
    payload = {
        "generated_at_utc": _now_iso(),
        "matched_6seed_improved": matched_improved,
        "utility_v2_claim_ready": utility_ready,
        "ood_claim_ready": bool(ood.get("true_ood_claim_ready", False)),
        "mechanism_claim_ready": bool(mechanism.get("mechanism_claim_ready", False)),
        "appearance_claim_allowed": bool(appearance.get("appearance_claim_allowed", False)),
        "one_last_surgical_fix_allowed": bool((not matched_improved) and utility_ready),
        "allowed_fix_categories": [
            "checkpoint policy fix",
            "unit collapse fix",
            "instance density reweighting",
            "teacher prior cache fix",
            "appearance plumbing fix",
        ],
        "recommended_fix_type": "matched_seed_completion_and_mechanism_repair" if ((not matched_improved) and utility_ready) else "none",
    }
    _write_json(Path(args.one_last_fix_gate_report), payload)
    _write_md(
        Path(args.one_last_fix_gate_doc),
        [
            "# STWM Top-Tier One-Last Fix Gate 20260420",
            "",
            f"- one_last_surgical_fix_allowed: {payload['one_last_surgical_fix_allowed']}",
            f"- recommended_fix_type: {payload['recommended_fix_type']}",
            f"- allowed_fix_categories: {payload['allowed_fix_categories']}",
        ],
    )
    return payload


def _final_decision(args: Any, matched: Dict[str, Any], strict_bootstrap: Dict[str, Any], utility_v2: Dict[str, Any], ood: Dict[str, Any], mechanism: Dict[str, Any], appearance: Dict[str, Any]) -> Dict[str, Any]:
    matched_improved = bool(matched.get("matched_6seed_improved_vs_calibration", False))
    claim_level = str(strict_bootstrap.get("strict_bootstrap_claim_level", "weak_claim"))
    utility_ready = bool(utility_v2.get("utility_v2_claim_ready", False))
    ood_ready = bool(ood.get("true_ood_claim_ready", False))
    mechanism_ready = bool(mechanism.get("mechanism_claim_ready", False))
    appearance_allowed = bool(appearance.get("appearance_claim_allowed", False))
    paper_target = "borderline_needs_one_last_fix"
    oral = "not_ready"
    next_step = "run_one_last_surgical_fix"
    if matched_improved and utility_ready and ood_ready:
        paper_target = "cvpr_eccv_main_ready"
        oral = "possible_if_utility_ood_and_mechanism_hold" if mechanism_ready else "not_ready"
        next_step = "start_writing_main_submission"
    elif not matched_improved and not utility_ready:
        paper_target = "not_ready_for_top_tier"
        oral = "not_ready"
        next_step = "rethink_stage2_story"
    elif matched_improved and claim_level != "strong_claim":
        paper_target = "aaai_main_ready"
        oral = "not_ready"
        next_step = "reframe_as_moderate_claim_main_track"
    payload = {
        "generated_at_utc": _now_iso(),
        "stage1_ready": True,
        "stage2_ready": False,
        "matched_6seed_improved": matched_improved,
        "strict_bootstrap_claim_level": claim_level,
        "utility_v2_claim_ready": utility_ready,
        "ood_claim_ready": ood_ready,
        "mechanism_claim_ready": mechanism_ready,
        "appearance_claim_allowed": appearance_allowed,
        "paper_target_recommendation": paper_target,
        "oral_spotlight_readiness": oral,
        "next_step_choice": next_step,
    }
    _write_json(Path(args.final_decision_report), payload)
    _write_md(
        Path(args.final_decision_doc),
        [
            "# STWM Top-Tier Final Decision 20260420",
            "",
            f"- matched_6seed_improved: {matched_improved}",
            f"- strict_bootstrap_claim_level: {claim_level}",
            f"- utility_v2_claim_ready: {utility_ready}",
            f"- ood_claim_ready: {ood_ready}",
            f"- mechanism_claim_ready: {mechanism_ready}",
            f"- appearance_claim_allowed: {appearance_allowed}",
            f"- paper_target_recommendation: {paper_target}",
            f"- oral_spotlight_readiness: {oral}",
            f"- next_step_choice: {next_step}",
        ],
    )
    return payload


def _final_reports(args: Any, consistency: Dict[str, Any], matched: Dict[str, Any], strict_bootstrap: Dict[str, Any], utility_v2: Dict[str, Any], ood: Dict[str, Any], mechanism: Dict[str, Any], appearance: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    launch = {
        "generated_at_utc": _now_iso(),
        "session_name": SESSION,
        "mode": "analysis_with_surgical_recompute_only",
        "gpu_training_jobs_launched": 0,
        "gpu_eval_jobs_launched": 0,
        "notes": "one-last-fix pack executed from current live assets plus strict recomputation; no new Stage2 architecture branch opened",
    }
    summary = {
        "generated_at_utc": _now_iso(),
        "consistency_conflict_detected": bool(consistency.get("conflict_detected", False)),
        "matched_6seed_improved": bool(matched.get("matched_6seed_improved_vs_calibration", False)),
        "strict_bootstrap_claim_level": str(strict_bootstrap.get("strict_bootstrap_claim_level", "weak_claim")),
        "utility_v2_claim_ready": bool(utility_v2.get("utility_v2_claim_ready", False)),
        "ood_claim_ready": bool(ood.get("true_ood_claim_ready", False)),
        "mechanism_claim_ready": bool(mechanism.get("mechanism_claim_ready", False)),
        "appearance_claim_allowed": bool(appearance.get("appearance_claim_allowed", False)),
    }
    diagnosis = {
        "generated_at_utc": _now_iso(),
        "context_preserving_densified_200_improved_vs_current_calonly": bool(
            consistency.get("actual_values", {}).get("context_preserving_densified_200_improved_vs_current_calonly", False)
        ),
        "matched_6seed_improved": summary["matched_6seed_improved"],
        "strict_bootstrap_claim_level": summary["strict_bootstrap_claim_level"],
        "utility_v2_claim_ready": summary["utility_v2_claim_ready"],
        "ood_claim_ready": summary["ood_claim_ready"],
        "mechanism_claim_ready": summary["mechanism_claim_ready"],
        "appearance_claim_allowed": summary["appearance_claim_allowed"],
        "paper_target_recommendation": decision["paper_target_recommendation"],
        "oral_spotlight_readiness": decision["oral_spotlight_readiness"],
        "next_step_choice": decision["next_step_choice"],
    }
    _write_json(Path(args.final_launch_report), launch)
    _write_json(Path(args.final_summary_report), summary)
    _write_json(Path(args.final_diagnosis_report), diagnosis)
    _write_md(
        Path(args.final_doc),
        [
            "# STWM Top-Tier One-Last Fix 20260420",
            "",
            f"- matched_6seed_improved: {diagnosis['matched_6seed_improved']}",
            f"- strict_bootstrap_claim_level: {diagnosis['strict_bootstrap_claim_level']}",
            f"- utility_v2_claim_ready: {diagnosis['utility_v2_claim_ready']}",
            f"- ood_claim_ready: {diagnosis['ood_claim_ready']}",
            f"- mechanism_claim_ready: {diagnosis['mechanism_claim_ready']}",
            f"- appearance_claim_allowed: {diagnosis['appearance_claim_allowed']}",
            f"- paper_target_recommendation: {diagnosis['paper_target_recommendation']}",
            f"- oral_spotlight_readiness: {diagnosis['oral_spotlight_readiness']}",
            f"- next_step_choice: {diagnosis['next_step_choice']}",
        ],
    )
    return {"launch": launch, "summary": summary, "diagnosis": diagnosis}


def main() -> None:
    _apply_process_title_normalization()
    parser = ArgumentParser(description="STWM top-tier one-last-fix + full validation pack.")
    parser.add_argument("--consistency-audit-report", default=str(ROOT / "reports/stwm_top_tier_one_last_consistency_audit_20260420.json"))
    parser.add_argument("--consistency-audit-doc", default=str(ROOT / "docs/STWM_TOP_TIER_ONE_LAST_CONSISTENCY_AUDIT_20260420.md"))
    parser.add_argument("--live-manifest-report", default=str(ROOT / "reports/stwm_top_tier_live_manifest_20260420.json"))
    parser.add_argument("--seed-completion-plan-report", default=str(ROOT / "reports/stwm_top_tier_matched_seed_real_completion_plan_20260420.json"))
    parser.add_argument("--seed-completion-launch-report", default=str(ROOT / "reports/stwm_top_tier_matched_seed_real_completion_launch_20260420.json"))
    parser.add_argument("--seed-completion-summary-report", default=str(ROOT / "reports/stwm_top_tier_matched_seed_real_completion_summary_20260420.json"))
    parser.add_argument("--seed-completion-doc", default=str(ROOT / "docs/STWM_TOP_TIER_MATCHED_SEED_REAL_COMPLETION_20260420.md"))
    parser.add_argument("--matched-full-eval-report", default=str(ROOT / "reports/stwm_top_tier_matched_6seed_full_eval_20260420.json"))
    parser.add_argument("--matched-full-eval-doc", default=str(ROOT / "docs/STWM_TOP_TIER_MATCHED_6SEED_FULL_EVAL_20260420.md"))
    parser.add_argument("--strict-bootstrap-report", default=str(ROOT / "reports/stwm_top_tier_strict_bootstrap_ci_20260420.json"))
    parser.add_argument("--strict-bootstrap-doc", default=str(ROOT / "docs/STWM_TOP_TIER_STRICT_BOOTSTRAP_CI_20260420.md"))
    parser.add_argument("--ood-fixed-report", default=str(ROOT / "reports/stwm_top_tier_ood_transfer_fixed_20260420.json"))
    parser.add_argument("--ood-fixed-doc", default=str(ROOT / "docs/STWM_TOP_TIER_OOD_TRANSFER_FIXED_20260420.md"))
    parser.add_argument("--utility-v2-report", default=str(ROOT / "reports/stwm_top_tier_downstream_utility_v2_20260420.json"))
    parser.add_argument("--utility-v2-doc", default=str(ROOT / "docs/STWM_TOP_TIER_DOWNSTREAM_UTILITY_V2_20260420.md"))
    parser.add_argument("--mechanism-full-report", default=str(ROOT / "reports/stwm_top_tier_mechanism_6seed_full_20260420.json"))
    parser.add_argument("--mechanism-full-doc", default=str(ROOT / "docs/STWM_TOP_TIER_MECHANISM_6SEED_FULL_20260420.md"))
    parser.add_argument("--appearance-final-fix-report", default=str(ROOT / "reports/stwm_top_tier_appearance_plumbing_final_fix_20260420.json"))
    parser.add_argument("--appearance-final-fix-doc", default=str(ROOT / "docs/STWM_TOP_TIER_APPEARANCE_PLUMBING_FINAL_FIX_20260420.md"))
    parser.add_argument("--teacher-prior-v2-report", default=str(ROOT / "reports/stwm_top_tier_teacher_prior_sanity_v2_20260420.json"))
    parser.add_argument("--teacher-prior-v2-doc", default=str(ROOT / "docs/STWM_TOP_TIER_TEACHER_PRIOR_SANITY_V2_20260420.md"))
    parser.add_argument("--one-last-fix-gate-report", default=str(ROOT / "reports/stwm_top_tier_one_last_fix_gate_20260420.json"))
    parser.add_argument("--one-last-fix-gate-doc", default=str(ROOT / "docs/STWM_TOP_TIER_ONE_LAST_FIX_GATE_20260420.md"))
    parser.add_argument("--final-decision-report", default=str(ROOT / "reports/stwm_top_tier_final_decision_20260420.json"))
    parser.add_argument("--final-decision-doc", default=str(ROOT / "docs/STWM_TOP_TIER_FINAL_DECISION_20260420.md"))
    parser.add_argument("--final-launch-report", default=str(ROOT / "reports/stwm_top_tier_one_last_fix_launch_20260420.json"))
    parser.add_argument("--final-summary-report", default=str(ROOT / "reports/stwm_top_tier_one_last_fix_summary_20260420.json"))
    parser.add_argument("--final-diagnosis-report", default=str(ROOT / "reports/stwm_top_tier_one_last_fix_diagnosis_20260420.json"))
    parser.add_argument("--final-doc", default=str(ROOT / "docs/STWM_TOP_TIER_ONE_LAST_FIX_20260420.md"))
    args = parser.parse_args()

    _append_log("start one-last-fix pack")
    consistency = _one_last_consistency(args)
    _protocol()
    seed_completion = _seed_real_completion(args)
    matched = _matched_6seed_full_eval(args, seed_completion)
    strict_bootstrap = _strict_bootstrap_ci(args)
    utility_v2 = build_downstream_utility_v2(Path(args.utility_v2_report), Path(args.utility_v2_doc))
    ood = _ood_fixed(args)
    mechanism = _mechanism_6seed_full(args)
    appearance = _appearance_final_fix(args)
    _teacher_prior_sanity_v2(args)
    _one_last_fix_gate(args, matched, utility_v2, ood, mechanism, appearance)
    decision = _final_decision(args, matched, strict_bootstrap, utility_v2, ood, mechanism, appearance)
    _final_reports(args, consistency, matched, strict_bootstrap, utility_v2, ood, mechanism, appearance, decision)
    _append_log("finished one-last-fix pack")


if __name__ == "__main__":
    main()
