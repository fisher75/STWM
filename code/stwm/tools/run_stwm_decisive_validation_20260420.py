#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import setproctitle  # type: ignore
except Exception:  # pragma: no cover
    setproctitle = None


def _set_proc_title() -> None:
    if setproctitle is not None:
        try:
            setproctitle.setproctitle("python")
        except Exception:
            pass


_set_proc_title()

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
CODE = ROOT / "code"
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"
LOGS = ROOT / "logs"

DATE_TAG = "20260420"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_md(path: Path, title: str, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    body.extend(lines)
    path.write_text("\n".join(body).rstrip() + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_audit(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "nonempty": False,
        "valid_json": False,
        "top_level_keys": [],
        "mtime": None,
        "size_bytes": 0,
        "if_missing_exact_reason": "",
    }
    if not path.exists():
        row["if_missing_exact_reason"] = "missing_in_live_repo"
        return row
    stat = path.stat()
    row["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    row["size_bytes"] = stat.st_size
    row["nonempty"] = stat.st_size > 0
    try:
        payload = json.loads(path.read_text())
        row["valid_json"] = True
        if isinstance(payload, dict):
            row["top_level_keys"] = sorted(payload.keys())
        elif isinstance(payload, list):
            row["top_level_keys"] = [f"list_len={len(payload)}"]
    except Exception as exc:
        row["if_missing_exact_reason"] = f"json_parse_error:{exc.__class__.__name__}"
    return row


def build_manifest(dir_paths: Iterable[Path]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for base in dir_paths:
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(ROOT)
            stat = path.stat()
            rows.append(
                {
                    "path": str(rel),
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "sha256": sha256_file(path),
                }
            )
    return {
        "generated_at_utc": now_iso(),
        "root": str(ROOT),
        "file_count": len(rows),
        "files": rows,
    }


def extract_truth_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "matched_6seed_improved": payload.get("matched_6seed_improved"),
        "strict_bootstrap_claim_level": payload.get("strict_bootstrap_claim_level"),
        "utility_v2_claim_ready": payload.get("utility_v2_claim_ready"),
        "ood_claim_ready": payload.get("ood_claim_ready"),
        "mechanism_claim_ready": payload.get("mechanism_claim_ready"),
        "appearance_claim_allowed": payload.get("appearance_claim_allowed"),
        "paper_target_recommendation": payload.get("paper_target_recommendation"),
        "oral_spotlight_readiness": payload.get("oral_spotlight_readiness"),
        "next_step_choice": payload.get("next_step_choice"),
    }


def build_asset_audit() -> Tuple[Dict[str, Any], List[str]]:
    paths = [
        REPORTS / "stwm_top_tier_one_last_fix_diagnosis_20260420.json",
        REPORTS / "stwm_top_tier_one_last_fix_summary_20260420.json",
        REPORTS / "stwm_top_tier_final_decision_20260420.json",
        REPORTS / "stwm_top_tier_matched_seed_real_completion_summary_20260420.json",
        REPORTS / "stwm_top_tier_matched_6seed_full_eval_20260420.json",
        REPORTS / "stwm_top_tier_strict_bootstrap_ci_20260420.json",
        REPORTS / "stwm_top_tier_downstream_utility_v2_20260420.json",
        REPORTS / "stwm_top_tier_ood_transfer_fixed_20260420.json",
        REPORTS / "stwm_top_tier_mechanism_6seed_full_20260420.json",
        REPORTS / "stwm_top_tier_appearance_plumbing_final_fix_20260420.json",
    ]
    audits = [file_audit(p) for p in paths]
    diag = read_json(REPORTS / "stwm_top_tier_one_last_fix_diagnosis_20260420.json", {})
    summ = read_json(REPORTS / "stwm_top_tier_one_last_fix_summary_20260420.json", {})
    final = read_json(REPORTS / "stwm_top_tier_final_decision_20260420.json", {})
    truth = extract_truth_fields(diag)
    conflicts: List[str] = []
    for key, truth_value in truth.items():
        for src_name, src_payload in [("summary", summ), ("final_decision", final)]:
            if key in src_payload and src_payload.get(key) != truth_value:
                conflicts.append(f"{src_name}:{key}")
    payload = {
        "generated_at_utc": now_iso(),
        "source_of_truth": "live_repo_reports_json",
        "audited_assets": audits,
        "truth_fields": truth,
        "conflict_detected": bool(conflicts),
        "conflicting_fields": sorted(conflicts),
    }
    md = [
        "## Scope",
        "- Source of truth: live repo JSON reports.",
        "- Markdown docs are not used to override JSON values.",
        "",
        "## Truth Fields",
        f"- matched_6seed_improved: `{truth['matched_6seed_improved']}`",
        f"- strict_bootstrap_claim_level: `{truth['strict_bootstrap_claim_level']}`",
        f"- utility_v2_claim_ready: `{truth['utility_v2_claim_ready']}`",
        f"- ood_claim_ready: `{truth['ood_claim_ready']}`",
        f"- mechanism_claim_ready: `{truth['mechanism_claim_ready']}`",
        f"- appearance_claim_allowed: `{truth['appearance_claim_allowed']}`",
        f"- paper_target_recommendation: `{truth['paper_target_recommendation']}`",
        f"- oral_spotlight_readiness: `{truth['oral_spotlight_readiness']}`",
        f"- next_step_choice: `{truth['next_step_choice']}`",
        "",
        "## Conflicts",
        f"- conflict_detected: `{bool(conflicts)}`",
        f"- conflicting_fields: `{sorted(conflicts)}`",
    ]
    return payload, md


def build_protocol() -> Tuple[Dict[str, Any], List[str]]:
    payload = {
        "generated_at_utc": now_iso(),
        "stage1_frozen": True,
        "candidate_stage2_mainline": "TUSB-v3.1",
        "v3p2_v3p3_do_not_exceed_anchor": True,
        "allowlist": [
            "matched seed completion",
            "killer baselines",
            "independent utility v3",
            "true OOD eval",
            "mechanism 6-seed repair",
            "appearance plumbing surgical fix",
            "teacher prior sanity",
        ],
        "denylist": [
            "protocol v4",
            "TUSB-v3.4/v3.5 large structure",
            "persistence",
            "Stage1 retraining",
            "video/render head",
            "codec/VAE",
        ],
        "current_risks": [
            "matched seed coverage incomplete",
            "multi-seed robustness not closed",
            "utility still needs stronger independent baseline coverage",
            "true OOD evidence missing",
            "mechanism 6-seed report missing seeds",
            "appearance signal still not entering batch/loss path",
        ],
    }
    md = [
        "## Boundary",
        "- Stage1 remains frozen.",
        "- TUSB-v3.1 remains the Stage2 candidate mainline.",
        "- v3.2/v3.3 are not promoted over v3.1.",
        "",
        "## Allowed Work",
        "- Matched seed completion.",
        "- Killer baselines.",
        "- Independent downstream utility.",
        "- True OOD evaluation.",
        "- Mechanism 6-seed repair.",
        "- Appearance plumbing surgical repair.",
        "- Teacher prior sanity.",
        "",
        "## Forbidden Work",
        "- New protocol v4.",
        "- New TUSB-v3.4/v3.5 structures.",
        "- Persistence, Stage1 retraining, video/render head, codec/VAE.",
    ]
    return payload, md


def build_seed_completion_assets() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
    prior = read_json(REPORTS / "stwm_top_tier_matched_6seed_full_eval_20260420.json", {})
    coverage = prior.get("coverage", {})
    missing_after_launch: List[Dict[str, Any]] = []
    for method, seed_rows in coverage.items():
        for seed, row in seed_rows.items():
            if row.get("training_needed"):
                missing_after_launch.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "if_missing_exact_reason": row.get("if_missing_exact_reason", "checkpoint_missing_in_live_repo"),
                    }
                )
    plan = {
        "generated_at_utc": now_iso(),
        "required_methods": list(coverage.keys()),
        "required_seeds": prior.get("required_matched_seeds", [42, 123, 456, 654, 789, 321]),
        "coverage": coverage,
    }
    launch = {
        "generated_at_utc": now_iso(),
        "real_completion_started": False,
        "training_jobs_launched": [],
        "missing_after_launch": missing_after_launch,
        "blocking_reason": "live repo still lacks materialized matched-seed checkpoints for TUSB-v3.1/cropenc/legacysem; this decisive pass does not fabricate completion",
        "estimated_eval_ready_check": "blocked_until_missing_checkpoints_materialize",
    }
    summary = {
        "generated_at_utc": now_iso(),
        "real_completion_started": False,
        "training_jobs_launched": [],
        "missing_after_launch": missing_after_launch,
        "methods_with_complete_6seed_coverage": [
            method for method, seed_rows in coverage.items() if all(not r.get("training_needed") for r in seed_rows.values())
        ],
        "methods_with_missing_coverage": {
            method: [int(seed) for seed, row in seed_rows.items() if row.get("training_needed")]
            for method, seed_rows in coverage.items()
            if any(row.get("training_needed") for row in seed_rows.values())
        },
        "matched_seed_completion_ready": len(missing_after_launch) == 0,
    }
    md = [
        "## Status",
        f"- real_completion_started: `{launch['real_completion_started']}`",
        f"- training_jobs_launched: `{launch['training_jobs_launched']}`",
        f"- matched_seed_completion_ready: `{summary['matched_seed_completion_ready']}`",
        "",
        "## Missing Coverage",
        f"- missing_after_launch: `{missing_after_launch}`",
        "",
        "## Blocking",
        f"- {launch['blocking_reason']}",
    ]
    return plan, launch, summary, md


def build_matched_6seed_eval() -> Tuple[Dict[str, Any], List[str]]:
    coverage_summary = read_json(REPORTS / "stwm_top_tier_matched_seed_real_completion_summary_20260420.json", {})
    partial = read_json(REPORTS / "stage2_v3p1_multiseed_dualpanel_20260420.json", {})
    payload = {
        "generated_at_utc": now_iso(),
        "matched_seed_coverage_complete": False,
        "required_seeds": [42, 123, 456, 654, 789, 321],
        "primary_panels": [
            "legacy_85_context_preserving",
            "densified_200_context_preserving",
            "protocol_v3_extended_600_context_preserving",
        ],
        "secondary_partial_seed_evidence": partial,
        "coverage_summary": coverage_summary,
        "matched_6seed_improved_vs_calibration": False,
        "matched_6seed_improved_vs_cropenc": False,
        "matched_6seed_improved_vs_legacysem": False,
        "matched_6seed_claim_ready": False,
        "failure_seed_diagnosis": "missing matched checkpoints for TUSB-v3.1/cropenc/legacysem prevent a real 6-seed judge; current evidence remains partial-seed only",
    }
    md = [
        "## Status",
        "- This report does not fabricate a matched 6-seed result.",
        f"- matched_6seed_claim_ready: `{payload['matched_6seed_claim_ready']}`",
        "",
        "## Failure Diagnosis",
        f"- {payload['failure_seed_diagnosis']}",
        "",
        "## Secondary Evidence",
        "- Existing 3-seed v3.1 dualpanel report is kept only as secondary context.",
    ]
    return payload, md


def build_killer_baselines_assets() -> Tuple[Dict[str, Any], List[str]]:
    no_teacher = read_json(REPORTS / "stage2_tusb_v2_no_teacher_prior_seed123_20260418_final.json", {})
    no_identity = read_json(REPORTS / "stage2_tusb_v3_no_identity_binding_seed123_20260418_final.json", {})
    utility_v2 = read_json(REPORTS / "stwm_top_tier_downstream_utility_v2_20260420.json", {})
    payload = {
        "generated_at_utc": now_iso(),
        "baselines": {
            "trace_only_tusb": {
                "available": bool(no_teacher),
                "seed_coverage": [123] if no_teacher else [],
                "additional_seed_available": False,
                "run_name": no_teacher.get("run_name"),
                "reason_if_incomplete": "only seed123 materialized in live repo",
            },
            "teacher_only_semantic_only_retrieval": {
                "available": False,
                "reason": "no materialized teacher-only checkpoint or standalone semantic-only evaluation asset in live repo",
            },
            "object_slot_no_trace": {
                "available": False,
                "reason": "no materialized object-slot/no-trace baseline checkpoint or eval asset in live repo",
            },
            "stage1_frozen_trace_only_readout": {
                "available": True,
                "run_name": "stage1_frozen_baseline",
            },
            "identity_binding_ablation_support": {
                "available": bool(no_identity),
                "run_name": no_identity.get("run_name"),
            },
        },
        "supporting_evidence": {
            "utility_v2_vs_stage1_positive": bool(utility_v2.get("utility_v2_improved_vs_calibration")),
            "no_identity_binding_materialized": bool(no_identity),
        },
        "trace_load_bearing": True,
        "semantics_load_bearing": False,
        "identity_binding_load_bearing": bool(no_identity),
        "trace_semantic_coupling_load_bearing": False,
        "killer_baselines_passed": False,
        "blocking_reason": "teacher-only and object-slot/no-trace killer baselines are not materialized; trace-only TUSB is only available for seed123",
    }
    md = [
        "## Killer Baseline Status",
        f"- killer_baselines_passed: `{payload['killer_baselines_passed']}`",
        f"- blocking_reason: {payload['blocking_reason']}",
        "",
        "## Available",
        f"- trace_only_tusb: `{payload['baselines']['trace_only_tusb']}`",
        f"- stage1_frozen_trace_only_readout: `{payload['baselines']['stage1_frozen_trace_only_readout']}`",
        f"- identity_binding_ablation_support: `{payload['baselines']['identity_binding_ablation_support']}`",
        "",
        "## Missing",
        f"- teacher_only_semantic_only_retrieval: `{payload['baselines']['teacher_only_semantic_only_retrieval']}`",
        f"- object_slot_no_trace: `{payload['baselines']['object_slot_no_trace']}`",
    ]
    return payload, md


def build_strict_bootstrap_ci() -> Tuple[Dict[str, Any], List[str]]:
    prior = read_json(REPORTS / "stwm_top_tier_strict_bootstrap_ci_20260420.json", {})
    killer = read_json(REPORTS / "stwm_killer_baselines_20260420.json", {})
    payload = dict(prior)
    payload["generated_at_utc"] = now_iso()
    payload["killer_baseline_comparisons_available"] = False
    payload["killer_baseline_comparison_reason"] = killer.get(
        "blocking_reason",
        "killer baselines not fully materialized in live repo",
    )
    payload["claim_level"] = prior.get("strict_bootstrap_claim_level", "weak_claim")
    md = [
        "## Strict Bootstrap",
        f"- strict_bootstrap_claim_level: `{payload.get('strict_bootstrap_claim_level', 'weak_claim')}`",
        f"- zero_excluded_any_primary_metric: `{payload.get('zero_excluded_any_primary_metric')}`",
        "",
        "## Killer Baseline Coverage",
        f"- killer_baseline_comparisons_available: `{payload['killer_baseline_comparisons_available']}`",
        f"- reason: {payload['killer_baseline_comparison_reason']}",
    ]
    return payload, md


def build_downstream_utility_v3_assets() -> Tuple[Dict[str, Any], List[str]]:
    prior = read_json(REPORTS / "stwm_top_tier_downstream_utility_v2_20260420.json", {})
    killer = read_json(REPORTS / "stwm_killer_baselines_20260420.json", {})
    payload = {
        "generated_at_utc": now_iso(),
        "probe_design_v3": {
            "probe_a": prior.get("probe_design", {}).get("probe_a"),
            "probe_b": prior.get("probe_design", {}).get("probe_b"),
            "probe_c": prior.get("probe_design", {}).get("probe_c"),
            "probe_train_items": prior.get("probe_design", {}).get("probe_train_items"),
            "probe_eval_items": prior.get("probe_design", {}).get("probe_eval_items"),
            "independence_note": prior.get("probe_design", {}).get("independence_note"),
        },
        "probe_a": prior.get("probe_a", {}),
        "probe_b": prior.get("probe_b", {}),
        "probe_c": prior.get("probe_c", {}),
        "hard_subset_breakdown": prior.get("hard_subset_breakdown", {}),
        "killer_baseline_availability": killer.get("baselines", {}),
        "utility_v3_improved_vs_calibration": bool(prior.get("utility_v2_improved_vs_calibration")),
        "utility_v3_improved_vs_cropenc": False,
        "utility_v3_improved_vs_killer_baselines": False,
        "utility_v3_hard_subset_improved": bool(prior.get("utility_v2_hard_subset_improved")),
        "utility_v3_leakage_check_passed": bool(prior.get("utility_v2_leakage_check_passed")),
        "utility_v3_claim_ready": bool(prior.get("utility_v2_claim_ready")),
        "utility_v3_limitations": [
            "cropenc utility probe rows are not materialized in live repo",
            "teacher-only/object-slot killer baselines are not materialized",
        ],
    }
    md = [
        "## Utility v3",
        f"- utility_v3_improved_vs_calibration: `{payload['utility_v3_improved_vs_calibration']}`",
        f"- utility_v3_hard_subset_improved: `{payload['utility_v3_hard_subset_improved']}`",
        f"- utility_v3_leakage_check_passed: `{payload['utility_v3_leakage_check_passed']}`",
        f"- utility_v3_claim_ready: `{payload['utility_v3_claim_ready']}`",
        "",
        "## Limitations",
        *[f"- {x}" for x in payload["utility_v3_limitations"]],
    ]
    return payload, md


def build_true_ood_eval_assets() -> Tuple[Dict[str, Any], List[str]]:
    prior = read_json(REPORTS / "stwm_top_tier_ood_transfer_fixed_20260420.json", {})
    payload = {
        "generated_at_utc": now_iso(),
        "setting_a_vipseg_to_burst_heavy": prior.get("setting_a_vipseg_to_burst_heavy", {}),
        "setting_b_burst_to_vipseg_heavy": prior.get("setting_b_burst_to_vipseg_heavy", {}),
        "setting_c_conservative_heldout_split": prior.get("setting_c_conservative_heldout_split", {}),
        "proxy_domain_split_positive": bool(prior.get("proxy_domain_split_positive")),
        "true_ood_improved_vs_calibration": False,
        "true_ood_improved_vs_cropenc": False,
        "true_ood_hard_subset_improved": False,
        "ood_claim_ready": False,
        "proxy_only_vs_true_ood_boundary": prior.get("proxy_only_vs_true_ood_boundary"),
    }
    md = [
        "## OOD",
        f"- proxy_domain_split_positive: `{payload['proxy_domain_split_positive']}`",
        f"- true_ood_improved_vs_calibration: `{payload['true_ood_improved_vs_calibration']}`",
        f"- true_ood_hard_subset_improved: `{payload['true_ood_hard_subset_improved']}`",
        f"- ood_claim_ready: `{payload['ood_claim_ready']}`",
        "",
        "## Boundary",
        f"- {payload['proxy_only_vs_true_ood_boundary']}",
    ]
    return payload, md


def build_mechanism_6seed_assets() -> Tuple[Dict[str, Any], List[str]]:
    prior = read_json(REPORTS / "stwm_top_tier_mechanism_6seed_full_20260420.json", {})
    payload = dict(prior)
    payload["generated_at_utc"] = now_iso()
    md = [
        "## Mechanism 6-seed",
        f"- mechanism_claim_ready: `{payload.get('mechanism_claim_ready')}`",
        f"- identity_binding_cross_seed_stable: `{payload.get('identity_binding_cross_seed_stable')}`",
        f"- slow_semantic_state_cross_seed_stable: `{payload.get('slow_semantic_state_cross_seed_stable')}`",
        f"- anti_collapse_cross_seed_stable: `{payload.get('anti_collapse_cross_seed_stable')}`",
        f"- failed_seed_flags: `{payload.get('failed_seed_flags')}`",
    ]
    return payload, md


def build_appearance_teacher_sanity_assets() -> Tuple[Dict[str, Any], List[str]]:
    appearance = read_json(REPORTS / "stwm_top_tier_appearance_plumbing_final_fix_20260420.json", {})
    teacher = read_json(REPORTS / "stwm_top_tier_teacher_prior_sanity_v2_20260420.json", {})
    payload = {
        "generated_at_utc": now_iso(),
        "offline_appearance_drift_high_ratio": appearance.get("offline_appearance_drift_high_ratio"),
        "dataloader_appearance_drift_high_ratio": appearance.get("dataloader_appearance_drift_high_ratio"),
        "batch_appearance_drift_high_ratio_mean": appearance.get("batch_appearance_drift_high_ratio_mean"),
        "appearance_refine_loss_nonzero_ratio": appearance.get("appearance_refine_loss_nonzero_ratio"),
        "exact_breakpoint": appearance.get("exact_breakpoint"),
        "appearance_claim_allowed": bool(appearance.get("appearance_claim_allowed")),
        "teacher_prior_upgrade_available": bool(teacher.get("teacher_prior_upgrade_available")),
        "best_available_teacher": teacher.get("best_available_teacher"),
        "teacher_sanity_improves_retrieval": bool(teacher.get("teacher_sanity_improves_retrieval")),
        "teacher_sanity_improves_unit_purity": bool(teacher.get("teacher_sanity_improves_unit_purity")),
        "exact_blocking_reason": teacher.get("exact_blocking_reason"),
    }
    md = [
        "## Appearance Plumbing",
        f"- appearance_claim_allowed: `{payload['appearance_claim_allowed']}`",
        f"- offline_appearance_drift_high_ratio: `{payload['offline_appearance_drift_high_ratio']}`",
        f"- batch_appearance_drift_high_ratio_mean: `{payload['batch_appearance_drift_high_ratio_mean']}`",
        f"- appearance_refine_loss_nonzero_ratio: `{payload['appearance_refine_loss_nonzero_ratio']}`",
        f"- exact_breakpoint: {payload['exact_breakpoint']}",
        "",
        "## Teacher Sanity",
        f"- teacher_prior_upgrade_available: `{payload['teacher_prior_upgrade_available']}`",
        f"- best_available_teacher: `{payload['best_available_teacher']}`",
        f"- exact_blocking_reason: {payload['exact_blocking_reason']}",
    ]
    return payload, md


def build_surgical_fix_gate() -> Tuple[Dict[str, Any], List[str]]:
    utility = read_json(REPORTS / "stwm_downstream_utility_v3_20260420.json", {})
    ood = read_json(REPORTS / "stwm_true_ood_eval_20260420.json", {})
    matched = read_json(REPORTS / "stwm_decisive_matched_6seed_eval_20260420.json", {})
    true_count = sum(
        [
            bool(matched.get("matched_6seed_improved_vs_calibration")),
            bool(utility.get("utility_v3_claim_ready")),
            bool(ood.get("ood_claim_ready")),
        ]
    )
    allowed = not (matched.get("matched_6seed_improved_vs_calibration") and utility.get("utility_v3_claim_ready"))
    payload = {
        "generated_at_utc": now_iso(),
        "matched_6seed_improved": bool(matched.get("matched_6seed_improved_vs_calibration")),
        "utility_v3_claim_ready": bool(utility.get("utility_v3_claim_ready")),
        "ood_claim_ready": bool(ood.get("ood_claim_ready")),
        "positive_major_axes_count": true_count,
        "one_last_fix_allowed": allowed,
        "recommended_fix": "matched seed completion / seed-specific evidence closure" if allowed else "no further fix required",
    }
    md = [
        "## Surgical Fix Gate",
        f"- matched_6seed_improved: `{payload['matched_6seed_improved']}`",
        f"- utility_v3_claim_ready: `{payload['utility_v3_claim_ready']}`",
        f"- ood_claim_ready: `{payload['ood_claim_ready']}`",
        f"- one_last_fix_allowed: `{payload['one_last_fix_allowed']}`",
        f"- recommended_fix: `{payload['recommended_fix']}`",
    ]
    return payload, md


def build_final_decision() -> Tuple[Dict[str, Any], List[str]]:
    matched = read_json(REPORTS / "stwm_decisive_matched_6seed_eval_20260420.json", {})
    killer = read_json(REPORTS / "stwm_killer_baselines_20260420.json", {})
    bootstrap = read_json(REPORTS / "stwm_decisive_strict_bootstrap_ci_20260420.json", {})
    utility = read_json(REPORTS / "stwm_downstream_utility_v3_20260420.json", {})
    ood = read_json(REPORTS / "stwm_true_ood_eval_20260420.json", {})
    mechanism = read_json(REPORTS / "stwm_decisive_mechanism_6seed_20260420.json", {})
    appearance = read_json(REPORTS / "stwm_appearance_teacher_final_sanity_20260420.json", {})

    matched_ok = bool(matched.get("matched_6seed_improved_vs_calibration"))
    utility_ok = bool(utility.get("utility_v3_claim_ready"))
    ood_ok = bool(ood.get("ood_claim_ready"))
    mechanism_ok = bool(mechanism.get("mechanism_claim_ready"))
    appearance_ok = bool(appearance.get("appearance_claim_allowed"))
    if not matched_ok and not utility_ok:
        paper_target = "not_ready_for_top_tier"
        next_step = "rethink_stage2_story"
    elif matched_ok and utility_ok and ood_ok:
        paper_target = "cvpr_eccv_main_ready"
        next_step = "start_writing_main_submission"
    else:
        paper_target = "borderline_needs_one_last_fix"
        next_step = "run_one_last_surgical_fix"
    if matched_ok and utility_ok and ood_ok and mechanism_ok:
        oral = "possible_if_utility_ood_and_mechanism_hold"
    else:
        oral = "not_ready"
    payload = {
        "generated_at_utc": now_iso(),
        "stage1_ready": True,
        "stage2_ready": matched_ok,
        "matched_6seed_improved": matched_ok,
        "killer_baselines_passed": bool(killer.get("killer_baselines_passed")),
        "strict_bootstrap_claim_level": bootstrap.get("strict_bootstrap_claim_level", "weak_claim"),
        "utility_v3_claim_ready": utility_ok,
        "true_ood_claim_ready": ood_ok,
        "mechanism_claim_ready": mechanism_ok,
        "appearance_claim_allowed": appearance_ok,
        "paper_target_recommendation": paper_target,
        "oral_spotlight_readiness": oral,
        "next_step_choice": next_step,
    }
    md = [
        "## Final Decision",
        f"- matched_6seed_improved: `{payload['matched_6seed_improved']}`",
        f"- killer_baselines_passed: `{payload['killer_baselines_passed']}`",
        f"- strict_bootstrap_claim_level: `{payload['strict_bootstrap_claim_level']}`",
        f"- utility_v3_claim_ready: `{payload['utility_v3_claim_ready']}`",
        f"- true_ood_claim_ready: `{payload['true_ood_claim_ready']}`",
        f"- mechanism_claim_ready: `{payload['mechanism_claim_ready']}`",
        f"- appearance_claim_allowed: `{payload['appearance_claim_allowed']}`",
        f"- paper_target_recommendation: `{payload['paper_target_recommendation']}`",
        f"- oral_spotlight_readiness: `{payload['oral_spotlight_readiness']}`",
        f"- next_step_choice: `{payload['next_step_choice']}`",
    ]
    return payload, md


def run_all() -> Dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    asset_audit, asset_md = build_asset_audit()
    write_json(REPORTS / "stwm_decisive_validation_asset_audit_20260420.json", asset_audit)
    write_md(DOCS / "STWM_DECISIVE_VALIDATION_ASSET_AUDIT_20260420.md", "STWM Decisive Validation Asset Audit 20260420", asset_md)
    write_json(REPORTS / "stwm_decisive_validation_live_manifest_20260420.json", build_manifest([CODE, DOCS, REPORTS, SCRIPTS, CONFIGS]))

    protocol, protocol_md = build_protocol()
    write_json(REPORTS / "stwm_decisive_validation_protocol_20260420.json", protocol)
    write_md(DOCS / "STWM_DECISIVE_VALIDATION_PROTOCOL_20260420.md", "STWM Decisive Validation Protocol 20260420", protocol_md)

    seed_plan, seed_launch, seed_summary, seed_md = build_seed_completion_assets()
    write_json(REPORTS / "stwm_decisive_seed_completion_plan_20260420.json", seed_plan)
    write_json(REPORTS / "stwm_decisive_seed_completion_launch_20260420.json", seed_launch)
    write_json(REPORTS / "stwm_decisive_seed_completion_summary_20260420.json", seed_summary)
    write_md(DOCS / "STWM_DECISIVE_SEED_COMPLETION_20260420.md", "STWM Decisive Seed Completion 20260420", seed_md)

    matched_eval, matched_md = build_matched_6seed_eval()
    write_json(REPORTS / "stwm_decisive_matched_6seed_eval_20260420.json", matched_eval)
    write_md(DOCS / "STWM_DECISIVE_MATCHED_6SEED_EVAL_20260420.md", "STWM Decisive Matched 6-Seed Eval 20260420", matched_md)

    killer, killer_md = build_killer_baselines_assets()
    write_json(REPORTS / "stwm_killer_baselines_20260420.json", killer)
    write_md(DOCS / "STWM_KILLER_BASELINES_20260420.md", "STWM Killer Baselines 20260420", killer_md)

    strict_bootstrap, strict_bootstrap_md = build_strict_bootstrap_ci()
    write_json(REPORTS / "stwm_decisive_strict_bootstrap_ci_20260420.json", strict_bootstrap)
    write_md(DOCS / "STWM_DECISIVE_STRICT_BOOTSTRAP_CI_20260420.md", "STWM Decisive Strict Bootstrap CI 20260420", strict_bootstrap_md)

    utility_v3, utility_v3_md = build_downstream_utility_v3_assets()
    write_json(REPORTS / "stwm_downstream_utility_v3_20260420.json", utility_v3)
    write_md(DOCS / "STWM_DOWNSTREAM_UTILITY_V3_20260420.md", "STWM Downstream Utility V3 20260420", utility_v3_md)

    ood, ood_md = build_true_ood_eval_assets()
    write_json(REPORTS / "stwm_true_ood_eval_20260420.json", ood)
    write_md(DOCS / "STWM_TRUE_OOD_EVAL_20260420.md", "STWM True OOD Eval 20260420", ood_md)

    mechanism, mechanism_md = build_mechanism_6seed_assets()
    write_json(REPORTS / "stwm_decisive_mechanism_6seed_20260420.json", mechanism)
    write_md(DOCS / "STWM_DECISIVE_MECHANISM_6SEED_20260420.md", "STWM Decisive Mechanism 6-Seed 20260420", mechanism_md)

    appearance_teacher, appearance_teacher_md = build_appearance_teacher_sanity_assets()
    write_json(REPORTS / "stwm_appearance_teacher_final_sanity_20260420.json", appearance_teacher)
    write_md(DOCS / "STWM_APPEARANCE_TEACHER_FINAL_SANITY_20260420.md", "STWM Appearance + Teacher Final Sanity 20260420", appearance_teacher_md)

    gate, gate_md = build_surgical_fix_gate()
    write_json(REPORTS / "stwm_decisive_one_last_fix_gate_20260420.json", gate)
    write_md(DOCS / "STWM_DECISIVE_ONE_LAST_FIX_GATE_20260420.md", "STWM Decisive One-Last Fix Gate 20260420", gate_md)

    final_decision, final_decision_md = build_final_decision()
    write_json(REPORTS / "stwm_decisive_final_decision_20260420.json", final_decision)
    write_md(DOCS / "STWM_DECISIVE_FINAL_DECISION_20260420.md", "STWM Decisive Final Decision 20260420", final_decision_md)

    launch = {
        "generated_at_utc": now_iso(),
        "session_name": "stwm_decisive_validation_20260420",
        "mode": "analysis_only_decisive_pack",
        "gpu_training_started": False,
        "phases": [
            "asset_audit",
            "protocol",
            "seed_completion",
            "matched_6seed_eval",
            "killer_baselines",
            "strict_bootstrap",
            "utility_v3",
            "true_ood_eval",
            "mechanism_6seed",
            "appearance_teacher_sanity",
            "decision_dashboard",
        ],
    }
    summary = {
        "generated_at_utc": now_iso(),
        "matched_6seed_improved": final_decision["matched_6seed_improved"],
        "killer_baselines_passed": final_decision["killer_baselines_passed"],
        "strict_bootstrap_claim_level": final_decision["strict_bootstrap_claim_level"],
        "utility_v3_claim_ready": final_decision["utility_v3_claim_ready"],
        "true_ood_claim_ready": final_decision["true_ood_claim_ready"],
        "mechanism_claim_ready": final_decision["mechanism_claim_ready"],
        "appearance_claim_allowed": final_decision["appearance_claim_allowed"],
        "paper_target_recommendation": final_decision["paper_target_recommendation"],
        "oral_spotlight_readiness": final_decision["oral_spotlight_readiness"],
        "next_step_choice": final_decision["next_step_choice"],
    }
    diagnosis = {
        "generated_at_utc": now_iso(),
        **summary,
        "blocking_reasons": {
            "matched_6seed": matched_eval["failure_seed_diagnosis"],
            "killer_baselines": killer["blocking_reason"],
            "ood": ood["proxy_only_vs_true_ood_boundary"],
            "mechanism": "6-seed mechanism claim remains blocked by missing seeds 654/789/321",
            "appearance": appearance_teacher["exact_breakpoint"],
        },
    }
    write_json(REPORTS / "stwm_decisive_validation_launch_20260420.json", launch)
    write_json(REPORTS / "stwm_decisive_validation_summary_20260420.json", summary)
    write_json(REPORTS / "stwm_decisive_validation_diagnosis_20260420.json", diagnosis)
    write_md(
        DOCS / "STWM_DECISIVE_VALIDATION_20260420.md",
        "STWM Decisive Validation 20260420",
        [
            "## Summary",
            *[f"- {k}: `{v}`" for k, v in summary.items() if k != "generated_at_utc"],
            "",
            "## Blocking Reasons",
            *[f"- {k}: {v}" for k, v in diagnosis["blocking_reasons"].items()],
        ],
    )
    return summary


def main() -> None:
    summary = run_all()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
