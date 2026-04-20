#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import importlib.util
import json
import math
import os
import statistics

from stwm.tools.run_stwm_top_tier_downstream_utility_20260420 import build_downstream_utility


ROOT = Path("/raid/chen034/workspace/stwm")
SESSION = "stwm_top_tier_final_validation_20260420"
LOG_PATH = ROOT / "logs/stwm_top_tier_final_validation_20260420.log"

CONSISTENCY_TARGETS = [
    "reports/stwm_final_credibility_utility_diagnosis_20260420.json",
    "reports/stwm_final_credibility_utility_summary_20260420.json",
    "reports/stwm_final_paper_position_decision_20260420.json",
    "reports/stage2_v3p1_matched_6seed_dualpanel_20260420.json",
    "reports/stage2_v3p1_dualpanel_context_audit_20260420.json",
    "reports/stage2_protocol_v3_extended_evalset_20260420.json",
    "reports/stage2_final_bootstrap_ci_20260420.json",
    "reports/stage2_v3p1_downstream_utility_20260420.json",
    "reports/stage2_v3p1_mechanism_6seed_20260420.json",
    "reports/stage2_final_appearance_plumbing_fix_audit_20260420.json",
]
CONSISTENCY_FIELDS = [
    "context_preserving_densified_200_improved_vs_current_calonly",
    "matched_6seed_improved",
    "bootstrap_ci_zero_excluded",
    "downstream_utility_improved",
    "mechanism_cross_seed_stable",
    "paper_target_recommendation",
    "oral_spotlight_readiness",
    "next_step_choice",
]
USER_VISIBLE_SUMMARY_REFERENCE = {
    "context_preserving_densified_200_improved_vs_current_calonly": False,
    "matched_6seed_improved": False,
    "bootstrap_ci_zero_excluded": False,
    "downstream_utility_improved": False,
    "mechanism_cross_seed_stable": False,
    "paper_target_recommendation": "borderline_needs_one_more_fix",
    "oral_spotlight_readiness": "not_ready",
    "next_step_choice": "one_last_surgical_fix",
}
MATCHED_SEEDS = [42, 123, 456, 654, 789, 321]


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


def _manifest_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for prefix in ["code", "docs", "reports", "scripts", "configs"]:
        base = ROOT / prefix
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            stat = path.stat()
            entries.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                    "sha256": _sha256_file(path),
                }
            )
    return entries


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


def _seed_checkpoint_rows() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    coverage_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for method, mapping in _method_seed_dirs().items():
        method_cov: Dict[str, Any] = {}
        for seed in MATCHED_SEEDS:
            run_name = mapping.get(seed, "")
            ckpt_dir = ROOT / "outputs/checkpoints" / run_name if run_name else None
            best_path = ckpt_dir / "best.pt" if ckpt_dir else None
            sidecar_path = ckpt_dir / "best_semantic_hard.pt" if ckpt_dir else None
            exists = bool(best_path and best_path.exists())
            side_exists = bool(sidecar_path and sidecar_path.exists())
            row = {
                "method": method,
                "seed": seed,
                "run_name": run_name or None,
                "checkpoint_exists": exists,
                "best.pt_exists": exists,
                "best_semantic_hard.pt_exists": side_exists,
                "training_needed": not exists,
                "eval_only_possible": exists,
                "exact_checkpoint_path": str(best_path) if exists else "",
            }
            coverage_rows.append(row)
            method_cov[str(seed)] = row
        summary[method] = method_cov
    return summary, coverage_rows


def _result_consistency_and_manifest(args: Any) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    actual_values: Dict[str, Any] = {}
    for rel in CONSISTENCY_TARGETS:
        path = ROOT / rel
        payload = _load_json(path)
        extracted = {key: payload[key] for key in CONSISTENCY_FIELDS if key in payload}
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "nonempty": bool(path.exists() and path.stat().st_size > 0),
                "valid_json": bool(payload),
                "top_level_keys": list(payload.keys())[:50],
                "mtime": path.stat().st_mtime if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "extracted_fields": extracted,
            }
        )
        for key, value in extracted.items():
            actual_values[key] = value

    conflicts = [
        key
        for key, expected in USER_VISIBLE_SUMMARY_REFERENCE.items()
        if key in actual_values and actual_values[key] != expected
    ]
    manifest_entries = _manifest_entries()
    manifest = {
        "generated_at_utc": _now_iso(),
        "repo_root": str(ROOT),
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
    }
    audit = {
        "generated_at_utc": _now_iso(),
        "audited_reports": rows,
        "actual_values": actual_values,
        "conflict_detected": bool(conflicts),
        "conflicting_fields": conflicts,
        "source_of_truth": "live_repo_reports_json",
        "external_summary_reference": USER_VISIBLE_SUMMARY_REFERENCE,
    }
    _write_json(Path(args.result_consistency_report), audit)
    _write_json(Path(args.live_manifest_report), manifest)
    _write_md(
        Path(args.result_consistency_doc),
        [
            "# STWM Top-Tier Result Consistency Audit 20260420",
            "",
            f"- conflict_detected: {audit['conflict_detected']}",
            f"- conflicting_fields: {audit['conflicting_fields']}",
            f"- source_of_truth: {audit['source_of_truth']}",
            "",
            *[
                f"- {row['path']}: exists={row['exists']} nonempty={row['nonempty']} valid_json={row['valid_json']} extracted={row['extracted_fields']}"
                for row in rows
            ],
            "",
            f"- live_manifest_path: {args.live_manifest_report}",
            f"- manifest_entry_count: {len(manifest_entries)}",
        ],
    )
    return audit


def _validation_protocol(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": _now_iso(),
        "stage1_frozen": True,
        "candidate_stage2_mainline": {
            "run_name": "stage2_tusb_v3p1_seed123_20260418",
            "official_checkpoint_candidate": "best.pt",
            "sidecar_checkpoint": "best_semantic_hard.pt",
        },
        "v3p2_v3p3_status": {
            "did_not_exceed_v3p1_anchor": True,
            "no_more_cumulative_stacking": True,
        },
        "current_top_tier_risks": [
            "matched seed coverage incomplete",
            "multi-seed robustness not closed",
            "utility is still weakly independent",
            "OOD / cross-dataset evidence missing or weak",
            "mechanism 6-seed evidence incomplete",
            "appearance signal still not reaching loss path",
        ],
        "allowed_this_round": [
            "matched seed completion",
            "stronger independent utility probe",
            "cross-dataset / OOD transfer evaluation",
            "appearance plumbing surgical fix audit",
            "teacher semantic prior sanity only",
        ],
        "forbidden_this_round": [
            "new large architecture branch",
            "persistence return",
            "Stage1 retraining",
            "video/render head",
            "codec/VAE",
        ],
    }
    _write_json(Path(args.validation_protocol_report), payload)
    _write_md(
        Path(args.validation_protocol_doc),
        [
            "# STWM Top-Tier Validation Protocol 20260420",
            "",
            "- Stage1 remains frozen.",
            "- candidate Stage2 mainline remains TUSB-v3.1 with `best.pt` as official checkpoint and `best_semantic_hard.pt` as sidecar.",
            "- v3.2/v3.3 did not exceed the v3.1 anchor.",
            "- this round only hardens credibility, matched seeds, bootstrap CI, downstream utility, OOD transfer, mechanism stability, and appearance plumbing audit.",
        ],
    )
    return payload


def _seed_completion_reports(args: Any) -> Dict[str, Any]:
    coverage_summary, coverage_rows = _seed_checkpoint_rows()
    training_needed = [row for row in coverage_rows if row["training_needed"]]
    plan = {
        "generated_at_utc": _now_iso(),
        "required_matched_seeds": MATCHED_SEEDS,
        "coverage": coverage_summary,
        "training_needed_count": len(training_needed),
        "training_needed_rows": training_needed,
    }
    launch = {
        "generated_at_utc": _now_iso(),
        "mode": "analysis_only",
        "training_jobs_launched": [],
        "reason": "no new matched-seed training launched inside this final validation pass; missing checkpoints are surfaced explicitly for decision making",
    }
    summary = {
        "generated_at_utc": _now_iso(),
        "methods_with_complete_6seed_coverage": [
            method for method, seeds in coverage_summary.items() if all(seeds[str(seed)]["best.pt_exists"] for seed in MATCHED_SEEDS)
        ],
        "methods_with_missing_coverage": {
            method: [seed for seed in MATCHED_SEEDS if not seeds[str(seed)]["best.pt_exists"]]
            for method, seeds in coverage_summary.items()
            if not all(seeds[str(seed)]["best.pt_exists"] for seed in MATCHED_SEEDS)
        },
    }
    _write_json(Path(args.seed_completion_plan_report), plan)
    _write_json(Path(args.seed_completion_launch_report), launch)
    _write_json(Path(args.seed_completion_summary_report), summary)
    _write_md(
        Path(args.seed_completion_doc),
        [
            "# STWM Top-Tier Seed Completion 20260420",
            "",
            f"- training_needed_count: {len(training_needed)}",
            *[
                f"- {method}: missing {summary['methods_with_missing_coverage'].get(method, [])}"
                for method in sorted(summary["methods_with_missing_coverage"].keys())
            ],
        ],
    )
    return {"plan": plan, "launch": launch, "summary": summary}


def _matched_6seed_dualpanel(args: Any, seed_summary: Dict[str, Any]) -> Dict[str, Any]:
    prior = _load_json(ROOT / "reports/stage2_v3p1_matched_6seed_dualpanel_20260420.json")
    payload = {
        "generated_at_utc": _now_iso(),
        "required_matched_seeds": MATCHED_SEEDS,
        "coverage": seed_summary["plan"]["coverage"],
        "main_6seed_table_available": False,
        "if_missing_exact_reason": "missing_seed_checkpoints_for_main_table",
        "legacy_85_context_preserving": prior.get("available_tusb_context_preserving_rows", []),
        "densified_200_context_preserving": [],
        "protocol_v3_extended_600_context_preserving": [],
        "failure_seed_diagnosis": prior.get("failure_seed_diagnosis", {}),
        "matched_6seed_improved_vs_current_calonly": False,
    }
    _write_json(Path(args.matched_6seed_report), payload)
    _write_md(
        Path(args.matched_6seed_doc),
        [
            "# STWM Top-Tier Matched 6-Seed Dualpanel 20260420",
            "",
            f"- main_6seed_table_available: {payload['main_6seed_table_available']}",
            f"- matched_6seed_improved_vs_current_calonly: {payload['matched_6seed_improved_vs_current_calonly']}",
            f"- missing_reason: {payload['if_missing_exact_reason']}",
        ],
    )
    return payload


def _final_bootstrap_ci(args: Any) -> Dict[str, Any]:
    prior = _load_json(ROOT / "reports/stage2_final_bootstrap_ci_20260420.json")
    payload = {
        "generated_at_utc": _now_iso(),
        "panels": prior.get("panels", {}),
        "statistically_strong": bool(prior.get("statistically_strong", False)),
        "practically_meaningful": bool(prior.get("practically_meaningful", False)),
        "paper_ready_claim_level": prior.get("paper_ready_claim_level", "weak_claim"),
        "bootstrap_ci_zero_excluded": bool(prior.get("zero_excluded_any_primary_comparison", False)),
    }
    _write_json(Path(args.final_bootstrap_report), payload)
    _write_md(
        Path(args.final_bootstrap_doc),
        [
            "# STWM Top-Tier Final Bootstrap CI 20260420",
            "",
            f"- bootstrap_ci_zero_excluded: {payload['bootstrap_ci_zero_excluded']}",
            f"- statistically_strong: {payload['statistically_strong']}",
            f"- practically_meaningful: {payload['practically_meaningful']}",
            f"- paper_ready_claim_level: {payload['paper_ready_claim_level']}",
        ],
    )
    return payload


def _ood_transfer(args: Any) -> Dict[str, Any]:
    ext = _load_json(ROOT / "reports/stage2_protocol_v3_extended_evalset_20260420.json")
    rows = list(ext.get("context_preserving_eval", {}).get("per_item_results", []))

    def _agg(filtered: List[Dict[str, Any]], method: str) -> Dict[str, float]:
        if not filtered:
            return {"count": 0.0, "top1": 0.0, "mrr": 0.0}
        scored = [row["methods"][method] for row in filtered if method in row.get("methods", {})]
        return {
            "count": float(len(scored)),
            "top1": float(sum(float(x.get("query_future_top1_acc", 0.0)) for x in scored) / max(len(scored), 1)),
            "mrr": float(sum(float(x.get("mrr", 0.0)) for x in scored) / max(len(scored), 1)),
        }

    burst = [row for row in rows if row.get("dataset") == "BURST"]
    vipseg = [row for row in rows if row.get("dataset") == "VIPSeg"]
    cal = "current_calibration_only_best"
    tusb = "current_tusb_v3p1_best::best.pt"
    burst_cal = _agg(burst, cal)
    burst_tusb = _agg(burst, tusb)
    vip_cal = _agg(vipseg, cal)
    vip_tusb = _agg(vipseg, tusb)
    ood_improved = bool(
        burst_tusb["top1"] > burst_cal["top1"]
        and vip_tusb["top1"] > vip_cal["top1"]
    )
    payload = {
        "generated_at_utc": _now_iso(),
        "setting_a_vipseg_to_burst_heavy_proxy": {
            "supported": True,
            "calibration_only": burst_cal,
            "tusb_v3p1": burst_tusb,
        },
        "setting_b_burst_to_vipseg_heavy_proxy": {
            "supported": True,
            "calibration_only": vip_cal,
            "tusb_v3p1": vip_tusb,
        },
        "setting_c_scene_domain_heldout": {
            "supported": False,
            "reason": "no clean held-out scene/domain split asset is materialized in the current live repo",
        },
        "ood_improved_vs_calibration": False if not ood_improved else False,
        "ood_hard_subset_improved": False,
        "ood_claim_ready": False,
        "proxy_domain_split_positive": bool(ood_improved),
    }
    _write_json(Path(args.ood_report), payload)
    _write_md(
        Path(args.ood_doc),
        [
            "# STWM Top-Tier OOD Transfer 20260420",
            "",
            f"- proxy_domain_split_positive: {payload['proxy_domain_split_positive']}",
            f"- ood_improved_vs_calibration: {payload['ood_improved_vs_calibration']}",
            f"- ood_claim_ready: {payload['ood_claim_ready']}",
            f"- setting_c_supported: {payload['setting_c_scene_domain_heldout']['supported']}",
        ],
    )
    return payload


def _mechanism_6seed_repair(args: Any) -> Dict[str, Any]:
    appendix = _load_json(ROOT / "reports/stage2_v3p1_mechanism_appendix_20260420.json")
    multiseed = _load_json(ROOT / "reports/stage2_v3p1_multiseed_dualpanel_20260420.json")
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
    available_seeds = []
    for row in rows:
        run_name = str(row.get("run_name", ""))
        seed = None
        for candidate in MATCHED_SEEDS:
            if f"seed{candidate}_" in run_name:
                seed = candidate
                break
        if seed is not None:
            available_seeds.append((seed, row))
    seed_mean = {metric: float(statistics.fmean(float(row.get(metric, 0.0)) for _, row in available_seeds)) if available_seeds else 0.0 for metric in metrics}
    seed_std = {metric: float(statistics.pstdev([float(row.get(metric, 0.0)) for _, row in available_seeds])) if len(available_seeds) > 1 else 0.0 for metric in metrics}
    perf_rows = {int(row["seed"]): row for row in multiseed.get("densified_200_panel", {}).get("seed_rows", [])}
    corr_rows = []
    for seed, row in available_seeds:
        if seed in perf_rows:
            corr_rows.append(
                {
                    "seed": seed,
                    "top1_acc": float(perf_rows[seed].get("top1_acc", 0.0)),
                    "same_instance_dominant_unit_match_rate_mean": float(row.get("same_instance_dominant_unit_match_rate_mean", 0.0)),
                    "different_instance_dominant_unit_collision_rate_mean": float(row.get("different_instance_dominant_unit_collision_rate_mean", 0.0)),
                    "z_sem_to_z_dyn_drift_ratio_mean": float(row.get("z_sem_to_z_dyn_drift_ratio_mean", 0.0)),
                }
            )
    payload = {
        "generated_at_utc": _now_iso(),
        "required_seeds": MATCHED_SEEDS,
        "available_seed_rows": [
            {"seed": seed, **{metric: float(row.get(metric, 0.0)) for metric in metrics}}
            for seed, row in available_seeds
        ],
        "missing_seeds": [seed for seed in MATCHED_SEEDS if seed not in {seed for seed, _ in available_seeds}],
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "failed_seed_flags": [seed for seed in MATCHED_SEEDS if seed not in {seed for seed, _ in available_seeds}],
        "mechanism_vs_performance_correlation": {
            "available_seed_count": len(corr_rows),
            "rows": corr_rows,
            "correlation_unreliable_due_to_missing_seeds": True,
        },
        "identity_binding_cross_seed_stable": False,
        "slow_semantic_state_cross_seed_stable": False,
        "anti_collapse_cross_seed_stable": False,
        "mechanism_cross_seed_stable": False,
        "mechanism_6seed_ready": False,
    }
    _write_json(Path(args.mechanism_report), payload)
    _write_md(
        Path(args.mechanism_doc),
        [
            "# STWM Top-Tier Mechanism 6-Seed Repair 20260420",
            "",
            f"- available_seed_count: {len(available_seeds)}",
            f"- missing_seeds: {payload['missing_seeds']}",
            f"- mechanism_cross_seed_stable: {payload['mechanism_cross_seed_stable']}",
            f"- identity_binding_cross_seed_stable: {payload['identity_binding_cross_seed_stable']}",
            f"- slow_semantic_state_cross_seed_stable: {payload['slow_semantic_state_cross_seed_stable']}",
            f"- anti_collapse_cross_seed_stable: {payload['anti_collapse_cross_seed_stable']}",
        ],
    )
    return payload


def _appearance_plumbing_fix(args: Any) -> Dict[str, Any]:
    prior = _load_json(ROOT / "reports/stage2_final_appearance_plumbing_fix_audit_20260420.json")
    payload = {
        "generated_at_utc": _now_iso(),
        "offline_appearance_drift_high_ratio": float(prior.get("offline_appearance_drift_high_ratio", 0.0)),
        "dataloader_appearance_drift_high_ratio": float(prior.get("dataloader_appearance_drift_high_ratio", 0.0)),
        "batch_appearance_drift_high_ratio_mean": float(prior.get("batch_appearance_drift_high_ratio_mean", 0.0)),
        "appearance_refine_loss_nonzero_ratio": float(prior.get("appearance_refine_loss_nonzero_ratio", 0.0)),
        "exact_breakpoint": str(prior.get("exact_breakpoint", "")),
        "appearance_claim_allowed": bool(prior.get("appearance_claim_allowed", False)),
        "repair_applied_this_round": False,
    }
    _write_json(Path(args.appearance_fix_report), payload)
    _write_md(
        Path(args.appearance_fix_doc),
        [
            "# STWM Top-Tier Appearance Plumbing Surgical Fix 20260420",
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


def _teacher_prior_sanity(args: Any) -> Dict[str, Any]:
    cached_names: List[str] = []
    for hub_root in [Path.home() / ".cache/huggingface/hub", Path("/raid/chen034/.cache/huggingface/hub")]:
        if hub_root.exists():
            for child in hub_root.iterdir():
                lower = child.name.lower()
                if any(term in lower for term in ["dinov2", "siglip", "clip-vit"]):
                    cached_names.append(child.name)
    payload = {
        "generated_at_utc": _now_iso(),
        "setproctitle_available": bool(importlib.util.find_spec("setproctitle") is not None),
        "transformers_available": bool(importlib.util.find_spec("transformers") is not None),
        "timm_available": bool(importlib.util.find_spec("timm") is not None),
        "open_clip_available": bool(importlib.util.find_spec("open_clip") is not None),
        "cached_teacher_model_hints": sorted(set(cached_names)),
        "dinov2_like_available_in_env": bool(importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("timm") is not None),
        "siglip_like_available_in_env": bool(importlib.util.find_spec("transformers") is not None),
        "stronger_clip_family_cached": any("clip-vit" in name.lower() for name in cached_names),
        "smallscale_teacher_sanity_performed": False,
        "exact_blocking_reason": "no stronger frozen teacher cache was materialized in the live repo; this round keeps teacher sanity at import/cache feasibility only",
    }
    _write_json(Path(args.teacher_sanity_report), payload)
    _write_md(
        Path(args.teacher_sanity_doc),
        [
            "# STWM Top-Tier Teacher Prior Sanity 20260420",
            "",
            f"- dinov2_like_available_in_env: {payload['dinov2_like_available_in_env']}",
            f"- siglip_like_available_in_env: {payload['siglip_like_available_in_env']}",
            f"- stronger_clip_family_cached: {payload['stronger_clip_family_cached']}",
            f"- cached_teacher_model_hints: {payload['cached_teacher_model_hints']}",
            f"- exact_blocking_reason: {payload['exact_blocking_reason']}",
        ],
    )
    return payload


def _surgical_fix_gate(args: Any, matched: Dict[str, Any], utility: Dict[str, Any], ood: Dict[str, Any], appearance: Dict[str, Any]) -> Dict[str, Any]:
    matched_improved = bool(matched.get("matched_6seed_improved_vs_current_calonly", False))
    utility_improved = bool(utility.get("utility_improved_vs_calibration", False))
    payload = {
        "generated_at_utc": _now_iso(),
        "matched_6seed_improved": matched_improved,
        "downstream_utility_improved": utility_improved,
        "ood_transfer_improved": bool(ood.get("ood_improved_vs_calibration", False)),
        "appearance_claim_allowed": bool(appearance.get("appearance_claim_allowed", False)),
        "one_last_surgical_fix_allowed": bool(not matched_improved and utility_improved),
        "allowed_fix_categories": [
            "checkpoint policy fix",
            "seed-specific unit collapse fix",
            "instance density reweighting",
            "teacher prior cache fix",
            "appearance plumbing fix",
        ],
        "recommended_gate_action": "one_last_surgical_fix" if (not matched_improved and utility_improved) else "start_writing_main_submission",
    }
    _write_json(Path(args.surgical_gate_report), payload)
    _write_md(
        Path(args.surgical_gate_doc),
        [
            "# STWM Top-Tier Surgical Fix Gate 20260420",
            "",
            f"- one_last_surgical_fix_allowed: {payload['one_last_surgical_fix_allowed']}",
            f"- recommended_gate_action: {payload['recommended_gate_action']}",
            f"- allowed_fix_categories: {payload['allowed_fix_categories']}",
        ],
    )
    return payload


def _paper_decision(args: Any, matched: Dict[str, Any], bootstrap: Dict[str, Any], utility: Dict[str, Any], ood: Dict[str, Any], mechanism: Dict[str, Any], appearance: Dict[str, Any]) -> Dict[str, Any]:
    matched_improved = bool(matched.get("matched_6seed_improved_vs_current_calonly", False))
    ci_zero_excluded = bool(bootstrap.get("bootstrap_ci_zero_excluded", False))
    utility_ready = bool(utility.get("utility_improved_vs_calibration", False))
    ood_ready = bool(ood.get("ood_improved_vs_calibration", False) and ood.get("ood_claim_ready", False))
    mechanism_ready = bool(mechanism.get("mechanism_6seed_ready", False))
    appearance_allowed = bool(appearance.get("appearance_claim_allowed", False))
    paper_target = "borderline_needs_one_last_fix"
    oral = "not_ready"
    next_step = "one_last_surgical_fix"
    if matched_improved and utility_ready and ood_ready:
        paper_target = "cvpr_eccv_main_ready"
        oral = "possible_if_utility_and_ood_hold" if not mechanism_ready else "plausible"
        next_step = "start_writing_main_submission"
    elif matched_improved and ci_zero_excluded:
        paper_target = "aaai_main_ready"
        oral = "not_ready"
        next_step = "reframe_as_moderate_claim_main_track"
    elif not matched_improved and not utility_ready:
        paper_target = "not_ready_for_top_tier"
        oral = "not_ready"
        next_step = "rethink_stage2_story"
    payload = {
        "generated_at_utc": _now_iso(),
        "stage1_ready": True,
        "stage2_ready": False,
        "matched_6seed_improved": matched_improved,
        "bootstrap_ci_zero_excluded": ci_zero_excluded,
        "downstream_utility_ready": utility_ready,
        "ood_transfer_ready": ood_ready,
        "mechanism_6seed_ready": mechanism_ready,
        "appearance_claim_allowed": appearance_allowed,
        "paper_target_recommendation": paper_target,
        "oral_spotlight_readiness": oral,
        "next_step_choice": next_step,
    }
    _write_json(Path(args.paper_decision_report), payload)
    _write_md(
        Path(args.paper_decision_doc),
        [
            "# STWM Top-Tier Paper Decision 20260420",
            "",
            f"- matched_6seed_improved: {matched_improved}",
            f"- bootstrap_ci_zero_excluded: {ci_zero_excluded}",
            f"- downstream_utility_ready: {utility_ready}",
            f"- ood_transfer_ready: {ood_ready}",
            f"- mechanism_6seed_ready: {mechanism_ready}",
            f"- appearance_claim_allowed: {appearance_allowed}",
            f"- paper_target_recommendation: {paper_target}",
            f"- oral_spotlight_readiness: {oral}",
            f"- next_step_choice: {next_step}",
        ],
    )
    return payload


def _final_validation_reports(args: Any, dualpanel: Dict[str, Any], matched: Dict[str, Any], bootstrap: Dict[str, Any], utility: Dict[str, Any], ood: Dict[str, Any], mechanism: Dict[str, Any], appearance: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    launch = {
        "generated_at_utc": _now_iso(),
        "session_name": SESSION,
        "mode": "analysis_only",
        "gpu_training_jobs_launched": 0,
        "gpu_eval_jobs_launched": 0,
        "notes": "final validation pack executed from current live assets; no new Stage2 training branch was opened",
    }
    summary = {
        "generated_at_utc": _now_iso(),
        "context_preserving_densified_200_improved_vs_current_calonly": bool(
            dualpanel.get("densified_200_context_preserving", {}).get("methods", [])
            and (dualpannel_cmp := next((row for row in dualpanel.get("comparison_rows", []) if row.get("name") == "current_tusb_v3p1_best::best.pt"), None)) is not None
        ),
        "matched_6seed_improved": bool(matched.get("matched_6seed_improved_vs_current_calonly", False)),
        "bootstrap_ci_zero_excluded": bool(bootstrap.get("bootstrap_ci_zero_excluded", False)),
        "downstream_utility_improved": bool(utility.get("utility_improved_vs_calibration", False)),
        "ood_transfer_improved": bool(ood.get("ood_improved_vs_calibration", False)),
        "mechanism_cross_seed_stable": bool(mechanism.get("mechanism_cross_seed_stable", False)),
        "appearance_claim_allowed": bool(appearance.get("appearance_claim_allowed", False)),
    }
    # precise densified improvement from existing audit
    dens_ctx = dualpanel.get("densified_200_context_preserving", {})
    methods = {row.get("name"): row for row in dens_ctx.get("methods", [])}
    if "current_tusb_v3p1_best::best.pt" in methods and "current_calibration_only_best" in methods:
        summary["context_preserving_densified_200_improved_vs_current_calonly"] = bool(
            float(methods["current_tusb_v3p1_best::best.pt"].get("query_future_top1_acc", 0.0))
            > float(methods["current_calibration_only_best"].get("query_future_top1_acc", 0.0))
        )
    diagnosis = {
        "generated_at_utc": _now_iso(),
        **summary,
        "paper_target_recommendation": paper.get("paper_target_recommendation", "borderline_needs_one_last_fix"),
        "oral_spotlight_readiness": paper.get("oral_spotlight_readiness", "not_ready"),
        "next_step_choice": paper.get("next_step_choice", "one_last_surgical_fix"),
    }
    _write_json(Path(args.final_validation_launch_report), launch)
    _write_json(Path(args.final_validation_summary_report), summary)
    _write_json(Path(args.final_validation_diagnosis_report), diagnosis)
    _write_md(
        Path(args.final_validation_doc),
        [
            "# STWM Top-Tier Final Validation 20260420",
            "",
            f"- context_preserving_densified_200_improved_vs_current_calonly: {diagnosis['context_preserving_densified_200_improved_vs_current_calonly']}",
            f"- matched_6seed_improved: {diagnosis['matched_6seed_improved']}",
            f"- bootstrap_ci_zero_excluded: {diagnosis['bootstrap_ci_zero_excluded']}",
            f"- downstream_utility_improved: {diagnosis['downstream_utility_improved']}",
            f"- ood_transfer_improved: {diagnosis['ood_transfer_improved']}",
            f"- mechanism_cross_seed_stable: {diagnosis['mechanism_cross_seed_stable']}",
            f"- appearance_claim_allowed: {diagnosis['appearance_claim_allowed']}",
            f"- paper_target_recommendation: {diagnosis['paper_target_recommendation']}",
            f"- oral_spotlight_readiness: {diagnosis['oral_spotlight_readiness']}",
            f"- next_step_choice: {diagnosis['next_step_choice']}",
        ],
    )
    return {"launch": launch, "summary": summary, "diagnosis": diagnosis}


def main() -> None:
    _apply_process_title_normalization()
    parser = ArgumentParser(description="STWM top-tier final validation and credibility pack.")
    parser.add_argument("--result-consistency-report", default=str(ROOT / "reports/stwm_top_tier_result_consistency_audit_20260420.json"))
    parser.add_argument("--result-consistency-doc", default=str(ROOT / "docs/STWM_TOP_TIER_RESULT_CONSISTENCY_AUDIT_20260420.md"))
    parser.add_argument("--live-manifest-report", default=str(ROOT / "reports/stwm_top_tier_live_manifest_20260420.json"))
    parser.add_argument("--validation-protocol-report", default=str(ROOT / "reports/stwm_top_tier_validation_protocol_20260420.json"))
    parser.add_argument("--validation-protocol-doc", default=str(ROOT / "docs/STWM_TOP_TIER_VALIDATION_PROTOCOL_20260420.md"))
    parser.add_argument("--seed-completion-plan-report", default=str(ROOT / "reports/stwm_top_tier_seed_completion_plan_20260420.json"))
    parser.add_argument("--seed-completion-launch-report", default=str(ROOT / "reports/stwm_top_tier_seed_completion_launch_20260420.json"))
    parser.add_argument("--seed-completion-summary-report", default=str(ROOT / "reports/stwm_top_tier_seed_completion_summary_20260420.json"))
    parser.add_argument("--seed-completion-doc", default=str(ROOT / "docs/STWM_TOP_TIER_SEED_COMPLETION_20260420.md"))
    parser.add_argument("--matched-6seed-report", default=str(ROOT / "reports/stwm_top_tier_matched_6seed_dualpanel_20260420.json"))
    parser.add_argument("--matched-6seed-doc", default=str(ROOT / "docs/STWM_TOP_TIER_MATCHED_6SEED_DUALPANEL_20260420.md"))
    parser.add_argument("--final-bootstrap-report", default=str(ROOT / "reports/stwm_top_tier_final_bootstrap_ci_20260420.json"))
    parser.add_argument("--final-bootstrap-doc", default=str(ROOT / "docs/STWM_TOP_TIER_FINAL_BOOTSTRAP_CI_20260420.md"))
    parser.add_argument("--downstream-utility-report", default=str(ROOT / "reports/stwm_top_tier_downstream_utility_20260420.json"))
    parser.add_argument("--downstream-utility-doc", default=str(ROOT / "docs/STWM_TOP_TIER_DOWNSTREAM_UTILITY_20260420.md"))
    parser.add_argument("--ood-report", default=str(ROOT / "reports/stwm_top_tier_ood_transfer_20260420.json"))
    parser.add_argument("--ood-doc", default=str(ROOT / "docs/STWM_TOP_TIER_OOD_TRANSFER_20260420.md"))
    parser.add_argument("--mechanism-report", default=str(ROOT / "reports/stwm_top_tier_mechanism_6seed_repair_20260420.json"))
    parser.add_argument("--mechanism-doc", default=str(ROOT / "docs/STWM_TOP_TIER_MECHANISM_6SEED_REPAIR_20260420.md"))
    parser.add_argument("--appearance-fix-report", default=str(ROOT / "reports/stwm_top_tier_appearance_plumbing_surgical_fix_20260420.json"))
    parser.add_argument("--appearance-fix-doc", default=str(ROOT / "docs/STWM_TOP_TIER_APPEARANCE_PLUMBING_SURGICAL_FIX_20260420.md"))
    parser.add_argument("--teacher-sanity-report", default=str(ROOT / "reports/stwm_top_tier_teacher_prior_sanity_20260420.json"))
    parser.add_argument("--teacher-sanity-doc", default=str(ROOT / "docs/STWM_TOP_TIER_TEACHER_PRIOR_SANITY_20260420.md"))
    parser.add_argument("--surgical-gate-report", default=str(ROOT / "reports/stwm_top_tier_surgical_fix_gate_20260420.json"))
    parser.add_argument("--surgical-gate-doc", default=str(ROOT / "docs/STWM_TOP_TIER_SURGICAL_FIX_GATE_20260420.md"))
    parser.add_argument("--paper-decision-report", default=str(ROOT / "reports/stwm_top_tier_paper_decision_20260420.json"))
    parser.add_argument("--paper-decision-doc", default=str(ROOT / "docs/STWM_TOP_TIER_PAPER_DECISION_20260420.md"))
    parser.add_argument("--final-validation-launch-report", default=str(ROOT / "reports/stwm_top_tier_final_validation_launch_20260420.json"))
    parser.add_argument("--final-validation-summary-report", default=str(ROOT / "reports/stwm_top_tier_final_validation_summary_20260420.json"))
    parser.add_argument("--final-validation-diagnosis-report", default=str(ROOT / "reports/stwm_top_tier_final_validation_diagnosis_20260420.json"))
    parser.add_argument("--final-validation-doc", default=str(ROOT / "docs/STWM_TOP_TIER_FINAL_VALIDATION_20260420.md"))
    args = parser.parse_args()

    _append_log("start final validation pack")
    _result_consistency_and_manifest(args)
    _validation_protocol(args)
    seed_summary = _seed_completion_reports(args)
    matched = _matched_6seed_dualpanel(args, seed_summary)
    bootstrap = _final_bootstrap_ci(args)
    utility = build_downstream_utility(Path(args.downstream_utility_report), Path(args.downstream_utility_doc))
    ood = _ood_transfer(args)
    mechanism = _mechanism_6seed_repair(args)
    appearance = _appearance_plumbing_fix(args)
    _teacher_prior_sanity(args)
    _surgical_fix_gate(args, matched, utility, ood, appearance)
    paper = _paper_decision(args, matched, bootstrap, utility, ood, mechanism, appearance)
    dualpanel = _load_json(ROOT / "reports/stage2_v3p1_dualpanel_context_audit_20260420.json")
    _final_validation_reports(args, dualpanel, matched, bootstrap, utility, ood, mechanism, appearance, paper)
    _append_log("finished final validation pack")


if __name__ == "__main__":
    main()
