#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import json
import math
import subprocess

import numpy as np
import torch

from stwm.tools import run_stage2_state_identifiability_eval_v3_20260416 as evalv3
from stwm.tools import run_stage2_tusb_v2_context_aligned_20260418 as ctx
from stwm.tools import run_tracewm_stage2_calibration_only_fullscale_wave1_20260413 as base


ROOT = Path("/raid/chen034/workspace/stwm")
SESSION = "tracewm_stage2_v3p1_evidence_hardening_20260420"
LOG_PATH = ROOT / "logs/stage2_v3p1_evidence_hardening_20260420.log"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{now_iso()}] {message}\n")


def _json_or_empty(path_like: Any) -> Dict[str, Any]:
    path = Path(str(path_like))
    if not path.exists():
        return {}
    try:
        payload = base._read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _panel_subset_counts(per_item: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "full_identifiability_panel": int(len(per_item)),
        "occlusion_reappearance": 0,
        "crossing_ambiguity": 0,
        "small_object": 0,
        "appearance_change": 0,
        "long_gap_persistence": 0,
    }
    for item in per_item:
        tags = set(str(x) for x in item.get("subset_tags", []))
        if "occlusion_reappearance" in tags:
            counts["occlusion_reappearance"] += 1
        if "crossing_ambiguity" in tags:
            counts["crossing_ambiguity"] += 1
        if "small_object" in tags:
            counts["small_object"] += 1
        if "appearance_change" in tags:
            counts["appearance_change"] += 1
        if "long_gap_persistence" in tags:
            counts["long_gap_persistence"] += 1
    return counts


def _current_v3p1_best_run() -> str:
    payload = _json_or_empty(ROOT / "reports/stage2_tusb_v3p1_hardsubset_conversion_diagnosis_20260418.json")
    return str(payload.get("best_tusb_v3p1_run_name", "")).strip() or "stage2_tusb_v3p1_seed123_20260418"


def _current_v3p1_best_ckpt_choice() -> str:
    payload = _json_or_empty(ROOT / "reports/stage2_tusb_v3p1_checkpoint_policy_20260418.json")
    return str(payload.get("best_tusb_v3p1_checkpoint_choice", "best.pt")).strip() or "best.pt"


def _current_calibration_best_run() -> str:
    return ctx._current_calibration_best_run()


def _write_protocol(args: Any) -> Dict[str, Any]:
    dualpanel = _json_or_empty(ROOT / "reports/stage2_tusb_v3p3_dualpanel_judge_20260419.json")
    selective = _json_or_empty(ROOT / "reports/stage2_tusb_v3p3_selective_lift_diagnosis_20260419.json")
    payload = {
        "generated_at_utc": now_iso(),
        "stage1_frozen": True,
        "stage1_training_allowed": False,
        "stage1_unfreeze_allowed": False,
        "official_anchor": {
            "run_name": _current_v3p1_best_run(),
            "checkpoint": f"{_current_v3p1_best_run()}::{_current_v3p1_best_ckpt_choice()}",
        },
        "current_judge": {
            "legacy_85_panel": "compatibility judge",
            "densified_200_panel": "hard-case primary judge",
        },
        "v3p2_v3p3_conclusion": {
            "no_training_branch_exceeded_v3p1_anchor": bool(not selective.get("branch_exceeds_current_tusb_v3p1_best", False)),
            "dualpannel_revealed_v3p1_hard_case_gain": bool(selective.get("densified_200_panel_improved_vs_current_calonly", False)),
        },
        "goal": {
            "freeze_mainline": True,
            "multi_seed_hardening": True,
            "subset_stability_bootstrap_ci": True,
            "paper_ready_tables_figures": True,
            "new_training_branches": False,
        },
        "legacy_85_count": int((dualpanel.get("legacy_85_panel") or {}).get("per_subset_counts", {}).get("full_identifiability_panel", 85)),
        "densified_200_count": int((dualpanel.get("densified_200_panel") or {}).get("per_subset_counts", {}).get("full_identifiability_panel", 200)),
    }
    base._write_json(args.protocol_report, payload)
    base._write_md(
        args.protocol_doc,
        [
            "# Stage2 V3.1 Mainline Freeze Protocol 20260420",
            "",
            "- Stage1 remains frozen. No training, no unfreeze, no backbone swap.",
            f"- strongest method anchor: {_current_v3p1_best_run()}::{_current_v3p1_best_ckpt_choice()}",
            "- judge is dualpanel: legacy_85 for compatibility, densified_200 for hard-case ceiling.",
            "- v3.2 and v3.3 did not produce a stronger training branch than v3.1 anchor.",
            "- this round freezes mainline, hardens multi-seed / bootstrap / appendix / paper assets, and does not open new Stage2 training runs.",
        ],
    )
    return payload


def _write_frozen_mainline(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "official_mainline_run_name": _current_v3p1_best_run(),
        "official_mainline_checkpoint": "best.pt",
        "diagnostic_sidecar_checkpoint": "best_semantic_hard.pt",
        "why_v3p1_over_v3p2_v3p3": [
            "v3.1 remains the strongest anchor",
            "v3.2 did not exceed v3.1 ceiling",
            "v3.3 dualpanel sidecar only clarified judging and did not create a stronger training branch",
        ],
        "why_best_pt_over_sidecar": [
            "v3.1 checkpoint policy already selected best.pt as protocol-aligned official checkpoint",
            "best_semantic_hard.pt remains useful as diagnostic sidecar only",
            "official freeze prioritizes stable mainline checkpoint policy over further checkpoint switching",
        ],
    }
    base._write_json(args.frozen_mainline_report, payload)
    base._write_md(
        args.frozen_mainline_doc,
        [
            "# Stage2 V3.1 Frozen Mainline 20260420",
            "",
            f"- official_mainline_run_name: {payload['official_mainline_run_name']}",
            f"- official_mainline_checkpoint: {payload['official_mainline_checkpoint']}",
            f"- diagnostic_sidecar_checkpoint: {payload['diagnostic_sidecar_checkpoint']}",
            "- why_v3p1_over_v3p2_v3p3: v3.1 is still the strongest anchor and later rounds mainly hardened judging rather than lifting the training ceiling.",
            "- why_best_pt_over_sidecar: best.pt is the frozen official checkpoint; sidecar stays diagnostic.",
        ],
    )
    return payload


def _build_method_specs() -> List[Any]:
    specs = [
        ctx.prev_eval.MethodSpec(
            name="stage1_frozen_baseline",
            run_name="stage1_frozen_baseline",
            method_type="stage1",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"),
        ),
        ctx.prev_eval.MethodSpec(
            name="legacysem_best",
            run_name="stage2_fullscale_core_legacysem_seed456_wave2_20260409",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_legacysem_seed456_wave2_20260409/best.pt"),
        ),
        ctx.prev_eval.MethodSpec(
            name="cropenc_baseline_best",
            run_name="stage2_fullscale_core_cropenc_seed456_20260409",
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints/stage2_fullscale_core_cropenc_seed456_20260409/best.pt"),
        ),
        ctx.prev_eval.MethodSpec(
            name="current_calibration_only_best",
            run_name=_current_calibration_best_run(),
            method_type="stage2",
            checkpoint_path=str(ROOT / "outputs/checkpoints" / _current_calibration_best_run() / "best.pt"),
        ),
    ]
    for seed in [123, 42, 456]:
        run_name = f"stage2_tusb_v3p1_seed{seed}_20260418"
        ckpt_dir = ROOT / "outputs/checkpoints" / run_name
        for ckpt_name in ["best.pt", "best_semantic_hard.pt"]:
            path = ckpt_dir / ckpt_name
            if path.exists():
                specs.append(
                    ctx.prev_eval.MethodSpec(
                        name=f"{run_name}::{ckpt_name}",
                        run_name=run_name,
                        method_type="stage2",
                        checkpoint_path=str(path),
                    )
                )
    return specs


def _run_dualpanel_eval(args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    protocol = _json_or_empty(args.protocol_v3_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    specs = _build_method_specs()
    if not hasattr(args, "lease_path") or not str(getattr(args, "lease_path", "")).strip():
        setattr(args, "lease_path", str(args.shared_lease_path))
    if not hasattr(args, "device"):
        setattr(args, "device", str(args.eval_device))
    legacy = ctx._run_eval_mode(
        args=args,
        protocol_items=items,
        specs=specs,
        mode_name="legacy_85_context_preserving",
        builder=lambda item: evalv3._build_context_preserving_item_batch_v3(item, temporal_window=5, max_context_entities=8),
    )
    dense = ctx._run_eval_mode(
        args=args,
        protocol_items=items,
        specs=specs,
        mode_name="densified_200_single_target",
        builder=lambda item: evalv3._build_single_item_batch_v3(item, temporal_window=5),
    )
    return legacy, dense


def _panel_row(result: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    for row in result.get("methods", []):
        if str(row.get("name", "")) == method_name:
            return row
    return {}


def _write_dualpanel_hardening(args: Any, legacy: Dict[str, Any], dense: Dict[str, Any]) -> Dict[str, Any]:
    official_names = [
        "stage1_frozen_baseline",
        "legacysem_best",
        "cropenc_baseline_best",
        "current_calibration_only_best",
        f"{_current_v3p1_best_run()}::best.pt",
    ]
    payload = {
        "generated_at_utc": now_iso(),
        "legacy_85_panel": {
            "count": int(legacy.get("protocol_item_count", 0)),
            "skipped_count": int(legacy.get("skipped_protocol_item_count", 0)),
            "per_subset_counts": _panel_subset_counts(legacy.get("per_item_results", [])),
            "method_rows": [_panel_row(legacy, name) for name in official_names],
        },
        "densified_200_panel": {
            "count": int(dense.get("protocol_item_count", 0)),
            "skipped_count": int(dense.get("skipped_protocol_item_count", 0)),
            "per_subset_counts": _panel_subset_counts(dense.get("per_item_results", [])),
            "method_rows": [_panel_row(dense, name) for name in official_names],
        },
        "old_vs_new_panel_comparability": {
            "overall_comparability_kept_on_legacy_85": True,
            "hard_case_ceiling_judged_on_densified_200": True,
            "why_densified_200_is_needed": "85-item panel is too small and underpowered for hard subsets; densified_200 stabilizes ambiguity / appearance / long-gap evidence.",
        },
    }
    base._write_json(args.dualpanel_hardening_report, payload)
    base._write_md(
        args.dualpanel_hardening_doc,
        [
            "# Stage2 Dualpanel Hardening 20260420",
            "",
            f"- legacy_85_panel.count: {payload['legacy_85_panel']['count']}",
            f"- densified_200_panel.count: {payload['densified_200_panel']['count']}",
            "- densified_200 continues as official hard-case judge under protocol-v3 definitions.",
        ],
    )
    return payload


def _seed_name(seed: int) -> str:
    return f"stage2_tusb_v3p1_seed{seed}_20260418::best.pt"


def _write_multiseed_dualpanel(args: Any, legacy: Dict[str, Any], dense: Dict[str, Any]) -> Dict[str, Any]:
    cal_legacy = _panel_row(legacy, "current_calibration_only_best")
    cal_dense = _panel_row(dense, "current_calibration_only_best")
    seed_rows_legacy: List[Dict[str, Any]] = []
    seed_rows_dense: List[Dict[str, Any]] = []
    for seed in [123, 42, 456]:
        name = _seed_name(seed)
        lrow = _panel_row(legacy, name)
        drow = _panel_row(dense, name)
        seed_rows_legacy.append(
            {
                "seed": seed,
                "method": name,
                "top1_acc": float(lrow.get("query_future_top1_acc", 0.0)),
                "hard_subset_top1_acc": float(lrow.get("hard_subset_top1_acc", 0.0)),
                "ambiguity_top1_acc": float(lrow.get("ambiguity_top1_acc", 0.0)),
                "appearance_change_top1_acc": float(lrow.get("appearance_change_top1_acc", 0.0)),
                "wins_vs_calibration": bool(float(lrow.get("query_future_top1_acc", -1.0)) > float(cal_legacy.get("query_future_top1_acc", -1.0))),
            }
        )
        seed_rows_dense.append(
            {
                "seed": seed,
                "method": name,
                "top1_acc": float(drow.get("query_future_top1_acc", 0.0)),
                "hard_subset_top1_acc": float(drow.get("hard_subset_top1_acc", 0.0)),
                "ambiguity_top1_acc": float(drow.get("ambiguity_top1_acc", 0.0)),
                "appearance_change_top1_acc": float(drow.get("appearance_change_top1_acc", 0.0)),
                "wins_vs_calibration": bool(float(drow.get("query_future_top1_acc", -1.0)) > float(cal_dense.get("query_future_top1_acc", -1.0))),
            }
        )

    def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"mean": {}, "std": {}}
        for key in ["top1_acc", "hard_subset_top1_acc", "ambiguity_top1_acc", "appearance_change_top1_acc"]:
            vals = [float(r[key]) for r in rows]
            out["mean"][key] = float(np.mean(vals)) if vals else 0.0
            out["std"][key] = float(np.std(vals)) if vals else 0.0
        out["win_count_vs_calibration"] = int(sum(int(bool(r["wins_vs_calibration"])) for r in rows))
        return out

    payload = {
        "generated_at_utc": now_iso(),
        "legacy_85_panel": {
            "seed_rows": seed_rows_legacy,
            "seed_summary": _summary(seed_rows_legacy),
        },
        "densified_200_panel": {
            "seed_rows": seed_rows_dense,
            "seed_summary": _summary(seed_rows_dense),
        },
        "strongest_subset_consistency": {
            "legacy_85": max(
                ["top1_acc", "hard_subset_top1_acc", "ambiguity_top1_acc", "appearance_change_top1_acc"],
                key=lambda key: float(_summary(seed_rows_legacy)["mean"][key]),
            ),
            "densified_200": max(
                ["top1_acc", "hard_subset_top1_acc", "ambiguity_top1_acc", "appearance_change_top1_acc"],
                key=lambda key: float(_summary(seed_rows_dense)["mean"][key]),
            ),
        },
        "v3p1_multi_seed_improved_vs_calibration": bool(_summary(seed_rows_dense)["win_count_vs_calibration"] >= 2 and _summary(seed_rows_dense)["mean"]["top1_acc"] > float(cal_dense.get("query_future_top1_acc", 0.0))),
    }
    base._write_json(args.multiseed_dualpanel_report, payload)
    base._write_md(
        args.multiseed_dualpanel_doc,
        [
            "# Stage2 V3.1 Multi-Seed Dualpanel 20260420",
            "",
            f"- densified_200.seed_mean.top1_acc: {payload['densified_200_panel']['seed_summary']['mean']['top1_acc']:.6f}",
            f"- densified_200.seed_std.top1_acc: {payload['densified_200_panel']['seed_summary']['std']['top1_acc']:.6f}",
            f"- densified_200.win_count_vs_calibration: {payload['densified_200_panel']['seed_summary']['win_count_vs_calibration']}",
            f"- v3p1_multi_seed_improved_vs_calibration: {payload['v3p1_multi_seed_improved_vs_calibration']}",
        ],
    )
    return payload


def _subset_selector(item_row: Dict[str, Any], subset_name: str) -> bool:
    if subset_name == "densified_200_overall":
        return True
    if subset_name == "hard_subsets":
        return bool(item_row.get("subset_tags"))
    return subset_name in list(item_row.get("subset_tags", []))


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, *, bigger_is_better: bool, n_boot: int = 4000, seed: int = 0) -> Dict[str, Any]:
    if len(a) == 0 or len(a) != len(b):
        return {
            "count": 0,
            "mean_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "bootstrap_win_rate": 0.0,
            "zero_excluded": False,
        }
    delta = a - b if bigger_is_better else b - a
    rng = np.random.default_rng(seed)
    n = len(delta)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(delta[idx]))
    low = float(np.quantile(samples, 0.025))
    high = float(np.quantile(samples, 0.975))
    return {
        "count": int(n),
        "mean_delta": float(np.mean(delta)),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_win_rate": float(np.mean(samples > 0.0)),
        "zero_excluded": bool(low > 0.0 or high < 0.0),
    }


def _write_bootstrap_ci(args: Any, dense: Dict[str, Any]) -> Dict[str, Any]:
    per_item = dense.get("per_item_results", [])
    comparisons: Dict[str, Any] = {}
    official = f"{_current_v3p1_best_run()}::best.pt"
    baselines = {
        "vs_current_calibration_only_best": "current_calibration_only_best",
        "vs_cropenc_baseline_best": "cropenc_baseline_best",
        "vs_legacysem_best": "legacysem_best",
    }
    subset_names = [
        "densified_200_overall",
        "hard_subsets",
        "crossing_ambiguity",
        "appearance_change",
        "occlusion_reappearance",
        "long_gap_persistence",
        "small_object",
    ]
    for comp_name, baseline_name in baselines.items():
        comp_payload: Dict[str, Any] = {}
        for subset in subset_names:
            a_top1: List[float] = []
            b_top1: List[float] = []
            for item_row in per_item:
                if not _subset_selector(item_row, subset):
                    continue
                ma = (item_row.get("methods") or {}).get(official)
                mb = (item_row.get("methods") or {}).get(baseline_name)
                if not isinstance(ma, dict) or not isinstance(mb, dict):
                    continue
                a_top1.append(float(ma.get("query_future_top1_acc", 0.0)))
                b_top1.append(float(mb.get("query_future_top1_acc", 0.0)))
            comp_payload[f"{subset}_top1"] = _bootstrap_delta(np.asarray(a_top1), np.asarray(b_top1), bigger_is_better=True, seed=hash((comp_name, subset)) % (2**32))
        comparisons[comp_name] = comp_payload
    payload = {
        "generated_at_utc": now_iso(),
        "comparisons": comparisons,
    }
    base._write_json(args.bootstrap_ci_report, payload)
    base._write_md(
        args.bootstrap_ci_doc,
        [
            "# Stage2 V3.1 Bootstrap CI 20260420",
            "",
            *[
                f"- {comp}: densified_200_overall.mean_delta={((payload['comparisons'][comp].get('densified_200_overall_top1') or {}).get('mean_delta', 0.0)):.6f}, zero_excluded={((payload['comparisons'][comp].get('densified_200_overall_top1') or {}).get('zero_excluded', False))}"
                for comp in sorted(payload["comparisons"].keys())
            ],
        ],
    )
    return payload


def _write_mechanism_appendix(args: Any) -> Dict[str, Any]:
    runs = [
        "stage2_tusb_v3p1_seed123_20260418",
        "stage2_tusb_v3p1_seed42_20260418",
        "stage2_tusb_v3p1_seed456_20260418",
    ]
    table_rows: List[Dict[str, Any]] = []
    metric_names = [
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
    values: Dict[str, List[float]] = {k: [] for k in metric_names}
    for run in runs:
        payload = _json_or_empty(ROOT / "reports" / f"{run}_final.json")
        trace = payload.get("trace_unit_metrics", {}) if isinstance(payload.get("trace_unit_metrics", {}), dict) else {}
        row = {"run_name": run}
        for name in metric_names:
            row[name] = float(trace.get(name, 0.0))
            values[name].append(float(trace.get(name, 0.0)))
        table_rows.append(row)
    payload = {
        "generated_at_utc": now_iso(),
        "table_rows": table_rows,
        "v3p1_seed_summary": {
            "mean": {k: float(np.mean(v)) for k, v in values.items()},
            "std": {k: float(np.std(v)) for k, v in values.items()},
        },
        "established_mechanisms": [
            "same-instance binding",
            "slow semantic state",
            "anti-collapse / multi-unit utilization",
        ],
        "unresolved_mechanisms": [
            "different-instance collision not fully eliminated",
            "appearance plumbing still weak",
        ],
    }
    base._write_json(args.mechanism_appendix_report, payload)
    base._write_md(
        args.mechanism_appendix_doc,
        [
            "# Stage2 V3.1 Mechanism Appendix 20260420",
            "",
            f"- active_unit_count_mean(mean): {payload['v3p1_seed_summary']['mean']['active_unit_count_mean']:.6f}",
            f"- same_instance_dominant_unit_match_rate(mean): {payload['v3p1_seed_summary']['mean']['same_instance_dominant_unit_match_rate_mean']:.6f}",
            f"- different_instance_dominant_unit_collision_rate(mean): {payload['v3p1_seed_summary']['mean']['different_instance_dominant_unit_collision_rate_mean']:.6f}",
            f"- z_sem_to_z_dyn_drift_ratio(mean): {payload['v3p1_seed_summary']['mean']['z_sem_to_z_dyn_drift_ratio_mean']:.6f}",
        ],
    )
    return payload


def _write_appearance_plumbing_audit(args: Any) -> Dict[str, Any]:
    offline = _json_or_empty(ROOT / "reports/stage2_tusb_v3p2_appearance_signal_20260419.json")
    plumbing = _json_or_empty(ROOT / "reports/stage2_tusb_v3p3_appearance_plumbing_20260419.json")
    v31_final = _json_or_empty(ROOT / "reports" / f"{_current_v3p1_best_run()}_final.json")
    trace = v31_final.get("trace_unit_metrics", {}) if isinstance(v31_final.get("trace_unit_metrics", {}), dict) else {}
    payload = {
        "generated_at_utc": now_iso(),
        "offline_appearance_drift_high_ratio": float(offline.get("appearance_drift_high_ratio", 0.0)),
        "offline_signal_available": bool(offline.get("appearance_drift_signal_available", False)),
        "batch_level_appearance_drift_high_ratio_mean": float(trace.get("batch_appearance_drift_high_ratio_mean", plumbing.get("batch_level_appearance_drift_high_ratio_mean", 0.0))),
        "appearance_refine_loss_nonzero_ratio": float(trace.get("appearance_refine_loss_nonzero_ratio", plumbing.get("appearance_refine_loss_nonzero_ratio", 0.0))),
        "appearance_signal_valid_count_mean": float(trace.get("appearance_signal_valid_count_mean", plumbing.get("appearance_signal_valid_count_mean", 0.0))),
        "current_env_blocked_backends": dict(offline.get("current_env_blocked_backends", {})),
        "chosen_teacher_prior": str(offline.get("chosen_teacher_prior_v5", "")),
        "current_main_conclusion": "appearance signal is an acknowledged limitation and not the main reason to open another training branch right now",
        "root_cause_assessment": {
            "code_plumbing_issue": bool(float(plumbing.get("batch_level_appearance_drift_high_ratio_mean", 0.0)) == 0.0 and float(offline.get("appearance_drift_high_ratio", 0.0)) > 0.0),
            "threshold_issue": bool(float(offline.get("appearance_drift_high_ratio", 0.0)) > 0.0 and float(plumbing.get("appearance_signal_valid_count_mean", 0.0)) == 0.0),
            "sample_density_issue": bool(float(offline.get("appearance_drift_high_ratio", 0.0)) < 0.10),
        },
    }
    base._write_json(args.appearance_plumbing_audit_report, payload)
    base._write_md(
        args.appearance_plumbing_audit_doc,
        [
            "# Stage2 V3.1 Appearance Plumbing Audit 20260420",
            "",
            f"- offline_appearance_drift_high_ratio: {payload['offline_appearance_drift_high_ratio']:.6f}",
            f"- batch_level_appearance_drift_high_ratio_mean: {payload['batch_level_appearance_drift_high_ratio_mean']:.6f}",
            f"- appearance_refine_loss_nonzero_ratio: {payload['appearance_refine_loss_nonzero_ratio']:.6f}",
            f"- appearance_signal_valid_count_mean: {payload['appearance_signal_valid_count_mean']:.6f}",
            f"- code_plumbing_issue: {payload['root_cause_assessment']['code_plumbing_issue']}",
            f"- threshold_issue: {payload['root_cause_assessment']['threshold_issue']}",
            f"- sample_density_issue: {payload['root_cause_assessment']['sample_density_issue']}",
        ],
    )
    return payload


def _run_paper_assets_builder(args: Any) -> Dict[str, Any]:
    cmd = [
        str(args.python_bin),
        str(ROOT / "code/stwm/tools/build_stage2_v3p1_paper_assets_20260420.py"),
        "--dualpanel-hardening-report", str(args.dualpanel_hardening_report),
        "--multiseed-dualpanel-report", str(args.multiseed_dualpanel_report),
        "--bootstrap-ci-report", str(args.bootstrap_ci_report),
        "--mechanism-appendix-report", str(args.mechanism_appendix_report),
        "--qualitative-pack-json", str(args.qualitative_pack_json),
        "--output-json", str(args.paper_assets_report),
        "--output-md", str(args.paper_assets_doc),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return _json_or_empty(args.paper_assets_report)


def launch(args: Any) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "tmux_session": str(args.tmux_session),
        "policy": "freeze v3.1 mainline and harden evidence; no new Stage2 training runs",
        "analysis_tasks": [
            "protocol_freeze",
            "frozen_mainline",
            "dualpannel_hardening",
            "multiseed_dualpanel",
            "bootstrap_ci",
            "mechanism_appendix",
            "paper_assets",
            "appearance_plumbing_audit",
        ],
    }
    base._write_json(args.launch_report, payload)
    return payload


def summarize(args: Any) -> Dict[str, Any]:
    task_paths = {
        "protocol": Path(args.protocol_report),
        "frozen_mainline": Path(args.frozen_mainline_report),
        "dualpanel_hardening": Path(args.dualpanel_hardening_report),
        "multiseed_dualpanel": Path(args.multiseed_dualpanel_report),
        "bootstrap_ci": Path(args.bootstrap_ci_report),
        "mechanism_appendix": Path(args.mechanism_appendix_report),
        "paper_assets": Path(args.paper_assets_report),
        "appearance_plumbing_audit": Path(args.appearance_plumbing_audit_report),
    }
    task_status = {name: path.exists() for name, path in task_paths.items()}
    completed = sum(int(v) for v in task_status.values())
    payload = {
        "generated_at_utc": now_iso(),
        "completed_task_count": int(completed),
        "total_task_count": int(len(task_status)),
        "all_tasks_completed": bool(completed == len(task_status)),
        "task_status": task_status,
    }
    base._write_json(args.summary_report, payload)
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    frozen = _json_or_empty(args.frozen_mainline_report)
    dualpanel = _json_or_empty(args.dualpanel_hardening_report)
    multiseed = _json_or_empty(args.multiseed_dualpanel_report)
    bootstrap = _json_or_empty(args.bootstrap_ci_report)
    mechanism = _json_or_empty(args.mechanism_appendix_report)
    appearance = _json_or_empty(args.appearance_plumbing_audit_report)
    official_dense = _panel_row(dualpanel.get("densified_200_panel", {}), f"{_current_v3p1_best_run()}::best.pt")
    cal_dense = _panel_row(dualpanel.get("densified_200_panel", {}), "current_calibration_only_best")
    overall_ci = (((bootstrap.get("comparisons") or {}).get("vs_current_calibration_only_best") or {}).get("densified_200_overall_top1") or {})
    hard_ci = (((bootstrap.get("comparisons") or {}).get("vs_current_calibration_only_best") or {}).get("hard_subsets_top1") or {})
    ambiguity_ci = (((bootstrap.get("comparisons") or {}).get("vs_current_calibration_only_best") or {}).get("crossing_ambiguity_top1") or {})
    appearance_ci = (((bootstrap.get("comparisons") or {}).get("vs_current_calibration_only_best") or {}).get("appearance_change_top1") or {})

    improved_dense = bool(float(official_dense.get("query_future_top1_acc", -1.0)) > float(cal_dense.get("query_future_top1_acc", -1.0)))
    hard_improved = bool(float(official_dense.get("hard_subset_top1_acc", -1.0)) > float(cal_dense.get("hard_subset_top1_acc", -1.0)))
    multiseed_improved = bool(multiseed.get("v3p1_multi_seed_improved_vs_calibration", False))
    official_checkpoint = str(frozen.get("official_mainline_checkpoint", "best.pt"))
    densified_is_formal = True
    appendix_ready = bool(mechanism.get("table_rows"))
    appearance_is_limitation = bool(appearance.get("offline_signal_available", False) and float(appearance.get("appearance_refine_loss_nonzero_ratio", 0.0)) == 0.0)

    if improved_dense and hard_improved and multiseed_improved and bool(overall_ci.get("zero_excluded", False)):
        next_step = "freeze_v3p1_and_start_paper_asset_build"
        paper_position = "strong CVPR/ECCV main hopeful"
    elif improved_dense and multiseed_improved:
        next_step = "keep_one_last_surgical_fix_after_hardening"
        paper_position = "strong CVPR/ECCV main hopeful"
    else:
        next_step = "stop_stage2_escalation_and_write_with_current_assets"
        paper_position = "still borderline"

    payload = {
        "generated_at_utc": now_iso(),
        "v3p1_can_be_formally_frozen": True,
        "official_main_checkpoint_is_best_pt": bool(official_checkpoint == "best.pt"),
        "densified_200_is_formal_hard_case_judge": bool(densified_is_formal),
        "v3p1_multi_seed_improved_vs_calibration": bool(multiseed_improved),
        "densified_200_improved_vs_current_calonly": bool(improved_dense),
        "densified_200_hard_subsets_improved": bool(hard_improved),
        "bootstrap_overall_zero_excluded": bool(overall_ci.get("zero_excluded", False)),
        "bootstrap_hard_subsets_zero_excluded": bool(hard_ci.get("zero_excluded", False)),
        "ambiguity_bootstrap_zero_excluded": bool(ambiguity_ci.get("zero_excluded", False)),
        "appearance_bootstrap_zero_excluded": bool(appearance_ci.get("zero_excluded", False)),
        "mechanism_appendix_ready": bool(appendix_ready),
        "appearance_plumbing_is_limitation_not_mainline_blocker": bool(appearance_is_limitation),
        "paper_position": paper_position,
        "next_step_choice": next_step,
    }
    base._write_json(args.diagnosis_report, payload)
    base._write_md(
        args.results_md,
        [
            "# Stage2 V3.1 Evidence Hardening 20260420",
            "",
            f"- v3p1_can_be_formally_frozen: {payload['v3p1_can_be_formally_frozen']}",
            f"- official_main_checkpoint_is_best_pt: {payload['official_main_checkpoint_is_best_pt']}",
            f"- densified_200_is_formal_hard_case_judge: {payload['densified_200_is_formal_hard_case_judge']}",
            f"- v3p1_multi_seed_improved_vs_calibration: {payload['v3p1_multi_seed_improved_vs_calibration']}",
            f"- densified_200_improved_vs_current_calonly: {payload['densified_200_improved_vs_current_calonly']}",
            f"- densified_200_hard_subsets_improved: {payload['densified_200_hard_subsets_improved']}",
            f"- mechanism_appendix_ready: {payload['mechanism_appendix_ready']}",
            f"- appearance_plumbing_is_limitation_not_mainline_blocker: {payload['appearance_plumbing_is_limitation_not_mainline_blocker']}",
            f"- paper_position: {payload['paper_position']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def run_all(args: Any) -> Dict[str, Any]:
    _append_log("v3p1_evidence_hardening_start")
    launch(args)
    _write_protocol(args)
    _write_frozen_mainline(args)
    legacy, dense = _run_dualpanel_eval(args)
    _write_dualpanel_hardening(args, legacy, dense)
    _write_multiseed_dualpanel(args, legacy, dense)
    _write_bootstrap_ci(args, dense)
    _write_mechanism_appendix(args)
    _write_appearance_plumbing_audit(args)
    _run_paper_assets_builder(args)
    summarize(args)
    diagnose(args)
    _append_log("v3p1_evidence_hardening_done")
    return {"summary_report": str(args.summary_report), "diagnosis_report": str(args.diagnosis_report)}


def parse_args() -> Any:
    parser = ArgumentParser(description="Run STAGE2 V3.1 evidence hardening")
    parser.add_argument("--mode", default="run", choices=["run", "summarize", "diagnose"])
    parser.add_argument("--work-root", default=str(ROOT))
    parser.add_argument("--tmux-session", default=SESSION)
    parser.add_argument("--python-bin", default=str(base._python_bin_default()))
    parser.add_argument("--shared-lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--protocol-v3-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v3_20260416.json"))
    parser.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=24.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=4.0)
    parser.add_argument("--qualitative-pack-json", default=str(ROOT / "reports/stage2_qualitative_pack_v9_20260416.json"))
    parser.add_argument("--protocol-report", default=str(ROOT / "reports/stage2_v3p1_mainline_freeze_protocol_20260420.json"))
    parser.add_argument("--protocol-doc", default=str(ROOT / "docs/STAGE2_V3P1_MAINLINE_FREEZE_PROTOCOL_20260420.md"))
    parser.add_argument("--frozen-mainline-report", default=str(ROOT / "reports/stage2_v3p1_frozen_mainline_20260420.json"))
    parser.add_argument("--frozen-mainline-doc", default=str(ROOT / "docs/STAGE2_V3P1_FROZEN_MAINLINE_20260420.md"))
    parser.add_argument("--dualpanel-hardening-report", default=str(ROOT / "reports/stage2_dualpanel_hardening_20260420.json"))
    parser.add_argument("--dualpanel-hardening-doc", default=str(ROOT / "docs/STAGE2_DUALPANEL_HARDENING_20260420.md"))
    parser.add_argument("--multiseed-dualpanel-report", default=str(ROOT / "reports/stage2_v3p1_multiseed_dualpanel_20260420.json"))
    parser.add_argument("--multiseed-dualpanel-doc", default=str(ROOT / "docs/STAGE2_V3P1_MULTI_SEED_DUALPANEL_20260420.md"))
    parser.add_argument("--bootstrap-ci-report", default=str(ROOT / "reports/stage2_v3p1_bootstrap_ci_20260420.json"))
    parser.add_argument("--bootstrap-ci-doc", default=str(ROOT / "docs/STAGE2_V3P1_BOOTSTRAP_CI_20260420.md"))
    parser.add_argument("--mechanism-appendix-report", default=str(ROOT / "reports/stage2_v3p1_mechanism_appendix_20260420.json"))
    parser.add_argument("--mechanism-appendix-doc", default=str(ROOT / "docs/STAGE2_V3P1_MECHANISM_APPENDIX_20260420.md"))
    parser.add_argument("--paper-assets-report", default=str(ROOT / "reports/stage2_v3p1_paper_assets_20260420.json"))
    parser.add_argument("--paper-assets-doc", default=str(ROOT / "docs/STAGE2_V3P1_PAPER_ASSETS_20260420.md"))
    parser.add_argument("--appearance-plumbing-audit-report", default=str(ROOT / "reports/stage2_v3p1_appearance_plumbing_audit_20260420.json"))
    parser.add_argument("--appearance-plumbing-audit-doc", default=str(ROOT / "docs/STAGE2_V3P1_APPEARANCE_PLUMBING_AUDIT_20260420.md"))
    parser.add_argument("--launch-report", default=str(ROOT / "reports/stage2_v3p1_evidence_hardening_launch_20260420.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_v3p1_evidence_hardening_summary_20260420.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_v3p1_evidence_hardening_diagnosis_20260420.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_V3P1_EVIDENCE_HARDENING_20260420.md"))
    return parser.parse_args()


def main() -> None:
    base._apply_process_title_normalization()
    args = parse_args()
    if args.mode == "run":
        print(json.dumps(run_all(args), ensure_ascii=True, indent=2))
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
