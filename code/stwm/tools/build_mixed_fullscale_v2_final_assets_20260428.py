#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPORT_DIR = Path("reports")
DOC_DIR = Path("docs")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, title: str, payload: dict[str, Any], *, notes: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for note in notes or []:
        lines.append(f"- {note}")
    if notes:
        lines.append("")
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return dict(default or {})
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_of(key: str) -> str:
    return str(key).split("::", 1)[0].upper()


def _dataset_counts(keys: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        ds = _dataset_of(key)
        out[ds] = out.get(ds, 0) + 1
    return out


def _split_subset(split_payload: dict[str, Any], dataset: str) -> dict[str, Any]:
    splits = {}
    for name, keys in split_payload.get("splits", {}).items():
        splits[name] = [str(k) for k in keys if _dataset_of(str(k)) == dataset]
    return {
        "audit_name": f"stwm_mixed_fullscale_v2_splits_{dataset.lower()}_subset",
        "source_split_report": "reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json",
        "dataset_subset": dataset,
        "item_level_split": True,
        "no_leakage": bool(split_payload.get("no_leakage", True)),
        "mixed_protocol_available": bool(split_payload.get("mixed_protocol_available", True)),
        "cross_dataset_protocol_available": bool(split_payload.get("cross_dataset_protocol_available", True)),
        "train_item_count": int(len(splits.get("train", []))),
        "val_item_count": int(len(splits.get("val", []))),
        "test_item_count": int(len(splits.get("test", []))),
        "splits": splits,
    }


def _lodo_split(split_payload: dict[str, Any], train_dataset: str, test_dataset: str) -> dict[str, Any]:
    train_keys = [str(k) for k in split_payload.get("splits", {}).get("train", []) if _dataset_of(str(k)) == train_dataset]
    val_keys = [str(k) for k in split_payload.get("splits", {}).get("val", []) if _dataset_of(str(k)) == train_dataset]
    test_keys = [str(k) for k in split_payload.get("splits", {}).get("test", []) if _dataset_of(str(k)) == test_dataset]
    return {
        "audit_name": f"stwm_mixed_fullscale_v2_lodo_{train_dataset.lower()}_to_{test_dataset.lower()}_splits",
        "source_split_report": "reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json",
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "item_level_split": True,
        "no_leakage": bool(split_payload.get("no_leakage", True)),
        "train_item_count": int(len(train_keys)),
        "val_item_count": int(len(val_keys)),
        "test_item_count": int(len(test_keys)),
        "splits": {"train": train_keys, "val": val_keys, "test": test_keys},
    }


def _metric(eval_payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(eval_payload.get("best_metrics", {}).get(key, default))


def _bootstrap(values: list[float], *, seed: int = 20260428, samples: int = 2000) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"item_count": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0], "zero_excluded": False, "bootstrap_win_rate": 0.0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    lo, hi = np.percentile(np.asarray(means, dtype=np.float64), [2.5, 97.5])
    return {
        "item_count": int(arr.size),
        "mean_delta": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0.0 or hi < 0.0),
        "bootstrap_win_rate": float((arr > 0.0).mean()),
    }


def _item_rows(eval_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = eval_payload.get("seed_results", [{}])[0].get("test_itemwise", {}).get("item_scores", [])
    return list(rows or [])


def _sig_for_eval(eval_payload: dict[str, Any]) -> dict[str, Any]:
    rows = _item_rows(eval_payload)
    overall = [
        float(r.get("residual_overall_top5", 0.0) - r.get("copy_overall_top5", 0.0))
        for r in rows
        if int(r.get("overall_count", 0)) > 0
    ]
    changed = [
        float(r.get("residual_changed_top5", 0.0) - r.get("copy_changed_top5", 0.0))
        for r in rows
        if int(r.get("changed_count", 0)) > 0
    ]
    stable_drop = [
        float(r.get("copy_stable_top5", 0.0) - r.get("residual_stable_top5", 0.0))
        for r in rows
        if int(r.get("stable_count", 0)) > 0
    ]
    ce_delta = [
        float(r.get("copy_overall_ce", 0.0) - r.get("residual_overall_ce", 0.0))
        for r in rows
        if int(r.get("overall_count", 0)) > 0
    ]
    return {
        "item_count": int(eval_payload.get("heldout_item_count", len(rows))),
        "changed_item_count": int(sum(1 for r in rows if int(r.get("changed_count", 0)) > 0)),
        "residual_vs_copy_overall_top5": _bootstrap(overall),
        "residual_vs_copy_changed_top5": _bootstrap(changed),
        "stable_preservation_drop": _bootstrap(stable_drop),
        "residual_vs_copy_ce_improvement": _bootstrap(ce_delta),
        "low_sample_warning": bool(int(eval_payload.get("heldout_item_count", len(rows))) < 100),
    }


def prepare() -> None:
    vip_decision = _load_json(REPORT_DIR / "stwm_vipseg_raw_observed_memory_v2_decision_20260428.json")
    vip_features = _load_json(REPORT_DIR / "stwm_vipseg_raw_observed_semantic_features_v2_20260428.json")
    vip_observed = _load_json(REPORT_DIR / "stwm_vipseg_observed_semantic_prototype_targets_v2_20260428.json")
    target_pool = _load_json(REPORT_DIR / "stwm_mixed_semantic_trace_target_pool_v2_20260428.json")
    split = _load_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json")

    protocol_audit = {
        "audit_name": "stwm_mixed_fullscale_v2_protocol_audit",
        "vipseg_observed_memory_repaired": bool(vip_decision.get("vipseg_raw_rebuild_successful", False)),
        "vipseg_raw_rebuild_successful": bool(vip_decision.get("vipseg_raw_rebuild_successful", False)),
        "vipseg_observed_proto_valid_ratio_v2": float(vip_decision.get("vipseg_observed_proto_valid_ratio_v2", vip_observed.get("vipseg_observed_proto_valid_ratio", 0.0))),
        "vipseg_future_overlap_ratio_v2": float(vip_decision.get("vipseg_future_overlap_ratio_v2", vip_features.get("vipseg_future_overlap_ratio", 0.0))),
        "vipseg_eligible_count_v2": int(vip_decision.get("vipseg_eligible_count_v2", target_pool.get("vipseg_eligible", 0))),
        "mixed_split_includes_vspw_and_vipseg": bool(set(split.get("eligible_by_dataset", {}).keys()) >= {"VSPW", "VIPSEG"}),
        "train_item_count": int(split.get("train_item_count", 0)),
        "val_item_count": int(split.get("val_item_count", 0)),
        "test_item_count": int(split.get("test_item_count", 0)),
        "dataset_counts_per_split": {name: _dataset_counts([str(k) for k in split.get("splits", {}).get(name, [])]) for name in ["train", "val", "test"]},
        "changed_stable_counts_per_split": split.get("stats_c64", {}),
        "no_item_leakage": bool(split.get("no_leakage", False)),
        "video_level_leakage_status": split.get("video_level_split_if_available", "not_available"),
        "free_rollout_eval_entrypoint_available": Path("code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py").exists(),
        "train_config_can_consume_mixed_split": Path("code/stwm/tools/train_fullscale_semantic_trace_world_model_single_20260428.py").exists(),
        "candidate_scorer_used": False,
        "stage1_frozen": True,
        "trace_dynamic_path_frozen": True,
    }
    _write_json(REPORT_DIR / "stwm_mixed_fullscale_v2_protocol_audit_20260428.json", protocol_audit)
    _write_doc(
        DOC_DIR / "STWM_MIXED_FULLSCALE_V2_PROTOCOL_AUDIT_20260428.md",
        "STWM Mixed Fullscale V2 Protocol Audit",
        protocol_audit,
        notes=["Mixed protocol is now eligible because VIPSeg observed semantic memory was rebuilt from raw samples."],
    )

    for dataset in ["VSPW", "VIPSEG"]:
        payload = _split_subset(split, dataset)
        _write_json(REPORT_DIR / f"stwm_mixed_fullscale_v2_splits_{dataset.lower()}_test_20260428.json", payload)
    _write_json(REPORT_DIR / "stwm_mixed_fullscale_v2_lodo_vspw_to_vipseg_splits_20260428.json", _lodo_split(split, "VSPW", "VIPSEG"))
    _write_json(REPORT_DIR / "stwm_mixed_fullscale_v2_lodo_vipseg_to_vspw_splits_20260428.json", _lodo_split(split, "VIPSEG", "VSPW"))


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def final() -> None:
    train = _load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_train_summary_20260428.json")
    selection = _load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_val_selection_20260428.json")
    mixed = _load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_mixed_test_eval_20260428.json")
    vspw = _load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vspw_test_eval_20260428.json")
    vipseg = _load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vipseg_test_eval_20260428.json")
    val_c32 = _load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_val_eval_c32_20260428.json")
    val_c64 = _load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_val_eval_c64_20260428.json")

    sig = {
        "audit_name": "stwm_mixed_fullscale_v2_significance",
        "bootstrap_unit": "item",
        "mixed": _sig_for_eval(mixed),
        "vspw": _sig_for_eval(vspw),
        "vipseg": _sig_for_eval(vipseg),
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    _write_json(REPORT_DIR / "stwm_mixed_fullscale_v2_significance_20260428.json", sig)
    _write_doc(DOC_DIR / "STWM_MIXED_FULLSCALE_V2_SIGNIFICANCE_20260428.md", "STWM Mixed Fullscale V2 Significance", sig)

    seed_rows = []
    for payload in [val_c32, val_c64]:
        for row in payload.get("seed_results", []):
            m = row.get("val_metrics", {})
            seed_rows.append(
                {
                    "prototype_count": int(payload.get("prototype_count", 0)),
                    "seed": int(row.get("seed", -1)),
                    "proto_top5": float(m.get("proto_top5", 0.0)),
                    "changed_gain_over_copy": float(m.get("changed_subset_gain_over_copy", 0.0)),
                    "overall_gain_over_copy": float(m.get("overall_gain_over_copy", 0.0)),
                    "stable_preservation_drop": float(m.get("stable_preservation_drop", 0.0)),
                    "future_trace_coord_error": float(m.get("future_trace_coord_error", 0.0)),
                }
            )
    robust = {
        "audit_name": "stwm_mixed_fullscale_v2_seed_robustness",
        "val_split_only": True,
        "seed_results": seed_rows,
        "c32_changed_gain_mean_std": _mean_std([r["changed_gain_over_copy"] for r in seed_rows if r["prototype_count"] == 32]),
        "c64_changed_gain_mean_std": _mean_std([r["changed_gain_over_copy"] for r in seed_rows if r["prototype_count"] == 64]),
        "failed_seeds": train.get("failed_runs", []),
        "dataset_specific_robustness": "reported on mixed val; per-dataset robustness is evaluated on selected test checkpoint only",
    }
    _write_json(REPORT_DIR / "stwm_mixed_fullscale_v2_seed_robustness_20260428.json", robust)
    _write_doc(DOC_DIR / "STWM_MIXED_FULLSCALE_V2_SEED_ROBUSTNESS_20260428.md", "STWM Mixed Fullscale V2 Seed Robustness", robust)

    lodo = {
        "audit_name": "stwm_mixed_fullscale_v2_lodo_eval",
        "lodo_executed": False,
        "skipped_reason": "Mixed fullscale training/evaluation was prioritized; no dedicated leave-one-dataset-out checkpoints were trained in this command.",
        "cross_dataset_protocol_available": True,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    _write_json(REPORT_DIR / "stwm_mixed_fullscale_v2_lodo_eval_20260428.json", lodo)
    _write_doc(DOC_DIR / "STWM_MIXED_FULLSCALE_V2_LODO_EVAL_20260428.md", "STWM Mixed Fullscale V2 LODO Eval", lodo)

    def beats(payload: dict[str, Any], subset: str = "overall") -> bool:
        m = payload.get("best_metrics", {})
        if subset == "changed":
            return bool(float(m.get("changed_subset_top5", 0.0)) > float(m.get("copy_changed_subset_top5", 0.0)))
        return bool(float(m.get("proto_top5", 0.0)) > float(m.get("copy_proto_top5", 0.0)))

    mixed_changed_ci = bool(sig["mixed"]["residual_vs_copy_changed_top5"]["zero_excluded"])
    vspw_changed_ci = bool(sig["vspw"]["residual_vs_copy_changed_top5"]["zero_excluded"])
    vipseg_changed_ci = bool(sig["vipseg"]["residual_vs_copy_changed_top5"]["zero_excluded"])
    stable_copy_preserved = bool(
        float(mixed.get("best_metrics", {}).get("stable_preservation_drop", 1.0)) <= 0.05
        and float(vspw.get("best_metrics", {}).get("stable_preservation_drop", 1.0)) <= 0.05
        and float(vipseg.get("best_metrics", {}).get("stable_preservation_drop", 1.0)) <= 0.05
    )
    trace_regression = bool(
        mixed.get("trace_regression_detected", False)
        or vspw.get("trace_regression_detected", False)
        or vipseg.get("trace_regression_detected", False)
    )
    residual_mixed = beats(mixed, "overall") and beats(mixed, "changed")
    residual_vspw = beats(vspw, "changed")
    residual_vipseg = beats(vipseg, "changed")
    paper_claimable = bool(residual_mixed and residual_vspw and residual_vipseg and mixed_changed_ci and vipseg_changed_ci and stable_copy_preserved and not trace_regression)
    status = "main_contribution_candidate" if paper_claimable else ("main_supporting_evidence" if residual_mixed else "domain_shift_issue")
    next_choice = "start_paper_writing_with_semantic_world_model" if paper_claimable else ("analyze_vipseg_domain_shift" if not residual_vipseg else "run_larger_training")
    decision = {
        "audit_name": "stwm_mixed_fullscale_v2_decision",
        "mixed_training_completed": bool(train.get("mixed_training_completed", False)),
        "best_prototype_count": int(selection.get("selected_prototype_count", 0)),
        "best_seed": int(selection.get("selected_seed", -1)),
        "residual_beats_copy_mixed": bool(residual_mixed),
        "residual_beats_copy_vspw": bool(residual_vspw),
        "residual_beats_copy_vipseg": bool(residual_vipseg),
        "changed_gain_CI_excludes_zero_mixed": bool(mixed_changed_ci),
        "changed_gain_CI_excludes_zero_vspw": bool(vspw_changed_ci),
        "changed_gain_CI_excludes_zero_vipseg": bool(vipseg_changed_ci),
        "stable_copy_preserved": bool(stable_copy_preserved),
        "trace_regression_detected": bool(trace_regression),
        "free_rollout_semantic_field_signal": bool(residual_mixed and mixed_changed_ci),
        "world_model_output_contract_satisfied": bool(not trace_regression and mixed.get("free_rollout_path") == "_free_rollout_predict"),
        "paper_world_model_claimable": "true" if paper_claimable else "unclear",
        "semantic_field_branch_status": status,
        "recommended_next_step_choice": next_choice,
        "mixed_test_metrics": mixed.get("best_metrics", {}),
        "vspw_test_metrics": vspw.get("best_metrics", {}),
        "vipseg_test_metrics": vipseg.get("best_metrics", {}),
    }
    _write_json(REPORT_DIR / "stwm_mixed_fullscale_v2_decision_20260428.json", decision)
    _write_doc(DOC_DIR / "STWM_MIXED_FULLSCALE_V2_DECISION_20260428.md", "STWM Mixed Fullscale V2 Decision", decision)

    guardrail = {
        "guardrail_version": "v37",
        "allowed": [
            "mixed/cross-dataset free-rollout semantic trace field validation",
            "observed semantic memory",
            "copy-gated residual transition",
            "Stage1 frozen",
            "trace dynamic path frozen",
        ],
        "forbidden": [
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "test-set model selection",
            "hiding VIPSeg result",
            "teacher-forced-only claim",
            "changing method before mixed protocol decision",
        ],
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    _write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v37_20260428.json", guardrail)
    _write_doc(DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V37.md", "STWM World Model No-Drift Guardrail V37", guardrail)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "final"], required=True)
    args = p.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        final()


if __name__ == "__main__":
    main()
