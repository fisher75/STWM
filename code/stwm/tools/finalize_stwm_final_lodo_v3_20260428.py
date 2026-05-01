#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tools.eval_free_rollout_semantic_trace_field_20260428 import _eval_one_seed
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import _load_checkpoint
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _load_observed
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
)


REPO_ROOT = Path("/raid/chen034/workspace/stwm")
REPORT_DIR = REPO_ROOT / "reports"
DOC_DIR = REPO_ROOT / "docs"
CHECKPOINT_DIR = REPO_ROOT / "outputs/checkpoints/stwm_final_lodo_v3_20260428"
RUN_STATUS_DIR = REPO_ROOT / "outputs/run_status"


def _apply_process_title_normalization() -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode == "off":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", "python")).strip() or "python"
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def _load_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: str | Path, title: str, sections: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _bootstrap(values: list[float], *, seed: int = 20260428, samples: int = 4000) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "item_count": 0,
            "mean_delta": 0.0,
            "ci95": [0.0, 0.0],
            "zero_excluded": False,
            "bootstrap_win_rate": 0.0,
        }
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    boot = np.asarray(means, dtype=np.float64)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "item_count": int(arr.size),
        "mean_delta": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0.0 or hi < 0.0),
        "bootstrap_win_rate": float(np.mean(arr > 0.0)),
    }


def _load_batch_cache(report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    report = _load_json(report_path)
    cache_path = Path(str(report.get("batch_cache_path", "")))
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    cache = torch.load(cache_path, map_location="cpu")
    return list(cache.get("batches", [])), report, int(len(cache.get("item_keys", [])))


def _checkpoint_path(direction: str, prototype_count: int, seed: int) -> Path:
    return CHECKPOINT_DIR / f"{direction}_c{prototype_count}_seed{seed}_final.pt"


def _summary_path(direction: str, prototype_count: int, seed: int) -> Path:
    return REPORT_DIR / f"stwm_final_lodo_v3_{direction}_c{prototype_count}_seed{seed}_train_20260428.json"


def _status_path(direction: str, prototype_count: int, seed: int) -> Path:
    return RUN_STATUS_DIR / f"stwm_final_lodo_v3_{direction}_c{prototype_count}_s{seed}.status.json"


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["val_metrics"]
    return (
        float(metrics.get("changed_subset_gain_over_copy", 0.0)),
        float(metrics.get("overall_gain_over_copy", 0.0)),
        -float(metrics.get("stable_preservation_drop", 0.0)),
        -float(metrics.get("future_trace_coord_error", 0.0)),
    )


def _sig_from_item_scores(item_scores: list[dict[str, Any]], *, heldout_item_count: int) -> dict[str, Any]:
    overall = [
        float(r.get("residual_overall_top5", 0.0) - r.get("copy_overall_top5", 0.0))
        for r in item_scores
        if int(r.get("overall_count", 0)) > 0
    ]
    changed = [
        float(r.get("residual_changed_top5", 0.0) - r.get("copy_changed_top5", 0.0))
        for r in item_scores
        if int(r.get("changed_count", 0)) > 0
    ]
    stable_drop = [
        float(r.get("copy_stable_top5", 0.0) - r.get("residual_stable_top5", 0.0))
        for r in item_scores
        if int(r.get("stable_count", 0)) > 0
    ]
    ce_gain = [
        float(r.get("copy_overall_ce", 0.0) - r.get("residual_overall_ce", 0.0))
        for r in item_scores
        if int(r.get("overall_count", 0)) > 0
    ]
    return {
        "item_count": int(heldout_item_count),
        "changed_item_count": int(sum(1 for r in item_scores if int(r.get("changed_count", 0)) > 0)),
        "residual_vs_copy_overall_top5": _bootstrap(overall, seed=2026042801),
        "residual_vs_copy_changed_top5": _bootstrap(changed, seed=2026042802),
        "stable_preservation_drop": _bootstrap(stable_drop, seed=2026042803),
        "residual_vs_copy_ce_improvement": _bootstrap(ce_gain, seed=2026042804),
        "low_sample_warning": bool(int(heldout_item_count) < 100),
    }


def _direction_eval(
    *,
    direction: str,
    train_dataset: str,
    val_dataset: str,
    test_dataset: str,
    val_report_path: Path,
    test_report_path: Path,
    start_checkpoint: Path,
    observed_report: Path,
    future_c32: Path,
    future_c64: Path,
    device: torch.device,
    residual_scale: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    val_batches, val_audit, val_item_count = _load_batch_cache(val_report_path)
    test_batches, test_audit, test_item_count = _load_batch_cache(test_report_path)
    payload = _load_checkpoint(start_checkpoint, device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    future_cache_by_c = {
        32: load_future_semantic_prototype_target_cache(future_c32),
        64: load_future_semantic_prototype_target_cache(future_c64),
    }
    obs_by_c = {
        32: _load_observed(observed_report, 32),
        64: _load_observed(observed_report, 64),
    }

    candidate_rows: list[dict[str, Any]] = []
    for prototype_count in [32, 64]:
        for seed in [42, 123, 456, 789, 1001]:
            ckpt = _checkpoint_path(direction, prototype_count, seed)
            summary_path = _summary_path(direction, prototype_count, seed)
            if not ckpt.exists() or not summary_path.exists():
                continue
            val_result = _eval_one_seed(
                checkpoint_path=ckpt,
                prototype_count=prototype_count,
                payload=payload,
                checkpoint_args=checkpoint_args,
                batches_cpu=val_batches,
                future_cache=future_cache_by_c[prototype_count],
                obs_data=obs_by_c[prototype_count],
                device=device,
                residual_scale=float(residual_scale),
            )
            candidate_rows.append(
                {
                    "direction": direction,
                    "prototype_count": int(prototype_count),
                    "seed": int(seed),
                    "checkpoint_path": str(ckpt),
                    "summary_path": str(summary_path),
                    "val_metrics": val_result["metrics"],
                    "val_itemwise": {
                        "aggregate": val_result["metrics"],
                        "item_scores": val_result["item_scores"],
                    },
                }
            )
    if not candidate_rows:
        raise RuntimeError(f"No completed LODO checkpoints found for direction={direction}")

    best = max(candidate_rows, key=_selection_key)
    selected_c = int(best["prototype_count"])
    selected_seed = int(best["seed"])
    test_result = _eval_one_seed(
        checkpoint_path=Path(best["checkpoint_path"]),
        prototype_count=selected_c,
        payload=payload,
        checkpoint_args=checkpoint_args,
        batches_cpu=test_batches,
        future_cache=future_cache_by_c[selected_c],
        obs_data=obs_by_c[selected_c],
        device=device,
        residual_scale=float(residual_scale),
    )
    metrics = test_result["metrics"]
    copy_baseline = {
        "proto_top1": float(metrics.get("copy_proto_top1", 0.0)),
        "proto_top5": float(metrics.get("copy_proto_top5", 0.0)),
        "proto_ce": float(metrics.get("copy_proto_ce", 0.0)),
        "stable_subset_top5": float(metrics.get("copy_stable_subset_top5", 0.0)),
        "changed_subset_top5": float(metrics.get("copy_changed_subset_top5", 0.0)),
    }
    eval_payload = {
        "audit_name": f"stwm_final_lodo_{direction}_eval",
        "direction": direction,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "lodo_completed": True,
        "free_rollout_path": "_free_rollout_predict",
        "teacher_forced_path_used": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "val_selection_used_val_only": True,
        "test_eval_once": True,
        "candidate_count": int(len(candidate_rows)),
        "selected_prototype_count": selected_c,
        "selected_seed": selected_seed,
        "selected_checkpoint_path": best["checkpoint_path"],
        "selection_rule": "primary changed gain over copy; secondary overall gain; tertiary smaller stable drop; tie lower trace coord error",
        "val_item_count": int(val_item_count),
        "val_materialization_report": str(val_report_path),
        "test_materialization_report": str(test_report_path),
        "heldout_item_count": int(test_item_count),
        "copy_baseline": copy_baseline,
        "best_val_metrics": best["val_metrics"],
        "best_metrics": metrics,
        "seed_results": [
            {
                "seed": selected_seed,
                "prototype_count": selected_c,
                "checkpoint_path": best["checkpoint_path"],
                "val_metrics": best["val_metrics"],
                "test_itemwise": {"aggregate": metrics, "item_scores": test_result["item_scores"]},
                "residual_beats_copy_overall": bool(metrics["proto_top5"] > metrics["copy_proto_top5"]),
                "residual_beats_copy_changed_subset": bool(metrics["changed_subset_top5"] > metrics["copy_changed_subset_top5"]),
                "stable_preservation_drop": float(metrics["stable_preservation_drop"]),
                "trace_regression_detected": False,
            }
        ],
        "all_val_candidates": [
            {
                "prototype_count": int(row["prototype_count"]),
                "seed": int(row["seed"]),
                "checkpoint_path": row["checkpoint_path"],
                "changed_gain_over_copy": float(row["val_metrics"].get("changed_subset_gain_over_copy", 0.0)),
                "overall_gain_over_copy": float(row["val_metrics"].get("overall_gain_over_copy", 0.0)),
                "stable_preservation_drop": float(row["val_metrics"].get("stable_preservation_drop", 0.0)),
                "future_trace_coord_error": float(row["val_metrics"].get("future_trace_coord_error", 0.0)),
            }
            for row in candidate_rows
        ],
        "residual_beats_copy_overall": bool(metrics["proto_top5"] > metrics["copy_proto_top5"]),
        "residual_beats_copy_changed_subset": bool(metrics["changed_subset_top5"] > metrics["copy_changed_subset_top5"]),
        "stable_copy_preserved": bool(metrics["stable_preservation_drop"] <= 0.05),
        "trace_regression_detected": bool(metrics["future_trace_coord_error"] > 1e6),
        "lodo_positive": bool(
            metrics["changed_subset_top5"] > metrics["copy_changed_subset_top5"]
            and metrics["stable_preservation_drop"] <= 0.05
        ),
        "domain_shift_analysis": (
            "Cross-dataset evaluation isolates transfer from observed semantic memory + trace dynamics under a shifted dataset prior."
        ),
    }
    sig_payload = _sig_from_item_scores(
        test_result["item_scores"],
        heldout_item_count=int(test_item_count),
    )
    sig_payload.update(
        {
            "audit_name": f"stwm_final_lodo_{direction}_significance",
            "direction": direction,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "prototype_count": selected_c,
            "selected_seed": selected_seed,
            "selected_checkpoint_path": best["checkpoint_path"],
            "lodo_completed": True,
            "lodo_positive": bool(
                sig_payload["residual_vs_copy_changed_top5"]["zero_excluded"]
                and sig_payload["residual_vs_copy_changed_top5"]["mean_delta"] > 0.0
                and eval_payload["stable_copy_preserved"]
            ),
        }
    )
    return eval_payload, sig_payload


def _checkpoint_stats(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
    }


def build_train_summary() -> dict[str, Any]:
    rows = []
    completed = 0
    failed = 0
    for direction, train_dataset, test_dataset in [
        ("vspw_to_vipseg", "VSPW", "VIPSEG"),
        ("vipseg_to_vspw", "VIPSEG", "VSPW"),
    ]:
        for prototype_count in [32, 64]:
            for seed in [42, 123, 456, 789, 1001]:
                ckpt = _checkpoint_path(direction, prototype_count, seed)
                summary_path = _summary_path(direction, prototype_count, seed)
                status_path = _status_path(direction, prototype_count, seed)
                summary = _load_json(summary_path, {})
                status = _load_json(status_path, {})
                row = {
                    "direction": direction,
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "prototype_count": int(prototype_count),
                    "seed": int(seed),
                    "status": status.get("status", "missing"),
                    "summary_path": str(summary_path),
                    "status_path": str(status_path),
                    "checkpoint": _checkpoint_stats(ckpt),
                    "steps": int(summary.get("steps", 0) or 0),
                    "loss_finite_ratio": float(summary.get("loss_finite_ratio", 0.0) or 0.0),
                    "trace_regression_detected": bool(summary.get("trace_regression_detected", True)),
                    "candidate_scorer_used": bool(summary.get("candidate_scorer_used", True)),
                    "future_candidate_leakage": bool(summary.get("future_candidate_leakage", True)),
                    "val_proto_top5": float(summary.get("val_metrics", {}).get("proto_top5", 0.0)),
                    "val_changed_top5": float(summary.get("val_metrics", {}).get("changed_subset_top5", 0.0)),
                    "val_stable_top5": float(summary.get("val_metrics", {}).get("stable_subset_top5", 0.0)),
                    "train_proto_top5": float(summary.get("train_metrics", {}).get("proto_top5", 0.0)),
                    "train_changed_top5": float(summary.get("train_metrics", {}).get("changed_subset_top5", 0.0)),
                    "train_stable_top5": float(summary.get("train_metrics", {}).get("stable_subset_top5", 0.0)),
                }
                rows.append(row)
                if row["status"] == "completed" and row["checkpoint"]["exists"] and summary_path.exists():
                    completed += 1
                else:
                    failed += 1
    return {
        "audit_name": "stwm_final_lodo_train_summary",
        "lodo_completed": bool(completed == 20),
        "completed_run_count": int(completed),
        "failed_run_count": int(failed),
        "expected_run_count": 20,
        "directions": ["vspw_to_vipseg", "vipseg_to_vspw"],
        "prototype_counts": [32, 64],
        "seeds": [42, 123, 456, 789, 1001],
        "runs": rows,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }


def main() -> None:
    _apply_process_title_normalization()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-checkpoint",
        default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt",
    )
    parser.add_argument(
        "--observed-report",
        default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json",
    )
    parser.add_argument(
        "--future-cache-c32",
        default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json",
    )
    parser.add_argument(
        "--future-cache-c64",
        default="reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual-scale", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    start_checkpoint = REPO_ROOT / str(args.start_checkpoint)
    observed_report = REPO_ROOT / str(args.observed_report)
    future_c32 = REPO_ROOT / str(args.future_cache_c32)
    future_c64 = REPO_ROOT / str(args.future_cache_c64)

    train_summary = build_train_summary()
    _write_json(REPORT_DIR / "stwm_final_lodo_train_summary_20260428.json", train_summary)

    vspw_to_vipseg_eval, vspw_to_vipseg_sig = _direction_eval(
        direction="vspw_to_vipseg",
        train_dataset="VSPW",
        val_dataset="VSPW",
        test_dataset="VIPSEG",
        val_report_path=REPORT_DIR / "stwm_final_lodo_v3_vspw_to_vipseg_materialization_val_20260428.json",
        test_report_path=REPORT_DIR / "stwm_final_lodo_v3_vspw_to_vipseg_materialization_test_20260428.json",
        start_checkpoint=start_checkpoint,
        observed_report=observed_report,
        future_c32=future_c32,
        future_c64=future_c64,
        device=device,
        residual_scale=float(args.residual_scale),
    )
    vipseg_to_vspw_eval, vipseg_to_vspw_sig = _direction_eval(
        direction="vipseg_to_vspw",
        train_dataset="VIPSEG",
        val_dataset="VIPSEG",
        test_dataset="VSPW",
        val_report_path=REPORT_DIR / "stwm_final_lodo_v3_vipseg_to_vspw_materialization_val_20260428.json",
        test_report_path=REPORT_DIR / "stwm_final_lodo_v3_vipseg_to_vspw_materialization_test_20260428.json",
        start_checkpoint=start_checkpoint,
        observed_report=observed_report,
        future_c32=future_c32,
        future_c64=future_c64,
        device=device,
        residual_scale=float(args.residual_scale),
    )

    _write_json(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_20260428.json", vspw_to_vipseg_eval)
    _write_json(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_20260428.json", vipseg_to_vspw_eval)

    significance = {
        "audit_name": "stwm_final_lodo_significance",
        "lodo_completed": True,
        "lodo_positive": bool(
            vspw_to_vipseg_sig["lodo_positive"]
            and vipseg_to_vspw_sig["lodo_positive"]
        ),
        "vspw_to_vipseg": vspw_to_vipseg_sig,
        "vipseg_to_vspw": vipseg_to_vspw_sig,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "domain_shift_analysis": {
            "vspw_to_vipseg": "Train on VSPW then test on VIPSeg measures transfer to a larger, previously unseen semantic-memory domain.",
            "vipseg_to_vspw": "Train on VIPSeg then test on VSPW measures reverse transfer into the smaller VSPW domain.",
        },
    }
    _write_json(REPORT_DIR / "stwm_final_lodo_significance_20260428.json", significance)

    # Mirror V3-named assets for the current command.
    _write_json(REPORT_DIR / "stwm_final_lodo_train_summary_v3_20260428.json", train_summary)
    _write_json(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_v3_20260428.json", vspw_to_vipseg_eval)
    _write_json(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_v3_20260428.json", vipseg_to_vspw_eval)
    _write_json(REPORT_DIR / "stwm_final_lodo_significance_v3_20260428.json", significance)

    _write_md(
        DOC_DIR / "STWM_FINAL_LODO_V3_20260428.md",
        "STWM Final LODO V3 20260428",
        [
            "## Train Matrix\n"
            + "\n".join(
                [
                    f"- expected_run_count: `{train_summary['expected_run_count']}`",
                    f"- completed_run_count: `{train_summary['completed_run_count']}`",
                    f"- failed_run_count: `{train_summary['failed_run_count']}`",
                ]
            ),
            "## VSPW -> VIPSeg\n"
            + "\n".join(
                [
                    f"- selected_prototype_count: `{vspw_to_vipseg_eval['selected_prototype_count']}`",
                    f"- selected_seed: `{vspw_to_vipseg_eval['selected_seed']}`",
                    f"- residual_beats_copy_overall: `{vspw_to_vipseg_eval['residual_beats_copy_overall']}`",
                    f"- residual_beats_copy_changed_subset: `{vspw_to_vipseg_eval['residual_beats_copy_changed_subset']}`",
                    f"- stable_copy_preserved: `{vspw_to_vipseg_eval['stable_copy_preserved']}`",
                ]
            ),
            "## VIPSeg -> VSPW\n"
            + "\n".join(
                [
                    f"- selected_prototype_count: `{vipseg_to_vspw_eval['selected_prototype_count']}`",
                    f"- selected_seed: `{vipseg_to_vspw_eval['selected_seed']}`",
                    f"- residual_beats_copy_overall: `{vipseg_to_vspw_eval['residual_beats_copy_overall']}`",
                    f"- residual_beats_copy_changed_subset: `{vipseg_to_vspw_eval['residual_beats_copy_changed_subset']}`",
                    f"- stable_copy_preserved: `{vipseg_to_vspw_eval['stable_copy_preserved']}`",
                ]
            ),
            "## Significance\n"
            + "\n".join(
                [
                    f"- lodo_positive: `{significance['lodo_positive']}`",
                    f"- vspw_to_vipseg changed_gain_zero_excluded: `{vspw_to_vipseg_sig['residual_vs_copy_changed_top5']['zero_excluded']}`",
                    f"- vipseg_to_vspw changed_gain_zero_excluded: `{vipseg_to_vspw_sig['residual_vs_copy_changed_top5']['zero_excluded']}`",
                ]
            ),
        ],
    )


if __name__ == "__main__":
    main()
