#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import json

import numpy as np
import torch

from stwm.tools import run_stage2_state_identifiability_eval_20260415 as prev


ROOT = prev.ROOT


def _load_method_specs(args: Any) -> List[prev.MethodSpec]:
    final_diag = prev.read_json(args.final_utility_diagnosis)
    final_summary = prev.read_json(args.final_utility_summary)
    mechanism_summary = prev.read_json(args.mechanism_summary)
    calibration_run = str(final_diag.get("overall_best_run_name", "stage2_calonly_topk1_seed123_longconfirm_v2_20260414"))

    final_rows = final_summary.get("run_rows", []) if isinstance(final_summary.get("run_rows", []), list) else []
    mech_rows = mechanism_summary.get("run_rows", []) if isinstance(mechanism_summary.get("run_rows", []), list) else []

    def metric_triplet(row: Dict[str, Any]) -> Tuple[float, float, float]:
        metrics = ((row.get("best_checkpoint_metric") or {}).get("metrics") or {}) if isinstance(row, dict) else {}
        return (
            float(metrics.get("free_rollout_endpoint_l2", 1e9)),
            float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
            float(metrics.get("teacher_forced_coord_loss", 1e9)),
        )

    def pick_failure(family: str, fallback: str) -> str:
        rows = [r for r in mech_rows if str(r.get("ablation_name", "")) == family and str(r.get("status", "")).lower() == "completed"]
        if not rows:
            return fallback
        return max(rows, key=metric_triplet).get("run_name", fallback)

    methods: List[prev.MethodSpec] = [
        prev.MethodSpec(
            name="stage1_frozen_baseline",
            run_name="stage1_frozen_baseline",
            method_type="stage1",
            checkpoint_path=str(args.stage1_checkpoint),
        )
    ]
    named_ckpts = [
        ("legacysem_best", "stage2_fullscale_core_legacysem_seed456_wave2_20260409"),
        ("cropenc_baseline_best", "stage2_fullscale_core_cropenc_seed456_20260409"),
        ("calibration_only_mainline_best", calibration_run),
        ("noalign_failure", pick_failure("noalign", "stage2_calonly_noalign_seed123_ablate_fix_20260415")),
        ("densegate_failure", pick_failure("densegate", "stage2_calonly_densegate_seed456_ablate_fix_20260415")),
        ("nodelay_failure", pick_failure("nodelay", "stage2_calonly_nodelay_seed456_ablate_fix_20260415")),
    ]
    for method_name, run_name in named_ckpts:
        ckpt = ROOT / "outputs/checkpoints" / run_name / "best.pt"
        if ckpt.exists():
            methods.append(
                prev.MethodSpec(
                    name=method_name,
                    run_name=run_name,
                    method_type="stage2",
                    checkpoint_path=str(ckpt),
                )
            )
    return methods


def _metric_rows(per_item: List[Dict[str, Any]], calibration_name: str, comparator_name: str, metric_key: str, higher_better: bool, subset_tag: str | None = None) -> List[float]:
    diffs: List[float] = []
    for item in per_item:
        if subset_tag and subset_tag not in list(item.get("subset_tags", [])):
            continue
        methods = item.get("methods", {}) if isinstance(item.get("methods", {}), dict) else {}
        cal = methods.get(calibration_name)
        comp = methods.get(comparator_name)
        if not isinstance(cal, dict) or not isinstance(comp, dict):
            continue
        a = float(cal.get(metric_key, 0.0 if higher_better else 1e9))
        b = float(comp.get(metric_key, 0.0 if higher_better else 1e9))
        diffs.append((a - b) if higher_better else (b - a))
    return diffs


def _bootstrap_summary(diffs: List[float], seed: int = 0, n_boot: int = 4000) -> Dict[str, Any]:
    if not diffs:
        return {
            "count": 0,
            "mean_diff": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "tie_rate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "ci95_width": 0.0,
            "significant_positive": False,
        }
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5]).tolist()
    return {
        "count": int(len(arr)),
        "mean_diff": float(arr.mean()),
        "win_rate": float(np.mean(arr > 0)),
        "loss_rate": float(np.mean(arr < 0)),
        "tie_rate": float(np.mean(arr == 0)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_width": float(high - low),
        "significant_positive": bool(low > 0.0),
    }


def _paired_comparison_bundle(per_item: List[Dict[str, Any]], calibration_name: str, comparator_name: str) -> Dict[str, Any]:
    return {
        "top1_acc": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_top1_acc", True), seed=7),
        "hit_rate": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_hit_rate", True), seed=11),
        "localization_error": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_localization_error", False), seed=13),
        "future_mask_iou_at_top1": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "future_mask_iou_at_top1", True), seed=17),
        "hard_top1_acc": _bootstrap_summary(_metric_rows(per_item, calibration_name, comparator_name, "query_future_top1_acc", True, subset_tag=None), seed=19),
    }


def _count_significant_top1(comparisons: Dict[str, Any], comparators: List[str]) -> Tuple[int, float]:
    count = 0
    widths: List[float] = []
    for comp in comparators:
        top1 = (((comparisons.get(comp) or {}).get("top1_acc")) or {})
        if not isinstance(top1, dict):
            continue
        if bool(top1.get("significant_positive", False)):
            count += 1
        widths.append(float(top1.get("ci95_width", 1e9)))
    return count, float(sum(widths) / max(len(widths), 1))


def parse_args() -> Any:
    parser = ArgumentParser(description="Run enlarged Stage2 state-identifiability / future grounding evaluation v2")
    parser.add_argument("--protocol-json", default=str(ROOT / "reports/stage2_state_identifiability_protocol_v2_20260416.json"))
    parser.add_argument("--output-json", default=str(ROOT / "reports/stage2_state_identifiability_eval_v2_20260416.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_STATE_IDENTIFIABILITY_EVAL_V2_20260416.md"))
    parser.add_argument("--stage1-checkpoint", default=str(ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt"))
    parser.add_argument("--final-utility-summary", default=str(ROOT / "reports/stage2_final_utility_closure_v2_summary_20260414.json"))
    parser.add_argument("--final-utility-diagnosis", default=str(ROOT / "reports/stage2_final_utility_closure_v2_diagnosis_20260414.json"))
    parser.add_argument("--mechanism-summary", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_summary_20260415.json"))
    parser.add_argument("--v1-eval-json", default=str(ROOT / "reports/stage2_state_identifiability_eval_20260415.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lease-path", default=str(ROOT / "reports/stage1_v2_gpu_lease_20260408.json"))
    parser.add_argument("--eval-required-mem-gb", type=float, default=40.0)
    parser.add_argument("--eval-safety-margin-gb", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = prev.read_json(args.protocol_json)
    items = protocol.get("items", []) if isinstance(protocol.get("items", []), list) else []
    device, device_info = prev._select_eval_device(args)
    specs = _load_method_specs(args)

    prepared_items: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]] = []
    per_item: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        batch, target_future_mask, future_masks = prev._build_single_item_batch(item)
        prepared_items.append((item, batch, target_future_mask, future_masks))
        per_item.append(
            {
                "protocol_item_id": str(item.get("protocol_item_id", "")),
                "dataset": str(item.get("dataset", "")),
                "clip_id": str(item.get("clip_id", "")),
                "subset_tags": list(item.get("subset_tags", [])),
                "target_id": str(item.get("target_id", "")),
                "methods": {},
            }
        )

    try:
        for spec in specs:
            method = prev._load_method(spec, device=device)
            for item_row, prepared in zip(per_item, prepared_items):
                item, batch, target_future_mask, future_masks = prepared
                item_row["methods"][method.name] = prev._evaluate_item(
                    method=method,
                    item=item,
                    batch=batch,
                    target_future_mask=target_future_mask,
                    future_masks=future_masks,
                    device=device,
                )
            prev._release_method(method)
    finally:
        lease_id = str(device_info.get("lease_id", "")).strip()
        if lease_id:
            try:
                prev.release_lease(lease_id=lease_id, lease_path=str(args.lease_path))
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    panel_names = [
        "full_identifiability_panel",
        "occlusion_reappearance",
        "crossing_ambiguity",
        "small_object",
        "appearance_change",
        "long_gap_persistence",
    ]
    method_rows: List[Dict[str, Any]] = []
    for spec in specs:
        panel_metrics: Dict[str, Any] = {}
        all_rows: List[Dict[str, Any]] = []
        hard_rows: List[Dict[str, Any]] = []
        for item_row in per_item:
            score = (item_row.get("methods") or {}).get(spec.name)
            if not isinstance(score, dict):
                continue
            all_rows.append(score)
            if item_row.get("subset_tags"):
                hard_rows.append(score)
        panel_metrics["full_identifiability_panel"] = prev._aggregate_item_metrics(all_rows)
        panel_metrics["hard_subsets"] = prev._aggregate_item_metrics(hard_rows)
        for panel in panel_names[1:]:
            subset_rows = [
                (item_row.get("methods") or {}).get(spec.name)
                for item_row in per_item
                if panel in list(item_row.get("subset_tags", []))
            ]
            panel_metrics[panel] = prev._aggregate_item_metrics([r for r in subset_rows if isinstance(r, dict)])
        method_rows.append(
            {
                "name": spec.name,
                "run_name": spec.run_name,
                "method_type": spec.method_type,
                "checkpoint_path": spec.checkpoint_path,
                "panels": panel_metrics,
                "query_future_top1_acc": float(panel_metrics["full_identifiability_panel"]["query_future_top1_acc"]),
                "query_future_hit_rate": float(panel_metrics["full_identifiability_panel"]["query_future_hit_rate"]),
                "query_future_localization_error": float(panel_metrics["full_identifiability_panel"]["query_future_localization_error"]),
                "future_mask_iou_at_top1": float(panel_metrics["full_identifiability_panel"]["future_mask_iou_at_top1"]),
                "hard_subset_top1_acc": float(panel_metrics["hard_subsets"]["query_future_top1_acc"]),
                "ambiguous_case_top1_acc": float(panel_metrics["crossing_ambiguity"]["query_future_top1_acc"]),
                "small_object_query_top1_acc": float(panel_metrics["small_object"]["query_future_top1_acc"]),
                "appearance_change_query_top1_acc": float(panel_metrics["appearance_change"]["query_future_top1_acc"]),
            }
        )

    by_name = {row["name"]: row for row in method_rows}
    calibration = by_name.get("calibration_only_mainline_best", {})
    stage1 = by_name.get("stage1_frozen_baseline", {})
    legacysem = by_name.get("legacysem_best", {})
    cropenc = by_name.get("cropenc_baseline_best", {})

    def _ge(a: Dict[str, Any], b: Dict[str, Any], key: str) -> bool:
        if not a or not b:
            return False
        return float(a.get(key, -1.0)) >= float(b.get(key, -1.0))

    improved_stage1 = _ge(calibration, stage1, "query_future_top1_acc")
    improved_legacysem = _ge(calibration, legacysem, "query_future_top1_acc")
    improved_cropenc = _ge(calibration, cropenc, "query_future_top1_acc")
    improved_all = bool(improved_stage1 and improved_legacysem and improved_cropenc)
    improved_hard = bool(
        calibration and stage1 and legacysem and cropenc and
        float(calibration.get("hard_subset_top1_acc", -1.0)) >= max(
            float(stage1.get("hard_subset_top1_acc", -1.0)),
            float(legacysem.get("hard_subset_top1_acc", -1.0)),
            float(cropenc.get("hard_subset_top1_acc", -1.0)),
        )
    )
    protocol_success = bool(
        items
        and int((protocol.get("panel_counts") or {}).get("full_identifiability_panel", 0)) >= 120
        and int((protocol.get("panel_counts") or {}).get("crossing_ambiguity", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("small_object", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("appearance_change", 0)) > 0
        and int((protocol.get("panel_counts") or {}).get("long_gap_persistence", 0)) > 0
    )

    comparisons = {
        "stage1_frozen_baseline": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "stage1_frozen_baseline"),
        "legacysem_best": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "legacysem_best"),
        "cropenc_baseline_best": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "cropenc_baseline_best"),
        "noalign_failure": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "noalign_failure"),
        "densegate_failure": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "densegate_failure"),
        "nodelay_failure": _paired_comparison_bundle(per_item, "calibration_only_mainline_best", "nodelay_failure"),
    }

    v1_eval = prev.read_json(args.v1_eval_json)
    v1_comparisons = v1_eval.get("paired_bootstrap_comparisons", {}) if isinstance(v1_eval.get("paired_bootstrap_comparisons", {}), dict) else {}
    baseline_names = ["stage1_frozen_baseline", "legacysem_best", "cropenc_baseline_best"]
    v2_sig_count, v2_avg_width = _count_significant_top1(comparisons, baseline_names)
    v1_sig_count, v1_avg_width = _count_significant_top1(v1_comparisons, baseline_names)
    v2_discriminative = bool(
        int(len(per_item)) >= 120
        and int(len(per_item)) > int(v1_eval.get("protocol_item_count", 0))
        and v2_sig_count >= v1_sig_count
        and (v2_avg_width < v1_avg_width or v2_sig_count > v1_sig_count or v1_avg_width <= 0.0)
    )
    discriminative_for_top_tier = bool(
        protocol_success
        and v2_discriminative
        and comparisons["stage1_frozen_baseline"]["top1_acc"]["significant_positive"]
        and comparisons["legacysem_best"]["top1_acc"]["significant_positive"]
        and comparisons["cropenc_baseline_best"]["top1_acc"]["significant_positive"]
    )

    payload = {
        "generated_at_utc": prev.now_iso(),
        "benchmark_scope": "real state-identifiability / future grounding with true instance continuity and future masks",
        "official_benchmark": False,
        "protocol_contribution": True,
        "selected_device": str(device),
        "device_info": device_info,
        "protocol_item_count": int(len(per_item)),
        "panel_counts": dict(protocol.get("panel_counts", {})),
        "methods": method_rows,
        "per_item_results": per_item,
        "paired_bootstrap_comparisons": comparisons,
        "state_identifiability_protocol_v2_success": bool(protocol_success),
        "state_identifiability_protocol_success": bool(protocol_success),
        "future_grounding_usefulness_improved_vs_stage1": bool(improved_stage1),
        "future_grounding_usefulness_improved_vs_legacysem": bool(improved_legacysem),
        "future_grounding_usefulness_improved_vs_cropenc": bool(improved_cropenc),
        "future_grounding_usefulness_improved_vs_baselines": bool(improved_all),
        "future_grounding_usefulness_improved_on_hard_subsets": bool(improved_hard),
        "protocol_v2_statistically_more_discriminative_than_v1": bool(v2_discriminative),
        "protocol_v2_discriminative_enough_for_top_tier": bool(discriminative_for_top_tier),
    }
    prev.write_json(args.output_json, payload)
    lines = [
        "# Stage2 State-Identifiability Eval V2 20260416",
        "",
        "- scope: real future grounding with true instance identity / future mask continuity",
        "- official_benchmark: False",
        "- protocol_contribution: True",
        f"- protocol_item_count: {len(per_item)}",
        f"- selected_device: {device}",
        f"- state_identifiability_protocol_v2_success: {protocol_success}",
        f"- future_grounding_usefulness_improved_vs_baselines: {improved_all}",
        f"- future_grounding_usefulness_improved_on_hard_subsets: {improved_hard}",
        f"- protocol_v2_statistically_more_discriminative_than_v1: {v2_discriminative}",
        f"- protocol_v2_discriminative_enough_for_top_tier: {discriminative_for_top_tier}",
        "",
        "| method | run_name | top1_acc | hit_rate | loc_error | top1_mask_iou | hard_top1 | ambiguity_top1 | small_top1 | appearance_top1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['name']} | {row['run_name']} | "
            f"{row['query_future_top1_acc']:.4f} | {row['query_future_hit_rate']:.4f} | "
            f"{row['query_future_localization_error']:.6f} | {row['future_mask_iou_at_top1']:.4f} | "
            f"{row['hard_subset_top1_acc']:.4f} | {row['ambiguous_case_top1_acc']:.4f} | "
            f"{row['small_object_query_top1_acc']:.4f} | {row['appearance_change_query_top1_acc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Bootstrap Comparisons",
            "",
            "| comparator | top1_mean_diff | top1_ci95 | top1_win_rate | locerr_mean_diff | locerr_ci95 |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for comp_name, block in comparisons.items():
        top1 = block["top1_acc"]
        loc = block["localization_error"]
        lines.append(
            f"| {comp_name} | {top1['mean_diff']:.4f} | [{top1['ci95_low']:.4f}, {top1['ci95_high']:.4f}] | "
            f"{top1['win_rate']:.4f} | {loc['mean_diff']:.6f} | [{loc['ci95_low']:.6f}, {loc['ci95_high']:.6f}] |"
        )
    prev.write_md(args.output_md, lines)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
