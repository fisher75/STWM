#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset
from stwm.tracewm_v2.tools.run_stage1_v2_220m_gap_closure import (
    GapRunSpec,
    _metric_key,
    _safe_float,
    _train_one_run,
    now_iso,
)
from stwm.tracewm_v2.tools.run_stage1_v2_scientific_revalidation import _load_runtime_config


def parse_args() -> Any:
    p = ArgumentParser(description="Stage1-v2 mainline confirmation round")
    p.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    p.add_argument("--recommended-runtime-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    p.add_argument("--stage1-minisplit-path", default="/home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json")
    p.add_argument("--data-root", default="/home/chen034/workspace/data")

    p.add_argument("--gap-closure-comparison-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_gap_closure_comparison_20260408.json")

    p.add_argument("--runs-summary-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_mainline_confirmation_runs_20260408.json")
    p.add_argument("--comparison-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_mainline_confirmation_comparison_20260408.json")
    p.add_argument("--results-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_MAINLINE_CONFIRMATION_RESULTS_20260408.md")

    p.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-samples-per-dataset-train", type=int, default=128)
    p.add_argument("--max-samples-per-dataset-val", type=int, default=64)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=96)
    p.add_argument("--eval-steps", type=int, default=12)

    p.add_argument("--ref-lr", type=float, default=1e-4)
    p.add_argument("--ref-weight-decay", type=float, default=0.0)
    p.add_argument("--ref-batch-size", type=int, default=2)
    p.add_argument("--ref-grad-accum", type=int, default=1)
    p.add_argument("--ref-clip-grad", type=float, default=1.0)

    p.add_argument("--eval-max-tapvid-samples", type=int, default=6)
    p.add_argument("--eval-max-tapvid3d-samples", type=int, default=12)

    p.add_argument("--gru-hidden-dim", type=int, default=384)
    p.add_argument("--gru-num-layers", type=int, default=2)

    p.add_argument("--seed", type=int, default=20260408)
    return p.parse_args()


def _load_gap_closure_best_recipe(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "stage1_v2_gap_220m_opt_lossweights"
    payload = json.loads(p.read_text(encoding="utf-8"))
    name = str(payload.get("best_220m_optimization_run", "stage1_v2_gap_220m_opt_lossweights"))
    if not name:
        name = "stage1_v2_gap_220m_opt_lossweights"
    return name


def _build_220m_bestrecipe_spec(recipe_name: str, args: Any) -> GapRunSpec:
    base_kwargs = {
        "run_name": "stage1_v2_confirm_220m_bestrecipe",
        "mode": "confirm_220m_bestrecipe",
        "backbone_variant": "stage1_v2_backbone_transformer_prototype220m",
        "state_variant": "multitoken",
        "loss_variant": "coord_visibility",
        "notes": f"220m best recipe reused from gap closure: {recipe_name}",
    }

    if recipe_name == "stage1_v2_gap_220m_opt_lr":
        return GapRunSpec(
            **base_kwargs,
            lr=6e-5,
            weight_decay=0.01,
            warmup_steps=8,
            batch_size=int(args.ref_batch_size),
            grad_accum_steps=int(args.ref_grad_accum),
            clip_grad_norm=float(args.ref_clip_grad),
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
        )

    if recipe_name == "stage1_v2_gap_220m_opt_batch":
        return GapRunSpec(
            **base_kwargs,
            lr=float(args.ref_lr),
            weight_decay=float(args.ref_weight_decay),
            warmup_steps=0,
            batch_size=1,
            grad_accum_steps=4,
            clip_grad_norm=0.8,
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
        )

    return GapRunSpec(
        **base_kwargs,
        lr=float(args.ref_lr),
        weight_decay=float(args.ref_weight_decay),
        warmup_steps=0,
        batch_size=int(args.ref_batch_size),
        grad_accum_steps=int(args.ref_grad_accum),
        clip_grad_norm=float(args.ref_clip_grad),
        coord_weight=1.2,
        visibility_weight=0.8,
        residual_weight=0.25,
        velocity_weight=0.25,
        endpoint_weight=0.1,
        enable_visibility=True,
        enable_residual=False,
        enable_velocity=False,
        enable_endpoint=False,
    )


def _extract_metrics(run: Dict[str, Any]) -> Dict[str, float]:
    ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}
    tap = ev.get("tapvid_eval", {}) if isinstance(ev.get("tapvid_eval", {}), dict) else {}
    tap3d = ev.get("tapvid3d_limited_eval", {}) if isinstance(ev.get("tapvid3d_limited_eval", {}), dict) else {}
    return {
        "teacher_forced_coord_loss": _safe_float(ev.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2": _safe_float(ev.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_endpoint_l2": _safe_float(ev.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid_endpoint_l2": _safe_float(tap.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid3d_limited_endpoint_l2": _safe_float(tap3d.get("free_rollout_endpoint_l2"), 1e9),
        "parameter_count": int((run.get("model", {}) if isinstance(run.get("model", {}), dict) else {}).get("estimated_parameter_count", 0)),
    }


def _gap(best_220m: Dict[str, Any], small: Dict[str, Any]) -> Dict[str, float]:
    bm = _extract_metrics(best_220m)
    sm = _extract_metrics(small)
    return {
        "primary_endpoint_l2_gap": float(bm["free_rollout_endpoint_l2"] - sm["free_rollout_endpoint_l2"]),
        "secondary_mean_l2_gap": float(bm["free_rollout_coord_mean_l2"] - sm["free_rollout_coord_mean_l2"]),
        "tertiary_tapvid_gap": float(bm["tapvid_endpoint_l2"] - sm["tapvid_endpoint_l2"]),
        "quaternary_tapvid3d_limited_gap": float(bm["tapvid3d_limited_endpoint_l2"] - sm["tapvid3d_limited_endpoint_l2"]),
    }


def main() -> None:
    args = parse_args()

    runtime = _load_runtime_config(args.recommended_runtime_json)
    print(
        "[stage1-v2-confirm] runtime "
        f"single_gpu_only={runtime.single_gpu_only} selected_gpu_id={runtime.selected_gpu_id} "
        f"workers={runtime.num_workers} pin_memory={runtime.pin_memory} "
        f"persistent_workers={runtime.persistent_workers} prefetch_factor={runtime.prefetch_factor}"
    )

    best_recipe_name = _load_gap_closure_best_recipe(args.gap_closure_comparison_json)
    print(f"[stage1-v2-confirm] selected_gap_recipe={best_recipe_name}")

    train_dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split="train",
        contract_path=str(args.contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset_train),
    )

    val_dataset = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split="val",
        contract_path=str(args.contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset_val),
    )

    specs: List[GapRunSpec] = [
        GapRunSpec(
            run_name="stage1_v2_confirm_debugsmall_mainline",
            mode="confirm_debugsmall_mainline",
            backbone_variant="stage1_v2_backbone_transformer_debugsmall",
            state_variant="multitoken",
            loss_variant="coord_visibility",
            lr=float(args.ref_lr),
            weight_decay=float(args.ref_weight_decay),
            warmup_steps=0,
            batch_size=int(args.ref_batch_size),
            grad_accum_steps=int(args.ref_grad_accum),
            clip_grad_norm=float(args.ref_clip_grad),
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="small mainline confirmation under longer budget",
        ),
        GapRunSpec(
            run_name="stage1_v2_confirm_220m_ref",
            mode="confirm_220m_ref",
            backbone_variant="stage1_v2_backbone_transformer_prototype220m",
            state_variant="multitoken",
            loss_variant="coord_visibility",
            lr=float(args.ref_lr),
            weight_decay=float(args.ref_weight_decay),
            warmup_steps=0,
            batch_size=int(args.ref_batch_size),
            grad_accum_steps=int(args.ref_grad_accum),
            clip_grad_norm=float(args.ref_clip_grad),
            coord_weight=1.0,
            visibility_weight=0.5,
            residual_weight=0.25,
            velocity_weight=0.25,
            endpoint_weight=0.1,
            enable_visibility=True,
            enable_residual=False,
            enable_velocity=False,
            enable_endpoint=False,
            notes="220m reference confirmation under matched budget",
        ),
        _build_220m_bestrecipe_spec(best_recipe_name, args),
    ]

    runs: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        print(f"[stage1-v2-confirm] run={spec.run_name} mode={spec.mode}")
        run = _train_one_run(
            spec=spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            runtime=runtime,
            args=args,
            seed=int(args.seed) + idx,
        )
        runs.append(run)

    runs_summary = {
        "generated_at_utc": now_iso(),
        "objective": "mainline confirmation under longer controlled budget without new search",
        "contract_path": str(args.contract_path),
        "recommended_runtime_path": str(args.recommended_runtime_json),
        "single_gpu_only": bool(runtime.single_gpu_only),
        "training_budget": {
            "optimizer_steps": int(args.train_steps),
            "epochs": int(args.epochs),
            "eval_steps": int(args.eval_steps),
        },
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "tapvid_eval.free_rollout_endpoint_l2",
            "quaternary": "tapvid3d_limited_eval.free_rollout_endpoint_l2",
            "total_loss_usage": "reference_only",
        },
        "runs": runs,
    }

    runs_path = Path(args.runs_summary_json)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps(runs_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    small = next(r for r in runs if str(r.get("run_name", "")) == "stage1_v2_confirm_debugsmall_mainline")
    all_220m = [r for r in runs if "220m" in str(r.get("run_name", ""))]
    best_220m = min(all_220m, key=_metric_key)

    gap = _gap(best_220m, small)

    does_surpass = bool(
        gap["primary_endpoint_l2_gap"] <= 0.0
        and gap["secondary_mean_l2_gap"] <= 0.0
        and gap["tertiary_tapvid_gap"] <= 0.0
        and gap["quaternary_tapvid3d_limited_gap"] <= 0.0
    )

    if does_surpass:
        final_mainline_decision = "promote_220m_as_mainline"
        why_promotion_justified = "220m is not worse on any ranked metric and surpasses/ties small across all four tiers."
        next_step_choice = "promote_220m_as_mainline"
    else:
        why_promotion_justified = ""
        # If only the quaternary metric is slightly behind while top-3 are not worse, keep small but continue 220m.
        top3_good = bool(
            gap["primary_endpoint_l2_gap"] <= 0.0
            and gap["secondary_mean_l2_gap"] <= 0.0
            and gap["tertiary_tapvid_gap"] <= 0.0
        )
        q4_small_gap = bool(0.0 < gap["quaternary_tapvid3d_limited_gap"] <= 0.03)
        if top3_good and q4_small_gap:
            final_mainline_decision = "keep_debugsmall_but_continue_220m_later"
            next_step_choice = "keep_debugsmall_but_continue_220m_later"
        else:
            final_mainline_decision = "keep_debugsmall_as_mainline"
            next_step_choice = "keep_debugsmall_as_mainline"

    comparison = {
        "generated_at_utc": now_iso(),
        "selection_policy": runs_summary["selection_policy"],
        "current_best_small_run": str(small.get("run_name", "")),
        "current_best_220m_run": str(best_220m.get("run_name", "")),
        "current_best_small_metrics": _extract_metrics(small),
        "current_best_220m_metrics": _extract_metrics(best_220m),
        "does_220m_now_surpass_small": bool(does_surpass),
        "gap_on_each_metric": {} if does_surpass else gap,
        "why_promotion_is_justified": why_promotion_justified,
        "final_mainline_decision": str(final_mainline_decision),
        "allowed_final_mainline_decision": [
            "keep_debugsmall_as_mainline",
            "promote_220m_as_mainline",
            "keep_debugsmall_but_continue_220m_later",
        ],
        "next_step_choice": str(next_step_choice),
        "evidence": {
            "runs_summary": str(args.runs_summary_json),
        },
    }

    comparison_path = Path(args.comparison_json)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    results_md = Path(args.results_md)
    results_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stage1-v2 Mainline Confirmation Results",
        "",
        f"- generated_at_utc: {comparison['generated_at_utc']}",
        f"- current_best_small_run: {comparison['current_best_small_run']}",
        f"- current_best_220m_run: {comparison['current_best_220m_run']}",
        f"- does_220m_now_surpass_small: {comparison['does_220m_now_surpass_small']}",
        f"- final_mainline_decision: {comparison['final_mainline_decision']}",
        f"- next_step_choice: {comparison['next_step_choice']}",
        "",
        "## Training Budget",
        f"- optimizer_steps: {int(args.train_steps)}",
        f"- epochs: {int(args.epochs)}",
        f"- eval_steps: {int(args.eval_steps)}",
        "",
        "## Ranked Metrics",
        "| run | primary_endpoint_l2 | secondary_mean_l2 | tertiary_tapvid | quaternary_tapvid3d_limited | teacher_forced_coord_loss | parameter_count | effective_batch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in sorted(runs, key=_metric_key):
        m = _extract_metrics(run)
        budget = run.get("train_budget", {}) if isinstance(run.get("train_budget", {}), dict) else {}
        lines.append(
            f"| {run.get('run_name', '-')} | {m['free_rollout_endpoint_l2']:.6f} | {m['free_rollout_coord_mean_l2']:.6f} | {m['tapvid_endpoint_l2']:.6f} | {m['tapvid3d_limited_endpoint_l2']:.6f} | {m['teacher_forced_coord_loss']:.6f} | {m['parameter_count']} | {int(budget.get('effective_batch', 0))} |"
        )

    if does_surpass:
        lines.extend([
            "",
            "## Promotion Justification",
            f"- {why_promotion_justified}",
        ])
    else:
        lines.extend([
            "",
            "## Gap On Each Metric",
            f"- primary_endpoint_l2_gap: {gap['primary_endpoint_l2_gap']:.6f}",
            f"- secondary_mean_l2_gap: {gap['secondary_mean_l2_gap']:.6f}",
            f"- tertiary_tapvid_gap: {gap['tertiary_tapvid_gap']:.6f}",
            f"- quaternary_tapvid3d_limited_gap: {gap['quaternary_tapvid3d_limited_gap']:.6f}",
        ])

    results_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-confirm] runs_summary={args.runs_summary_json}")
    print(f"[stage1-v2-confirm] comparison={args.comparison_json}")
    print(f"[stage1-v2-confirm] current_best_small_run={comparison['current_best_small_run']}")
    print(f"[stage1-v2-confirm] current_best_220m_run={comparison['current_best_220m_run']}")
    print(f"[stage1-v2-confirm] does_220m_now_surpass_small={comparison['does_220m_now_surpass_small']}")
    print(f"[stage1-v2-confirm] final_mainline_decision={comparison['final_mainline_decision']}")
    print(f"[stage1-v2-confirm] next_step_choice={comparison['next_step_choice']}")


if __name__ == "__main__":
    main()
