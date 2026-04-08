#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List
import json

from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset
from stwm.tracewm_v2.tools.run_stage1_v2_220m_gap_closure import (
    GapRunSpec,
    _safe_float,
    _train_one_run,
    now_iso,
)
from stwm.tracewm_v2.tools.run_stage1_v2_scientific_revalidation import _load_runtime_config


def parse_args() -> Any:
    p = ArgumentParser(description="Stage1-v2 220M mainline freeze round")
    p.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    p.add_argument("--recommended-runtime-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    p.add_argument("--stage1-minisplit-path", default="/home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json")
    p.add_argument("--data-root", default="/home/chen034/workspace/data")

    p.add_argument(
        "--mainline-confirmation-comparison-json",
        default="/home/chen034/workspace/stwm/reports/stage1_v2_mainline_confirmation_comparison_20260408.json",
    )
    p.add_argument(
        "--mainline-confirmation-runs-json",
        default="/home/chen034/workspace/stwm/reports/stage1_v2_mainline_confirmation_runs_20260408.json",
    )

    p.add_argument(
        "--runs-summary-json",
        default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_mainline_freeze_runs_20260408.json",
    )
    p.add_argument(
        "--comparison-json",
        default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_mainline_freeze_comparison_20260408.json",
    )
    p.add_argument(
        "--results-md",
        default="/home/chen034/workspace/stwm/docs/STAGE1_V2_220M_MAINLINE_FREEZE_RESULTS_20260408.md",
    )

    p.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-samples-per-dataset-train", type=int, default=128)
    p.add_argument("--max-samples-per-dataset-val", type=int, default=64)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=192)
    p.add_argument("--eval-steps", type=int, default=16)

    p.add_argument("--eval-max-tapvid-samples", type=int, default=6)
    p.add_argument("--eval-max-tapvid3d-samples", type=int, default=12)

    p.add_argument("--gru-hidden-dim", type=int, default=384)
    p.add_argument("--gru-num-layers", type=int, default=2)

    p.add_argument("--seed", type=int, default=20260408)
    return p.parse_args()


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"required json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be object: {p}")
    return payload


def _find_run(runs_payload: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    runs = runs_payload.get("runs", [])
    if not isinstance(runs, list):
        raise RuntimeError("mainline confirmation runs json missing list field 'runs'")
    for item in runs:
        if not isinstance(item, dict):
            continue
        if str(item.get("run_name", "")) == str(run_name):
            return item
    raise RuntimeError(f"run not found in mainline confirmation runs: {run_name}")


def _spec_from_run(
    source_run: Dict[str, Any],
    freeze_run_name: str,
    freeze_mode: str,
    note: str,
) -> GapRunSpec:
    train_cfg = source_run.get("train_config", {}) if isinstance(source_run.get("train_config", {}), dict) else {}

    return GapRunSpec(
        run_name=freeze_run_name,
        mode=freeze_mode,
        backbone_variant=str(source_run.get("backbone_variant", "")),
        state_variant=str(source_run.get("state_variant", "")),
        loss_variant=str(source_run.get("loss_variant", "coord_visibility")),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        batch_size=int(train_cfg.get("batch_size", 2)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        clip_grad_norm=float(train_cfg.get("clip_grad_norm", 1.0)),
        coord_weight=float(train_cfg.get("coord_weight", 1.0)),
        visibility_weight=float(train_cfg.get("visibility_weight", 0.5)),
        residual_weight=float(train_cfg.get("residual_weight", 0.25)),
        velocity_weight=float(train_cfg.get("velocity_weight", 0.25)),
        endpoint_weight=float(train_cfg.get("endpoint_weight", 0.1)),
        enable_visibility=bool(train_cfg.get("enable_visibility", True)),
        enable_residual=bool(train_cfg.get("enable_residual", False)),
        enable_velocity=bool(train_cfg.get("enable_velocity", False)),
        enable_endpoint=bool(train_cfg.get("enable_endpoint", False)),
        notes=note,
    )


def _extract_metrics(run: Dict[str, Any]) -> Dict[str, float]:
    ev = run.get("evaluation", {}) if isinstance(run.get("evaluation", {}), dict) else {}
    tap = ev.get("tapvid_eval", {}) if isinstance(ev.get("tapvid_eval", {}), dict) else {}
    tap3d = ev.get("tapvid3d_limited_eval", {}) if isinstance(ev.get("tapvid3d_limited_eval", {}), dict) else {}
    model = run.get("model", {}) if isinstance(run.get("model", {}), dict) else {}
    return {
        "teacher_forced_coord_loss": _safe_float(ev.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2": _safe_float(ev.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_endpoint_l2": _safe_float(ev.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid_endpoint_l2": _safe_float(tap.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid3d_limited_endpoint_l2": _safe_float(tap3d.get("free_rollout_endpoint_l2"), 1e9),
        "parameter_count": int(model.get("estimated_parameter_count", 0)),
    }


def _ranking_key(metrics: Dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics["free_rollout_endpoint_l2"]),
        float(metrics["free_rollout_coord_mean_l2"]),
        float(metrics["tapvid_endpoint_l2"]),
        float(metrics["tapvid3d_limited_endpoint_l2"]),
    )


def _metric_gap(lead: Dict[str, float], ref: Dict[str, float]) -> Dict[str, float]:
    return {
        "primary_endpoint_l2_gap": float(lead["free_rollout_endpoint_l2"] - ref["free_rollout_endpoint_l2"]),
        "secondary_mean_l2_gap": float(lead["free_rollout_coord_mean_l2"] - ref["free_rollout_coord_mean_l2"]),
        "tertiary_tapvid_gap": float(lead["tapvid_endpoint_l2"] - ref["tapvid_endpoint_l2"]),
        "quaternary_tapvid3d_limited_gap": float(
            lead["tapvid3d_limited_endpoint_l2"] - ref["tapvid3d_limited_endpoint_l2"]
        ),
    }


def _decide(
    freeze_220m_metrics: Dict[str, float],
    freeze_small_metrics: Dict[str, float],
) -> Dict[str, Any]:
    gap = _metric_gap(freeze_220m_metrics, freeze_small_metrics)
    still_better = bool(_ranking_key(freeze_220m_metrics) < _ranking_key(freeze_small_metrics))

    if still_better:
        return {
            "is_220m_mainline_still_better_than_debugsmall": True,
            "final_stage1_backbone_decision": "freeze_220m_as_stage1_backbone",
            "next_step_choice": "freeze_stage1_and_prepare_stage2",
            "blocking_reason_if_not_freeze": "",
            "gap_on_each_metric": {},
        }

    # Keep as candidate only for strict primary tie with slight secondary lag.
    primary_tied = bool(abs(gap["primary_endpoint_l2_gap"]) <= 1e-12)
    secondary_slightly_worse = bool(0.0 < gap["secondary_mean_l2_gap"] <= 0.03)
    if primary_tied and secondary_slightly_worse:
        return {
            "is_220m_mainline_still_better_than_debugsmall": False,
            "final_stage1_backbone_decision": "keep_220m_as_candidate_but_not_frozen",
            "next_step_choice": "do_one_last_stage1_followup",
            "blocking_reason_if_not_freeze": (
                "primary is tied but 220m is still behind on secondary free-rollout mean L2 under freeze budget"
            ),
            "gap_on_each_metric": gap,
        }

    return {
        "is_220m_mainline_still_better_than_debugsmall": False,
        "final_stage1_backbone_decision": "revert_to_debugsmall",
        "next_step_choice": "revert_to_small_backbone",
        "blocking_reason_if_not_freeze": (
            "220m is behind debugsmall on one or more ranked core metrics under freeze budget"
        ),
        "gap_on_each_metric": gap,
    }


def main() -> None:
    args = parse_args()

    confirmation_cmp = _load_json(args.mainline_confirmation_comparison_json)
    confirmation_runs = _load_json(args.mainline_confirmation_runs_json)

    best_220m_name = str(confirmation_cmp.get("current_best_220m_run", "")).strip()
    best_small_name = str(confirmation_cmp.get("current_best_small_run", "")).strip()
    if not best_220m_name or not best_small_name:
        raise RuntimeError("mainline confirmation comparison missing best run names")

    best_220m_source = _find_run(confirmation_runs, best_220m_name)
    best_small_source = _find_run(confirmation_runs, best_small_name)

    runtime = _load_runtime_config(args.recommended_runtime_json)
    print(
        "[stage1-v2-freeze] runtime "
        f"single_gpu_only={runtime.single_gpu_only} selected_gpu_id={runtime.selected_gpu_id} "
        f"workers={runtime.num_workers} pin_memory={runtime.pin_memory} "
        f"persistent_workers={runtime.persistent_workers} prefetch_factor={runtime.prefetch_factor}"
    )

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

    specs = [
        _spec_from_run(
            source_run=best_220m_source,
            freeze_run_name="stage1_v2_freeze_220m_mainline",
            freeze_mode="freeze_220m_mainline",
            note=(
                "freeze mainline 220m run reused from confirmation best recipe "
                f"source={best_220m_name}"
            ),
        ),
        _spec_from_run(
            source_run=best_small_source,
            freeze_run_name="stage1_v2_freeze_debugsmall_ref",
            freeze_mode="freeze_debugsmall_ref",
            note=(
                "freeze debugsmall reference reused from confirmation best small recipe "
                f"source={best_small_name}"
            ),
        ),
    ]

    if len(specs) != 2:
        raise RuntimeError(f"freeze round must contain exactly 2 runs, got {len(specs)}")

    runs: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        print(f"[stage1-v2-freeze] run={spec.run_name} mode={spec.mode}")
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
        "objective": "Stage1-v2 220M mainline freeze confirmation under longer controlled budget",
        "contract_path": str(args.contract_path),
        "recommended_runtime_path": str(args.recommended_runtime_json),
        "single_gpu_only": bool(runtime.single_gpu_only),
        "source_confirmation_comparison": str(args.mainline_confirmation_comparison_json),
        "source_confirmation_runs": str(args.mainline_confirmation_runs_json),
        "source_best_220m_run": best_220m_name,
        "source_best_small_run": best_small_name,
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

    run_map = {str(r.get("run_name", "")): r for r in runs}
    freeze_220m = run_map["stage1_v2_freeze_220m_mainline"]
    freeze_small = run_map["stage1_v2_freeze_debugsmall_ref"]

    freeze_220m_metrics = _extract_metrics(freeze_220m)
    freeze_small_metrics = _extract_metrics(freeze_small)
    decision = _decide(freeze_220m_metrics, freeze_small_metrics)

    winner = "stage1_v2_freeze_220m_mainline"
    if _ranking_key(freeze_small_metrics) < _ranking_key(freeze_220m_metrics):
        winner = "stage1_v2_freeze_debugsmall_ref"

    comparison = {
        "generated_at_utc": now_iso(),
        "selection_policy": runs_summary["selection_policy"],
        "freeze_220m_run": "stage1_v2_freeze_220m_mainline",
        "freeze_debugsmall_ref_run": "stage1_v2_freeze_debugsmall_ref",
        "winner_by_ranked_policy": winner,
        "freeze_220m_mainline_metrics": freeze_220m_metrics,
        "freeze_debugsmall_ref_metrics": freeze_small_metrics,
        "is_220m_mainline_still_better_than_debugsmall": bool(
            decision["is_220m_mainline_still_better_than_debugsmall"]
        ),
        "final_stage1_backbone_decision": str(decision["final_stage1_backbone_decision"]),
        "allowed_final_stage1_backbone_decision": [
            "freeze_220m_as_stage1_backbone",
            "keep_220m_as_candidate_but_not_frozen",
            "revert_to_debugsmall",
        ],
        "blocking_reason_if_not_freeze": str(decision["blocking_reason_if_not_freeze"]),
        "next_step_choice": str(decision["next_step_choice"]),
        "allowed_next_step_choice": [
            "freeze_stage1_and_prepare_stage2",
            "do_one_last_stage1_followup",
            "revert_to_small_backbone",
        ],
        "gap_on_each_metric": decision["gap_on_each_metric"],
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
        "# Stage1-v2 220M Mainline Freeze Results",
        "",
        f"- generated_at_utc: {comparison['generated_at_utc']}",
        f"- freeze_220m_run: {comparison['freeze_220m_run']}",
        f"- freeze_debugsmall_ref_run: {comparison['freeze_debugsmall_ref_run']}",
        f"- winner_by_ranked_policy: {comparison['winner_by_ranked_policy']}",
        (
            "- is_220m_mainline_still_better_than_debugsmall: "
            f"{comparison['is_220m_mainline_still_better_than_debugsmall']}"
        ),
        f"- final_stage1_backbone_decision: {comparison['final_stage1_backbone_decision']}",
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

    for run_name in ["stage1_v2_freeze_220m_mainline", "stage1_v2_freeze_debugsmall_ref"]:
        run = run_map[run_name]
        m = _extract_metrics(run)
        budget = run.get("train_budget", {}) if isinstance(run.get("train_budget", {}), dict) else {}
        lines.append(
            f"| {run_name} | {m['free_rollout_endpoint_l2']:.6f} | {m['free_rollout_coord_mean_l2']:.6f} | {m['tapvid_endpoint_l2']:.6f} | {m['tapvid3d_limited_endpoint_l2']:.6f} | {m['teacher_forced_coord_loss']:.6f} | {m['parameter_count']} | {int(budget.get('effective_batch', 0))} |"
        )

    blocking = str(comparison["blocking_reason_if_not_freeze"])
    if blocking:
        lines.extend(["", "## Blocking Reason", f"- {blocking}"])

    results_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-freeze] runs_summary={args.runs_summary_json}")
    print(f"[stage1-v2-freeze] comparison={args.comparison_json}")
    print(
        "[stage1-v2-freeze] "
        f"is_220m_mainline_still_better_than_debugsmall={comparison['is_220m_mainline_still_better_than_debugsmall']}"
    )
    print(f"[stage1-v2-freeze] final_stage1_backbone_decision={comparison['final_stage1_backbone_decision']}")
    print(f"[stage1-v2-freeze] next_step_choice={comparison['next_step_choice']}")


if __name__ == "__main__":
    main()
