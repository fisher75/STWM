#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from stwm.tools.eval_stwm_fstf_trace_conditioning_audit_v10_20260502 import (
    _dump,
    _eval_modes,
    _paired,
)
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import _load_checkpoint
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _load_observed
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import load_future_semantic_prototype_target_cache


MODES = [
    "STWM_full_selected",
    "STWM_zero_future_hidden_to_semantic_head",
    "STWM_shuffle_future_hidden_across_items",
    "STWM_random_future_hidden_same_stats",
]


def _apply_process_title() -> None:
    if str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).lower() == "off":
        return
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _run_one_horizon(
    *,
    horizon: int,
    batch_cache_report: Path,
    future_cache_report: Path,
    observed_report: Path,
    start_checkpoint: Path,
    selected_checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    mat = json.loads(batch_cache_report.read_text(encoding="utf-8"))
    cache = torch.load(mat["batch_cache_path"], map_location="cpu")
    payload = _load_checkpoint(start_checkpoint, device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    checkpoint_args = dict(checkpoint_args)
    checkpoint_args["fut_len"] = int(horizon)
    future_cache = load_future_semantic_prototype_target_cache(future_cache_report)
    obs_data = _load_observed(observed_report, 32)
    results = _eval_modes(
        modes=MODES,
        checkpoint_path=selected_checkpoint,
        prototype_count=32,
        payload=payload,
        checkpoint_args=checkpoint_args,
        batches_cpu=cache["batches"],
        future_cache=future_cache,
        obs_data=obs_data,
        device=device,
        residual_scale=0.25,
    )
    full_scores = results["STWM_full_selected"]["item_scores"]
    boot = {mode: _paired(full_scores, res["item_scores"]) for mode, res in results.items()}

    def full_beats(mode: str) -> bool:
        changed = boot[mode]["a_minus_b_changed_top5"]
        overall = boot[mode]["a_minus_b_overall_top5"]
        return bool(
            changed.get("zero_excluded")
            and (changed.get("mean_delta") or 0.0) > 0.0
            and overall.get("zero_excluded")
            and (overall.get("mean_delta") or 0.0) > 0.0
        )

    zero_harmful = full_beats("STWM_zero_future_hidden_to_semantic_head")
    shuffle_harmful = full_beats("STWM_shuffle_future_hidden_across_items")
    random_harmful = full_beats("STWM_random_future_hidden_same_stats")
    return {
        "horizon": int(horizon),
        "batch_cache_report": str(batch_cache_report),
        "future_cache_report": str(future_cache_report),
        "selected_checkpoint": str(selected_checkpoint),
        "results": results,
        "bootstrap_vs_full": boot,
        "zero_future_hidden_harmful": zero_harmful,
        "shuffle_future_hidden_harmful": shuffle_harmful,
        "random_future_hidden_harmful": random_harmful,
        "future_hidden_load_bearing": bool(zero_harmful or shuffle_harmful or random_harmful),
    }


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Trace Conditioning Horizon Audit V13", ""]
    for key in [
        "future_hidden_load_bearing_at_H16",
        "future_hidden_load_bearing_at_H24",
        "long_horizon_trace_condition_claim_allowed",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    for horizon, result in payload.get("horizon_results", {}).items():
        lines.append(f"## {horizon}")
        lines.append(f"- future_hidden_load_bearing: `{result.get('future_hidden_load_bearing')}`")
        lines.append(f"- zero_future_hidden_harmful: `{result.get('zero_future_hidden_harmful')}`")
        lines.append(f"- shuffle_future_hidden_harmful: `{result.get('shuffle_future_hidden_harmful')}`")
        lines.append(f"- random_future_hidden_harmful: `{result.get('random_future_hidden_harmful')}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    parser.add_argument("--selected-checkpoint", default="outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/c32_seed456_final.pt")
    parser.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="reports/stwm_fstf_trace_conditioning_horizon_v13_20260502.json")
    parser.add_argument("--doc", default="docs/STWM_FSTF_TRACE_CONDITIONING_HORIZON_V13_20260502.md")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    specs = {
        "H16": {
            "horizon": 16,
            "batch": Path("reports/stwm_fstf_horizon_h16_batch_test_v12_20260502.json"),
            "future": Path("reports/stwm_fstf_horizon_h16_prototype_targets_c32_v12_20260502.json"),
        },
        "H24": {
            "horizon": 24,
            "batch": Path("reports/stwm_fstf_horizon_h24_batch_test_v12_20260502.json"),
            "future": Path("reports/stwm_fstf_horizon_h24_prototype_targets_c32_v12_20260502.json"),
        },
    }
    horizon_results: dict[str, Any] = {}
    for name, spec in specs.items():
        horizon_results[name] = _run_one_horizon(
            horizon=int(spec["horizon"]),
            batch_cache_report=spec["batch"],
            future_cache_report=spec["future"],
            observed_report=Path(args.observed_report),
            start_checkpoint=Path(args.start_checkpoint),
            selected_checkpoint=Path(args.selected_checkpoint),
            device=device,
        )
    h16 = bool(horizon_results["H16"]["future_hidden_load_bearing"])
    h24 = bool(horizon_results["H24"]["future_hidden_load_bearing"])
    payload = {
        "audit_name": "stwm_fstf_trace_conditioning_horizon_v13",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "free_rollout_path": True,
        "teacher_forced_path_used": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "horizon_results": horizon_results,
        "future_hidden_load_bearing_at_H16": h16,
        "future_hidden_load_bearing_at_H24": h24,
        "long_horizon_trace_condition_claim_allowed": bool(h16 and h24),
        "note": "This audits future_hidden interventions at H16/H24; it does not make future trace coordinate or temporal-order load-bearing claims.",
    }
    _dump(Path(args.output), payload)
    _write_doc(Path(args.doc), payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
