#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _free_rollout_predict, _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.eval_free_rollout_semantic_trace_field_20260428 import _cache_bool_for_batch
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import (
    _batch_slot_count,
    _load_checkpoint,
    _make_forward_kwargs,
    _merge_args,
    _proto_loss_and_metrics,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _binary_metrics, _load_observed, _observed_for_batch
from stwm.tools.run_semantic_memory_world_model_v3_20260428 import _load_trained_models


MODES = [
    "STWM_full_selected",
    "STWM_copy_only",
    "STWM_residual_only_no_copy",
    "STWM_no_gate_or_gate_one",
    "STWM_gate_zero",
    "STWM_no_observed_semantic_memory",
    "STWM_no_trace_condition",
]


class ConstantChangeHead(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*hidden.shape[:-1], 1, device=hidden.device, dtype=hidden.dtype) + self.value


def _apply_process_title() -> None:
    if str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).lower() == "off":
        return
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_logits(obs_dist: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.log(obs_dist.clamp_min(1e-6))[:, None].expand(-1, int(horizon), -1, -1)


def _item_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, float, float, float]:
    valid = mask.to(torch.bool) & (target >= 0)
    if not bool(valid.any().item()):
        return 0, 0.0, 0.0, 0.0
    _loss, metrics = _proto_loss_and_metrics(logits, target, valid)
    return int(metrics["valid_count"]), float(metrics["proto_accuracy"]), float(metrics["proto_top5"]), float(metrics["proto_ce"])


def _configure_mode(models: dict[str, Any], mode: str) -> dict[str, Any]:
    head = models["future_semantic_state_head"]
    notes: dict[str, Any] = {"mode": mode, "implemented": True}
    if mode == "STWM_full_selected":
        return notes
    if mode == "STWM_copy_only":
        head.cfg.semantic_proto_prediction_mode = "copy_only"
        notes["ablation"] = "force final semantic logits to observed semantic memory copy"
    elif mode == "STWM_residual_only_no_copy":
        head.cfg.semantic_proto_prediction_mode = "direct_logits"
        notes["ablation"] = "use direct semantic residual logits while preserving observed-memory hidden conditioning"
    elif mode == "STWM_no_gate_or_gate_one":
        head.semantic_change_head = None
        notes["ablation"] = "copy-gated mode with semantic change gate forced to one"
    elif mode == "STWM_gate_zero":
        head.semantic_change_head = ConstantChangeHead(-80.0).to(next(head.parameters()).device)
        notes["ablation"] = "copy-gated mode with gate sigmoid approximately zero"
    elif mode == "STWM_no_observed_semantic_memory":
        head.cfg.semantic_proto_prediction_mode = "direct_logits"
        notes["ablation"] = "drop observed semantic memory tensors and use direct logits"
    elif mode == "STWM_no_trace_condition":
        notes["ablation"] = "zero observed trace state before free rollout; semantic memory remains available"
    else:
        notes["implemented"] = False
    return notes


def _eval_mode(
    *,
    mode: str,
    checkpoint_path: Path,
    prototype_count: int,
    payload: dict[str, Any],
    checkpoint_args: dict[str, Any],
    batches_cpu: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    residual_scale: float,
) -> dict[str, Any]:
    args, models = _load_trained_models(
        checkpoint_path=checkpoint_path,
        prototype_count=prototype_count,
        payload=payload,
        checkpoint_args=checkpoint_args,
        device=device,
        residual_scale=residual_scale,
    )
    notes = _configure_mode(models, mode)
    item_scores: list[dict[str, Any]] = []
    sums = {k: 0.0 for k in ["res_o_t1", "res_o_t5", "res_o_ce", "copy_o_t1", "copy_o_t5", "copy_o_ce", "res_s_t5", "copy_s_t5", "res_c_t5", "copy_c_t5"]}
    counts = {"overall": 0, "stable": 0, "changed": 0}
    change_scores: list[float] = []
    change_labels: list[int] = []
    vis_scores: list[float] = []
    vis_labels: list[int] = []
    reap_scores: list[float] = []
    reap_labels: list[int] = []
    coord_errors: list[float] = []
    pass_observed = mode != "STWM_no_observed_semantic_memory"
    zero_trace = mode == "STWM_no_trace_condition"
    with torch.no_grad():
        for batch_cpu in batches_cpu:
            batch = _to_device(batch_cpu, device, non_blocking=False)
            if zero_trace and "obs_state" in batch:
                batch = dict(batch)
                batch["obs_state"] = torch.zeros_like(batch["obs_state"])
            target, _dist, future_mask, _ = prototype_tensors_for_batch(
                future_cache,
                batch,
                horizon=int(getattr(args, "fut_len", 8)),
                slot_count=_batch_slot_count(batch),
                device=device,
            )
            obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
            change_target, change_mask, _event_target, _event_mask, _info = semantic_change_tensors(
                future_proto_target=target,
                future_proto_mask=future_mask,
                observed_proto_target=obs_target,
                observed_proto_mask=obs_mask,
            )
            out = _free_rollout_predict(
                **_make_forward_kwargs(models, args, batch),
                fut_len=int(getattr(args, "fut_len", 8)),
                observed_semantic_proto_target=obs_target if pass_observed else None,
                observed_semantic_proto_distribution=obs_dist if pass_observed else None,
                observed_semantic_proto_mask=obs_mask if pass_observed else None,
            )
            state = out["future_semantic_trace_state"]
            logits = state.future_semantic_proto_logits
            copy_logits = _copy_logits(obs_dist, int(target.shape[1]))
            valid_coord = out["valid_mask"].to(torch.bool)
            if bool(valid_coord.any().item()):
                coord = torch.sqrt(((out["pred_coord"] - out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
                coord_errors.append(float(coord[valid_coord].mean().detach().cpu().item()))
            if state.future_semantic_change_logit is not None and bool(change_mask.any().item()):
                change_scores.extend(torch.sigmoid(state.future_semantic_change_logit[change_mask]).detach().cpu().flatten().tolist())
                change_labels.extend(change_target[change_mask].to(torch.int64).detach().cpu().flatten().tolist())
            if state.future_visibility_logit is not None and bool(future_mask.any().item()):
                vis_target = _cache_bool_for_batch(future_cache, "visibility", batch, horizon=int(target.shape[1]), slot_count=_batch_slot_count(batch), device=device)
                vis_scores.extend(torch.sigmoid(state.future_visibility_logit[future_mask]).detach().cpu().flatten().tolist())
                vis_labels.extend(vis_target[future_mask].to(torch.int64).detach().cpu().flatten().tolist())
            if state.future_reappearance_logit is not None and bool(future_mask.any().item()):
                reap_target = _cache_bool_for_batch(future_cache, "reappearance", batch, horizon=int(target.shape[1]), slot_count=_batch_slot_count(batch), device=device)
                reap_scores.extend(torch.sigmoid(state.future_reappearance_logit[future_mask]).detach().cpu().flatten().tolist())
                reap_labels.extend(reap_target[future_mask].to(torch.int64).detach().cpu().flatten().tolist())
            for b, meta in enumerate(batch["meta"]):
                per_item: dict[str, Any] = {"item_key": stage2_item_key(meta)}
                for name, mask in [
                    ("overall", change_mask[b]),
                    ("stable", change_mask[b] & (~change_target[b])),
                    ("changed", change_mask[b] & change_target[b]),
                ]:
                    n, acc, top5, ce = _item_metrics(logits[b : b + 1], target[b : b + 1], mask[None, ...])
                    _nc, accc, top5c, cec = _item_metrics(copy_logits[b : b + 1], target[b : b + 1], mask[None, ...])
                    per_item[f"{name}_count"] = int(n)
                    per_item[f"residual_{name}_top1"] = float(acc)
                    per_item[f"residual_{name}_top5"] = float(top5)
                    per_item[f"residual_{name}_ce"] = float(ce)
                    per_item[f"copy_{name}_top1"] = float(accc)
                    per_item[f"copy_{name}_top5"] = float(top5c)
                    per_item[f"copy_{name}_ce"] = float(cec)
                    if n > 0:
                        counts[name] += n
                        if name == "overall":
                            sums["res_o_t1"] += acc * n
                            sums["res_o_t5"] += top5 * n
                            sums["res_o_ce"] += ce * n
                            sums["copy_o_t1"] += accc * n
                            sums["copy_o_t5"] += top5c * n
                            sums["copy_o_ce"] += cec * n
                        elif name == "stable":
                            sums["res_s_t5"] += top5 * n
                            sums["copy_s_t5"] += top5c * n
                        elif name == "changed":
                            sums["res_c_t5"] += top5 * n
                            sums["copy_c_t5"] += top5c * n
                item_scores.append(per_item)
    metrics = {
        "proto_top1": float(sums["res_o_t1"] / max(counts["overall"], 1)),
        "proto_top5": float(sums["res_o_t5"] / max(counts["overall"], 1)),
        "proto_ce": float(sums["res_o_ce"] / max(counts["overall"], 1)),
        "copy_proto_top1": float(sums["copy_o_t1"] / max(counts["overall"], 1)),
        "copy_proto_top5": float(sums["copy_o_t5"] / max(counts["overall"], 1)),
        "copy_proto_ce": float(sums["copy_o_ce"] / max(counts["overall"], 1)),
        "stable_subset_top5": float(sums["res_s_t5"] / max(counts["stable"], 1)),
        "copy_stable_subset_top5": float(sums["copy_s_t5"] / max(counts["stable"], 1)),
        "changed_subset_top5": float(sums["res_c_t5"] / max(counts["changed"], 1)),
        "copy_changed_subset_top5": float(sums["copy_c_t5"] / max(counts["changed"], 1)),
        "valid_count": int(counts["overall"]),
        "stable_subset_count": int(counts["stable"]),
        "changed_subset_count": int(counts["changed"]),
        "changed_subset_gain_over_copy": float(sums["res_c_t5"] / max(counts["changed"], 1) - sums["copy_c_t5"] / max(counts["changed"], 1)),
        "overall_gain_over_copy": float(sums["res_o_t5"] / max(counts["overall"], 1) - sums["copy_o_t5"] / max(counts["overall"], 1)),
        "stable_preservation_drop": float(sums["copy_s_t5"] / max(counts["stable"], 1) - sums["res_s_t5"] / max(counts["stable"], 1)),
        "future_trace_coord_error": float(np.mean(coord_errors)) if coord_errors else 0.0,
        "change_detection": _binary_metrics(change_scores, change_labels),
        "visibility": _binary_metrics(vis_scores, vis_labels),
        "reappearance": _binary_metrics(reap_scores, reap_labels),
    }
    return {"mode": mode, "metrics": metrics, "item_scores": item_scores, "implementation_notes": notes}


def _bootstrap(values: list[float], seed: int = 20260501, n_boot: int = 5000) -> dict[str, Any]:
    if not values:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return {"item_count": n, "mean_delta": float(statistics.fmean(values)), "ci95": [float(lo), float(hi)], "zero_excluded": bool(lo > 0 or hi < 0)}


def _paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    da = {str(x["item_key"]): x for x in a}
    db = {str(x["item_key"]): x for x in b}
    keys = sorted(set(da) & set(db))
    changed, overall, stable_drop = [], [], []
    for k in keys:
        x, y = da[k], db[k]
        overall.append(float(x["residual_overall_top5"]) - float(y["residual_overall_top5"]))
        if float(x.get("changed_count", 0)) > 0 and float(y.get("changed_count", 0)) > 0:
            changed.append(float(x["residual_changed_top5"]) - float(y["residual_changed_top5"]))
        if float(x.get("stable_count", 0)) > 0 and float(y.get("stable_count", 0)) > 0:
            dx = float(x["copy_stable_top5"]) - float(x["residual_stable_top5"])
            dy = float(y["copy_stable_top5"]) - float(y["residual_stable_top5"])
            stable_drop.append(dx - dy)
    return {
        "common_item_count": len(keys),
        "a_minus_b_overall_top5": _bootstrap(overall),
        "a_minus_b_changed_top5": _bootstrap(changed),
        "a_minus_b_stable_drop": _bootstrap(stable_drop),
        "stable_drop_note": "Positive means first run has larger stable drop.",
    }


def main() -> int:
    _apply_process_title()
    p = argparse.ArgumentParser()
    p.add_argument("--batch-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json")
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--selected-checkpoint", default="outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/c32_seed456_final.pt")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--strong-baseline-suite", default="reports/stwm_fstf_strong_copyaware_baseline_suite_v8_20260501.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="reports/stwm_fstf_mechanism_ablation_v9_20260501.json")
    p.add_argument("--bootstrap-output", default="reports/stwm_fstf_mechanism_ablation_bootstrap_v9_20260501.json")
    p.add_argument("--doc", default="docs/STWM_FSTF_MECHANISM_ABLATION_V9_20260501.md")
    args = p.parse_args()
    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    materialization = json.loads(Path(args.batch_cache_report).read_text(encoding="utf-8"))
    batch_cache = torch.load(materialization["batch_cache_path"], map_location="cpu")
    batches_cpu = batch_cache["batches"]
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_report))
    obs_data = _load_observed(Path(args.observed_report), 32)
    results = {}
    for mode in MODES:
        print(f"[mechanism-v9] evaluating {mode}", flush=True)
        results[mode] = _eval_mode(
            mode=mode,
            checkpoint_path=Path(args.selected_checkpoint),
            prototype_count=32,
            payload=payload,
            checkpoint_args=checkpoint_args,
            batches_cpu=batches_cpu,
            future_cache=future_cache,
            obs_data=obs_data,
            device=device,
            residual_scale=0.25,
        )
    blocked = {
        "STWM_no_semantic_trace_unit_coupling": {
            "implemented": False,
            "exact_blocking_reason": "No safe non-invasive eval hook exists to disable only TUSB semantic trace-unit coupling while preserving the selected checkpoint and frozen trace dynamics. Requires a dedicated forward-mode flag or retraining-free module bypass audit.",
        }
    }
    full_scores = results["STWM_full_selected"]["item_scores"]
    boot = {"audit_name": "stwm_fstf_mechanism_ablation_v9_bootstrap", "vs_full": {}, "vs_copy_residual_mlp": {}}
    for mode, res in results.items():
        boot["vs_full"][mode] = _paired(full_scores, res["item_scores"])
    suite = json.loads(Path(args.strong_baseline_suite).read_text(encoding="utf-8"))
    strongest = suite.get("strongest_copyaware_baseline", "copy_residual_mlp")
    strong_paths = sorted(Path("outputs/checkpoints/fstf_strong_copyaware_baselines_v8_20260501").joinpath(strongest).glob("*/eval_test.json"))
    strong_group: dict[str, list[dict[str, Any]]] = {}
    for ep in strong_paths:
        ed = json.loads(ep.read_text(encoding="utf-8"))
        for item in ed.get("item_scores", []):
            strong_group.setdefault(str(item["item_key"]), []).append(item)
    strong_avg = []
    for key, rows in strong_group.items():
        out = {"item_key": key}
        for k, v in rows[0].items():
            if isinstance(v, (int, float)):
                out[k] = float(statistics.fmean(float(r[k]) for r in rows))
        strong_avg.append(out)
    for mode, res in results.items():
        boot["vs_copy_residual_mlp"][mode] = _paired(res["item_scores"], strong_avg)
    m = {mode: res["metrics"] for mode, res in results.items()}
    judgments = {
        "copy_base_load_bearing": bool(m["STWM_residual_only_no_copy"]["proto_top5"] < m["STWM_full_selected"]["proto_top5"]),
        "residual_load_bearing": bool(m["STWM_copy_only"]["changed_subset_top5"] < m["STWM_full_selected"]["changed_subset_top5"]),
        "gate_load_bearing": bool(m["STWM_no_gate_or_gate_one"]["stable_preservation_drop"] > m["STWM_full_selected"]["stable_preservation_drop"]),
        "trace_condition_load_bearing": bool(m["STWM_no_trace_condition"]["changed_subset_top5"] < m["STWM_full_selected"]["changed_subset_top5"]),
        "observed_semantic_memory_load_bearing": bool(m["STWM_no_observed_semantic_memory"]["proto_top5"] < m["STWM_full_selected"]["proto_top5"]),
        "semantic_trace_unit_coupling_load_bearing": None,
        "full_STWM_beats_all_internal_ablations": bool(
            all(m["STWM_full_selected"]["proto_top5"] >= vals["proto_top5"] for name, vals in m.items() if name != "STWM_full_selected")
        ),
    }
    report = {
        "audit_name": "stwm_fstf_mechanism_ablation_v9",
        "selected_checkpoint": args.selected_checkpoint,
        "prototype_count": 32,
        "horizon": 8,
        "trace_unit_count": 8,
        "free_rollout_path": "_free_rollout_predict",
        "teacher_forced_path_used": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "results": results,
        "blocked_ablations": blocked,
        "judgments": judgments,
        "visibility_reappearance_status": "metric_invalid_or_untrained_for_claims",
    }
    _dump(Path(args.output), report)
    _dump(Path(args.bootstrap_output), boot)
    doc = Path(args.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "\n".join(
            [
                "# STWM FSTF Mechanism Ablation V9",
                "",
                f"- mechanism_ablation_completed: `True`",
                f"- full_STWM_beats_all_internal_ablations: `{judgments['full_STWM_beats_all_internal_ablations']}`",
                f"- copy_base_load_bearing: `{judgments['copy_base_load_bearing']}`",
                f"- residual_load_bearing: `{judgments['residual_load_bearing']}`",
                f"- gate_load_bearing: `{judgments['gate_load_bearing']}`",
                f"- trace_condition_load_bearing: `{judgments['trace_condition_load_bearing']}`",
                f"- observed_semantic_memory_load_bearing: `{judgments['observed_semantic_memory_load_bearing']}`",
                "- semantic_trace_unit_coupling_load_bearing: `blocked_non_invasive_eval_hook_missing`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[mechanism-v9] report={args.output}")
    print(f"[mechanism-v9] bootstrap={args.bootstrap_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
