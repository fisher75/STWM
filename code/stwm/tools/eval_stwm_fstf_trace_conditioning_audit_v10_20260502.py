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
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import (
    _binary_metrics,
    _load_observed,
    _observed_for_batch,
)
from stwm.tools.run_semantic_memory_world_model_v3_20260428 import _load_trained_models


TRACE_MODES = [
    "STWM_full_selected",
    "STWM_no_trace_condition_old",
    "STWM_zero_future_hidden_to_semantic_head",
    "STWM_shuffle_future_hidden_across_items",
    "STWM_shuffle_future_trace_coord_across_items",
    "STWM_time_reverse_future_hidden",
    "STWM_random_future_hidden_same_stats",
    "STWM_disable_trace_unit_binding_if_hook_available",
    "memory_only_same_capacity_head",
    "trace_only_same_capacity_head",
]


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


def _intervene(
    *,
    mode: str,
    future_hidden: torch.Tensor,
    future_trace_coord: torch.Tensor,
    batch_index_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    notes: dict[str, Any] = {"mode": mode, "intervention_applied": True}
    hidden = future_hidden
    coord = future_trace_coord
    if mode in {"STWM_full_selected", "STWM_no_trace_condition_old"}:
        notes["intervention_applied"] = False
    elif mode in {"STWM_zero_future_hidden_to_semantic_head", "memory_only_same_capacity_head"}:
        hidden = torch.zeros_like(future_hidden)
        notes["intervention"] = "future_hidden zeroed immediately before future_semantic_state_head; observed semantic memory retained"
    elif mode == "STWM_shuffle_future_hidden_across_items":
        if int(future_hidden.shape[0]) > 1:
            g = torch.Generator(device=future_hidden.device)
            g.manual_seed(int(batch_index_seed))
            perm = torch.randperm(int(future_hidden.shape[0]), generator=g, device=future_hidden.device)
            hidden = future_hidden[perm]
            notes["batch_permutation"] = [int(x) for x in perm.detach().cpu().tolist()]
        else:
            notes["intervention_applied"] = False
            notes["skip_reason"] = "batch_size_one"
    elif mode == "STWM_shuffle_future_trace_coord_across_items":
        if int(future_trace_coord.shape[0]) > 1:
            g = torch.Generator(device=future_trace_coord.device)
            g.manual_seed(int(batch_index_seed) + 17)
            perm = torch.randperm(int(future_trace_coord.shape[0]), generator=g, device=future_trace_coord.device)
            coord = future_trace_coord[perm]
            notes["batch_permutation"] = [int(x) for x in perm.detach().cpu().tolist()]
        else:
            notes["intervention_applied"] = False
            notes["skip_reason"] = "batch_size_one"
    elif mode == "STWM_time_reverse_future_hidden":
        hidden = torch.flip(future_hidden, dims=[1])
        notes["intervention"] = "future_hidden horizon dimension reversed"
    elif mode == "STWM_random_future_hidden_same_stats":
        mean = future_hidden.detach().mean()
        std = future_hidden.detach().std().clamp_min(1e-6)
        g = torch.Generator(device=future_hidden.device)
        g.manual_seed(int(batch_index_seed) + 101)
        hidden = torch.randn(future_hidden.shape, generator=g, device=future_hidden.device, dtype=future_hidden.dtype) * std + mean
        notes["intervention"] = "future_hidden replaced by Gaussian noise with same batch mean/std"
        notes["source_mean"] = float(mean.detach().cpu().item())
        notes["source_std"] = float(std.detach().cpu().item())
    elif mode == "trace_only_same_capacity_head":
        notes["intervention"] = "same STWM semantic head on future_hidden with observed semantic memory removed; no copy base"
    elif mode == "STWM_disable_trace_unit_binding_if_hook_available":
        notes["intervention"] = "free-rollout recomputed with structure_mode disabled before semantic head"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return hidden, coord, notes


def _eval_modes(
    *,
    modes: list[str],
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
    head = models["future_semantic_state_head"]
    results: dict[str, Any] = {}
    accum: dict[str, dict[str, Any]] = {}
    for mode in modes:
        accum[mode] = {
            "sums": {k: 0.0 for k in ["res_o_t1", "res_o_t5", "res_o_ce", "copy_o_t1", "copy_o_t5", "copy_o_ce", "res_s_t5", "copy_s_t5", "res_c_t5", "copy_c_t5"]},
            "counts": {"overall": 0, "stable": 0, "changed": 0},
            "item_scores": [],
            "change_scores": [],
            "change_labels": [],
            "vis_scores": [],
            "vis_labels": [],
            "reap_scores": [],
            "reap_labels": [],
            "coord_errors": [],
            "notes": [],
        }

    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(batches_cpu):
            batch = _to_device(batch_cpu, device, non_blocking=False)
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

            def rollout_for(mode: str) -> dict[str, Any]:
                batch_for_rollout = batch
                structure_mode = str(_make_forward_kwargs(models, args, batch)["structure_mode"])
                if mode == "STWM_no_trace_condition_old":
                    batch_for_rollout = dict(batch)
                    batch_for_rollout["obs_state"] = torch.zeros_like(batch["obs_state"])
                if mode == "STWM_disable_trace_unit_binding_if_hook_available":
                    structure_mode = "none"
                kwargs = _make_forward_kwargs(models, args, batch_for_rollout)
                kwargs["future_semantic_state_head"] = None
                kwargs["structure_mode"] = structure_mode
                return _free_rollout_predict(
                    **kwargs,
                    fut_len=int(getattr(args, "fut_len", 8)),
                    observed_semantic_proto_target=None,
                    observed_semantic_proto_distribution=None,
                    observed_semantic_proto_mask=None,
                )

            rollout_cache: dict[str, dict[str, Any]] = {}
            for mode in modes:
                rollout_key = mode if mode in {"STWM_no_trace_condition_old", "STWM_disable_trace_unit_binding_if_hook_available"} else "default"
                if rollout_key not in rollout_cache:
                    rollout_cache[rollout_key] = rollout_for(mode)
                out = rollout_cache[rollout_key]
                hidden, coord, notes = _intervene(
                    mode=mode,
                    future_hidden=out["future_hidden"],
                    future_trace_coord=out["pred_coord"],
                    batch_index_seed=20260502 + batch_idx,
                )
                pass_observed = mode != "trace_only_same_capacity_head"
                state = head(
                    hidden,
                    future_trace_coord=coord,
                    observed_semantic_proto_target=obs_target if pass_observed else None,
                    observed_semantic_proto_distribution=obs_dist if pass_observed else None,
                    observed_semantic_proto_mask=obs_mask if pass_observed else None,
                )
                logits = state.future_semantic_proto_logits
                copy = _copy_logits(obs_dist, int(target.shape[1]))
                rec = accum[mode]
                rec["notes"].append(notes)
                valid_coord = out["valid_mask"].to(torch.bool)
                if bool(valid_coord.any().item()):
                    err = torch.sqrt(((coord - out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
                    rec["coord_errors"].append(float(err[valid_coord].mean().detach().cpu().item()))
                if state.future_semantic_change_logit is not None and bool(change_mask.any().item()):
                    rec["change_scores"].extend(torch.sigmoid(state.future_semantic_change_logit[change_mask]).detach().cpu().flatten().tolist())
                    rec["change_labels"].extend(change_target[change_mask].to(torch.int64).detach().cpu().flatten().tolist())
                if state.future_visibility_logit is not None and bool(future_mask.any().item()):
                    vis_target = _cache_bool_for_batch(future_cache, "visibility", batch, horizon=int(target.shape[1]), slot_count=_batch_slot_count(batch), device=device)
                    rec["vis_scores"].extend(torch.sigmoid(state.future_visibility_logit[future_mask]).detach().cpu().flatten().tolist())
                    rec["vis_labels"].extend(vis_target[future_mask].to(torch.int64).detach().cpu().flatten().tolist())
                if state.future_reappearance_logit is not None and bool(future_mask.any().item()):
                    reap_target = _cache_bool_for_batch(future_cache, "reappearance", batch, horizon=int(target.shape[1]), slot_count=_batch_slot_count(batch), device=device)
                    rec["reap_scores"].extend(torch.sigmoid(state.future_reappearance_logit[future_mask]).detach().cpu().flatten().tolist())
                    rec["reap_labels"].extend(reap_target[future_mask].to(torch.int64).detach().cpu().flatten().tolist())
                for b, meta in enumerate(batch["meta"]):
                    per_item: dict[str, Any] = {"item_key": stage2_item_key(meta)}
                    for name, mask in [
                        ("overall", change_mask[b]),
                        ("stable", change_mask[b] & (~change_target[b])),
                        ("changed", change_mask[b] & change_target[b]),
                    ]:
                        n, acc, top5, ce = _item_metrics(logits[b : b + 1], target[b : b + 1], mask[None, ...])
                        _nc, accc, top5c, cec = _item_metrics(copy[b : b + 1], target[b : b + 1], mask[None, ...])
                        per_item[f"{name}_count"] = int(n)
                        per_item[f"residual_{name}_top1"] = float(acc)
                        per_item[f"residual_{name}_top5"] = float(top5)
                        per_item[f"residual_{name}_ce"] = float(ce)
                        per_item[f"copy_{name}_top1"] = float(accc)
                        per_item[f"copy_{name}_top5"] = float(top5c)
                        per_item[f"copy_{name}_ce"] = float(cec)
                        if n > 0:
                            rec["counts"][name] += n
                            if name == "overall":
                                rec["sums"]["res_o_t1"] += acc * n
                                rec["sums"]["res_o_t5"] += top5 * n
                                rec["sums"]["res_o_ce"] += ce * n
                                rec["sums"]["copy_o_t1"] += accc * n
                                rec["sums"]["copy_o_t5"] += top5c * n
                                rec["sums"]["copy_o_ce"] += cec * n
                            elif name == "stable":
                                rec["sums"]["res_s_t5"] += top5 * n
                                rec["sums"]["copy_s_t5"] += top5c * n
                            elif name == "changed":
                                rec["sums"]["res_c_t5"] += top5 * n
                                rec["sums"]["copy_c_t5"] += top5c * n
                    rec["item_scores"].append(per_item)

    for mode, rec in accum.items():
        counts, sums = rec["counts"], rec["sums"]
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
            "future_trace_coord_error": float(np.mean(rec["coord_errors"])) if rec["coord_errors"] else 0.0,
            "change_detection": _binary_metrics(rec["change_scores"], rec["change_labels"]),
            "visibility": _binary_metrics(rec["vis_scores"], rec["vis_labels"]),
            "reappearance": _binary_metrics(rec["reap_scores"], rec["reap_labels"]),
        }
        results[mode] = {
            "mode": mode,
            "metrics": metrics,
            "item_scores": rec["item_scores"],
            "implementation_notes": rec["notes"][:5],
        }
    return results


def _bootstrap(values: list[float], *, seed: int = 20260502, n_boot: int = 5000) -> dict[str, Any]:
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
    overall, changed, stable_drop = [], [], []
    for key in keys:
        x, y = da[key], db[key]
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
        "stable_drop_note": "Positive means first item set has larger stable drop.",
    }


def _average_item_scores(paths: list[Path]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("item_scores", []):
            groups.setdefault(str(item["item_key"]), []).append(item)
    out = []
    for key, rows in groups.items():
        row: dict[str, Any] = {"item_key": key}
        for k, v in rows[0].items():
            if isinstance(v, (int, float)):
                row[k] = float(statistics.fmean(float(r[k]) for r in rows))
        out.append(row)
    return out


def main() -> int:
    _apply_process_title()
    p = argparse.ArgumentParser()
    p.add_argument("--batch-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json")
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--selected-checkpoint", default="outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/c32_seed456_final.pt")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="reports/stwm_fstf_trace_conditioning_audit_v10_20260502.json")
    p.add_argument("--bootstrap-output", default="reports/stwm_fstf_trace_conditioning_bootstrap_v10_20260502.json")
    p.add_argument("--doc", default="docs/STWM_FSTF_TRACE_CONDITIONING_AUDIT_V10_20260502.md")
    args = p.parse_args()
    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    mat = json.loads(Path(args.batch_cache_report).read_text(encoding="utf-8"))
    cache = torch.load(mat["batch_cache_path"], map_location="cpu")
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_report))
    obs_data = _load_observed(Path(args.observed_report), 32)
    results = _eval_modes(
        modes=TRACE_MODES,
        checkpoint_path=Path(args.selected_checkpoint),
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
    boot = {"audit_name": "stwm_fstf_trace_conditioning_bootstrap_v10", "vs_full": {}, "vs_copy_residual_mlp": {}}
    for mode, res in results.items():
        boot["vs_full"][mode] = _paired(full_scores, res["item_scores"])
    strong_paths = sorted(Path("outputs/checkpoints/fstf_strong_copyaware_baselines_v8_20260501/copy_residual_mlp").glob("*/eval_test.json"))
    strong_scores = _average_item_scores(strong_paths)
    for mode, res in results.items():
        boot["vs_copy_residual_mlp"][mode] = _paired(res["item_scores"], strong_scores)

    def sig_full_beats(mode: str, metric: str = "changed") -> bool:
        key = "a_minus_b_changed_top5" if metric == "changed" else "a_minus_b_overall_top5"
        val = boot["vs_full"][mode][key]
        return bool(val["zero_excluded"] and (val["mean_delta"] or 0.0) > 0)

    old_valid = bool(not sig_full_beats("STWM_no_trace_condition_old", "changed") and abs(results["STWM_full_selected"]["metrics"]["proto_top5"] - results["STWM_no_trace_condition_old"]["metrics"]["proto_top5"]) > 1e-8)
    future_hidden_load = bool(
        sig_full_beats("STWM_zero_future_hidden_to_semantic_head", "changed")
        or sig_full_beats("STWM_shuffle_future_hidden_across_items", "changed")
        or sig_full_beats("STWM_random_future_hidden_same_stats", "changed")
    )
    trace_coord_load = bool(sig_full_beats("STWM_shuffle_future_trace_coord_across_items", "changed"))
    temporal_load = bool(sig_full_beats("STWM_time_reverse_future_hidden", "changed"))
    binding_load = bool(sig_full_beats("STWM_disable_trace_unit_binding_if_hook_available", "changed"))
    full_changed = results["STWM_full_selected"]["metrics"]["changed_subset_top5"]
    memory_changed = results["STWM_zero_future_hidden_to_semantic_head"]["metrics"]["changed_subset_top5"]
    semantic_memory_dominates = bool((full_changed - memory_changed) < 0.01)
    trace_claim_allowed = bool(future_hidden_load or temporal_load or binding_load)
    judgments = {
        "old_no_trace_ablation_valid": old_valid,
        "future_hidden_load_bearing": future_hidden_load,
        "future_trace_coord_load_bearing": trace_coord_load,
        "trace_temporal_order_load_bearing": temporal_load,
        "trace_unit_binding_load_bearing": binding_load,
        "semantic_memory_dominates_prediction": semantic_memory_dominates,
        "trace_condition_claim_allowed": trace_claim_allowed,
    }
    report = {
        "audit_name": "stwm_fstf_trace_conditioning_audit_v10",
        "selected_checkpoint": args.selected_checkpoint,
        "prototype_count": 32,
        "horizon": 8,
        "trace_unit_count": 8,
        "free_rollout_path": "_free_rollout_predict",
        "teacher_forced_path_used": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "results": results,
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
                "# STWM FSTF Trace Conditioning Audit V10",
                "",
                f"- old_no_trace_ablation_valid: `{old_valid}`",
                f"- future_hidden_load_bearing: `{future_hidden_load}`",
                f"- future_trace_coord_load_bearing: `{trace_coord_load}`",
                f"- trace_temporal_order_load_bearing: `{temporal_load}`",
                f"- trace_unit_binding_load_bearing: `{binding_load}`",
                f"- semantic_memory_dominates_prediction: `{semantic_memory_dominates}`",
                f"- trace_condition_claim_allowed: `{trace_claim_allowed}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[trace-v10] report={args.output}")
    print(f"[trace-v10] bootstrap={args.bootstrap_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
