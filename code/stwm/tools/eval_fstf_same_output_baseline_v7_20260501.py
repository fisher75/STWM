#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import _load_observed, _observed_for_batch
from stwm.tools.train_fstf_same_output_baseline_v7_20260501 import (
    FSTFSameOutputBaseline,
    batch_slot_count,
    baseline_officiality_metadata,
    load_batches,
    proto_loss_and_metrics,
    write_json,
)


def apply_process_title() -> None:
    title = os.environ.get("STWM_PROC_TITLE", "python") or "python"
    mode = os.environ.get("STWM_PROC_TITLE_MODE", "generic")
    if str(mode).lower() == "off":
        return
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def binary_metrics(scores: list[float], labels: list[int]) -> dict[str, Any]:
    if not scores or len(set(labels)) < 2:
        return {"eligible": False, "auroc": 0.0, "ap": 0.0, "status": "metric_invalid_or_untrained"}
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores_np)
    y = labels_np[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float((precision * (y == 1)).sum() / max(int((labels_np == 1).sum()), 1))
    pos = scores_np[labels_np == 1]
    neg = scores_np[labels_np == 0]
    auroc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    return {"eligible": True, "auroc": auroc, "ap": ap, "status": "computed"}


def bootstrap(values: list[float], *, seed: int = 20260501, samples: int = 2000) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"item_count": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0], "zero_excluded": False}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5])
    return {
        "item_count": int(arr.size),
        "mean_delta": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0.0 or hi < 0.0),
        "bootstrap_win_rate": float(np.mean(arr > 0.0)),
    }


def copy_logits(obs_dist: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.log(obs_dist.clamp_min(1e-6))[:, None].expand(-1, int(horizon), -1, -1)


def item_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, float, float, float]:
    loss, metrics = proto_loss_and_metrics(logits, target, mask)
    return (
        int(metrics["valid_count"]),
        float(metrics["proto_accuracy"]),
        float(metrics["proto_top5"]),
        float(metrics["proto_ce"]),
    )


def load_model(checkpoint: Path, device: torch.device) -> tuple[FSTFSameOutputBaseline, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = dict(payload["config"])
    model = FSTFSameOutputBaseline(**cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def evaluate(
    *,
    model: FSTFSameOutputBaseline,
    batches: list[dict[str, Any]],
    future_cache: Any,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sums = {k: 0.0 for k in ["res_o_t1", "res_o_t5", "res_o_ce", "copy_o_t1", "copy_o_t5", "copy_o_ce", "res_s_t5", "copy_s_t5", "res_c_t5", "copy_c_t5"]}
    counts = {"overall": 0, "stable": 0, "changed": 0}
    item_scores: list[dict[str, Any]] = []
    trace_errors: list[float] = []
    change_scores: list[float] = []
    change_labels: list[int] = []
    with torch.no_grad():
        for batch_cpu in batches:
            batch = _to_device(batch_cpu, device, non_blocking=False)
            target, _dist, future_mask, _info = prototype_tensors_for_batch(
                future_cache,
                batch,
                horizon=int(horizon),
                slot_count=batch_slot_count(batch),
                device=device,
            )
            obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
            change_target, change_mask, _event_target, _event_mask, _change_info = semantic_change_tensors(
                future_proto_target=target,
                future_proto_mask=future_mask,
                observed_proto_target=obs_target,
                observed_proto_mask=obs_mask,
            )
            logits = model(batch, obs_dist)
            copy = copy_logits(obs_dist, int(horizon))
            change_prob = (1.0 - torch.softmax(logits, dim=-1).gather(-1, obs_target[:, None, :, None].clamp_min(0).expand(-1, int(horizon), -1, 1)).squeeze(-1)).clamp(0.0, 1.0)
            if bool(change_mask.any().item()):
                change_scores.extend(change_prob[change_mask].detach().cpu().flatten().tolist())
                change_labels.extend(change_target[change_mask].to(torch.int64).detach().cpu().flatten().tolist())
            fut_valid = batch.get("fut_valid")
            if isinstance(fut_valid, torch.Tensor) and bool(fut_valid.any().item()):
                pred_coord = batch["obs_state"][:, -1:, :, :2].expand(-1, int(horizon), -1, -1)
                target_coord = batch["fut_state"][:, : int(horizon), :, :2]
                valid = fut_valid[:, : int(horizon)].bool()
                err = torch.sqrt(((pred_coord - target_coord) ** 2).sum(dim=-1).clamp_min(1e-12))
                trace_errors.append(float(err[valid].mean().detach().cpu().item()))
            for b, meta in enumerate(batch["meta"]):
                per: dict[str, Any] = {"item_key": str(meta.get("item_key", ""))}
                for name, mask in (
                    ("overall", change_mask[b]),
                    ("stable", change_mask[b] & (~change_target[b])),
                    ("changed", change_mask[b] & change_target[b]),
                ):
                    n, acc, top5, ce = item_metrics(logits[b : b + 1], target[b : b + 1], mask[None])
                    nc, accc, top5c, cec = item_metrics(copy[b : b + 1], target[b : b + 1], mask[None])
                    per[f"{name}_count"] = int(n)
                    per[f"residual_{name}_top1"] = float(acc)
                    per[f"residual_{name}_top5"] = float(top5)
                    per[f"residual_{name}_ce"] = float(ce)
                    per[f"copy_{name}_top1"] = float(accc)
                    per[f"copy_{name}_top5"] = float(top5c)
                    per[f"copy_{name}_ce"] = float(cec)
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
                        else:
                            sums["res_c_t5"] += top5 * n
                            sums["copy_c_t5"] += top5c * n
                item_scores.append(per)
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
        "overall_gain_over_copy": float(sums["res_o_t5"] / max(counts["overall"], 1) - sums["copy_o_t5"] / max(counts["overall"], 1)),
        "changed_subset_gain_over_copy": float(sums["res_c_t5"] / max(counts["changed"], 1) - sums["copy_c_t5"] / max(counts["changed"], 1)),
        "stable_preservation_drop": float(sums["copy_s_t5"] / max(counts["stable"], 1) - sums["res_s_t5"] / max(counts["stable"], 1)),
        "future_trace_coord_error": float(np.mean(trace_errors)) if trace_errors else 0.0,
        "visibility": {"eligible": False, "status": "metric_invalid_or_untrained", "ap": 0.0, "auroc": 0.0},
        "reappearance": {"eligible": False, "status": "metric_invalid_or_untrained", "ap": 0.0, "auroc": 0.0},
        "change_detection": binary_metrics(change_scores, change_labels),
    }
    return metrics, item_scores


def aggregate_suite(args: argparse.Namespace) -> None:
    root = Path(args.suite_dir)
    baselines = [
        "trace_only_ar_transformer",
        "semantic_only_memory_transition",
        "trace_semantic_transformer",
        "slotformer_like_trace_unit_dynamics",
        "dino_wm_like_latent_dynamics_proxy",
    ]
    seeds = [42, 123, 456]
    per_seed = []
    missing = []
    failed = []
    for baseline in baselines:
        for seed in seeds:
            ev = root / baseline / str(seed) / "eval_test.json"
            tr = root / baseline / str(seed) / "train_summary.json"
            ck = root / baseline / str(seed) / "checkpoint.pt"
            log = Path(args.log_dir) / f"{baseline}_seed{seed}.log"
            if not ev.exists() or not tr.exists() or not ck.exists() or not log.exists() or log.stat().st_size <= 0:
                missing.append({"baseline": baseline, "seed": seed, "eval": str(ev), "train": str(tr), "checkpoint": str(ck), "log": str(log)})
                continue
            payload = json.loads(ev.read_text(encoding="utf-8"))
            per_seed.append(payload)
    by_base: dict[str, list[dict[str, Any]]] = {}
    for row in per_seed:
        by_base.setdefault(row["baseline"], []).append(row)
    seed_mean_std = {}
    for name, rows in by_base.items():
        vals = np.asarray([r["metrics"]["changed_subset_gain_over_copy"] for r in rows], dtype=np.float64)
        seed_mean_std[name] = {
            "seed_count": len(rows),
            "changed_gain_mean": float(vals.mean()) if vals.size else 0.0,
            "changed_gain_std": float(vals.std(ddof=0)) if vals.size else 0.0,
            "overall_top5_delta_mean": float(np.mean([r["metrics"]["overall_gain_over_copy"] for r in rows])) if rows else 0.0,
            "stable_drop_mean": float(np.mean([r["metrics"]["stable_preservation_drop"] for r in rows])) if rows else 0.0,
        }
    strongest = max(seed_mean_std.items(), key=lambda kv: kv[1]["changed_gain_mean"], default=("none", {"changed_gain_mean": -1e9}))[0]
    stwm = json.loads(Path(args.stwm_eval).read_text(encoding="utf-8"))
    stwm_changed = float(stwm.get("best_metrics", {}).get("changed_subset_gain_over_copy", 0.0))
    strongest_changed = float(seed_mean_std.get(strongest, {}).get("changed_gain_mean", 0.0))
    suite_completed = len(missing) == 0 and len(by_base) == len(baselines)
    same_output_table = [
        {
            "method": "copy_semantic_memory_baseline",
            "baseline_name": "copy_semantic_memory_baseline",
            "baseline_family": "semantic_persistence_lower_bound",
            "baseline_type": "same_output_fstf",
            "evidence_level": "trivial_lower_bound",
            "official_repo_used": False,
            "official_repo_url_or_path": "",
            "official_commit_hash": "",
            "official_checkpoint_used": False,
            "pretrained_checkpoint_path": "",
            "adaptation_changes": "No learned dynamics; copies observed semantic memory as the persistence lower bound.",
            "output_contract_matched": True,
            "uses_future_candidate_measurement": False,
            "allowed_table_placement": "main_fstf_table",
            "should_appear_in_main_fstf_table": True,
            "should_appear_in_external_boundary_table": False,
            "allowed_claim": "Strong semantic persistence lower bound for STWM-FSTF.",
            "forbidden_claim": "Do not treat as learned dynamics or an official external baseline.",
            "role": "copy_prior_reference",
        }
    ]
    for name in baselines:
        meta = baseline_officiality_metadata(name)
        same_output_table.append(
            {
                **meta,
                "method": meta["baseline_name"],
                "baseline_type": "same_output_fstf",
                "should_appear_in_main_fstf_table": meta["allowed_table_placement"] == "main_fstf_table",
                "should_appear_in_external_boundary_table": False,
                "should_appear_in_appendix_proxy_table": meta["allowed_table_placement"] == "appendix_proxy_table",
                "seed_mean_std": seed_mean_std.get(name, {"seed_count": 0}),
            }
        )
    external_boundary_table = [
        {
            "method": "SAM2",
            "baseline_name": "SAM2 official external",
            "baseline_type": "external_association",
            "baseline_family": "external_mask_tracking_boundary",
            "evidence_level": "official_external",
            "official_repo_used": True,
            "official_repo_url_or_path": "see prior external association artifact",
            "official_commit_hash": "see prior external association artifact",
            "official_checkpoint_used": True,
            "pretrained_checkpoint_path": "see prior external association artifact",
            "adaptation_changes": "External downstream association/reacquisition diagnostic; not adapted to output future_semantic_proto_logits.",
            "output_contract_matched": False,
            "uses_future_candidate_measurement": True,
            "allowed_table_placement": "external_boundary_table",
            "should_appear_in_main_fstf_table": False,
            "should_appear_in_external_boundary_table": True,
            "allowed_claim": "Official external association boundary/utility comparison.",
            "forbidden_claim": "Do not present as a same-output STWM-FSTF world-model baseline.",
            "role": "external_mask_tracking_boundary",
        },
        {
            "method": "CoTracker",
            "baseline_name": "CoTracker official external",
            "baseline_type": "external_association",
            "baseline_family": "external_point_tracking_boundary",
            "evidence_level": "official_external",
            "official_repo_used": True,
            "official_repo_url_or_path": "see prior external association artifact",
            "official_commit_hash": "see prior external association artifact",
            "official_checkpoint_used": True,
            "pretrained_checkpoint_path": "see prior external association artifact",
            "adaptation_changes": "External downstream association/reacquisition diagnostic; not adapted to output future_semantic_proto_logits.",
            "output_contract_matched": False,
            "uses_future_candidate_measurement": True,
            "allowed_table_placement": "external_boundary_table",
            "should_appear_in_main_fstf_table": False,
            "should_appear_in_external_boundary_table": True,
            "allowed_claim": "Official external association boundary/utility comparison.",
            "forbidden_claim": "Do not present as a same-output STWM-FSTF world-model baseline.",
            "role": "external_point_tracking_boundary",
        },
        {
            "method": "Cutie",
            "baseline_name": "Cutie official external",
            "baseline_type": "external_association",
            "baseline_family": "external_video_object_segmentation_boundary",
            "evidence_level": "official_external",
            "official_repo_used": True,
            "official_repo_url_or_path": "see prior external association artifact",
            "official_commit_hash": "see prior external association artifact",
            "official_checkpoint_used": True,
            "pretrained_checkpoint_path": "see prior external association artifact",
            "adaptation_changes": "External downstream association/reacquisition diagnostic; not adapted to output future_semantic_proto_logits.",
            "output_contract_matched": False,
            "uses_future_candidate_measurement": True,
            "allowed_table_placement": "external_boundary_table",
            "should_appear_in_main_fstf_table": False,
            "should_appear_in_external_boundary_table": True,
            "allowed_claim": "Official external association boundary/utility comparison.",
            "forbidden_claim": "Do not present as a same-output STWM-FSTF world-model baseline.",
            "role": "external_video_object_segmentation_boundary",
        },
    ]
    report = {
        "audit_name": "stwm_fstf_same_output_baseline_suite_v7",
        "baseline_suite_completed": suite_completed,
        "completed_baseline_count": int(sum(1 for b in baselines if len(by_base.get(b, [])) == len(seeds))),
        "missing_baselines": missing,
        "failed_baselines": failed,
        "new_checkpoint_count": int(sum(1 for b in baselines for s in seeds if (root / b / str(s) / "checkpoint.pt").exists())),
        "new_eval_summary_count": int(len(per_seed)),
        "gpu_jobs_launched": bool(per_seed or args.gpu_jobs_launched == "true"),
        "gpu_job_evidence": json.loads(Path(args.manifest).read_text(encoding="utf-8")) if Path(args.manifest).exists() else {},
        "STWM_beats_same_output_baselines": bool(suite_completed and stwm_changed > strongest_changed),
        "strongest_same_output_baseline": strongest,
        "per_seed_results": per_seed,
        "seed_mean_std": seed_mean_std,
        "main_fstf_same_output_table": same_output_table,
        "external_boundary_utility_table": external_boundary_table,
        "external_boundary_note": "SAM2/CoTracker/Cutie are preserved as downstream association/reacquisition diagnostics only; they are not same-output FSTF world-model baselines and are not mixed into the main FSTF table.",
        "paired_bootstrap_CI": {},
        "reviewer_risk_after_baselines": "low" if suite_completed and stwm_changed > strongest_changed else "high",
        "next_step_choice": "run_scaling_laws" if suite_completed and stwm_changed > strongest_changed else ("fix_training_or_objective" if suite_completed else "block_due_to_baseline_failure"),
        "visibility_reappearance_status": "metric_invalid_or_untrained",
    }
    write_json(Path(args.output), report)
    write_json(Path(args.bootstrap_output), {"audit_name": "stwm_fstf_same_output_baseline_bootstrap_v7", "paired_bootstrap_CI": report["paired_bootstrap_CI"], "status": "pending_pairwise_bootstrap" if suite_completed else "incomplete"})
    doc = [
        "# STWM FSTF Same Output Baseline Suite V7",
        "",
        f"- baseline_suite_completed: `{suite_completed}`",
        f"- completed_baseline_count: `{report['completed_baseline_count']}`",
        f"- new_checkpoint_count: `{report['new_checkpoint_count']}`",
        f"- new_eval_summary_count: `{report['new_eval_summary_count']}`",
        f"- strongest_same_output_baseline: `{strongest}`",
        f"- STWM_beats_same_output_baselines: `{report['STWM_beats_same_output_baselines']}`",
        "- external trackers are external_boundary_only, not same-output FSTF baselines.",
        f"- next_step_choice: `{report['next_step_choice']}`",
    ]
    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text("\n".join(doc) + "\n", encoding="utf-8")


def main() -> None:
    apply_process_title()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--test-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="")
    p.add_argument("--aggregate-suite", action="store_true")
    p.add_argument("--suite-dir", default="outputs/checkpoints/fstf_same_output_baselines_v7_20260501")
    p.add_argument("--log-dir", default="logs/fstf_same_output_baselines_v7_20260501")
    p.add_argument("--manifest", default="reports/stwm_fstf_same_output_baseline_run_manifest_v7_20260501.json")
    p.add_argument("--stwm-eval", default="reports/stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    p.add_argument("--bootstrap-output", default="reports/stwm_fstf_same_output_baseline_bootstrap_v7_20260501.json")
    p.add_argument("--doc", default="docs/STWM_FSTF_SAME_OUTPUT_BASELINE_SUITE_V7_20260501.md")
    p.add_argument("--gpu-jobs-launched", default="true")
    args = p.parse_args()
    if args.aggregate_suite:
        aggregate_suite(args)
        return
    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    model, cfg = load_model(Path(args.checkpoint), device)
    batches, report = load_batches(Path(args.test_cache_report))
    future_cache = load_future_semantic_prototype_target_cache(Path(args.future_cache_report))
    obs_data = _load_observed(Path(args.observed_report), int(cfg["prototype_count"]))
    metrics, item_scores = evaluate(
        model=model,
        batches=batches,
        future_cache=future_cache,
        obs_data=obs_data,
        device=device,
        horizon=int(cfg["horizon"]),
    )
    deltas_overall = [r["residual_overall_top5"] - r["copy_overall_top5"] for r in item_scores if r.get("overall_count", 0) > 0]
    deltas_changed = [r["residual_changed_top5"] - r["copy_changed_top5"] for r in item_scores if r.get("changed_count", 0) > 0]
    stable_drop = [r["copy_stable_top5"] - r["residual_stable_top5"] for r in item_scores if r.get("stable_count", 0) > 0]
    payload = {
        "audit_name": "stwm_fstf_same_output_baseline_v7_eval",
        **baseline_officiality_metadata(str(cfg["baseline"])),
        "baseline": cfg["baseline"],
        "baseline_type": "same_output_fstf",
        "output_contract_matched": True,
        "uses_future_candidate_measurement": False,
        "should_appear_in_main_fstf_table": baseline_officiality_metadata(str(cfg["baseline"]))["allowed_table_placement"] == "main_fstf_table",
        "should_appear_in_external_boundary_table": False,
        "should_appear_in_appendix_proxy_table": baseline_officiality_metadata(str(cfg["baseline"]))["allowed_table_placement"] == "appendix_proxy_table",
        "prototype_count": int(cfg["prototype_count"]),
        "seed": int(torch.load(Path(args.checkpoint), map_location="cpu").get("seed", -1)),
        "checkpoint_path": str(args.checkpoint),
        "test_cache_report": str(args.test_cache_report),
        "heldout_item_count": int(report.get("final_eval_item_count", len(batches))),
        "free_rollout_path": "baseline_forward_observed_inputs_only",
        "teacher_forced_path_used": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "metrics": metrics,
        "paired_bootstrap_CI": {
            "overall_top5_delta": bootstrap(deltas_overall, seed=2026050101),
            "changed_top5_delta": bootstrap(deltas_changed, seed=2026050102),
            "stable_drop": bootstrap(stable_drop, seed=2026050103),
        },
        "visibility_reappearance_status": "metric_invalid_or_untrained",
        "item_scores": item_scores,
    }
    write_json(Path(args.output), payload)
    print(f"[fstf-baseline-eval] done baseline={cfg['baseline']} seed={payload['seed']} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
