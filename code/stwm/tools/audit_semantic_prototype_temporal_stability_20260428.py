#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.semantic_prototype_predictability_common_20260428 import (
    build_or_load_observed_feature_cache,
    frequency_scores,
    l2_normalize,
    load_npz_from_report,
    topk_metrics,
    write_doc,
    write_json,
)


def _conditional_entropy(obs_proto: np.ndarray, labels: np.ndarray, c: int) -> float:
    counts = np.zeros((c, c), dtype=np.float64)
    for o, y in zip(obs_proto.astype(np.int64), labels.astype(np.int64)):
        if 0 <= o < c and 0 <= y < c:
            counts[o, y] += 1.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    entropy = 0.0
    for row in counts:
        rs = row.sum()
        if rs <= 0:
            continue
        p_o = rs / total
        py = row[row > 0] / rs
        entropy += p_o * float(-(py * np.log(py)).sum())
    return float(entropy)


def _slot_stability(target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mode_fracs: list[float] = []
    switches = 0
    transitions = 0
    for i in range(target.shape[0]):
        for k in range(target.shape[2]):
            valid_h = np.where(mask[i, :, k])[0]
            if valid_h.size < 2:
                continue
            labels = target[i, valid_h, k].astype(np.int64)
            counts = np.bincount(labels[labels >= 0])
            if counts.size:
                mode_fracs.append(float(counts.max() / max(labels.size, 1)))
            for a, b in zip(labels[:-1], labels[1:]):
                if a >= 0 and b >= 0:
                    transitions += 1
                    switches += int(a != b)
    return {
        "same_slot_mode_consistency": float(np.mean(mode_fracs)) if mode_fracs else 0.0,
        "prototype_switch_rate_over_horizon": float(switches / max(transitions, 1)),
        "slot_sequence_count": int(len(mode_fracs)),
        "future_transition_count": int(transitions),
    }


def _identity_consistency(target: np.ndarray, mask: np.ndarray, identity: np.ndarray) -> dict[str, float]:
    buckets: dict[int, list[int]] = {}
    for i in range(target.shape[0]):
        for k in range(target.shape[2]):
            ident = int(identity[i, k]) if identity.ndim == 2 else -1
            if ident < 0:
                continue
            labels = target[i, mask[i, :, k], k].astype(np.int64)
            buckets.setdefault(ident, []).extend([int(x) for x in labels if int(x) >= 0])
    fracs = []
    for labels in buckets.values():
        if len(labels) < 2:
            continue
        counts = np.bincount(np.asarray(labels, dtype=np.int64))
        fracs.append(float(counts.max() / max(len(labels), 1)))
    return {
        "identity_level_mode_consistency": float(np.mean(fracs)) if fracs else 0.0,
        "identity_bucket_count": int(len(fracs)),
    }


def _audit_one(
    *,
    target_report: Path,
    observed_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    payload, data, _ = load_npz_from_report(target_report, key="target_cache_path")
    target = np.asarray(data["future_semantic_proto_target"], dtype=np.int64)
    mask = np.asarray(data["target_mask"], dtype=bool) & (target >= 0)
    identity = np.asarray(data["identity_target"], dtype=np.int64)
    prototypes = l2_normalize(np.asarray(data["prototypes"], dtype=np.float32))
    c = int(payload.get("prototype_count") or prototypes.shape[0])
    labels = target[mask].astype(np.int64)
    freq = topk_metrics(frequency_scores(labels, labels.shape[0], c), labels)
    obs_mask = np.asarray(observed_cache["observed_feature_mask"], dtype=bool)
    obs_last = l2_normalize(np.asarray(observed_cache["observed_last_feature"], dtype=np.float32))
    obs_mean = l2_normalize(np.asarray(observed_cache["observed_mean_feature"], dtype=np.float32))
    obs_last_scores = obs_last @ prototypes.T
    obs_mean_scores = obs_mean @ prototypes.T

    eval_last_scores: list[np.ndarray] = []
    eval_mean_scores: list[np.ndarray] = []
    eval_labels: list[int] = []
    eval_obs_proto: list[int] = []
    for i in range(target.shape[0]):
        for h in range(target.shape[1]):
            for k in range(target.shape[2]):
                if not mask[i, h, k] or not obs_mask[i, k]:
                    continue
                eval_labels.append(int(target[i, h, k]))
                eval_last_scores.append(obs_last_scores[i, k])
                eval_mean_scores.append(obs_mean_scores[i, k])
                eval_obs_proto.append(int(obs_last_scores[i, k].argmax()))
    y = np.asarray(eval_labels, dtype=np.int64)
    last_scores = np.stack(eval_last_scores, axis=0) if eval_last_scores else np.zeros((0, c), dtype=np.float32)
    mean_scores = np.stack(eval_mean_scores, axis=0) if eval_mean_scores else np.zeros((0, c), dtype=np.float32)
    obs_proto = np.asarray(eval_obs_proto, dtype=np.int64)
    stability = _slot_stability(target, mask)
    ident = _identity_consistency(target, mask, identity)
    observed_last = topk_metrics(last_scores, y)
    observed_mean = topk_metrics(mean_scores, y)
    cond_entropy = _conditional_entropy(obs_proto, y, c)
    observed_beats_frequency = bool(observed_mean["top5"] > freq["top5"] or observed_last["top5"] > freq["top5"])
    target_noise_warning = bool((not observed_beats_frequency) or stability["prototype_switch_rate_over_horizon"] > 0.5)
    return {
        "prototype_count": c,
        "target_report": str(target_report),
        "valid_record_count": int(labels.shape[0]),
        **stability,
        **ident,
        "frequency_baseline_top1": freq["top1"],
        "frequency_baseline_top5": freq["top5"],
        "observed_last_prototype_top1": observed_last["top1"],
        "observed_last_prototype_top5": observed_last["top5"],
        "observed_mean_feature_nearest_proto_top1": observed_mean["top1"],
        "observed_mean_feature_nearest_proto_top5": observed_mean["top5"],
        "conditional_entropy_future_proto_given_observed_proto": cond_entropy,
        "conditional_entropy_normalized": float(cond_entropy / max(math.log(max(c, 2)), 1e-8)),
        "observed_baseline_beats_frequency": observed_beats_frequency,
        "target_noise_warning": target_noise_warning,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-report", default="reports/stwm_semantic_trace_field_decoder_v2_feature_targets_large_20260428.json")
    p.add_argument("--checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--target-reports", nargs="+", default=[
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json",
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json",
        "reports/stwm_future_semantic_trace_prototype_targets_v2_20260428.json",
    ])
    p.add_argument("--observed-cache", default="outputs/cache/stwm_semantic_target_temporal_stability_v1_20260428/observed_features.npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples-per-dataset", type=int, default=128)
    p.add_argument("--output", default="reports/stwm_semantic_target_temporal_stability_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_TARGET_TEMPORAL_STABILITY_V1_20260428.md")
    args = p.parse_args()
    observed_cache, observed_meta = build_or_load_observed_feature_cache(
        feature_report=Path(args.feature_report),
        checkpoint_path=Path(args.checkpoint),
        output_cache=Path(args.observed_cache),
        device=str(args.device),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )
    results = [_audit_one(target_report=Path(path), observed_cache=observed_cache) for path in args.target_reports]
    c64 = next((r for r in results if int(r["prototype_count"]) == 64), results[0] if results else {})
    target_temporally_stable = bool(
        c64.get("same_slot_mode_consistency", 0.0) >= 0.6
        and c64.get("prototype_switch_rate_over_horizon", 1.0) <= 0.5
    )
    payload = {
        "audit_name": "stwm_semantic_target_temporal_stability_v1",
        "observed_feature_meta": observed_meta,
        "results_by_prototype_count": results,
        "target_temporally_stable": target_temporally_stable,
        "target_noise_warning": bool(any(r.get("target_noise_warning") for r in results)),
    }
    write_json(Path(args.output), payload)
    write_doc(
        Path(args.doc),
        "STWM Semantic Target Temporal Stability V1",
        payload,
        bullets=[
            "Observed-last and observed-mean CLIP crop features are used only for target predictability diagnosis.",
            "No STWM backbone is updated and no future candidate is used as rollout input.",
        ],
    )


if __name__ == "__main__":
    main()
