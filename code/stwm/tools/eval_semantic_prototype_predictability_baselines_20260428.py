#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from stwm.tools.semantic_prototype_predictability_common_20260428 import (
    build_or_load_observed_feature_cache,
    frequency_scores,
    l2_normalize,
    load_npz_from_report,
    topk_metrics,
    write_doc,
    write_json,
)


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.LayerNorm(int(hidden)),
            nn.Linear(int(hidden), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _records(
    *,
    target_data: dict[str, np.ndarray],
    feature_data: dict[str, np.ndarray],
    observed_cache: dict[str, np.ndarray],
    prototypes: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    target = np.asarray(target_data["future_semantic_proto_target"], dtype=np.int64)
    mask = np.asarray(target_data["target_mask"], dtype=bool) & (target >= 0)
    splits = [str(x) for x in target_data["splits"].tolist()]
    obs_last = l2_normalize(np.asarray(observed_cache["observed_last_feature"], dtype=np.float32))
    obs_mean = l2_normalize(np.asarray(observed_cache["observed_mean_feature"], dtype=np.float32))
    obs_mask = np.asarray(observed_cache["observed_feature_mask"], dtype=bool)
    trace_summary = np.asarray(observed_cache["trace_summary"], dtype=np.float32)
    future_feat = l2_normalize(np.asarray(feature_data["future_semantic_feature_target"], dtype=np.float32))
    prototypes = l2_normalize(prototypes.astype(np.float32))
    out: dict[str, dict[str, list[Any]]] = {
        "train": {"trace": [], "semantic": [], "combined": [], "label": [], "last_scores": [], "mean_scores": [], "oracle_scores": []},
        "val": {"trace": [], "semantic": [], "combined": [], "label": [], "last_scores": [], "mean_scores": [], "oracle_scores": []},
    }
    n, h_len, k_len = target.shape
    for i in range(n):
        split = "val" if str(splits[i]).lower() == "val" else "train"
        for h in range(h_len):
            for k in range(k_len):
                if not mask[i, h, k] or not obs_mask[i, k]:
                    continue
                h_norm = float(h / max(h_len - 1, 1))
                k_norm = float(k / max(k_len - 1, 1))
                trace = np.concatenate([trace_summary[i, k], np.asarray([h_norm, k_norm], dtype=np.float32)], axis=0)
                semantic = np.concatenate([obs_mean[i, k], np.asarray([h_norm, k_norm], dtype=np.float32)], axis=0)
                combined = np.concatenate([trace, semantic], axis=0)
                out[split]["trace"].append(trace)
                out[split]["semantic"].append(semantic)
                out[split]["combined"].append(combined)
                out[split]["label"].append(int(target[i, h, k]))
                out[split]["last_scores"].append(obs_last[i, k] @ prototypes.T)
                out[split]["mean_scores"].append(obs_mean[i, k] @ prototypes.T)
                out[split]["oracle_scores"].append(future_feat[i, h, k] @ prototypes.T)
    final: dict[str, dict[str, np.ndarray]] = {}
    for split, values in out.items():
        final[split] = {}
        for key, arrs in values.items():
            if key == "label":
                final[split][key] = np.asarray(arrs, dtype=np.int64)
            else:
                final[split][key] = np.stack(arrs, axis=0).astype(np.float32) if arrs else np.zeros((0, 1), dtype=np.float32)
    return final


def _train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    out_dim: int,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(int(seed))
    model = TinyMLP(x_train.shape[1], int(out_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xb = torch.from_numpy(x_train).float().to(device)
    yb = torch.from_numpy(y_train).long().to(device)
    xv = torch.from_numpy(x_val).float().to(device)
    for _ in range(50):
        perm = torch.randperm(xb.shape[0], device=device)
        for start in range(0, xb.shape[0], 512):
            idx = perm[start : start + 512]
            logits = model(xb[idx])
            loss = nn.functional.cross_entropy(logits, yb[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        scores = model(xv).detach().cpu().numpy()
    return topk_metrics(scores, y_val)


def _eval_one(
    *,
    target_report: Path,
    feature_report: Path,
    observed_cache: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    payload, target_data, _ = load_npz_from_report(target_report, key="target_cache_path")
    _, feature_data, _ = load_npz_from_report(feature_report, key="cache_path")
    c = int(payload.get("prototype_count"))
    prototypes = np.asarray(target_data["prototypes"], dtype=np.float32)
    rec = _records(target_data=target_data, feature_data=feature_data, observed_cache=observed_cache, prototypes=prototypes)
    y_train = rec["train"]["label"]
    y_val = rec["val"]["label"]
    freq = topk_metrics(frequency_scores(y_train, y_val.shape[0], c), y_val)
    observed_last = topk_metrics(rec["val"]["last_scores"], y_val)
    observed_mean = topk_metrics(rec["val"]["mean_scores"], y_val)
    oracle = topk_metrics(rec["val"]["oracle_scores"], y_val)
    trace = _train_probe(rec["train"]["trace"], y_train, rec["val"]["trace"], y_val, c, device=device, seed=20260428)
    semantic = _train_probe(rec["train"]["semantic"], y_train, rec["val"]["semantic"], y_val, c, device=device, seed=20260429)
    combined = _train_probe(rec["train"]["combined"], y_train, rec["val"]["combined"], y_val, c, device=device, seed=20260430)
    return {
        "prototype_count": c,
        "train_record_count": int(y_train.shape[0]),
        "val_record_count": int(y_val.shape[0]),
        "frequency": freq,
        "observed_last_prototype": observed_last,
        "observed_mean_feature_nearest_prototype": observed_mean,
        "trace_only_mlp": trace,
        "semantic_only_mlp": semantic,
        "semantic_trace_mlp": combined,
        "oracle_future_feature_nearest_prototype": oracle,
        "baseline_gain_over_frequency_top5": {
            "observed_last": float(observed_last["top5"] - freq["top5"]),
            "observed_mean": float(observed_mean["top5"] - freq["top5"]),
            "trace_only_mlp": float(trace["top5"] - freq["top5"]),
            "semantic_only_mlp": float(semantic["top5"] - freq["top5"]),
            "semantic_trace_mlp": float(combined["top5"] - freq["top5"]),
        },
        "semantic_only_vs_trace_only_top5_gain": float(semantic["top5"] - trace["top5"]),
        "semantic_trace_vs_semantic_only_top5_gain": float(combined["top5"] - semantic["top5"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-report", default="reports/stwm_semantic_trace_field_decoder_v2_feature_targets_large_20260428.json")
    p.add_argument("--checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--target-reports", nargs="+", default=[
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json",
        "reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json",
    ])
    p.add_argument("--observed-cache", default="outputs/cache/stwm_semantic_target_temporal_stability_v1_20260428/observed_features.npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples-per-dataset", type=int, default=128)
    p.add_argument("--output", default="reports/stwm_semantic_prototype_predictability_baselines_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_PROTOTYPE_PREDICTABILITY_BASELINES_V1_20260428.md")
    args = p.parse_args()
    observed_cache, observed_meta = build_or_load_observed_feature_cache(
        feature_report=Path(args.feature_report),
        checkpoint_path=Path(args.checkpoint),
        output_cache=Path(args.observed_cache),
        device=str(args.device),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )
    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    results = [_eval_one(target_report=Path(path), feature_report=Path(args.feature_report), observed_cache=observed_cache, device=device) for path in args.target_reports]
    c64 = next((r for r in results if int(r["prototype_count"]) == 64), results[-1])
    simple_probe_beats_frequency = bool(c64["semantic_trace_mlp"]["top5"] > c64["frequency"]["top5"])
    target_predictable = bool(
        c64["observed_mean_feature_nearest_prototype"]["top5"] > c64["frequency"]["top5"]
        or c64["semantic_trace_mlp"]["top5"] > c64["frequency"]["top5"]
    )
    trace_only_sufficient = bool(c64["trace_only_mlp"]["top5"] > c64["frequency"]["top5"] + 0.05)
    semantic_load_bearing = (
        True
        if c64["semantic_only_vs_trace_only_top5_gain"] > 0.02
        else False
        if c64["semantic_only_vs_trace_only_top5_gain"] < -0.02
        else "unclear"
    )
    payload = {
        "audit_name": "stwm_semantic_prototype_predictability_baselines_v1",
        "observed_feature_meta": observed_meta,
        "results_by_prototype_count": results,
        "target_predictable_from_observed_semantics": target_predictable,
        "trace_only_sufficient": trace_only_sufficient,
        "semantic_input_load_bearing": semantic_load_bearing,
        "simple_probe_beats_frequency": simple_probe_beats_frequency,
        "no_stwm_backbone_update": True,
        "candidate_scorer_used": False,
        "future_candidate_input_used": False,
    }
    write_json(Path(args.output), payload)
    write_doc(
        Path(args.doc),
        "STWM Semantic Prototype Predictability Baselines V1",
        payload,
        bullets=[
            "Simple MLP probes are trained only as diagnostics and do not update STWM.",
            "Observed crop features are measurement diagnostics, not candidate scorer inputs.",
        ],
    )


if __name__ == "__main__":
    main()
