#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, batch_from_samples, dump_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import analytic_constant_velocity_predict, build_v18_rows
from stwm.tools.ostf_multimodal_metrics_v21 import (
    aggregate_item_rows,
    hypothesis_diversity_valid,
    multimodal_item_scores,
    paired_bootstrap_from_rows,
)
from stwm.tools.ostf_v20_common_20260502 import hard_subset_flags, load_context_cache, sample_key
from stwm.tools.train_ostf_context_residual_v20_20260502 import _build_model


def _subset_flags(samples: list[Any], ctx_map: dict[tuple[str, int], dict[str, Any]]) -> dict[str, np.ndarray]:
    records = []
    for s in samples:
        c = ctx_map[sample_key(s)]
        records.append(
            {
                "cv_point_l1_proxy": c["cv_point_l1_proxy"],
                "curvature_proxy": c["curvature_proxy"],
                "occlusion_ratio": c["occlusion_ratio"],
                "interaction_proxy": c["interaction_proxy"],
            }
        )
    return hard_subset_flags(records)


def _load_v20_run(run_name: str) -> dict[str, Any]:
    path = ROOT / f"reports/stwm_ostf_v20_runs/{run_name}.json"
    return __import__("json").load(open(path))


def _infer_v20(run_name: str, combo: str, context_cache_rel: str, batch_size: int) -> dict[str, Any]:
    report = _load_v20_run(run_name)
    rows, proto_centers = build_v18_rows(combo, seed=42)
    samples = rows["test"]
    ctx_map = load_context_cache(ROOT / context_cache_rel)
    flags = _subset_flags(samples, ctx_map)

    model = _build_model(report["model_kind"], int(report["horizon"]))
    ckpt = torch.load(ROOT / report["best_checkpoint_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    point_modes = []
    mode_logits = []
    point_pred = []
    vis_logits = []
    sem_logits = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_rows = samples[start : start + batch_size]
            batch = batch_from_samples(batch_rows, torch.device("cpu"))
            crop = np.stack([ctx_map[sample_key(s)]["crop_feat"] for s in batch_rows], axis=0)
            box = np.stack([ctx_map[sample_key(s)]["box_feat"] for s in batch_rows], axis=0)
            neigh = np.stack([ctx_map[sample_key(s)]["neighbor_feat"] for s in batch_rows], axis=0)
            glob = np.stack([ctx_map[sample_key(s)]["global_feat"] for s in batch_rows], axis=0)
            out = model(
                obs_points=batch["obs_points"],
                obs_vis=batch["obs_vis"],
                rel_xy=batch["rel_xy"],
                anchor_obs=batch["anchor_obs"],
                anchor_obs_vel=batch["anchor_obs_vel"],
                semantic_feat=batch["semantic_feat"],
                crop_feat=torch.tensor(crop, dtype=torch.float32),
                box_feat=torch.tensor(box, dtype=torch.float32),
                neighbor_feat=torch.tensor(neigh, dtype=torch.float32),
                global_feat=torch.tensor(glob, dtype=torch.float32),
            )
            point_modes.append(out["point_hypotheses"].detach().cpu().numpy())
            mode_logits.append(out["hypothesis_logits"].detach().cpu().numpy())
            point_pred.append(out["point_pred"].detach().cpu().numpy())
            vis_logits.append(out["visibility_logits"].detach().cpu().numpy())
            sem_logits.append(out["semantic_logits"].detach().cpu().numpy())
    point_modes_np = np.concatenate(point_modes, axis=0)
    mode_logits_np = np.concatenate(mode_logits, axis=0)
    point_pred_np = np.concatenate(point_pred, axis=0)
    vis_logits_np = np.concatenate(vis_logits, axis=0)
    sem_logits_np = np.concatenate(sem_logits, axis=0)
    rows_mm = multimodal_item_scores(
        samples,
        point_modes=point_modes_np,
        mode_logits=mode_logits_np,
        point_pred=point_pred_np,
        pred_vis_logits=vis_logits_np,
        pred_proto_logits=sem_logits_np,
        subset_flags=flags,
        cv_mode_index=0,
    )
    return {
        "run_name": run_name,
        "combo": combo,
        "model_kind": report["model_kind"],
        "point_hypotheses_shape_example": list(point_modes_np[0].shape),
        "weighted_average_metrics": aggregate_item_rows(rows_mm),
        "hard_subset_metrics": aggregate_item_rows(rows_mm, subset_key="top20_cv_hard"),
        "occlusion_subset_metrics": aggregate_item_rows(rows_mm, subset_key="occlusion_hard"),
        "nonlinear_subset_metrics": aggregate_item_rows(rows_mm, subset_key="nonlinear_hard"),
        "interaction_subset_metrics": aggregate_item_rows(rows_mm, subset_key="interaction_hard"),
        "rows": rows_mm,
        "diversity_valid": hypothesis_diversity_valid(rows_mm),
    }


def _cv_rows(combo: str, context_cache_rel: str) -> list[dict[str, Any]]:
    rows, proto_centers = build_v18_rows(combo, seed=42)
    samples = rows["test"]
    ctx_map = load_context_cache(ROOT / context_cache_rel)
    flags = _subset_flags(samples, ctx_map)
    pred_points, pred_vis, pred_sem = analytic_constant_velocity_predict(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    modes = pred_points[:, :, :, None, :]
    logits = np.zeros((pred_points.shape[0], 1), dtype=np.float32)
    return multimodal_item_scores(
        samples,
        point_modes=modes,
        mode_logits=logits,
        point_pred=pred_points,
        pred_vis_logits=pred_vis,
        pred_proto_logits=pred_sem,
        subset_flags=flags,
        cv_mode_index=0,
    )


def _metric_bootstrap_same_rows(rows: list[dict[str, Any]], better_metric: str, worse_metric: str, subset_key: str | None = None, higher_better: bool = False) -> dict[str, Any]:
    a = []
    b = []
    for r in rows:
        if subset_key is not None and not r.get(subset_key, False):
            continue
        if r.get(better_metric) is None or r.get(worse_metric) is None:
            continue
        a.append(float(r[better_metric]))
        b.append(float(r[worse_metric]))
    if not a:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not higher_better:
        a = -a
        b = -b
    delta = a - b
    rng = np.random.default_rng(42)
    means = []
    for _ in range(1000):
        idx = rng.integers(0, len(delta), size=len(delta))
        means.append(float(delta[idx].mean()))
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5]).tolist()
    return {
        "item_count": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool((lo > 0.0) or (hi < 0.0)),
    }


def main() -> int:
    m128 = _infer_v20(
        "v20_context_residual_m128_seed42_h8",
        "M128_H8",
        "outputs/cache/stwm_ostf_context_features_v20/M128_H8_context_features.npz",
        4,
    )
    m512 = _infer_v20(
        "v20_context_residual_m512_seed42_h8",
        "M512_H8",
        "outputs/cache/stwm_ostf_context_features_v20/M512_H8_context_features.npz",
        2,
    )
    cv128_rows = _cv_rows("M128_H8", "outputs/cache/stwm_ostf_context_features_v20/M128_H8_context_features.npz")
    cv512_rows = _cv_rows("M512_H8", "outputs/cache/stwm_ostf_context_features_v20/M512_H8_context_features.npz")

    payload = {
        "audit_name": "stwm_ostf_multimodal_eval_gap_v21",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "point_hypotheses_exist": True,
        "deterministic_eval_used_weighted_average_point_pred": True,
        "M128": {
            "summary": {
                "weighted_average_metrics": m128["weighted_average_metrics"],
                "hard_subset_metrics": m128["hard_subset_metrics"],
                "diversity_valid": m128["diversity_valid"],
            },
            "bestofk_vs_weighted_all_minFDE_vs_endpoint": _metric_bootstrap_same_rows(m128["rows"], "minFDE_K_px", "weighted_endpoint_error_px", higher_better=False),
            "bestofk_vs_weighted_hard_minFDE_vs_endpoint": _metric_bootstrap_same_rows(m128["rows"], "minFDE_K_px", "weighted_endpoint_error_px", subset_key="top20_cv_hard", higher_better=False),
            "bestofk_vs_cv_hard_minFDE": paired_bootstrap_from_rows(m128["rows"], cv128_rows, metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard"),
        },
        "M512": {
            "summary": {
                "weighted_average_metrics": m512["weighted_average_metrics"],
                "hard_subset_metrics": m512["hard_subset_metrics"],
                "diversity_valid": m512["diversity_valid"],
            },
            "bestofk_vs_weighted_all_minFDE_vs_endpoint": _metric_bootstrap_same_rows(m512["rows"], "minFDE_K_px", "weighted_endpoint_error_px", higher_better=False),
            "bestofk_vs_weighted_hard_minFDE_vs_endpoint": _metric_bootstrap_same_rows(m512["rows"], "minFDE_K_px", "weighted_endpoint_error_px", subset_key="top20_cv_hard", higher_better=False),
            "bestofk_vs_cv_hard_minFDE": paired_bootstrap_from_rows(m512["rows"], cv512_rows, metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard"),
        },
        "hypothesis_diversity": {
            "M128_pairwise_endpoint_diversity_px": m128["weighted_average_metrics"]["pairwise_endpoint_diversity_px"],
            "M512_pairwise_endpoint_diversity_px": m512["weighted_average_metrics"]["pairwise_endpoint_diversity_px"],
            "M128_collapse_rate_8px": m128["weighted_average_metrics"]["collapse_rate_8px"],
            "M512_collapse_rate_8px": m512["weighted_average_metrics"]["collapse_rate_8px"],
            "M128_best_mode_non_cv_rate": m128["weighted_average_metrics"]["best_mode_non_cv_rate"],
            "M512_best_mode_non_cv_rate": m512["weighted_average_metrics"]["best_mode_non_cv_rate"],
        },
        "best_of_K_beats_weighted_average": bool(
            m512["diversity_valid"]
            and (payload := _metric_bootstrap_same_rows(m512["rows"], "minFDE_K_px", "weighted_endpoint_error_px", higher_better=False))["zero_excluded"]
            and (payload["mean_delta"] or 0.0) > 0.0
        ),
    }
    payload["best_of_K_beats_CV_on_hard_subset"] = bool(
        payload["M512"]["bestofk_vs_cv_hard_minFDE"]["zero_excluded"]
        and (payload["M512"]["bestofk_vs_cv_hard_minFDE"]["mean_delta"] or 0.0) > 0.0
    )
    payload["current_deterministic_eval_invalidates_multihypothesis_claim"] = bool(
        payload["best_of_K_beats_weighted_average"] and not payload["best_of_K_beats_CV_on_hard_subset"]
    )

    out = ROOT / "reports/stwm_ostf_multimodal_eval_gap_v21_20260502.json"
    dump_json(out, payload)
    write_doc(
        ROOT / "docs/STWM_OSTF_MULTIMODAL_EVAL_GAP_V21_20260502.md",
        "STWM OSTF Multimodal Eval Gap V21",
        payload,
        [
            "point_hypotheses_exist",
            "deterministic_eval_used_weighted_average_point_pred",
            "best_of_K_beats_weighted_average",
            "best_of_K_beats_CV_on_hard_subset",
            "current_deterministic_eval_invalidates_multihypothesis_claim",
        ],
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
