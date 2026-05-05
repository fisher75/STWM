#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


REPORT_PATH = ROOT / "reports/stwm_traceanything_vs_cotracker_teacher_comparison_v24_20260502.json"
DOC_PATH = ROOT / "docs/STWM_TRACEANYTHING_VS_COTRACKER_TEACHER_COMPARISON_V24_20260502.md"
HARDBENCH_PATH = ROOT / "reports/stwm_traceanything_hardbench_cache_v24_20260502.json"


def _scalar(x: Any) -> Any:
    a = np.asarray(x)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _cv_errors(z: np.lib.npyio.NpzFile, fut_steps: int = 16) -> dict[str, float]:
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    vis = np.asarray(z["visibility"]).astype(bool)
    obs_len = int(_scalar(z["obs_len"]))
    horizon = int(_scalar(z["horizon"]))
    take = min(fut_steps, horizon)
    obs = tracks[:, :, :obs_len]
    fut = tracks[:, :, obs_len : obs_len + take]
    vis_fut = vis[:, :, obs_len : obs_len + take]
    vel = obs[:, :, -1] - obs[:, :, -2] if obs.shape[2] >= 2 else np.zeros_like(obs[:, :, -1])
    pred = np.stack([obs[:, :, -1] + vel * (t + 1) for t in range(take)], axis=2)
    err = np.linalg.norm(pred - fut, axis=-1)
    endpoint = np.linalg.norm(pred[:, :, -1] - fut[:, :, -1], axis=-1)
    return {
        "cv_point_l1_px": float(err[vis_fut].mean()) if np.any(vis_fut) else float(err.mean()),
        "cv_endpoint_px": float(endpoint.mean()),
    }


def _compare_pair(trace_path: Path, cot_path: Path) -> dict[str, Any]:
    ta = np.load(trace_path, allow_pickle=True)
    co = np.load(cot_path, allow_pickle=True)
    ta_tracks = np.asarray(ta["tracks_xy"], dtype=np.float32)
    co_tracks = np.asarray(co["tracks_xy"], dtype=np.float32)
    ta_vis = np.asarray(ta["visibility"]).astype(bool)
    co_vis = np.asarray(co["visibility"]).astype(bool)
    steps = min(ta_tracks.shape[2], co_tracks.shape[2])
    ta_tracks = ta_tracks[:, :, :steps]
    co_tracks = co_tracks[:, :, :steps]
    ta_vis = ta_vis[:, :, :steps]
    co_vis = co_vis[:, :, :steps]
    both = np.logical_and(ta_vis, co_vis)
    dist = np.linalg.norm(ta_tracks - co_tracks, axis=-1)
    ta_var = float(np.var(ta_tracks, axis=2).mean())
    co_var = float(np.var(co_tracks, axis=2).mean())
    return {
        "trajectory_disagreement_l2_px": float(dist[both].mean()) if np.any(both) else float(dist.mean()),
        "endpoint_disagreement_l2_px": float(np.linalg.norm(ta_tracks[:, :, -1] - co_tracks[:, :, -1], axis=-1).mean()),
        "visibility_agreement": float((ta_vis == co_vis).mean()),
        "traceanything_trajectory_variance": ta_var,
        "cotracker_trajectory_variance": co_var,
        "traceanything_valid_point_ratio": float(ta_vis.mean()) if ta_vis.size else 0.0,
        "cotracker_valid_point_ratio": float(co_vis.mean()) if co_vis.size else 0.0,
        "traceanything_same_trajectory_fraction": float(_scalar(ta["same_trajectory_fraction"])) if "same_trajectory_fraction" in ta.files else None,
        "cotracker_same_trajectory_fraction": float(_scalar(co["same_trajectory_fraction"])) if "same_trajectory_fraction" in co.files else None,
        "traceanything_cv_errors": _cv_errors(ta),
        "cotracker_cv_errors": _cv_errors(co),
        "teacher_more_nonlinear": "traceanything" if ta_var > co_var else "cotracker",
        "teacher_harder_for_cv": "traceanything" if _cv_errors(ta)["cv_endpoint_px"] > _cv_errors(co)["cv_endpoint_px"] else "cotracker",
    }


def main() -> int:
    hardbench = json.loads(HARDBENCH_PATH.read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for combo, combo_info in hardbench.get("combo_summary", {}).items():
        for row in combo_info.get("rows", []):
            cot = row.get("matching_cotracker_h16_path")
            ta = row.get("cache_path")
            if cot and ta:
                cot_path = ROOT / cot
                ta_path = ROOT / ta
                if cot_path.exists() and ta_path.exists():
                    pairs.append(
                        {
                            "combo": combo,
                            "item_key": row["item_key"],
                            "dataset": row["dataset"],
                            **_compare_pair(ta_path, cot_path),
                        }
                    )
    if not pairs:
        payload = {
            "audit_name": "stwm_traceanything_vs_cotracker_teacher_comparison_v24",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "comparison_available": False,
            "recommended_primary_teacher": "unresolved",
        }
        dump_json(REPORT_PATH, payload)
        write_doc(DOC_PATH, "STWM TraceAnything vs CoTracker Teacher Comparison V24", payload, list(payload.keys()))
        return 1
    metrics = {
        "mean_trajectory_disagreement_l2_px": float(np.mean([p["trajectory_disagreement_l2_px"] for p in pairs])),
        "mean_endpoint_disagreement_l2_px": float(np.mean([p["endpoint_disagreement_l2_px"] for p in pairs])),
        "mean_visibility_agreement": float(np.mean([p["visibility_agreement"] for p in pairs])),
        "mean_traceanything_trajectory_variance": float(np.mean([p["traceanything_trajectory_variance"] for p in pairs])),
        "mean_cotracker_trajectory_variance": float(np.mean([p["cotracker_trajectory_variance"] for p in pairs])),
        "mean_traceanything_cv_endpoint_px": float(np.mean([p["traceanything_cv_errors"]["cv_endpoint_px"] for p in pairs])),
        "mean_cotracker_cv_endpoint_px": float(np.mean([p["cotracker_cv_errors"]["cv_endpoint_px"] for p in pairs])),
    }
    nonlin_votes = Counter(p["teacher_more_nonlinear"] for p in pairs)
    hard_votes = Counter(p["teacher_harder_for_cv"] for p in pairs)
    if nonlin_votes.get("traceanything", 0) > 0.6 * len(pairs) and hard_votes.get("traceanything", 0) > 0.6 * len(pairs):
        rec = "traceanything"
    elif nonlin_votes.get("traceanything", 0) > 0.4 * len(pairs) and hard_votes.get("traceanything", 0) > 0.4 * len(pairs):
        rec = "hybrid teacher"
    else:
        rec = "cotracker"
    payload = {
        "audit_name": "stwm_traceanything_vs_cotracker_teacher_comparison_v24",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_available": True,
        "pair_count": len(pairs),
        "per_dataset_counts": dict(Counter(p["dataset"] for p in pairs)),
        "metrics": metrics,
        "teacher_more_nonlinear_votes": dict(nonlin_votes),
        "teacher_harder_for_cv_votes": dict(hard_votes),
        "recommended_primary_teacher": rec,
        "pairs": pairs[:200],
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM TraceAnything vs CoTracker Teacher Comparison V24",
        payload,
        ["pair_count", "per_dataset_counts", "metrics", "teacher_more_nonlinear_votes", "teacher_harder_for_cv_votes", "recommended_primary_teacher"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
