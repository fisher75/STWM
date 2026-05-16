#!/usr/bin/env python3
"""V36.1: 拆解 V30 causal trace rollout 为什么输给 strongest prior。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

ROLLOUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_v30_past_only_future_trace_rollout/M128_H32"
CAUSAL_SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_causal_unified_semantic_identity_slice/M128_H32"
MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
ROLLOUT_REPORT = ROOT / "reports/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v36_1_trace_rollout_failure_atlas_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_1_TRACE_ROLLOUT_FAILURE_ATLAS_20260516.md"

METHOD_KEYS = {
    "v30": "predicted_future_points",
    "last_visible_copy": "prior_last_visible_copy",
    "last_observed_copy": "prior_last_observed_copy",
    "constant_velocity": "prior_constant_velocity",
    "damped_velocity": "prior_damped_velocity",
}


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return arr.item() if arr.shape == () else arr


def ade(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    mask = valid.astype(bool)
    if not mask.any():
        return None
    return float(np.linalg.norm(pred - target, axis=-1)[mask].mean())


def fde(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    vals = []
    for i in range(pred.shape[0]):
        idx = np.where(valid[i])[0]
        if idx.size:
            t = int(idx[-1])
            vals.append(float(np.linalg.norm(pred[i, t] - target[i, t])))
    return float(np.mean(vals)) if vals else None


def bool_ratio(z: np.lib.npyio.NpzFile, key: str) -> float:
    if key not in z.files:
        return 0.0
    arr = np.asarray(z[key]).astype(bool)
    return float(arr.mean()) if arr.size else 0.0


def motion_stats(obs: np.ndarray) -> dict[str, float]:
    step = np.linalg.norm(np.diff(obs, axis=1), axis=-1)
    point_motion = step.mean(axis=1) if step.size else np.zeros(obs.shape[0], dtype=np.float32)
    global_disp = np.linalg.norm(np.median(obs[:, -1] - obs[:, 0], axis=0))
    return {
        "motion_mean": float(point_motion.mean()) if point_motion.size else 0.0,
        "motion_p90": float(np.percentile(point_motion, 90)) if point_motion.size else 0.0,
        "camera_motion_proxy": float(global_disp),
    }


def manifest_map() -> dict[str, dict[str, Any]]:
    if not MANIFEST.exists():
        return {}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for s in data.get("samples", []):
        name = Path(str(s.get("source_unified_npz", s.get("expected_rerun_trace_path", "")))).name
        if name:
            out[name] = s
    return out


def bin_by_quantile(value: float, cuts: tuple[float, float], prefix: str) -> str:
    if value <= cuts[0]:
        return f"{prefix}_low"
    if value <= cuts[1]:
        return f"{prefix}_mid"
    return f"{prefix}_high"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_count": len(rows)}
    for method in METHOD_KEYS:
        vals = [float(r[f"{method}_ADE"]) for r in rows if r.get(f"{method}_ADE") is not None]
        fdes = [float(r[f"{method}_FDE"]) for r in rows if r.get(f"{method}_FDE") is not None]
        out[f"{method}_ADE_mean"] = float(np.mean(vals)) if vals else None
        out[f"{method}_FDE_mean"] = float(np.mean(fdes)) if fdes else None
    prior_methods = [m for m in METHOD_KEYS if m != "v30"]
    best_prior = min(prior_methods, key=lambda m: out[f"{m}_ADE_mean"] if out[f"{m}_ADE_mean"] is not None else 1e18)
    out["strongest_prior"] = best_prior
    out["strongest_prior_ADE_mean"] = out[f"{best_prior}_ADE_mean"]
    if out["v30_ADE_mean"] is not None and out["strongest_prior_ADE_mean"] is not None:
        out["v30_minus_strongest_prior_ADE"] = float(out["v30_ADE_mean"]) - float(out["strongest_prior_ADE_mean"])
        out["v30_beats_strongest_prior"] = bool(out["v30_minus_strongest_prior_ADE"] <= 0)
    else:
        out["v30_minus_strongest_prior_ADE"] = None
        out["v30_beats_strongest_prior"] = False
    return out


def main() -> int:
    mmap = manifest_map()
    raw_rows: list[dict[str, Any]] = []
    for p in sorted(ROLLOUT_ROOT.glob("*/*.npz")):
        rz = np.load(p, allow_pickle=True)
        split = str(scalar(rz, "split"))
        cp = CAUSAL_SLICE_ROOT / split / p.name
        cz = np.load(cp, allow_pickle=True) if cp.exists() else None
        target = np.asarray(rz["future_trace_teacher_points"], dtype=np.float32)
        valid = np.asarray(rz["future_trace_teacher_vis"], dtype=bool)
        obs = np.asarray(rz["obs_points"], dtype=np.float32)
        obs_vis = np.asarray(rz["obs_vis"], dtype=bool)
        obs_conf = np.asarray(rz["obs_conf"], dtype=np.float32)
        sample_meta = mmap.get(p.name, {})
        mstats = motion_stats(obs)
        row: dict[str, Any] = {
            "sample_uid": str(scalar(rz, "sample_uid")),
            "path": rel(p),
            "dataset": str(scalar(rz, "dataset")),
            "split": split,
            "point_count": int(obs.shape[0]),
            "object_count_est": int(max(1, round(obs.shape[0] / 128))),
            "obs_visibility_mean": float(obs_vis.mean()),
            "obs_confidence_mean": float(obs_conf.mean()),
            "future_teacher_visibility_mean": float(valid.mean()),
            "motion_mean": mstats["motion_mean"],
            "motion_p90": mstats["motion_p90"],
            "camera_motion_proxy": mstats["camera_motion_proxy"],
            "category_tags_manifest": sample_meta.get("category_tags", []),
            "identity_claim_allowed": bool(sample_meta.get("identity_claim_allowed", False)),
        }
        if cz is not None:
            row.update(
                {
                    "semantic_changed_ratio": bool_ratio(cz, "semantic_changed_mask"),
                    "semantic_hard_ratio": bool_ratio(cz, "semantic_hard_mask"),
                    "semantic_stable_ratio": bool_ratio(cz, "semantic_stable_mask"),
                    "occlusion_ratio": bool_ratio(cz, "identity_occlusion_reappear_point_mask"),
                    "crossing_pair_ratio": bool_ratio(cz, "identity_trajectory_crossing_pair_mask"),
                }
            )
        for method, key in METHOD_KEYS.items():
            pred = np.asarray(rz[key], dtype=np.float32)
            row[f"{method}_ADE"] = ade(pred, target, valid)
            row[f"{method}_FDE"] = fde(pred, target, valid)
        prior_methods = [m for m in METHOD_KEYS if m != "v30"]
        best = min(prior_methods, key=lambda m: row[f"{m}_ADE"] if row[f"{m}_ADE"] is not None else 1e18)
        row["sample_strongest_prior"] = best
        row["v30_minus_sample_strongest_prior_ADE"] = float(row["v30_ADE"]) - float(row[f"{best}_ADE"]) if row["v30_ADE"] is not None and row[f"{best}_ADE"] is not None else None
        row["v30_wins_sample"] = bool(row["v30_minus_sample_strongest_prior_ADE"] is not None and row["v30_minus_sample_strongest_prior_ADE"] <= 0)
        raw_rows.append(row)

    motion_vals = np.asarray([r["motion_mean"] for r in raw_rows], dtype=np.float32)
    camera_vals = np.asarray([r["camera_motion_proxy"] for r in raw_rows], dtype=np.float32)
    conf_vals = np.asarray([r["obs_confidence_mean"] for r in raw_rows], dtype=np.float32)
    vis_vals = np.asarray([r["obs_visibility_mean"] for r in raw_rows], dtype=np.float32)
    cuts = {
        "motion": tuple(np.quantile(motion_vals, [1 / 3, 2 / 3]).tolist()) if len(motion_vals) else (0.0, 0.0),
        "camera_motion": tuple(np.quantile(camera_vals, [1 / 3, 2 / 3]).tolist()) if len(camera_vals) else (0.0, 0.0),
        "confidence": tuple(np.quantile(conf_vals, [1 / 3, 2 / 3]).tolist()) if len(conf_vals) else (0.0, 0.0),
        "visibility": tuple(np.quantile(vis_vals, [1 / 3, 2 / 3]).tolist()) if len(vis_vals) else (0.0, 0.0),
    }

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in raw_rows:
        cats = {
            f"dataset_{str(r['dataset']).lower()}",
            f"split_{r['split']}",
            "real_instance_identity" if r["identity_claim_allowed"] else "pseudo_or_unknown_identity",
            "changed_present" if r.get("semantic_changed_ratio", 0.0) > 0 else "changed_absent",
            "hard_present" if r.get("semantic_hard_ratio", 0.0) > 0 else "hard_absent",
            "stable_heavy" if r.get("semantic_stable_ratio", 0.0) >= 0.6 else "stable_light",
            "occlusion_present" if r.get("occlusion_ratio", 0.0) > 0 else "occlusion_absent",
            "crossing_present" if r.get("crossing_pair_ratio", 0.0) > 0 else "crossing_absent",
            bin_by_quantile(float(r["motion_mean"]), cuts["motion"], "motion"),
            bin_by_quantile(float(r["camera_motion_proxy"]), cuts["camera_motion"], "camera_motion"),
            bin_by_quantile(float(r["obs_confidence_mean"]), cuts["confidence"], "confidence"),
            bin_by_quantile(float(r["obs_visibility_mean"]), cuts["visibility"], "visibility"),
            "object_count_1" if r["object_count_est"] <= 1 else "object_count_2_4" if r["object_count_est"] <= 4 else "object_count_5plus",
        }
        cats.update(str(c) for c in r.get("category_tags_manifest", []))
        r["failure_atlas_categories"] = sorted(cats)
        buckets["all"].append(r)
        for c in cats:
            buckets[c].append(r)
    category_summary = {k: aggregate(v) for k, v in sorted(buckets.items())}
    fragile = {
        k: v
        for k, v in category_summary.items()
        if v["sample_count"] >= 5 and v.get("v30_minus_strongest_prior_ADE") is not None and float(v["v30_minus_strongest_prior_ADE"]) > 0
    }
    robust = {
        k: v
        for k, v in category_summary.items()
        if v["sample_count"] >= 5 and v.get("v30_minus_strongest_prior_ADE") is not None and float(v["v30_minus_strongest_prior_ADE"]) <= 0
    }
    top_fragile = sorted(fragile.items(), key=lambda kv: float(kv[1]["v30_minus_strongest_prior_ADE"]), reverse=True)[:20]
    sample_win_rate = float(np.mean([r["v30_wins_sample"] for r in raw_rows])) if raw_rows else 0.0
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_rollout_failure_atlas_done": True,
        "input_rollout_root": rel(ROLLOUT_ROOT),
        "source_rollout_report": rel(ROLLOUT_REPORT),
        "sample_count": len(raw_rows),
        "global_summary": category_summary.get("all", {}),
        "v30_sample_win_rate_vs_sample_strongest_prior": sample_win_rate,
        "quantile_thresholds": cuts,
        "category_summary": category_summary,
        "top_fragile_categories": [{"category": k, **v} for k, v in top_fragile],
        "robust_category_count": len(robust),
        "fragile_category_count": len(fragile),
        "exact_blockers": []
        if raw_rows
        else ["没有找到 V36 rollout npz，无法构建 failure atlas。"],
        "recommended_next_step": "build_v36_1_strongest_prior_downstream_slice",
        "中文总结": (
            "V36.1 failure atlas 显示 V30 causal trace rollout 在 full 325 上整体没有赢 strongest analytic prior；需要进一步看 downstream semantic/identity 是否仍优于 strongest-prior slice。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# STWM OSTF V36.1 Trace Rollout Failure Atlas",
        "",
        f"- sample_count: {len(raw_rows)}",
        f"- global_v30_ADE_mean: {report['global_summary'].get('v30_ADE_mean')}",
        f"- global_strongest_prior: {report['global_summary'].get('strongest_prior')}",
        f"- global_strongest_prior_ADE_mean: {report['global_summary'].get('strongest_prior_ADE_mean')}",
        f"- global_v30_minus_strongest_prior_ADE: {report['global_summary'].get('v30_minus_strongest_prior_ADE')}",
        f"- v30_sample_win_rate_vs_sample_strongest_prior: {sample_win_rate}",
        f"- fragile_category_count: {len(fragile)}",
        f"- robust_category_count: {len(robust)}",
        "",
        "## 高风险类别 Top",
    ]
    for item in top_fragile[:10]:
        lines.append(
            f"- {item[0]}: sample_count={item[1]['sample_count']}, "
            f"v30_minus_prior_ADE={item[1].get('v30_minus_strongest_prior_ADE')}, strongest_prior={item[1].get('strongest_prior')}"
        )
    lines += ["", "## 中文总结", report["中文总结"], ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"V36_1_trace_failure_atlas完成": True, "sample_count": len(raw_rows), "global_v30_beats_prior": report["global_summary"].get("v30_beats_strongest_prior")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
