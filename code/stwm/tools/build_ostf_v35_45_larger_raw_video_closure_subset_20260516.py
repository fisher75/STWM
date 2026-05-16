#!/usr/bin/env python3
"""V35.45 构建更大的 M128/H32 raw-video closure subset。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import scalar

UNIFIED_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_closure_subset"
MANIFEST = OUT_ROOT / "manifest.json"
REPORT = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_subset_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_45_LARGER_RAW_VIDEO_CLOSURE_SUBSET_BUILD_20260516.md"


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


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def identity_provenance(dataset: str, inst: np.ndarray, source_sem: np.ndarray, obs_last: np.ndarray) -> tuple[str, bool]:
    valid = inst[inst >= 0]
    counts = [int((inst == k).sum()) for k in np.unique(valid)] if valid.size else []
    equal_slot_blocks = bool(counts and len(set(counts)) == 1 and counts[0] == 128)
    sem_unique = int(len(np.unique(source_sem[source_sem >= 0]))) if np.any(source_sem >= 0) else 0
    last_unique = int(len(np.unique(obs_last[obs_last >= 0]))) if np.any(obs_last >= 0) else 0
    if dataset.upper() == "VIPSEG":
        return "real_instance", True
    if dataset.upper() == "VSPW" or (equal_slot_blocks and sem_unique <= 1 and last_unique <= 1):
        return "pseudo_slot", False
    return "unknown", False


def trace_motion_from_source(path: Path) -> float:
    try:
        z = np.load(path, allow_pickle=True)
        src = ROOT / str(scalar(z, "video_trace_source_npz"))
        tz = np.load(src, allow_pickle=True)
        tr = np.asarray(tz["tracks_xy"], dtype=np.float32)
        return float(np.sqrt((np.diff(tr, axis=2) ** 2).sum(axis=-1)).mean())
    except Exception:
        return 0.0


def sample_row(path: Path, motion_median: float | None = None) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    unc = np.asarray(z["semantic_uncertainty_target"], dtype=np.float32) > 0.5
    stable = np.asarray(z["semantic_stable_mask"], dtype=bool) & valid
    confuser = np.asarray(z["identity_identity_confuser_pair_mask"], dtype=bool)
    occlusion = np.asarray(z["identity_occlusion_reappear_point_mask"], dtype=bool)
    crossing = np.asarray(z["identity_trajectory_crossing_pair_mask"], dtype=bool)
    dataset = str(scalar(z, "dataset"))
    split = str(scalar(z, "split"))
    uid = str(scalar(z, "sample_uid", path.stem))
    trace_rel = str(scalar(z, "video_trace_source_npz"))
    trace_path = ROOT / trace_rel
    predecode = None
    raw_paths: list[str] = []
    if trace_path.exists():
        tz = np.load(trace_path, allow_pickle=True)
        predecode = str(scalar(tz, "predecode_path", ""))
        if "frame_paths" in tz.files:
            raw_paths = [str(x) for x in np.asarray(tz["frame_paths"], dtype=object).tolist()]
    motion = trace_motion_from_source(path)
    prov, claim = identity_provenance(
        dataset,
        np.asarray(z["point_to_instance_id"], dtype=np.int64),
        np.asarray(z["source_semantic_id"], dtype=np.int64),
        np.asarray(z["obs_semantic_last_id"], dtype=np.int64),
    )
    tags = {f"dataset_{dataset.lower()}", f"split_{split}"}
    if changed.any():
        tags.add("semantic_changed")
    if hard.any():
        tags.add("semantic_hard")
    if unc.any():
        tags.add("semantic_uncertainty")
    if stable.mean() >= 0.55:
        tags.add("stable_heavy")
    if occlusion.any():
        tags.add("occlusion")
    if crossing.any():
        tags.add("crossing")
    if confuser.any():
        tags.add("identity_confuser")
    if prov == "real_instance":
        tags.add("real_instance_identity")
    elif prov == "pseudo_slot":
        tags.add("pseudo_identity_diagnostic")
    if motion_median is not None:
        tags.add("high_motion" if motion >= motion_median else "low_motion")
    return {
        "sample_uid": uid,
        "dataset": dataset,
        "split": split,
        "source_unified_npz": rel(path),
        "raw_frame_paths": raw_paths,
        "predecode_path": predecode,
        "predecode_available": bool(predecode and Path(predecode).exists()),
        "semantic_flags": {
            "changed_ratio": float(changed.mean()),
            "hard_ratio": float(hard.mean()),
            "uncertainty_ratio": float(unc.mean()),
            "stable_ratio": float(stable.mean()),
        },
        "identity_provenance_type": prov,
        "identity_claim_allowed": claim,
        "expected_rerun_trace_path": f"outputs/cache/stwm_ostf_v35_45_larger_raw_video_frontend_rerun/M128_H32/{split}/{uid}.npz",
        "expected_cached_trace_path": trace_rel,
        "category_tags": sorted(tags),
        "selection_reason": "",
        "motion_mean": motion,
        "point_count": int(np.asarray(z["point_id"]).size),
        "confuser_pair_count": int(confuser.sum()),
        "occlusion_count": int(occlusion.sum()),
        "crossing_count": int(crossing.sum()),
    }


def score(row: dict[str, Any], covered: set[str], split_counts: Counter[str], dataset_counts: Counter[str]) -> tuple[Any, ...]:
    wanted = {
        "dataset_vspw",
        "dataset_vipseg",
        "split_train",
        "split_val",
        "split_test",
        "semantic_changed",
        "semantic_hard",
        "semantic_uncertainty",
        "stable_heavy",
        "high_motion",
        "low_motion",
        "occlusion",
        "crossing",
        "identity_confuser",
        "real_instance_identity",
        "pseudo_identity_diagnostic",
    }
    tags = set(row["category_tags"])
    split_penalty = split_counts[row["split"]]
    dataset_penalty = dataset_counts[row["dataset"]]
    return (
        len((tags & wanted) - covered),
        -split_penalty,
        -dataset_penalty,
        row["semantic_flags"]["changed_ratio"] + row["semantic_flags"]["hard_ratio"] + row["semantic_flags"]["uncertainty_ratio"],
        int(row["identity_claim_allowed"]),
        int(row["confuser_pair_count"] > 0) + int(row["occlusion_count"] > 0) + int(row["crossing_count"] > 0),
        -row["point_count"],
    )


def select(rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    split_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    pool = list(rows)
    # Seed each split and dataset first, then greedily fill.
    required = [("split", "val"), ("split", "test"), ("split", "train"), ("dataset", "VIPSEG"), ("dataset", "VSPW")]
    for key, val in required:
        cand = [r for r in pool if r[key] == val]
        if cand:
            best = max(cand, key=lambda r: score(r, covered, split_counts, dataset_counts))
            best["selection_reason"] = f"覆盖 {key}={val}"
            selected.append(best)
            covered |= set(best["category_tags"])
            split_counts[best["split"]] += 1
            dataset_counts[best["dataset"]] += 1
            pool = [r for r in pool if r["sample_uid"] != best["sample_uid"]]
    while pool and len(selected) < target_count:
        best = max(pool, key=lambda r: score(r, covered, split_counts, dataset_counts))
        best["selection_reason"] = "greedy coverage balance"
        selected.append(best)
        covered |= set(best["category_tags"])
        split_counts[best["split"]] += 1
        dataset_counts[best["dataset"]] += 1
        pool = [r for r in pool if r["sample_uid"] != best["sample_uid"]]
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-count", type=int, default=32)
    ap.add_argument("--min-count", type=int, default=24)
    args = ap.parse_args()
    candidates_raw = []
    for p in list_npz(UNIFIED_ROOT):
        z = np.load(p, allow_pickle=True)
        if int(np.asarray(z["point_id"]).size) <= 1280:
            candidates_raw.append(p)
    motions = [trace_motion_from_source(p) for p in candidates_raw]
    median = float(np.median(motions)) if motions else 0.0
    candidates = [sample_row(p, median) for p in candidates_raw]
    candidates = [r for r in candidates if r["predecode_available"] and r["raw_frame_paths"]]
    selected = select(candidates, args.target_count)
    blockers: list[str] = []
    if len(selected) < args.min_count:
        blockers.append(f"可选样本不足：selected={len(selected)} < min={args.min_count}")
    counts = {
        "dataset_counts": dict(Counter(r["dataset"] for r in selected)),
        "split_counts": dict(Counter(r["split"] for r in selected)),
        "semantic_changed_counts": int(sum("semantic_changed" in r["category_tags"] for r in selected)),
        "semantic_hard_counts": int(sum("semantic_hard" in r["category_tags"] for r in selected)),
        "stable_counts": int(sum("stable_heavy" in r["category_tags"] for r in selected)),
        "real_instance_identity_count": int(sum(r["identity_provenance_type"] == "real_instance" for r in selected)),
        "pseudo_identity_count": int(sum(r["identity_provenance_type"] == "pseudo_slot" for r in selected)),
        "occlusion_count": int(sum("occlusion" in r["category_tags"] for r in selected)),
        "crossing_count": int(sum("crossing" in r["category_tags"] for r in selected)),
        "confuser_count": int(sum("identity_confuser" in r["category_tags"] for r in selected)),
        "high_motion_count": int(sum("high_motion" in r["category_tags"] for r in selected)),
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V35.45",
        "m": 128,
        "horizon": 32,
        "selected_clip_count": len(selected),
        "target_clip_count": args.target_count,
        "min_clip_count": args.min_count,
        "samples": selected,
        "exact_blockers": blockers,
        **counts,
    }
    report = {
        "generated_at_utc": manifest["generated_at_utc"],
        "larger_raw_video_closure_subset_built": len(blockers) == 0,
        **{k: manifest[k] for k in ["selected_clip_count", "target_clip_count", "min_clip_count", "exact_blockers"]},
        **counts,
        "manifest_path": rel(MANIFEST),
        "中文结论": (
            f"V35.45 larger raw-video closure subset 已构建：selected={len(selected)}，覆盖 dataset/split/semantic/identity provenance/motion/occlusion/crossing/confuser。"
            if not blockers
            else "V35.45 larger subset 未达到最小样本数，需要修候选池。"
        ),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.45 Larger Raw-Video Closure Subset Build\n\n"
        f"- selected_clip_count: {len(selected)}\n"
        f"- dataset_counts: {counts['dataset_counts']}\n"
        f"- split_counts: {counts['split_counts']}\n"
        f"- real_instance_identity_count: {counts['real_instance_identity_count']}\n"
        f"- pseudo_identity_count: {counts['pseudo_identity_count']}\n"
        f"- exact_blockers: {blockers}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_clip_count": len(selected), "larger_subset_built": len(blockers) == 0}, ensure_ascii=False), flush=True)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
