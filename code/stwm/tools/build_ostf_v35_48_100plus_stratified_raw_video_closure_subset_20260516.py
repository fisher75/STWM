#!/usr/bin/env python3
"""V35.48 构建 100+ stratified M128/H32 raw-video closure subset。"""
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

from stwm.tools.build_ostf_v35_45_larger_raw_video_closure_subset_20260516 import (  # noqa: E402
    UNIFIED_ROOT,
    list_npz,
    sample_row,
    trace_motion_from_source,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_48_100plus_stratified_raw_video_closure_subset"
MANIFEST = OUT_ROOT / "manifest.json"
REPORT = ROOT / "reports/stwm_ostf_v35_48_100plus_stratified_raw_video_closure_subset_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_48_100PLUS_STRATIFIED_RAW_VIDEO_CLOSURE_SUBSET_BUILD_20260516.md"


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


def retarget(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    split = out["split"]
    uid = out["sample_uid"]
    out["expected_rerun_trace_path"] = f"outputs/cache/stwm_ostf_v35_48_100plus_raw_video_frontend_rerun/M128_H32/{split}/{uid}.npz"
    return out


def risk_tags(row: dict[str, Any]) -> set[str]:
    tags = set(row["category_tags"])
    if row["dataset"] == "VIPSEG" and "semantic_changed" in tags:
        tags.add("risk_vipseg_changed")
    if "high_motion" in tags and "semantic_hard" in tags:
        tags.add("risk_high_motion_hard")
    if row["identity_claim_allowed"] and "semantic_changed" in tags:
        tags.add("risk_real_instance_semantic_changed")
    return tags


def target_split_counts(target: int, available: Counter[str]) -> dict[str, int]:
    test = min(available.get("test", 0), max(24, int(round(target * 0.27))))
    val = min(available.get("val", 0), max(32, int(round(target * 0.35))))
    train = min(available.get("train", 0), target - test - val)
    while train + val + test < target:
        if train < available.get("train", 0):
            train += 1
        elif val < available.get("val", 0):
            val += 1
        elif test < available.get("test", 0):
            test += 1
        else:
            break
    return {"train": train, "val": val, "test": test}


def row_score(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    tag_counts: Counter[str],
    dataset_counts: Counter[str],
    split_counts: Counter[str],
    dataset_targets: dict[str, int],
    split_targets: dict[str, int],
    real_target: int,
) -> tuple[Any, ...]:
    tags = risk_tags(row)
    selected_real = sum(int(r["identity_claim_allowed"]) for r in selected)
    target_tags = {
        "risk_vipseg_changed": 34,
        "risk_high_motion_hard": 34,
        "risk_real_instance_semantic_changed": real_target,
        "occlusion": 70,
        "crossing": 64,
        "identity_confuser": 64,
        "stable_heavy": 52,
        "high_motion": 48,
        "low_motion": 48,
        "semantic_uncertainty": 80,
    }
    tag_deficit = sum(max(v - tag_counts.get(k, 0), 0) for k, v in target_tags.items() if k in tags)
    dataset_deficit = max(dataset_targets.get(row["dataset"], 0) - dataset_counts.get(row["dataset"], 0), 0)
    split_deficit = max(split_targets.get(row["split"], 0) - split_counts.get(row["split"], 0), 0)
    real_deficit = max(real_target - selected_real, 0) if row["identity_claim_allowed"] else 0
    return (
        split_deficit,
        dataset_deficit,
        real_deficit,
        tag_deficit,
        int(row["identity_claim_allowed"]),
        row["semantic_flags"]["changed_ratio"] + row["semantic_flags"]["hard_ratio"] + row["semantic_flags"]["uncertainty_ratio"],
        int(row["confuser_pair_count"] > 0) + int(row["occlusion_count"] > 0) + int(row["crossing_count"] > 0),
        -row["point_count"],
    )


def select_stratified(rows: list[dict[str, Any]], target: int, real_target: int) -> list[dict[str, Any]]:
    available_dataset = Counter(r["dataset"] for r in rows)
    available_split = Counter(r["split"] for r in rows)
    dataset_targets = {"VSPW": min(available_dataset.get("VSPW", 0), target // 2), "VIPSEG": min(available_dataset.get("VIPSEG", 0), target - target // 2)}
    split_targets = target_split_counts(target, available_split)
    selected: list[dict[str, Any]] = []
    pool = list(rows)
    tag_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    while pool and len(selected) < target:
        best = max(pool, key=lambda r: row_score(r, selected, tag_counts, dataset_counts, split_counts, dataset_targets, split_targets, real_target))
        best = retarget(best)
        best["selection_reason"] = "V35.48 stratified greedy: dataset/split balance + fragile category oversampling"
        selected.append(best)
        tag_counts.update(risk_tags(best))
        dataset_counts[best["dataset"]] += 1
        split_counts[best["split"]] += 1
        pool = [r for r in pool if r["sample_uid"] != best["sample_uid"]]
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-count", type=int, default=128)
    ap.add_argument("--min-count", type=int, default=96)
    ap.add_argument("--real-instance-min", type=int, default=30)
    args = ap.parse_args()
    paths = []
    for p in list_npz(UNIFIED_ROOT):
        z = np.load(p, allow_pickle=True)
        if int(np.asarray(z["point_id"]).size) <= 1280:
            paths.append(p)
    motions = [trace_motion_from_source(p) for p in paths]
    median = float(np.median(motions)) if motions else 0.0
    rows = [sample_row(p, median) for p in paths]
    rows = [r for r in rows if r["predecode_available"] and r["raw_frame_paths"]]
    selected = select_stratified(rows, args.target_count, max(args.real_instance_min, 50))
    blockers: list[str] = []
    if len(selected) < args.min_count:
        blockers.append(f"selected_clip_count={len(selected)} < min_count={args.min_count}")
    real_count = sum(1 for r in selected if r["identity_claim_allowed"])
    if real_count < args.real_instance_min:
        blockers.append(f"real_instance_identity_count={real_count} < required={args.real_instance_min}")
    counts = {
        "dataset_counts": dict(Counter(r["dataset"] for r in selected)),
        "split_counts": dict(Counter(r["split"] for r in selected)),
        "semantic_changed_counts": int(sum("semantic_changed" in r["category_tags"] for r in selected)),
        "semantic_hard_counts": int(sum("semantic_hard" in r["category_tags"] for r in selected)),
        "stable_counts": int(sum("stable_heavy" in r["category_tags"] for r in selected)),
        "real_instance_identity_count": int(real_count),
        "pseudo_identity_count": int(sum(r["identity_provenance_type"] == "pseudo_slot" for r in selected)),
        "occlusion_count": int(sum("occlusion" in r["category_tags"] for r in selected)),
        "crossing_count": int(sum("crossing" in r["category_tags"] for r in selected)),
        "confuser_count": int(sum("identity_confuser" in r["category_tags"] for r in selected)),
        "high_motion_count": int(sum("high_motion" in r["category_tags"] for r in selected)),
        "risk_vipseg_changed_count": int(sum("risk_vipseg_changed" in risk_tags(r) for r in selected)),
        "risk_high_motion_hard_count": int(sum("risk_high_motion_hard" in risk_tags(r) for r in selected)),
        "risk_real_instance_semantic_changed_count": int(sum("risk_real_instance_semantic_changed" in risk_tags(r) for r in selected)),
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V35.48",
        "m": 128,
        "horizon": 32,
        "selected_clip_count": len(selected),
        "target_clip_count": args.target_count,
        "min_clip_count": args.min_count,
        "real_instance_min": args.real_instance_min,
        "samples": selected,
        "exact_blockers": blockers,
        **counts,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.48 100+ Stratified Raw-Video Closure Subset Build\n\n"
        f"- selected_clip_count: {len(selected)}\n"
        f"- target_clip_count: {args.target_count}\n"
        f"- min_clip_count: {args.min_count}\n"
        f"- dataset_counts: {counts['dataset_counts']}\n"
        f"- split_counts: {counts['split_counts']}\n"
        f"- real_instance_identity_count: {real_count}\n"
        f"- pseudo_identity_count: {counts['pseudo_identity_count']}\n"
        f"- risk_vipseg_changed_count: {counts['risk_vipseg_changed_count']}\n"
        f"- risk_high_motion_hard_count: {counts['risk_high_motion_hard_count']}\n"
        f"- risk_real_instance_semantic_changed_count: {counts['risk_real_instance_semantic_changed_count']}\n"
        f"- exact_blockers: {blockers}\n\n"
        "## 中文总结\n"
        f"V35.48 已构建 100+ stratified subset：{len(selected)} clips，重点过采样 VIPSeg changed、高运动 hard、真实 instance semantic changed、occlusion、crossing、identity confuser。"
        "本阶段仍只做 M128/H32 raw-video closure，不训练新模型，不跑 H64/H96/M512/M1024。\n",
        encoding="utf-8",
    )
    print(json.dumps({"v35_48_subset_built": not blockers, "selected_clip_count": len(selected), "real_instance_identity_count": real_count}, ensure_ascii=False), flush=True)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
