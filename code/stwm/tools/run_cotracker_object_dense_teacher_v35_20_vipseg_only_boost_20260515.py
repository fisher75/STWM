#!/usr/bin/env python3
"""V35.20 只扩 VIPSeg source-domain 的 M128/H32 CoTracker trace cache。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

setproctitle.setproctitle("python")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cotracker_object_dense_teacher_v15c_20260502 as v15c  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_BASE = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16"


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


def dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def combo_root(m: int, h: int) -> Path:
    return OUT_BASE / f"M{m}_H{h}"


def expected_cache_path(pre_path: Path, split_map: dict[str, str], root: Path) -> Path:
    key = v15c._norm_key(pre_path)
    split = split_map.get(key, pre_path.parent.name)
    return root / split / f"{key.replace('::', '__')}.npz"


def summarize_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    tracks = np.asarray(z["tracks_xy"])
    vis = np.asarray(z["visibility"]).astype(bool)
    return {
        "item_key": str(scalar(z["item_key"])),
        "split": str(scalar(z["split"])),
        "dataset": str(scalar(z["dataset"])),
        "cache_path": str(path.relative_to(ROOT)),
        "object_count": int(tracks.shape[0]),
        "point_count": int(tracks.shape[0] * tracks.shape[1]),
        "valid_point_ratio": float(vis.mean()) if vis.size else 0.0,
        "resumed_existing": True,
    }


def vipseg_candidates(split_map: dict[str, str], quotas: dict[str, int]) -> list[Path]:
    buckets: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(v15c.PREDECODE_ROOT.glob("*/*.npz")):
        key = v15c._norm_key(path)
        ds = key.split("::", 1)[0]
        split = split_map.get(key, path.parent.name)
        if ds == "VIPSEG" and split in quotas:
            buckets[split].append(path)
    rng = np.random.default_rng(20260515)
    selected: list[Path] = []
    for split in ["train", "val", "test"]:
        rows = buckets.get(split, [])
        if not rows:
            continue
        order = rng.permutation(len(rows))
        # Oversample candidates because many short clips fail the horizon window test.
        selected.extend([rows[i] for i in order[: max(quotas[split] * 12, quotas[split])]])
    return selected


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# STWM V35.20 VIPSeg-only CoTracker Boost\n\n"
        f"- processed_clip_count: {payload.get('processed_clip_count')}\n"
        f"- vipseg_processed_split_counts: {payload.get('vipseg_processed_split_counts')}\n"
        f"- skipped_existing_clip_count: {payload.get('skipped_existing_clip_count')}\n"
        f"- failed_clip_count: {payload.get('failed_clip_count')}\n"
        f"- cache_root: {payload.get('cache_root')}\n\n"
        "## 中文总结\n"
        "本轮只补 VIPSeg source-domain 的 M128/H32 trace cache，目的是修 VIPSeg→VSPW 的 source 覆盖不足；没有跑 H64/H96/M512，也没有训练 semantic adapter。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default="baselines/repos/co-tracker")
    parser.add_argument("--checkpoint", default="baselines/checkpoints/cotracker/scaled_offline.pth")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--target-vipseg-train", type=int, default=120)
    parser.add_argument("--target-vipseg-val", type=int, default=30)
    parser.add_argument("--target-vipseg-test", type=int, default=9)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.repo_path = str(ROOT / args.repo_path)
    args.checkpoint = str(ROOT / args.checkpoint)
    combo = f"M{args.m}_H{args.horizon}"
    out_root = combo_root(args.m, args.horizon)
    v15c.OUT_ROOT = out_root
    split_map = v15c._mixed_split_map()
    quotas = {"train": args.target_vipseg_train, "val": args.target_vipseg_val, "test": args.target_vipseg_test}
    repo = Path(args.repo_path)
    ckpt = Path(args.checkpoint)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_existing = 0
    if not repo.exists() or not ckpt.exists():
        payload = {"teacher_run_success": False, "blocker": "cotracker_repo_or_checkpoint_missing", "official_repo_path": str(repo), "checkpoint_path": str(ckpt)}
        dump(ROOT / f"reports/stwm_cotracker_object_dense_teacher_v16_{combo}_v35_20_vipseg_only_boost_20260515.json", payload)
        return 1
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor  # type: ignore

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = CoTrackerPredictor(checkpoint=str(ckpt), offline=True, window_len=60).to(device).eval()
    selected = vipseg_candidates(split_map, quotas)
    success_by_split: Counter[str] = Counter()
    for idx, pre_path in enumerate(selected):
        split_name = split_map.get(v15c._norm_key(pre_path), pre_path.parent.name)
        expected = expected_cache_path(pre_path, split_map, out_root)
        existing = None if args.force else summarize_existing(expected)
        if existing is not None:
            rows.append(existing)
            success_by_split[split_name] += 1
            skipped_existing += 1
        else:
            row, fail = v15c._run_clip(model, pre_path, args, device, split_map)
            if row:
                rows.append(row)
                success_by_split[row["split"]] += 1
            if fail:
                failures.append(fail)
        print(json.dumps({"进度": f"{idx + 1}/{len(selected)}", "VIPSeg成功": dict(success_by_split), "失败": len(failures), "clip": pre_path.name}, ensure_ascii=False), flush=True)
        if all(success_by_split[s] >= q for s, q in quotas.items()):
            break
    payload = {
        "teacher_run_success": bool(rows),
        "combo": combo,
        "M": args.m,
        "horizon": args.horizon,
        "obs_len": args.obs_len,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exact_command": " ".join(sys.argv),
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "requested_vipseg_split_counts": quotas,
        "vipseg_processed_split_counts": dict(success_by_split),
        "processed_clip_count": len(rows),
        "skipped_existing_clip_count": skipped_existing,
        "failed_clip_count": len(failures),
        "runtime_seconds": float(time.time() - t0),
        "cache_root": str(out_root.relative_to(ROOT)),
        "clip_caches": rows,
        "failed_clips": failures,
        "failure_reason_top": dict(Counter(str(f["reason"]) for f in failures).most_common(20)),
        "v30_backbone_frozen": True,
        "no_h64_h96_m512": True,
        "semantic_adapter_training_ran": False,
        "中文结论": "已只针对 VIPSeg source-domain 扩展 M128/H32 video-derived trace cache；下一步需要补 measurement、重建 target、重跑 VIPSeg→VSPW predictability。",
    }
    report = ROOT / f"reports/stwm_cotracker_object_dense_teacher_v16_{combo}_v35_20_vipseg_only_boost_20260515.json"
    dump(report, payload)
    write_doc(ROOT / "docs/STWM_COTRACKER_OBJECT_DENSE_TEACHER_V16_M128_H32_V35_20_VIPSEG_ONLY_BOOST_20260515.md", payload)
    print(json.dumps({"报告": str(report.relative_to(ROOT)), "VIPSeg成功": dict(success_by_split)}, ensure_ascii=False), flush=True)
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
