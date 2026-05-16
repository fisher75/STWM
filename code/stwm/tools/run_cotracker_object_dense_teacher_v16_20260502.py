#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM CoTracker Object-Dense Teacher V16", ""]
    for key in [
        "teacher_run_success",
        "combo",
        "teacher_source",
        "processed_clip_count",
        "skipped_existing_clip_count",
        "failed_clip_count",
        "object_count",
        "point_count",
        "valid_point_ratio",
        "runtime_seconds",
        "next_step_if_failed",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.extend(
        [
            "",
            "## 中文总结",
            (
                "本次只构建 video-derived object-dense trace cache；"
                "CoTracker 作为 video trace teacher 生成观测/未来 trace 监督，STWM 主模型输入仍限制为 observed trace。"
            ),
            f"- 已处理 clip 数: `{payload.get('processed_clip_count')}`",
            f"- 已跳过已有 cache 数: `{payload.get('skipped_existing_clip_count')}`",
            f"- 失败 clip 数: `{payload.get('failed_clip_count')}`",
            f"- 输出 cache: `{payload.get('cache_root')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _combo_root(m: int, h: int) -> Path:
    return OUT_BASE / f"M{m}_H{h}"


def _expected_cache_path(pre_path: Path, split_map: dict[str, str], root: Path) -> Path:
    key = v15c._norm_key(pre_path)
    split = split_map.get(key, pre_path.parent.name)
    return root / split / f"{key.replace('::', '__')}.npz"


def _summarize_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    tracks = np.asarray(z["tracks_xy"])
    vis = np.asarray(z["visibility"]).astype(bool)
    return {
        "item_key": str(_scalar(z["item_key"])),
        "split": str(_scalar(z["split"])),
        "dataset": str(_scalar(z["dataset"])),
        "cache_path": str(path.relative_to(ROOT)),
        "object_count": int(tracks.shape[0]),
        "point_count": int(tracks.shape[0] * tracks.shape[1]),
        "valid_point_ratio": float(vis.mean()) if vis.size else 0.0,
        "query_frame": int(_scalar(z["query_frame"])),
        "raw_frame_paths_available": True,
        "peak_gpu_memory_bytes": 0,
        "resumed_existing": True,
    }


def _target_counts(total: int, train: int | None, val: int | None, test: int | None) -> tuple[int, int, int]:
    if train is not None or val is not None or test is not None:
        tr = int(train or 0)
        va = int(val or 0)
        te = int(test or 0)
        return tr, va, te
    tr = int(round(total * 0.70))
    va = int(round(total * 0.15))
    te = max(0, total - tr - va)
    return tr, va, te


def main() -> int:
    v15c._apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default="baselines/repos/co-tracker")
    parser.add_argument("--checkpoint", default="baselines/checkpoints/cotracker/scaled_offline.pth")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--target-total", type=int, default=500)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidate-order", choices=["forward", "reverse"], default=os.environ.get("STWM_CANDIDATE_ORDER", "forward"))
    parser.add_argument("--worker-id", default=os.environ.get("STWM_WORKER_ID", "main"))
    args = parser.parse_args()
    max_train, max_val, max_test = _target_counts(args.target_total, args.max_train, args.max_val, args.max_test)
    args.max_train = max_train
    args.max_val = max_val
    args.max_test = max_test
    args.repo_path = str(ROOT / args.repo_path)
    args.checkpoint = str(ROOT / args.checkpoint)
    combo = f"M{args.m}_H{args.horizon}"
    out_root = _combo_root(args.m, args.horizon)
    v15c.OUT_ROOT = out_root
    t0 = time.time()
    repo = Path(args.repo_path)
    ckpt = Path(args.checkpoint)
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    skipped_existing = 0
    if not repo.exists() or not ckpt.exists():
        payload = {
            "teacher_run_success": False,
            "combo": combo,
            "teacher_source": "missing",
            "official_repo_path": str(repo),
            "checkpoint_path": str(ckpt),
            "blocker": "cotracker_repo_or_checkpoint_missing",
            "next_step_if_failed": "fix_cotracker_integration",
        }
        _dump(ROOT / f"reports/stwm_cotracker_object_dense_teacher_v16_{combo}_20260502.json", payload)
        return 1
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor  # type: ignore

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = CoTrackerPredictor(checkpoint=str(ckpt), offline=True, window_len=60).to(device).eval()
    split_map = v15c._mixed_split_map()
    selected = v15c._select_items(max_train, max_val, max_test, split_map)
    if args.candidate_order == "reverse":
        selected = list(reversed(selected))
    target_quotas = {"train": max_train, "val": max_val, "test": max_test}
    success_by_split: Counter[str] = Counter()
    for idx, pre_path in enumerate(selected):
        split_name = split_map.get(v15c._norm_key(pre_path), pre_path.parent.name)
        if success_by_split[split_name] >= target_quotas.get(split_name, 0):
            continue
        if all(success_by_split[s] >= q for s, q in target_quotas.items()):
            break
        expected = _expected_cache_path(pre_path, split_map, out_root)
        existing = None if args.force else _summarize_existing(expected)
        if existing is not None:
            rows.append(existing)
            success_by_split[existing["split"]] += 1
            skipped_existing += 1
            print(f"[{idx+1}/{len(selected)}] 已恢复={len(rows)} 失败={len(failures)} {pre_path.name}", flush=True)
            continue
        row, fail = v15c._run_clip(model, pre_path, args, device, split_map)
        if row:
            rows.append(row)
            success_by_split[row["split"]] += 1
        if fail:
            failures.append(fail)
        print(f"[{idx+1}/{len(selected)}] 成功={len(rows)} 失败={len(failures)} {pre_path.name}", flush=True)
    split_counts = Counter(row["split"] for row in rows)
    dataset_counts = Counter(row["dataset"] for row in rows)
    point_count = int(sum(row["point_count"] for row in rows))
    object_count = int(sum(row["object_count"] for row in rows))
    payload = {
        "teacher_run_success": bool(rows),
        "combo": combo,
        "M": args.m,
        "horizon": args.horizon,
        "obs_len": args.obs_len,
        "teacher_source": "cotracker_official" if rows else "none",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo_path": str(repo),
        "official_commit_hash": v15c._repo_commit(repo),
        "checkpoint_path": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "exact_command": " ".join(sys.argv),
        "worker_id": args.worker_id,
        "candidate_order": args.candidate_order,
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "runtime_seconds": float(time.time() - t0),
        "processed_clip_count": len(rows),
        "skipped_existing_clip_count": skipped_existing,
        "failed_clip_count": len(failures),
        "requested_split_counts": target_quotas,
        "processed_split_counts": dict(split_counts),
        "processed_dataset_counts": dict(dataset_counts),
        "object_count": object_count,
        "point_count": point_count,
        "valid_point_ratio": float(np.mean([row["valid_point_ratio"] for row in rows])) if rows else 0.0,
        "average_points_per_object": float(point_count / max(object_count, 1)),
        "average_points_per_scene": float(point_count / max(len(rows), 1)),
        "peak_memory_bytes_max": int(max([row.get("peak_gpu_memory_bytes", 0) for row in rows], default=0)),
        "cache_root": str(out_root.relative_to(ROOT)),
        "clip_caches": rows,
        "failed_clips": failures,
        "failure_reason_top": dict(Counter(str(f["reason"]) for f in failures).most_common(20)),
        "no_future_box_projection": True,
        "teacher_uses_full_obs_future_clip_as_target": True,
        "stwm_input_restricted_to_observed": True,
        "next_step_if_failed": "fix_window_selection" if len(rows) < args.target_total else None,
    }
    suffix = "" if args.worker_id == "main" else f"_{args.worker_id}"
    report = ROOT / f"reports/stwm_cotracker_object_dense_teacher_v16_{combo}{suffix}_20260502.json"
    _dump(report, payload)
    _write_doc(ROOT / f"docs/STWM_COTRACKER_OBJECT_DENSE_TEACHER_V16_{combo}_20260502.md", payload)
    # Also write/update a lightweight aggregate index for running jobs.
    _dump(ROOT / "reports/stwm_cotracker_object_dense_teacher_v16_20260502.json", {"latest_combo_report": str(report.relative_to(ROOT)), **payload})
    print(f"报告路径: {report.relative_to(ROOT)}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
