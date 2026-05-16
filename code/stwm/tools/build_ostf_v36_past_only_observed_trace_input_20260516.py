#!/usr/bin/env python3
"""V36: 从 V35.49 full-clip trace 中只取 observed 段作为 past-only 输入。"""
from __future__ import annotations

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

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
RERUN_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_past_only_observed_trace_input/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v36_past_only_observed_trace_input_build_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_PAST_ONLY_OBSERVED_TRACE_INPUT_BUILD_20260516.md"


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


def sample_path(sample: dict[str, Any]) -> Path:
    if sample.get("expected_rerun_trace_path"):
        p = ROOT / str(sample["expected_rerun_trace_path"])
        if p.exists():
            return p
    return RERUN_ROOT / str(sample["split"]) / (Path(str(sample.get("source_unified_npz", sample["sample_uid"]))).stem + ".npz")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    for s in samples:
        p = sample_path(s)
        if not p.exists():
            failures.append({"sample_uid": s.get("sample_uid"), "reason": "rerun_trace_npz_missing", "path": rel(p)})
            continue
        z = np.load(p, allow_pickle=True)
        tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
        vis = np.asarray(z["visibility"], dtype=bool)
        conf = np.asarray(z["confidence"], dtype=np.float32)
        obs_len = int(scalar(z, "obs_len", 8))
        horizon = int(scalar(z, "horizon", 32))
        point_n = int(tracks.shape[0] * tracks.shape[1])
        out_dir = OUT_ROOT / str(s["split"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / p.name
        frame_paths = np.asarray(z["frame_paths"], dtype=object)
        payload = {
            "sample_uid": np.asarray(str(s["sample_uid"])),
            "dataset": np.asarray(str(s["dataset"])),
            "split": np.asarray(str(s["split"])),
            "point_id": np.arange(point_n, dtype=np.int64),
            "object_id": np.repeat(np.asarray(z["object_id"], dtype=np.int64), tracks.shape[1]).astype(np.int64) if "object_id" in z.files else np.zeros(point_n, dtype=np.int64),
            "semantic_id": np.repeat(np.asarray(z["semantic_id"], dtype=np.int64), tracks.shape[1]).astype(np.int64) if "semantic_id" in z.files else np.full(point_n, -1, dtype=np.int64),
            "obs_points": tracks[:, :, :obs_len].reshape(point_n, obs_len, 2).astype(np.float32),
            "obs_vis": vis[:, :, :obs_len].reshape(point_n, obs_len).astype(bool),
            "obs_conf": conf[:, :, :obs_len].reshape(point_n, obs_len).astype(np.float32),
            "future_trace_teacher_points": tracks[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon, 2).astype(np.float32),
            "future_trace_teacher_vis": vis[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon).astype(bool),
            "future_trace_teacher_conf": conf[:, :, obs_len : obs_len + horizon].reshape(point_n, horizon).astype(np.float32),
            "raw_frame_paths_obs_only": frame_paths[:obs_len],
            "raw_frame_paths_future_for_target_only": frame_paths[obs_len : obs_len + horizon],
            "source_v35_49_full_clip_trace_npz": np.asarray(rel(p)),
            "future_trace_teacher_input_allowed": np.asarray(False),
            "leakage_safe": np.asarray(True),
            "teacher_uses_full_obs_future_clip_as_target": np.asarray(bool(scalar(z, "teacher_uses_full_obs_future_clip_as_target", True))),
        }
        np.savez_compressed(out_path, **payload)
        split_counts[str(s["split"])] += 1
        rows.append({"sample_uid": s["sample_uid"], "split": s["split"], "output_path": rel(out_path), "point_count": point_n})

    ok = bool(rows and not failures)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "past_only_observed_trace_input_build_done": True,
        "sample_count": len(rows),
        "manifest_sample_count": len(samples),
        "split_counts": dict(split_counts),
        "obs_only_input_built": ok,
        "future_trace_teacher_target_available": ok,
        "future_trace_teacher_input_allowed": False,
        "leakage_safe": True,
        "output_root": rel(OUT_ROOT),
        "rows": rows,
        "exact_blockers": failures,
        "中文总结": (
            "已从 V35.49 full-clip teacher trace 中只抽取 observed 段作为 V36 输入；future trace 仅保留为 target/upper-bound 对比，不允许作为模型输入。"
            if ok
            else "V36 past-only 输入构建存在缺失样本；需要先补齐 rerun trace cache。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36 Past-Only Observed Trace Input Build\n\n"
        f"- sample_count: {len(rows)}\n"
        f"- obs_only_input_built: {ok}\n"
        f"- future_trace_teacher_target_available: {ok}\n"
        "- future_trace_teacher_input_allowed: false\n"
        "- leakage_safe: true\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"past_only输入构建完成": ok, "样本数": len(rows), "失败数": len(failures)}, ensure_ascii=False), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
