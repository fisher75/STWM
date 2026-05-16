#!/usr/bin/env python3
"""V35.45 对 larger subset 重跑 raw-video frontend，M128/H32。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import run_cotracker_object_dense_teacher_v15c_20260502 as v15c
from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import compare_trace

MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_closure_subset/manifest.json"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_frontend_rerun/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_frontend_rerun_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_45_LARGER_RAW_VIDEO_FRONTEND_RERUN_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_45_larger_raw_video_frontend_rerun_20260516.log"


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


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--repo-path", default="baselines/repos/co-tracker")
    ap.add_argument("--checkpoint", default="baselines/checkpoints/cotracker/scaled_offline.pth")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    repo = ROOT / args.repo_path
    ckpt = ROOT / args.checkpoint
    if not repo.exists() or not ckpt.exists():
        raise RuntimeError(f"CoTracker repo/checkpoint 不存在：repo={repo}, ckpt={ckpt}")
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor  # type: ignore

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    log(f"开始 V35.45 raw frontend rerun：samples={len(samples)} device={device}")
    model = CoTrackerPredictor(checkpoint=str(ckpt), offline=True, window_len=60).to(device).eval()
    v15c.OUT_ROOT = OUT_ROOT
    split_map = v15c._mixed_split_map()
    run_args = SimpleNamespace(repo_path=str(repo), checkpoint=str(ckpt), m=128, horizon=32, obs_len=8, max_side=args.max_side)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        uid = sample["sample_uid"]
        cached = ROOT / sample["expected_cached_trace_path"]
        predecode = Path(sample["predecode_path"])
        log(f"重跑 [{idx}/{len(samples)}] {uid} split={sample['split']} dataset={sample['dataset']}")
        row, fail = v15c._run_clip(model, predecode, run_args, device, split_map)
        if row:
            rows.append(row)
            rerun_path = ROOT / row["cache_path"]
            drift = compare_trace(cached, rerun_path)
            drift.update(
                {
                    "sample_uid": uid,
                    "split": sample["split"],
                    "dataset": sample["dataset"],
                    "identity_provenance_type": sample["identity_provenance_type"],
                    "identity_claim_allowed": sample["identity_claim_allowed"],
                    "cached_trace": sample["expected_cached_trace_path"],
                    "rerun_trace": row["cache_path"],
                }
            )
            drift_rows.append(drift)
            log(f"完成 {uid} mean_delta={drift.get('mean_l2_trace_delta_px')} vis_agree={drift.get('visibility_agreement')}")
        if fail:
            failures.append({"sample_uid": uid, **fail})
            log(f"失败 {uid} reason={fail.get('reason')}")
    success_rate = len(rows) / max(len(samples), 1)
    trace_mean = float(np.mean([r.get("mean_l2_trace_delta_px", 0.0) for r in drift_rows])) if drift_rows else None
    trace_max = float(np.max([r.get("max_l2_trace_delta_px", 0.0) for r in drift_rows])) if drift_rows else None
    vis_mean = float(np.mean([r.get("visibility_agreement", 0.0) for r in drift_rows])) if drift_rows else None
    conf_delta = float(np.mean([r.get("confidence_mae", 0.0) for r in drift_rows])) if drift_rows else None
    motion_delta = float(np.mean([r.get("motion_mean_abs_delta", 0.0) for r in drift_rows])) if drift_rows else None
    frame_ok = bool(drift_rows and all(r.get("frame_paths_aligned", False) for r in drift_rows))
    shape_ok = bool(drift_rows and all(r.get("shape_match", False) for r in drift_rows))
    drift_ok = bool(shape_ok and frame_ok and (trace_mean or 0.0) <= 8.0 and (vis_mean or 0.0) >= 0.80)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_frontend_rerun_attempted": True,
        "raw_frontend_rerun_success_count": len(rows),
        "raw_frontend_rerun_fail_count": len(failures),
        "raw_frontend_rerun_success_rate": success_rate,
        "trace_drift_vs_cache_mean": trace_mean,
        "trace_drift_vs_cache_max": trace_max,
        "visibility_agreement_mean": vis_mean,
        "confidence_delta_mean": conf_delta,
        "motion_delta_mean": motion_delta,
        "frame_path_alignment_passed": frame_ok,
        "trace_shape_alignment_passed": shape_ok,
        "trace_drift_ok": drift_ok,
        "old_trace_cache_used_as_input_result": False,
        "old_trace_cache_used_for_comparison_only": True,
        "raw_video_frame_paths_rerun_used": True,
        "rerun_trace_root": rel(OUT_ROOT),
        "manifest_path": rel(manifest_path),
        "rerun_rows": rows,
        "drift_rows": drift_rows,
        "exact_failures": failures,
        "runtime_seconds": float(time.time() - t0),
        "中文结论": (
            f"V35.45 larger raw-video frontend rerun 完成：success_rate={success_rate:.3f}，trace_drift_ok={drift_ok}。"
            if success_rate >= 0.95 and drift_ok
            else "V35.45 frontend rerun 未通过 success/drift gate，需要修 frontend reproducibility。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.45 Larger Raw-Video Frontend Rerun\n\n"
        f"- raw_frontend_rerun_attempted: true\n"
        f"- raw_frontend_rerun_success_count: {len(rows)}\n"
        f"- raw_frontend_rerun_fail_count: {len(failures)}\n"
        f"- raw_frontend_rerun_success_rate: {success_rate}\n"
        f"- trace_drift_vs_cache_mean: {trace_mean}\n"
        f"- trace_drift_vs_cache_max: {trace_max}\n"
        f"- visibility_agreement_mean: {vis_mean}\n"
        f"- frame_path_alignment_passed: {frame_ok}\n"
        f"- trace_drift_ok: {drift_ok}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    log(f"V35.45 frontend rerun 完成 success_rate={success_rate:.3f} drift_ok={drift_ok}")
    print(json.dumps({"raw_frontend_rerun_done": True, "raw_frontend_rerun_success_rate": success_rate, "trace_drift_ok": drift_ok}, ensure_ascii=False), flush=True)
    return 0 if success_rate >= 0.95 and drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
