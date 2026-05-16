#!/usr/bin/env python3
"""V35.41 评估 trace-instance cues 是否能补 identity hard retrieval。"""
from __future__ import annotations

import argparse
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

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import load_identity_sample
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import aggregate, retrieval_metrics_for_sample

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_unified_slice/M128_H32"
V35_38_REPORT = ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json"
V35_40_REPORT = ROOT / "reports/stwm_ostf_v35_40_identity_hard_case_failure_modes_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_41_trace_instance_cue_identity_probe_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_41_TRACE_INSTANCE_CUE_IDENTITY_PROBE_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_41_trace_instance_cue_identity_probe_20260516.log"


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


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-4)


def trace_signature(sample: dict[str, np.ndarray | str]) -> np.ndarray:
    obs = np.asarray(sample["obs_points"], dtype=np.float32)
    fut = np.asarray(sample["future_points"], dtype=np.float32)
    pts = np.concatenate([obs, fut], axis=1)
    center = obs[:, -1:, :]
    rel = (pts - center) / 512.0
    vel = np.diff(pts, axis=1) / 64.0
    obs_vis = np.ones((obs.shape[0], obs.shape[1]), dtype=np.float32)
    fut_vis = np.ones((fut.shape[0], fut.shape[1]), dtype=np.float32)
    if "obs_vis" in sample:
        obs_vis = np.asarray(sample["obs_vis"], dtype=np.float32)
    if "future_vis" in sample:
        fut_vis = np.asarray(sample["future_vis"], dtype=np.float32)
    vis = np.concatenate([obs_vis, fut_vis], axis=1)
    feat = np.concatenate(
        [
            rel.reshape(rel.shape[0], -1),
            vel.reshape(vel.shape[0], -1),
            vis,
            (obs[:, -1] / 512.0),
            ((fut[:, -1] - obs[:, -1]) / 512.0),
        ],
        axis=1,
    ).astype(np.float32)
    return norm(standardize(feat))


def fused_embedding(sample: dict[str, np.ndarray | str], trace_weight: float) -> np.ndarray:
    meas = np.asarray(sample["measurement"], dtype=np.float32)
    meas = norm(meas)
    tr = trace_signature(sample)
    return norm(np.concatenate([meas, trace_weight * tr], axis=1).astype(np.float32))


def method_embeddings(sample: dict[str, np.ndarray | str]) -> dict[str, np.ndarray]:
    out = {
        "measurement_only": norm(np.asarray(sample["measurement"], dtype=np.float32)),
        "trace_shape_only": trace_signature(sample),
    }
    for w in [0.10, 0.25, 0.50, 1.00, 2.00, 4.00]:
        out[f"measurement_trace_fused_w{w:g}"] = fused_embedding(sample, w)
    return out


def sample_paths(smoke: dict[str, Any], root: Path) -> list[Path]:
    return [root / str(r["split"]) / f"{r['sample_uid']}.npz" for r in smoke.get("selected_samples", [])]


def eval_methods(samples: list[dict[str, np.ndarray | str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_by_method_split: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    sample_rows: list[dict[str, Any]] = []
    for s in samples:
        method_rows = {}
        for name, emb in method_embeddings(s).items():
            m = aggregate([retrieval_metrics_for_sample(emb, s)])
            method_rows[name] = m
            rows_by_method_split[name][str(s["split"])].append(m)
        sample_rows.append({"sample_uid": str(s["sample_uid"]), "split": str(s["split"]), "dataset": str(s["dataset"]), "methods": method_rows})
    split_metrics = {
        method: {split: average_metrics(rows) for split, rows in split_rows.items()}
        for method, split_rows in rows_by_method_split.items()
    }
    return split_metrics, sample_rows


def average_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    keys = sorted({k for r in rows for k in r})
    out: dict[str, float | None] = {}
    for k in keys:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def pass_identity(m: dict[str, float | None]) -> bool:
    return bool(
        (m.get("identity_retrieval_exclude_same_point_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_instance_pooled_top1") or 0.0) >= 0.65
        and (m.get("identity_confuser_avoidance_top1") or 0.0) >= 0.65
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-root", default=str(SLICE_ROOT))
    ap.add_argument("--v35-38-report", default=str(V35_38_REPORT))
    ap.add_argument("--v35-40-report", default=str(V35_40_REPORT))
    args = ap.parse_args()
    root = Path(args.slice_root)
    if not root.is_absolute():
        root = ROOT / root
    smoke_path = Path(args.v35_38_report)
    if not smoke_path.is_absolute():
        smoke_path = ROOT / smoke_path
    v35_40_path = Path(args.v35_40_report)
    if not v35_40_path.is_absolute():
        v35_40_path = ROOT / v35_40_path
    LOG.write_text("", encoding="utf-8")
    log("开始 V35.41 trace-instance cue identity probe。")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    v35_40 = json.loads(v35_40_path.read_text(encoding="utf-8")) if v35_40_path.exists() else {}
    samples = [load_identity_sample(p) for p in sample_paths(smoke, root)]
    split_metrics, sample_rows = eval_methods(samples)
    best_by_val = max(
        split_metrics,
        key=lambda name: (
            split_metrics[name].get("val", {}).get("identity_retrieval_instance_pooled_top1") or 0.0,
            split_metrics[name].get("val", {}).get("identity_retrieval_exclude_same_point_top1") or 0.0,
        ),
    )
    best_val = split_metrics[best_by_val].get("val", {})
    best_test = split_metrics[best_by_val].get("test", {})
    measurement_val = split_metrics["measurement_only"].get("val", {})
    measurement_test = split_metrics["measurement_only"].get("test", {})
    trace_instance_cues_help = bool(
        (best_val.get("identity_retrieval_instance_pooled_top1") or 0.0) >= (measurement_val.get("identity_retrieval_instance_pooled_top1") or 0.0) + 0.05
        and (best_test.get("identity_retrieval_instance_pooled_top1") or 0.0) >= (measurement_test.get("identity_retrieval_instance_pooled_top1") or 0.0) - 0.03
    )
    best_passes_val_test = bool(pass_identity(best_val) and pass_identity(best_test))
    recommended = "train_trace_augmented_identity_retrieval_head" if trace_instance_cues_help else "fix_identity_targets_or_video_instance_supervision"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_41_trace_instance_cue_identity_probe_done": True,
        "slice_root": rel(root),
        "v35_38_report": rel(smoke_path),
        "v35_40_report": rel(v35_40_path),
        "identity_feature_alignment_ok": bool(v35_40.get("identity_feature_alignment_ok", False)),
        "method_split_metrics": split_metrics,
        "sample_rows": sample_rows,
        "best_trace_instance_method_by_val": best_by_val,
        "trace_instance_cues_help_identity": trace_instance_cues_help,
        "best_trace_instance_method_passes_val_test": best_passes_val_test,
        "m128_h32_video_system_benchmark_claim_allowed": False,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": recommended,
        "中文结论": (
            "V35.41 完成 trace-instance cue probe；如果融合 trace shape 的方法能稳定超过 measurement baseline，下一步才训练 trace-augmented identity retrieval head。"
            if trace_instance_cues_help
            else "V35.41 显示简单 trace-shape 融合还不足以修复 hard identity；下一步应回到 identity target / video instance supervision，而不是继续调 semantic head。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.41 Trace-Instance Cue Identity Probe\n\n"
        f"- v35_41_trace_instance_cue_identity_probe_done: true\n"
        f"- best_trace_instance_method_by_val: {best_by_val}\n"
        f"- trace_instance_cues_help_identity: {trace_instance_cues_help}\n"
        f"- best_trace_instance_method_passes_val_test: {best_passes_val_test}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: false\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## Claim boundary\n"
        "本轮是 identity 输入上界 probe；未训练新模型，不能开放完整系统 claim。\n",
        encoding="utf-8",
    )
    log(f"V35.41 完成；recommended_next_step={recommended}")
    print(json.dumps({"v35_41_trace_instance_cue_identity_probe_done": True, "recommended_next_step": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
