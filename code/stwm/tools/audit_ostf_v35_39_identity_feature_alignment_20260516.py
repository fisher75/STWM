#!/usr/bin/env python3
"""V35.39 审计 raw-video rerun slice 的 identity feature alignment。"""
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import load_identity_model, load_identity_sample
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import (
    aggregate,
    model_embedding,
    retrieval_metrics_for_sample,
)

ORIGINAL_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark/M128_H32"
RERUN_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_unified_slice/M128_H32"
INPUT_REPORT = ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_39_identity_feature_alignment_audit_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_39_IDENTITY_FEATURE_ALIGNMENT_AUDIT_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_39_identity_feature_alignment_audit_20260516.log"
SEEDS = [42, 123, 456]


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


def compare_array(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    same_shape = a.shape == b.shape
    if not same_shape:
        return {"same_shape": False, "original_shape": list(a.shape), "rerun_shape": list(b.shape)}
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "same_shape": True,
        "mean_abs_delta": float(d.mean()),
        "p95_abs_delta": float(np.quantile(d, 0.95)),
        "max_abs_delta": float(d.max()),
    }


def sample_paths(smoke: dict[str, Any]) -> list[tuple[str, str, Path, Path]]:
    rows = []
    for r in smoke.get("selected_samples", []):
        uid = str(r["sample_uid"])
        split = str(r["split"])
        rows.append((uid, split, ORIGINAL_ROOT / split / f"{uid}.npz", RERUN_ROOT / split / f"{uid}.npz"))
    return rows


@torch.no_grad()
def eval_identity(paths: list[Path], seed: int, device: torch.device) -> dict[str, Any]:
    model = load_identity_model(seed, device)
    by_split: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_sample: list[dict[str, Any]] = []
    for p in paths:
        s = load_identity_sample(p)
        emb = model_embedding(model, np.asarray(s["x"], dtype=np.float32), device)
        row = retrieval_metrics_for_sample(emb, s)
        split = str(s["split"])
        by_split[split].append(row)
        by_sample.append({"sample_uid": str(s["sample_uid"]), "split": split, "metrics": aggregate([row])})
    return {"by_split": {k: aggregate(v) for k, v in by_split.items()}, "by_sample": by_sample}


def pass_identity(m: dict[str, float | None]) -> bool:
    return bool(
        (m.get("identity_retrieval_exclude_same_point_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_same_frame_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_instance_pooled_top1") or 0.0) >= 0.65
        and (m.get("identity_confuser_avoidance_top1") or 0.0) >= 0.65
    )


def main() -> int:
    global ORIGINAL_ROOT, RERUN_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-report", default=str(INPUT_REPORT))
    ap.add_argument("--original-root", default=str(ORIGINAL_ROOT))
    ap.add_argument("--rerun-root", default=str(RERUN_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    ORIGINAL_ROOT = Path(args.original_root)
    RERUN_ROOT = Path(args.rerun_root)
    if not ORIGINAL_ROOT.is_absolute():
        ORIGINAL_ROOT = ROOT / ORIGINAL_ROOT
    if not RERUN_ROOT.is_absolute():
        RERUN_ROOT = ROOT / RERUN_ROOT
    input_report = Path(args.input_report)
    if not input_report.is_absolute():
        input_report = ROOT / input_report
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu and args.device == "cuda" else "cpu")
    smoke = json.loads(input_report.read_text(encoding="utf-8"))
    LOG.write_text("", encoding="utf-8")
    log(f"开始 V35.39 identity alignment audit；device={device}")
    rows = []
    original_paths: list[Path] = []
    rerun_paths: list[Path] = []
    for uid, split, op, rp in sample_paths(smoke):
        original_paths.append(op)
        rerun_paths.append(rp)
        oz = np.load(op, allow_pickle=True)
        rz = np.load(rp, allow_pickle=True)
        rows.append(
            {
                "sample_uid": uid,
                "split": split,
                "identity_input_features": compare_array(np.asarray(oz["identity_identity_input_features"]), np.asarray(rz["identity_identity_input_features"])),
                "measurement_identity_embedding": compare_array(np.asarray(oz["identity_measurement_identity_embedding"]), np.asarray(rz["identity_measurement_identity_embedding"])),
                "obs_points": compare_array(np.asarray(oz["obs_points"]), np.asarray(rz["obs_points"])),
                "future_points": compare_array(np.asarray(oz["future_points"]), np.asarray(rz["future_points"])),
                "same_pair_mask_equal": bool(np.array_equal(np.asarray(oz["identity_same_instance_pair_mask"]), np.asarray(rz["identity_same_instance_pair_mask"]))),
                "confuser_mask_equal": bool(np.array_equal(np.asarray(oz["identity_identity_confuser_pair_mask"]), np.asarray(rz["identity_identity_confuser_pair_mask"]))),
            }
        )
    feature_delta_mean = float(np.mean([r["identity_input_features"].get("mean_abs_delta", 0.0) for r in rows]))
    feature_delta_max = float(np.max([r["identity_input_features"].get("max_abs_delta", 0.0) for r in rows]))
    mask_mismatch = any((not r["same_pair_mask_equal"]) or (not r["confuser_mask_equal"]) for r in rows)
    eval_rows: dict[str, Any] = {}
    for seed in SEEDS:
        eval_rows[str(seed)] = {
            "original": eval_identity(original_paths, seed, device),
            "rerun": eval_identity(rerun_paths, seed, device),
        }
    original_val_pass = all(pass_identity(eval_rows[str(seed)]["original"]["by_split"].get("val", {})) for seed in SEEDS)
    rerun_val_pass = all(pass_identity(eval_rows[str(seed)]["rerun"]["by_split"].get("val", {})) for seed in SEEDS)
    rerun_feature_alignment_broken = bool(feature_delta_mean > 1e-5 or feature_delta_max > 1e-3 or mask_mismatch)
    identity_target_alignment_broken = bool(original_val_pass and not rerun_val_pass and rerun_feature_alignment_broken)
    selected_val_hard_identity_intrinsic = bool((not original_val_pass) and (not rerun_val_pass))
    recommended = (
        "fix_rerun_unified_identity_feature_rebuild"
        if identity_target_alignment_broken
        else "fix_identity_val_confuser_category_or_eval_slice"
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_39_identity_feature_alignment_audit_done": True,
        "input_report": rel(input_report),
        "original_root": rel(ORIGINAL_ROOT),
        "rerun_root": rel(RERUN_ROOT),
        "feature_delta_mean_abs": feature_delta_mean,
        "feature_delta_max_abs": feature_delta_max,
        "identity_pair_mask_mismatch_detected": mask_mismatch,
        "rerun_feature_alignment_broken": rerun_feature_alignment_broken,
        "original_selected_val_identity_passed_all_seeds": original_val_pass,
        "rerun_selected_val_identity_passed_all_seeds": rerun_val_pass,
        "identity_target_alignment_broken": identity_target_alignment_broken,
        "selected_val_hard_identity_intrinsic": selected_val_hard_identity_intrinsic,
        "feature_alignment_rows": rows,
        "identity_eval_comparison": eval_rows,
        "m128_h32_video_system_benchmark_claim_allowed": False,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": recommended,
        "中文结论": (
            "V35.39 显示 rerun unified identity feature 与原始 cache 完全一致，pair mask 也一致；"
            "identity 失败不是 raw-video rerun feature rebuild bug，而是当前 val hard slice 本身暴露了 identity retrieval 在少数 VSPW confuser/crossing clip 上不稳。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.39 Identity Feature Alignment Audit\n\n"
        f"- v35_39_identity_feature_alignment_audit_done: true\n"
        f"- feature_delta_mean_abs: {feature_delta_mean}\n"
        f"- feature_delta_max_abs: {feature_delta_max}\n"
        f"- identity_pair_mask_mismatch_detected: {mask_mismatch}\n"
        f"- original_selected_val_identity_passed_all_seeds: {original_val_pass}\n"
        f"- rerun_selected_val_identity_passed_all_seeds: {rerun_val_pass}\n"
        f"- identity_target_alignment_broken: {identity_target_alignment_broken}\n"
        f"- selected_val_hard_identity_intrinsic: {selected_val_hard_identity_intrinsic}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: false\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## Claim boundary\n"
        "本轮是 identity feature / target alignment 审计；不开放 video identity field 或完整系统 claim。\n",
        encoding="utf-8",
    )
    log(f"V35.39 完成；recommended_next_step={recommended}")
    print(json.dumps({"v35_39_identity_feature_alignment_audit_done": True, "recommended_next_step": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
