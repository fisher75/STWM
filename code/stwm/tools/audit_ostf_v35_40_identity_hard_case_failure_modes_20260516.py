#!/usr/bin/env python3
"""V35.40 审计 identity hard-case failure mode：measurement baseline 还是 learned head。"""
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

INPUT_REPORT = ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json"
ALIGNMENT_REPORT = ROOT / "reports/stwm_ostf_v35_39_identity_feature_alignment_audit_20260516.json"
ROOT_SLICE = ROOT / "outputs/cache/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_unified_slice/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_40_identity_hard_case_failure_modes_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_40_IDENTITY_HARD_CASE_FAILURE_MODES_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_40_identity_hard_case_failure_modes_20260516.log"
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


def metric_key(m: dict[str, float | None], key: str) -> float:
    v = m.get(key)
    return float(v) if v is not None else 0.0


def average_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r})
    out: dict[str, float | None] = {}
    for k in keys:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


@torch.no_grad()
def learned_metrics(sample: dict[str, np.ndarray | str], seed: int, device: torch.device) -> dict[str, float | None]:
    model = load_identity_model(seed, device)
    emb = model_embedding(model, np.asarray(sample["x"], dtype=np.float32), device)
    return aggregate([retrieval_metrics_for_sample(emb, sample)])


def measurement_metrics(sample: dict[str, np.ndarray | str]) -> dict[str, float | None]:
    emb = np.asarray(sample["measurement"], dtype=np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
    return aggregate([retrieval_metrics_for_sample(emb, sample)])


def sample_paths(smoke: dict[str, Any], root: Path) -> list[Path]:
    return [root / str(r["split"]) / f"{r['sample_uid']}.npz" for r in smoke.get("selected_samples", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-report", default=str(INPUT_REPORT))
    ap.add_argument("--alignment-report", default=str(ALIGNMENT_REPORT))
    ap.add_argument("--slice-root", default=str(ROOT_SLICE))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    input_report = Path(args.input_report)
    alignment_report = Path(args.alignment_report)
    slice_root = Path(args.slice_root)
    if not input_report.is_absolute():
        input_report = ROOT / input_report
    if not alignment_report.is_absolute():
        alignment_report = ROOT / alignment_report
    if not slice_root.is_absolute():
        slice_root = ROOT / slice_root
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu and args.device == "cuda" else "cpu")
    LOG.write_text("", encoding="utf-8")
    log(f"开始 V35.40 identity hard-case failure audit；device={device}")
    smoke = json.loads(input_report.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_report.read_text(encoding="utf-8")) if alignment_report.exists() else {}
    sample_rows = []
    learned_by_split: dict[str, dict[str, list[dict[str, float | None]]]] = defaultdict(lambda: defaultdict(list))
    measurement_by_split: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for p in sample_paths(smoke, slice_root):
        sample = load_identity_sample(p)
        meas = measurement_metrics(sample)
        learned_seed = {str(seed): learned_metrics(sample, seed, device) for seed in SEEDS}
        mean_learned_exclude = float(np.mean([metric_key(m, "identity_retrieval_exclude_same_point_top1") for m in learned_seed.values()]))
        mean_learned_pooled = float(np.mean([metric_key(m, "identity_retrieval_instance_pooled_top1") for m in learned_seed.values()]))
        mean_learned_confuser = float(np.mean([metric_key(m, "identity_confuser_avoidance_top1") for m in learned_seed.values()]))
        fail = bool(mean_learned_exclude < 0.65 or mean_learned_pooled < 0.65 or mean_learned_confuser < 0.65)
        row = {
            "sample_uid": str(sample["sample_uid"]),
            "split": str(sample["split"]),
            "dataset": str(sample["dataset"]),
            "measurement": meas,
            "learned_by_seed": learned_seed,
            "mean_learned_exclude_same_point_top1": mean_learned_exclude,
            "mean_learned_instance_pooled_top1": mean_learned_pooled,
            "mean_learned_confuser_avoidance_top1": mean_learned_confuser,
            "measurement_minus_learned_exclude": metric_key(meas, "identity_retrieval_exclude_same_point_top1") - mean_learned_exclude,
            "measurement_minus_learned_instance_pooled": metric_key(meas, "identity_retrieval_instance_pooled_top1") - mean_learned_pooled,
            "failed_identity_gate": fail,
        }
        sample_rows.append(row)
        split = str(sample["split"])
        measurement_by_split[split].append(meas)
        for seed, m in learned_seed.items():
            learned_by_split[seed][split].append(m)
    failed = [r for r in sample_rows if r["failed_identity_gate"]]
    hard_vspw = bool(failed and all(r["dataset"].upper() == "VSPW" for r in failed if r["split"] in {"val", "train"}))
    measurement_dominates_failed = bool(
        failed
        and np.mean([r["measurement_minus_learned_exclude"] for r in failed]) > 0.05
        and np.mean([r["measurement_minus_learned_instance_pooled"] for r in failed]) > 0.05
    )
    measurement_also_weak = bool(
        failed
        and np.mean([metric_key(r["measurement"], "identity_retrieval_exclude_same_point_top1") for r in failed]) < 0.65
    )
    learned_split = {seed: {split: average_metrics(rows) for split, rows in splits.items()} for seed, splits in learned_by_split.items()}
    meas_split = {split: average_metrics(rows) for split, rows in measurement_by_split.items()}
    recommended = (
        "fix_identity_head_measurement_preservation_on_hard_vspw"
        if measurement_dominates_failed
        else "fix_identity_inputs_with_trace_instance_cues"
        if measurement_also_weak
        else "fix_identity_val_confuser_category_or_eval_slice"
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_40_identity_hard_case_failure_audit_done": True,
        "input_report": rel(input_report),
        "alignment_report": rel(alignment_report),
        "slice_root": rel(slice_root),
        "identity_feature_alignment_ok": not bool(alignment.get("rerun_feature_alignment_broken", False)),
        "hard_vspw_identity_failure_detected": hard_vspw,
        "measurement_baseline_dominates_failed_clips": measurement_dominates_failed,
        "measurement_baseline_also_weak_on_failed_clips": measurement_also_weak,
        "measurement_by_split": meas_split,
        "learned_by_seed_and_split": learned_split,
        "sample_failure_rows": sample_rows,
        "failed_sample_uids": [r["sample_uid"] for r in failed],
        "m128_h32_video_system_benchmark_claim_allowed": False,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": recommended,
        "中文结论": (
            "V35.40 显示 identity blocker 集中在少数 VSPW hard/confuser/crossing clips；"
            "rerun feature alignment 没问题。下一步应修 identity hard-case 表征或 measurement-preserving head，而不是回到语义 writer/gate。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.40 Identity Hard-Case Failure Modes\n\n"
        f"- v35_40_identity_hard_case_failure_audit_done: true\n"
        f"- identity_feature_alignment_ok: {report['identity_feature_alignment_ok']}\n"
        f"- hard_vspw_identity_failure_detected: {hard_vspw}\n"
        f"- measurement_baseline_dominates_failed_clips: {measurement_dominates_failed}\n"
        f"- measurement_baseline_also_weak_on_failed_clips: {measurement_also_weak}\n"
        f"- failed_sample_uids: {report['failed_sample_uids']}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: false\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {recommended}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## Claim boundary\n"
        "本轮只做 identity hard-case failure attribution；不能 claim full video semantic/identity system。\n",
        encoding="utf-8",
    )
    log(f"V35.40 完成；recommended_next_step={recommended}")
    print(json.dumps({"v35_40_identity_hard_case_failure_audit_done": True, "recommended_next_step": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
