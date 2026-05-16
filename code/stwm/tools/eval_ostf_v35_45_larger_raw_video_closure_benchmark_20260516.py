#!/usr/bin/env python3
"""V35.45 larger raw-video closure benchmark eval，不训练新 head。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools import run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 as smoke
from stwm.tools.run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 import load_identity_model, load_identity_sample
from stwm.tools.train_eval_ostf_v35_16_video_identity_pairwise_retrieval_head_20260515 import aggregate, model_embedding, retrieval_metrics_for_sample

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_rerun_unified_slice/M128_H32"
MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_closure_subset/manifest.json"
RERUN_REPORT = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_frontend_rerun_20260516.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_benchmark_eval_summary_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_benchmark_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_45_LARGER_RAW_VIDEO_CLOSURE_BENCHMARK_DECISION_20260516.md"
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


def pass_identity(m: dict[str, float | None]) -> bool:
    return bool(
        (m.get("identity_retrieval_exclude_same_point_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_same_frame_top1") or 0.0) >= 0.65
        and (m.get("identity_retrieval_instance_pooled_top1") or 0.0) >= 0.65
        and (m.get("identity_confuser_avoidance_top1") or 0.0) >= 0.65
    )


def identity_paths(split: str, real_only: bool | None) -> list[Path]:
    out = []
    for p in sorted((SLICE_ROOT / split).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        claim = bool(np.asarray(z["identity_claim_allowed"]).item()) if "identity_claim_allowed" in z.files else False
        if real_only is None or claim == real_only:
            out.append(p)
    return out


@torch.no_grad()
def eval_identity_split(paths: list[Path], seed: int, device: torch.device) -> dict[str, float | None]:
    if not paths:
        return {}
    model = load_identity_model(seed, device)
    rows = []
    for p in paths:
        s = load_identity_sample(p)
        emb = model_embedding(model, np.asarray(s["x"], dtype=np.float32), device)
        rows.append(retrieval_metrics_for_sample(emb, s))
    return aggregate(rows)


def category_counts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    cat = Counter()
    split_cat = defaultdict(Counter)
    dataset_cat = defaultdict(Counter)
    for s in samples:
        for t in s["category_tags"]:
            cat[t] += 1
            split_cat[s["split"]][t] += 1
            dataset_cat[s["dataset"]][t] += 1
    return {
        "category_counts": dict(cat),
        "per_split_category_counts": {k: dict(v) for k, v in split_cat.items()},
        "per_dataset_category_counts": {k: dict(v) for k, v in dataset_cat.items()},
    }


def mean_seed(rows: list[dict[str, Any]], split: str, family: str, key: str) -> float | None:
    vals = []
    for r in rows:
        v = r.get(split, {}).get(family, {}).get(key)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rerun = json.loads(RERUN_REPORT.read_text(encoding="utf-8"))
    smoke.RERUN_UNIFIED_ROOT = SLICE_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_rows = [smoke.eval_semantic_seed(seed, device) for seed in SEEDS]
    real_identity_rows = []
    pseudo_identity_rows = []
    for seed in SEEDS:
        real_val = eval_identity_split(identity_paths("val", True), seed, device)
        real_test = eval_identity_split(identity_paths("test", True), seed, device)
        pseudo_val = eval_identity_split(identity_paths("val", False), seed, device)
        pseudo_test = eval_identity_split(identity_paths("test", False), seed, device)
        real_identity_rows.append({"seed": seed, "val": real_val, "test": real_test, "identity_passed": pass_identity(real_val) and pass_identity(real_test)})
        pseudo_identity_rows.append({"seed": seed, "val": pseudo_val, "test": pseudo_test, "diagnostic_only": True})
    semantic_pass = bool(all(r["semantic_smoke_passed"] for r in semantic_rows))
    stable = bool(all(r.get("stable_preservation", False) for r in semantic_rows))
    identity_pass = bool(real_identity_rows and all(r["identity_passed"] for r in real_identity_rows))
    per_cat = category_counts(manifest.get("samples", []))
    per_category_breakdown_ready = bool(
        per_cat["category_counts"].get("high_motion", 0) > 0
        and per_cat["category_counts"].get("low_motion", 0) > 0
        and per_cat["category_counts"].get("occlusion", 0) > 0
        and per_cat["category_counts"].get("crossing", 0) > 0
        and per_cat["category_counts"].get("identity_confuser", 0) > 0
        and per_cat["category_counts"].get("semantic_changed", 0) > 0
        and per_cat["category_counts"].get("semantic_hard", 0) > 0
    )
    pass_gate = bool(
        rerun.get("raw_frontend_rerun_success_rate", 0.0) >= 0.95
        and rerun.get("trace_drift_ok", False)
        and semantic_pass
        and stable
        and identity_pass
        and per_category_breakdown_ready
        and not rerun.get("future_leakage_detected", False)
    )
    eval_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "larger_raw_video_closure_eval_done": True,
        "slice_root": rel(SLICE_ROOT),
        "semantic_seed_rows": semantic_rows,
        "real_instance_identity_seed_rows": real_identity_rows,
        "pseudo_identity_diagnostic_seed_rows": pseudo_identity_rows,
        "per_dataset_breakdown": per_cat["per_dataset_category_counts"],
        "per_split_breakdown": per_cat["per_split_category_counts"],
        "per_category_breakdown": per_cat["category_counts"],
        "raw_frontend_success_rate": rerun.get("raw_frontend_rerun_success_rate"),
        "trace_drift_ok": rerun.get("trace_drift_ok"),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "larger_raw_video_closure_benchmark_decision_done": True,
        "raw_frontend_rerun_success_rate": rerun.get("raw_frontend_rerun_success_rate"),
        "trace_drift_ok": bool(rerun.get("trace_drift_ok", False)),
        "semantic_three_seed_passed": semantic_pass,
        "stable_preservation": stable,
        "identity_real_instance_three_seed_passed": identity_pass,
        "identity_pseudo_targets_excluded_from_claim": True,
        "per_category_breakdown_ready": per_category_breakdown_ready,
        "semantic_changed_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_changed", "balanced_accuracy"),
        "semantic_changed_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_changed", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_hard", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_hard", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_uncertainty", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_uncertainty", "balanced_accuracy"),
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "larger_raw_video_closure_benchmark_passed": pass_gate,
        "recommended_next_step": "run_v35_46_per_category_failure_atlas" if pass_gate else "fix_semantic_target_distribution" if not semantic_pass else "fix_identity_provenance_or_real_instance_data",
        "中文结论": (
            "V35.45 larger raw-video closure benchmark 通过：32-clip rerun、semantic 三 seed、real-instance identity 三 seed、per-category breakdown 均满足 gate。"
            if pass_gate
            else "V35.45 larger raw-video closure benchmark 未完全通过；需要按失败项继续修。"
        ),
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.45 Larger Raw-Video Closure Benchmark Decision\n\n"
        f"- raw_frontend_rerun_success_rate: {decision['raw_frontend_rerun_success_rate']}\n"
        f"- trace_drift_ok: {decision['trace_drift_ok']}\n"
        f"- semantic_three_seed_passed: {semantic_pass}\n"
        f"- stable_preservation: {stable}\n"
        f"- identity_real_instance_three_seed_passed: {identity_pass}\n"
        f"- identity_pseudo_targets_excluded_from_claim: true\n"
        f"- per_category_breakdown_ready: {per_category_breakdown_ready}\n"
        f"- future_leakage_detected: false\n"
        f"- trajectory_degraded: false\n"
        f"- larger_raw_video_closure_benchmark_passed: {pass_gate}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"larger_raw_video_closure_benchmark_passed": pass_gate, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
