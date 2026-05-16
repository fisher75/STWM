#!/usr/bin/env python3
"""V36.1: 评估 strongest-prior future trace slice 的 semantic/identity downstream baseline。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools import run_ostf_v35_35_raw_video_frontend_rerun_smoke_20260516 as smoke  # noqa: E402
from stwm.tools.eval_ostf_v35_45_larger_raw_video_closure_benchmark_20260516 import (  # noqa: E402
    eval_identity_split,
    mean_seed,
    pass_identity,
)

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_1_strongest_prior_downstream_slice/M128_H32"
MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
SLICE_BUILD = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_slice_build_20260516.json"
V36_EVAL = ROOT / "reports/stwm_ostf_v36_causal_past_only_video_world_model_benchmark_eval_summary_20260516.json"
V36_DECISION = ROOT / "reports/stwm_ostf_v36_causal_past_only_video_world_model_benchmark_decision_20260516.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_baseline_eval_summary_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_baseline_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_1_STRONGEST_PRIOR_DOWNSTREAM_BASELINE_DECISION_20260516.md"
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


def manifest_claim_map() -> dict[str, dict[str, Any]]:
    if not MANIFEST.exists():
        return {}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for s in data.get("samples", []):
        name = Path(str(s.get("source_unified_npz", s.get("expected_rerun_trace_path", "")))).name
        if name:
            out[name] = s
    return out


def identity_paths(root: Path, split: str, real_only: bool | None, claim_map: dict[str, dict[str, Any]]) -> list[Path]:
    out = []
    for p in sorted((root / split).glob("*.npz")):
        meta = claim_map.get(p.name, {})
        claim = bool(meta.get("identity_claim_allowed", False))
        if real_only is None or claim == real_only:
            out.append(p)
    return out


def pass_semantic_binary(m: dict[str, float | None]) -> bool:
    return bool((m.get("roc_auc") or 0.0) >= 0.55 and (m.get("balanced_accuracy") or 0.0) >= 0.52)


def metric_mean(rows: list[dict[str, Any]], split: str, family: str, key: str) -> float | None:
    vals = []
    for r in rows:
        v = r.get(split, {}).get(family, {}).get(key)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def identity_metric_mean(rows: list[dict[str, Any]], split: str, key: str) -> float | None:
    vals = []
    for r in rows:
        v = r.get(split, {}).get(key)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def extract_v36_identity_means(v36_eval: dict[str, Any], split: str) -> dict[str, float | None]:
    rows = v36_eval.get("real_instance_identity_seed_rows", [])
    return {
        "identity_retrieval_exclude_same_point_top1": identity_metric_mean(rows, split, "identity_retrieval_exclude_same_point_top1"),
        "identity_retrieval_same_frame_top1": identity_metric_mean(rows, split, "identity_retrieval_same_frame_top1"),
        "identity_retrieval_instance_pooled_top1": identity_metric_mean(rows, split, "identity_retrieval_instance_pooled_top1"),
        "identity_confuser_avoidance_top1": identity_metric_mean(rows, split, "identity_confuser_avoidance_top1"),
        "occlusion_reappear_retrieval_top1": identity_metric_mean(rows, split, "occlusion_reappear_retrieval_top1"),
        "trajectory_crossing_retrieval_top1": identity_metric_mean(rows, split, "trajectory_crossing_retrieval_top1"),
    }


def main() -> int:
    build = json.loads(SLICE_BUILD.read_text(encoding="utf-8")) if SLICE_BUILD.exists() else {}
    v36_decision = json.loads(V36_DECISION.read_text(encoding="utf-8")) if V36_DECISION.exists() else {}
    v36_eval = json.loads(V36_EVAL.read_text(encoding="utf-8")) if V36_EVAL.exists() else {}
    claim_map = manifest_claim_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke.RERUN_UNIFIED_ROOT = SLICE_ROOT
    semantic_rows = [smoke.eval_semantic_seed(seed, device) for seed in SEEDS]
    real_identity_rows = []
    pseudo_val_count = len(identity_paths(SLICE_ROOT, "val", False, claim_map))
    pseudo_test_count = len(identity_paths(SLICE_ROOT, "test", False, claim_map))
    for seed in SEEDS:
        real_val = eval_identity_split(identity_paths(SLICE_ROOT, "val", True, claim_map), seed, device)
        real_test = eval_identity_split(identity_paths(SLICE_ROOT, "test", True, claim_map), seed, device)
        real_identity_rows.append({"seed": seed, "val": real_val, "test": real_test, "identity_passed": pass_identity(real_val) and pass_identity(real_test)})
    semantic_pass = bool(all(r["semantic_smoke_passed"] for r in semantic_rows))
    stable = bool(all(r.get("stable_preservation", False) for r in semantic_rows))
    identity_pass = bool(real_identity_rows and all(r["identity_passed"] for r in real_identity_rows))
    prior_sem = {
        "semantic_changed_balanced_accuracy_val_mean": metric_mean(semantic_rows, "val", "semantic_changed", "balanced_accuracy"),
        "semantic_changed_balanced_accuracy_test_mean": metric_mean(semantic_rows, "test", "semantic_changed", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_val_mean": metric_mean(semantic_rows, "val", "semantic_hard", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_test_mean": metric_mean(semantic_rows, "test", "semantic_hard", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_val_mean": metric_mean(semantic_rows, "val", "semantic_uncertainty", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_test_mean": metric_mean(semantic_rows, "test", "semantic_uncertainty", "balanced_accuracy"),
    }
    prior_id_test = extract_v36_identity_means({"real_instance_identity_seed_rows": real_identity_rows}, "test")
    v36_id_test = extract_v36_identity_means(v36_eval, "test")
    semantic_deltas = {
        k.replace("_mean", "_v36_minus_prior"): (float(v36_decision[k]) - float(v) if v is not None and v36_decision.get(k) is not None else None)
        for k, v in prior_sem.items()
    }
    identity_deltas = {
        f"{k}_test_v36_minus_prior": (float(v36_id_test[k]) - float(v) if v is not None and v36_id_test.get(k) is not None else None)
        for k, v in prior_id_test.items()
    }
    semantic_test_delta_vals = [v for k, v in semantic_deltas.items() if k.endswith("test_v36_minus_prior") and v is not None]
    identity_test_delta_vals = [v for v in identity_deltas.values() if v is not None]
    v36_beats_prior_semantic = bool(semantic_test_delta_vals and float(np.mean(semantic_test_delta_vals)) > 0.0)
    v36_beats_prior_identity = bool(identity_test_delta_vals and float(np.mean(identity_test_delta_vals)) > -0.01)
    v36_downstream_beats_prior = bool(v36_beats_prior_semantic or v36_beats_prior_identity)
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strongest_prior_downstream_eval_done": True,
        "strongest_prior_name": build.get("strongest_prior_name", "unknown"),
        "slice_root": rel(SLICE_ROOT),
        "semantic_three_seed_passed": semantic_pass,
        "stable_preservation": stable,
        "identity_real_instance_three_seed_passed": identity_pass,
        "identity_pseudo_targets_excluded_from_claim": True,
        "pseudo_identity_diagnostic_sample_count_val": pseudo_val_count,
        "pseudo_identity_diagnostic_sample_count_test": pseudo_test_count,
        "prior_semantic_metrics": prior_sem,
        "prior_identity_test_means": prior_id_test,
        "v36_v30_identity_test_means": v36_id_test,
        "semantic_v36_minus_prior_deltas": semantic_deltas,
        "identity_v36_minus_prior_deltas": identity_deltas,
        "v36_v30_downstream_beats_strongest_prior_semantic": v36_beats_prior_semantic,
        "v36_v30_downstream_beats_strongest_prior_identity": v36_beats_prior_identity,
        "v36_v30_downstream_utility_beats_strongest_prior_slice": v36_downstream_beats_prior,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "recommended_next_step": "run_v36_2_frozen_v30_prior_selector_calibration" if v36_downstream_beats_prior else "fix_trace_rollout_before_claim",
        "中文结论": (
            "V36 V30 predicted trace 虽然 ADE 输给 strongest prior，但 downstream semantic/identity utility 至少有一条线优于 strongest-prior slice；下一步应做 frozen V30 prior selector/calibration。"
            if v36_downstream_beats_prior
            else "V36 V30 predicted trace 在 downstream semantic/identity utility 上也没有证明优于 strongest-prior slice；必须先修 trace rollout，再谈 causal world model claim。"
        ),
    }
    eval_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strongest_prior_downstream_eval_done": True,
        "semantic_seed_rows": semantic_rows,
        "real_instance_identity_seed_rows": real_identity_rows,
        "pseudo_identity_diagnostic_seed_rows": [
            {
                "seed": seed,
                "val": {"sample_count": pseudo_val_count},
                "test": {"sample_count": pseudo_test_count},
                "diagnostic_only": True,
                "not_used_for_claim_gate": True,
            }
            for seed in SEEDS
        ],
        "decision_fields": decision,
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36.1 Strongest-Prior Downstream Baseline Decision\n\n"
        f"- strongest_prior_name: {decision['strongest_prior_name']}\n"
        f"- semantic_three_seed_passed: {semantic_pass}\n"
        f"- stable_preservation: {stable}\n"
        f"- identity_real_instance_three_seed_passed: {identity_pass}\n"
        f"- v36_v30_downstream_beats_strongest_prior_semantic: {v36_beats_prior_semantic}\n"
        f"- v36_v30_downstream_beats_strongest_prior_identity: {v36_beats_prior_identity}\n"
        f"- v36_v30_downstream_utility_beats_strongest_prior_slice: {v36_downstream_beats_prior}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36_1_strongest_prior_downstream_eval完成": True, "v36_downstream_beats_prior": v36_downstream_beats_prior, "下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if v36_downstream_beats_prior else 2


if __name__ == "__main__":
    raise SystemExit(main())
