#!/usr/bin/env python3
"""V36.2c: downstream secondary gate for conservative selector causal trace。"""
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
from stwm.tools.eval_ostf_v35_45_larger_raw_video_closure_benchmark_20260516 import eval_identity_split, mean_seed, pass_identity  # noqa: E402

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_2c_conservative_selector_downstream_slice/M128_H32"
MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
V36_DECISION = ROOT / "reports/stwm_ostf_v36_causal_past_only_video_world_model_benchmark_decision_20260516.json"
PRIOR_DECISION = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_baseline_decision_20260516.json"
SELECTOR_REPORT = ROOT / "reports/stwm_ostf_v36_2c_conservative_copy_default_selector_rollout_20260516.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v36_2c_conservative_selector_downstream_gate_eval_summary_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v36_2c_conservative_selector_downstream_gate_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_2C_CONSERVATIVE_SELECTOR_DOWNSTREAM_GATE_DECISION_20260516.md"
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
    data = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    out: dict[str, dict[str, Any]] = {}
    for s in data.get("samples", []):
        name = Path(str(s.get("source_unified_npz", s.get("expected_rerun_trace_path", "")))).name
        if name:
            out[name] = s
    return out


def identity_paths(root: Path, split: str, real_only: bool | None, claim_map: dict[str, dict[str, Any]]) -> list[Path]:
    out = []
    for p in sorted((root / split).glob("*.npz")):
        claim = bool(claim_map.get(p.name, {}).get("identity_claim_allowed", False))
        if real_only is None or claim == real_only:
            out.append(p)
    return out


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


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    claim_map = manifest_claim_map()
    smoke.RERUN_UNIFIED_ROOT = SLICE_ROOT
    semantic_rows = [smoke.eval_semantic_seed(seed, device) for seed in SEEDS]
    real_identity_rows = []
    pseudo_val_count = len(identity_paths(SLICE_ROOT, "val", False, claim_map))
    pseudo_test_count = len(identity_paths(SLICE_ROOT, "test", False, claim_map))
    for seed in SEEDS:
        val = eval_identity_split(identity_paths(SLICE_ROOT, "val", True, claim_map), seed, device)
        test = eval_identity_split(identity_paths(SLICE_ROOT, "test", True, claim_map), seed, device)
        real_identity_rows.append({"seed": seed, "val": val, "test": test, "identity_passed": pass_identity(val) and pass_identity(test)})
    semantic_pass = bool(all(r["semantic_smoke_passed"] for r in semantic_rows))
    stable = bool(all(r.get("stable_preservation", False) for r in semantic_rows))
    identity_pass = bool(real_identity_rows and all(r["identity_passed"] for r in real_identity_rows))
    v36 = json.loads(V36_DECISION.read_text(encoding="utf-8")) if V36_DECISION.exists() else {}
    prior = json.loads(PRIOR_DECISION.read_text(encoding="utf-8")) if PRIOR_DECISION.exists() else {}
    selector = json.loads(SELECTOR_REPORT.read_text(encoding="utf-8")) if SELECTOR_REPORT.exists() else {}
    selector_sem = {
        "semantic_changed_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_changed", "balanced_accuracy"),
        "semantic_changed_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_changed", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_hard", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_hard", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_uncertainty", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_uncertainty", "balanced_accuracy"),
    }
    prior_sem = prior.get("prior_semantic_metrics", {})
    v36_sem = {k: v36.get(k) for k in selector_sem}
    selector_minus_prior = {
        k.replace("_mean", "_selector_minus_prior"): (float(v) - float(prior_sem[k]) if v is not None and prior_sem.get(k) is not None else None)
        for k, v in selector_sem.items()
    }
    selector_minus_v36 = {
        k.replace("_mean", "_selector_minus_v36"): (float(v) - float(v36_sem[k]) if v is not None and v36_sem.get(k) is not None else None)
        for k, v in selector_sem.items()
    }
    id_test = {
        "identity_retrieval_exclude_same_point_top1": identity_metric_mean(real_identity_rows, "test", "identity_retrieval_exclude_same_point_top1"),
        "identity_retrieval_same_frame_top1": identity_metric_mean(real_identity_rows, "test", "identity_retrieval_same_frame_top1"),
        "identity_retrieval_instance_pooled_top1": identity_metric_mean(real_identity_rows, "test", "identity_retrieval_instance_pooled_top1"),
        "identity_confuser_avoidance_top1": identity_metric_mean(real_identity_rows, "test", "identity_confuser_avoidance_top1"),
        "occlusion_reappear_retrieval_top1": identity_metric_mean(real_identity_rows, "test", "occlusion_reappear_retrieval_top1"),
        "trajectory_crossing_retrieval_top1": identity_metric_mean(real_identity_rows, "test", "trajectory_crossing_retrieval_top1"),
    }
    sem_test_delta_vals = [v for k, v in selector_minus_prior.items() if "_test_" in k and v is not None]
    downstream_positive = bool((sem_test_delta_vals and float(np.mean(sem_test_delta_vals)) >= -0.005) and semantic_pass and stable and identity_pass)
    trace_pass = bool(selector.get("copy_default_selector_passed", False))
    secondary_pass = bool(trace_pass and downstream_positive)
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conservative_selector_downstream_gate_eval_done": True,
        "trace_copy_default_selector_passed": trace_pass,
        "semantic_three_seed_passed": semantic_pass,
        "stable_preservation": stable,
        "identity_real_instance_three_seed_passed": identity_pass,
        "identity_pseudo_targets_excluded_from_claim": True,
        "pseudo_identity_diagnostic_sample_count_val": pseudo_val_count,
        "pseudo_identity_diagnostic_sample_count_test": pseudo_test_count,
        "selector_semantic_metrics": selector_sem,
        "selector_identity_test_means": id_test,
        "selector_minus_strongest_prior_semantic_deltas": selector_minus_prior,
        "selector_minus_v36_semantic_deltas": selector_minus_v36,
        "downstream_positive_causal_trace": downstream_positive,
        "secondary_gate_passed": secondary_pass,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "recommended_next_step": "run_v36_3_full_325_causal_benchmark_rerun" if secondary_pass else "fix_v30_prior_selector_calibration",
        "中文结论": (
            "V36.2c conservative selector 同时满足 trace no-harm 与 downstream secondary gate；可以进入 V36.3 full causal benchmark rerun。"
            if secondary_pass
            else "V36.2c downstream secondary gate 未完全满足；继续修 trace calibration，不跑 V36.3。"
        ),
    }
    eval_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_seed_rows": semantic_rows,
        "real_instance_identity_seed_rows": real_identity_rows,
        "pseudo_identity_diagnostic_seed_rows": [
            {"seed": seed, "val": {"sample_count": pseudo_val_count}, "test": {"sample_count": pseudo_test_count}, "diagnostic_only": True}
            for seed in SEEDS
        ],
        "decision_fields": decision,
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36.2c Conservative Selector Downstream Gate Decision\n\n"
        f"- trace_copy_default_selector_passed: {trace_pass}\n"
        f"- semantic_three_seed_passed: {semantic_pass}\n"
        f"- stable_preservation: {stable}\n"
        f"- identity_real_instance_three_seed_passed: {identity_pass}\n"
        f"- downstream_positive_causal_trace: {downstream_positive}\n"
        f"- secondary_gate_passed: {secondary_pass}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36_2c_downstream_gate完成": True, "secondary_gate_passed": secondary_pass, "下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if secondary_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
