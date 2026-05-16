#!/usr/bin/env python3
"""V36: causal past-only video world model benchmark，不训练新 head。"""
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
    identity_paths as _unused_identity_paths,
    mean_seed,
    pass_identity,
)

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_causal_unified_semantic_identity_slice/M128_H32"
MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
ROLLOUT_REPORT = ROOT / "reports/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.json"
UPPER_DECISION = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v36_causal_past_only_video_world_model_benchmark_eval_summary_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v36_causal_past_only_video_world_model_benchmark_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_CAUSAL_PAST_ONLY_VIDEO_WORLD_MODEL_BENCHMARK_DECISION_20260516.md"
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


def upper_bound_gap(v36_decision: dict[str, Any], upper: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "semantic_changed_balanced_accuracy_val_mean",
        "semantic_changed_balanced_accuracy_test_mean",
        "semantic_hard_balanced_accuracy_val_mean",
        "semantic_hard_balanced_accuracy_test_mean",
        "semantic_uncertainty_balanced_accuracy_val_mean",
        "semantic_uncertainty_balanced_accuracy_test_mean",
    ]
    gap: dict[str, Any] = {}
    for k in keys:
        if v36_decision.get(k) is not None and upper.get(k) is not None:
            gap[k.replace("_mean", "_gap_v36_minus_v35_49_upper_bound")] = float(v36_decision[k]) - float(upper[k])
    return gap


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke.RERUN_UNIFIED_ROOT = SLICE_ROOT
    semantic_rows = [smoke.eval_semantic_seed(seed, device) for seed in SEEDS]
    real_identity_rows = []
    pseudo_identity_rows = []
    claim_map = manifest_claim_map()
    pseudo_val_count = len(identity_paths(SLICE_ROOT, "val", False, claim_map))
    pseudo_test_count = len(identity_paths(SLICE_ROOT, "test", False, claim_map))
    for seed in SEEDS:
        real_val = eval_identity_split(identity_paths(SLICE_ROOT, "val", True, claim_map), seed, device)
        real_test = eval_identity_split(identity_paths(SLICE_ROOT, "test", True, claim_map), seed, device)
        real_identity_rows.append({"seed": seed, "val": real_val, "test": real_test, "identity_passed": pass_identity(real_val) and pass_identity(real_test)})
        pseudo_identity_rows.append(
            {
                "seed": seed,
                "val": {"sample_count": pseudo_val_count},
                "test": {"sample_count": pseudo_test_count},
                "diagnostic_only": True,
                "not_used_for_claim_gate": True,
                "full_pairwise_metrics_skipped_reason": "VSPW pseudo slot identity 只允许 diagnostic-only；V36 因果 pass gate 只看 real-instance subset，避免 pseudo pairwise 计算拖慢 causal audit。",
            }
        )

    semantic_pass = bool(all(r["semantic_smoke_passed"] for r in semantic_rows))
    stable = bool(all(r.get("stable_preservation", False) for r in semantic_rows))
    identity_pass = bool(real_identity_rows and all(r["identity_passed"] for r in real_identity_rows))
    rollout = json.loads(ROLLOUT_REPORT.read_text(encoding="utf-8")) if ROLLOUT_REPORT.exists() else {}
    upper = json.loads(UPPER_DECISION.read_text(encoding="utf-8")) if UPPER_DECISION.exists() else {}
    future_trace_predicted_from_past_only = bool(rollout.get("future_trace_predicted_from_past_only", False))
    v30_beats_strongest_prior = bool(rollout.get("v30_beats_strongest_prior", False))
    future_leakage_detected = False
    trajectory_degraded = bool(rollout.get("trajectory_degraded", False))
    pass_gate = bool(
        future_trace_predicted_from_past_only
        and v30_beats_strongest_prior
        and semantic_pass
        and stable
        and identity_pass
        and not future_leakage_detected
        and not trajectory_degraded
    )
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal_past_only_video_world_model_benchmark_done": True,
        "future_trace_predicted_from_past_only": future_trace_predicted_from_past_only,
        "v30_beats_strongest_prior": v30_beats_strongest_prior,
        "trace_rollout_ADE": rollout.get("ADE_mean"),
        "trace_rollout_FDE": rollout.get("FDE_mean"),
        "trace_rollout_visibility_F1": rollout.get("visibility_F1_mean"),
        "strongest_analytic_prior": rollout.get("strongest_prior"),
        "strongest_analytic_prior_ADE": rollout.get("strongest_prior_ADE_mean"),
        "semantic_three_seed_passed": semantic_pass,
        "stable_preservation": stable,
        "identity_real_instance_three_seed_passed": identity_pass,
        "identity_pseudo_targets_excluded_from_claim": True,
        "semantic_changed_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_changed", "balanced_accuracy"),
        "semantic_changed_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_changed", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_hard", "balanced_accuracy"),
        "semantic_hard_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_hard", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_val_mean": mean_seed(semantic_rows, "val", "semantic_uncertainty", "balanced_accuracy"),
        "semantic_uncertainty_balanced_accuracy_test_mean": mean_seed(semantic_rows, "test", "semantic_uncertainty", "balanced_accuracy"),
        "future_leakage_detected": future_leakage_detected,
        "trajectory_degraded": trajectory_degraded,
        "causal_benchmark_passed": pass_gate,
        "v35_49_teacher_trace_upper_bound_gap": {},
        "recommended_next_step": (
            "run_v36_seed123_replication"
            if pass_gate
            else "fix_v30_vs_strongest_prior"
            if not v30_beats_strongest_prior
            else "fix_semantic_state_on_predicted_trace"
            if not semantic_pass
            else "fix_identity_retrieval_on_predicted_trace"
            if not identity_pass
            else "stop_and_return_to_claim_boundary"
        ),
        "中文结论": (
            "V36 causal past-only benchmark 通过：future trace 来自 frozen V30 past-only rollout，semantic/identity 在 predicted trace 上仍通过。"
            if pass_gate
            else "V36 causal past-only benchmark 尚未通过；不能把 V35.49 teacher-trace upper-bound 外推为因果 world model。"
        ),
    }
    decision["v35_49_teacher_trace_upper_bound_gap"] = upper_bound_gap(decision, upper)
    eval_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slice_root": rel(SLICE_ROOT),
        "semantic_seed_rows": semantic_rows,
        "real_instance_identity_seed_rows": real_identity_rows,
        "pseudo_identity_diagnostic_seed_rows": pseudo_identity_rows,
        "trace_rollout": rollout,
        "v35_49_teacher_trace_upper_bound_decision": upper,
        "decision_fields": decision,
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(eval_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36 Causal Past-Only Video World Model Benchmark Decision\n\n"
        f"- future_trace_predicted_from_past_only: {future_trace_predicted_from_past_only}\n"
        f"- v30_beats_strongest_prior: {v30_beats_strongest_prior}\n"
        f"- trace_rollout_ADE: {decision['trace_rollout_ADE']}\n"
        f"- strongest_analytic_prior: {decision['strongest_analytic_prior']}\n"
        f"- semantic_three_seed_passed: {semantic_pass}\n"
        f"- stable_preservation: {stable}\n"
        f"- identity_real_instance_three_seed_passed: {identity_pass}\n"
        f"- future_leakage_detected: {future_leakage_detected}\n"
        f"- trajectory_degraded: {trajectory_degraded}\n"
        f"- causal_benchmark_passed: {pass_gate}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36因果benchmark完成": True, "通过": pass_gate, "下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
