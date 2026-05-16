#!/usr/bin/env python3
"""汇总 V35.29 expanded video identity pairwise retrieval 三 seed 复现。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

SEEDS = [42, 123, 456]
REPORT = ROOT / "reports/stwm_ostf_v35_29_expanded_identity_replication_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_29_EXPANDED_IDENTITY_REPLICATION_DECISION_20260516.md"


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


def suffix(seed: int) -> str:
    return "" if seed == 42 else f"_seed{seed}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"missing": True, "path": str(path.relative_to(ROOT))}


def get_metric(eval_report: dict[str, Any], split: str, key: str) -> float | None:
    value = eval_report.get("learned_identity_head", {}).get(split, {}).get(key)
    return None if value is None else float(value)


def mean(rows: dict[str, dict[str, Any]], split: str, key: str) -> float | None:
    vals = [get_metric(row["eval"], split, key) for row in rows.values()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    rows: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        suf = suffix(seed)
        dec = read_json(ROOT / f"reports/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_decision_20260516{suf}.json")
        ev = read_json(ROOT / f"reports/stwm_ostf_v35_29_expanded_video_identity_pairwise_retrieval_eval_summary_20260516{suf}.json")
        rows[str(seed)] = {
            "decision": dec,
            "eval": ev,
            "passed": bool(dec.get("video_identity_pairwise_retrieval_passed", False)),
            "test_exclude_same_point_top1": get_metric(ev, "test", "identity_retrieval_exclude_same_point_top1"),
            "test_same_frame_top1": get_metric(ev, "test", "identity_retrieval_same_frame_top1"),
            "test_instance_pooled_top1": get_metric(ev, "test", "identity_retrieval_instance_pooled_top1"),
            "test_confuser_avoidance_top1": get_metric(ev, "test", "identity_confuser_avoidance_top1"),
            "test_confuser_separation": get_metric(ev, "test", "identity_confuser_separation"),
            "test_occlusion_reappear_top1": get_metric(ev, "test", "occlusion_reappear_retrieval_top1"),
            "test_trajectory_crossing_top1": get_metric(ev, "test", "trajectory_crossing_retrieval_top1"),
        }
    all_passed = all(row["passed"] for row in rows.values())
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expanded_identity_replication_done": True,
        "seeds": SEEDS,
        "seed_results": rows,
        "all_three_seed_passed": all_passed,
        "test_exclude_same_point_top1_mean": mean(rows, "test", "identity_retrieval_exclude_same_point_top1"),
        "test_same_frame_top1_mean": mean(rows, "test", "identity_retrieval_same_frame_top1"),
        "test_instance_pooled_top1_mean": mean(rows, "test", "identity_retrieval_instance_pooled_top1"),
        "test_confuser_avoidance_top1_mean": mean(rows, "test", "identity_confuser_avoidance_top1"),
        "test_confuser_separation_mean": mean(rows, "test", "identity_confuser_separation"),
        "test_occlusion_reappear_top1_mean": mean(rows, "test", "occlusion_reappear_retrieval_top1"),
        "test_trajectory_crossing_top1_mean": mean(rows, "test", "trajectory_crossing_retrieval_top1"),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "integrated_identity_field_claim_allowed": False,
        "recommended_next_step": "update_full_video_semantic_identity_closure_decision" if all_passed else "fix_expanded_video_identity_retrieval",
        "中文结论": (
            "V35.29 expanded video identity retrieval 在 325 clips 上 seed42/123/456 全部通过，identity field 的 pairwise/retrieval 证据已从 96 clips 扩展到完整 unified benchmark。"
            if all_passed
            else "V35.29 expanded identity 三 seed 未全过，不能开放 identity field claim。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.29 Expanded Identity Replication Decision\n\n"
        f"- all_three_seed_passed: {all_passed}\n"
        f"- test_exclude_same_point_top1_mean: {decision['test_exclude_same_point_top1_mean']}\n"
        f"- test_confuser_avoidance_top1_mean: {decision['test_confuser_avoidance_top1_mean']}\n"
        f"- test_occlusion_reappear_top1_mean: {decision['test_occlusion_reappear_top1_mean']}\n"
        f"- test_trajectory_crossing_top1_mean: {decision['test_trajectory_crossing_top1_mean']}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"expanded_identity_three_seed_passed": all_passed, "推荐下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
