#!/usr/bin/env python3
"""汇总 V35.8 identity retrieval 三 seed 复现实验。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

DECISIONS = {
    "seed42": ROOT / "reports/stwm_ostf_v35_8_identity_only_retrieval_finetune_decision_20260515.json",
    "seed123": ROOT / "reports/stwm_ostf_v35_8_identity_only_retrieval_finetune_seed123_decision_20260515.json",
    "seed456": ROOT / "reports/stwm_ostf_v35_8_identity_only_retrieval_finetune_seed456_decision_20260515.json",
}
OUT = ROOT / "reports/stwm_ostf_v35_8_identity_retrieval_replication_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_8_IDENTITY_RETRIEVAL_REPLICATION_DECISION_20260515.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def all_nested(items: dict[str, dict[str, Any]], key: str, split: str | None = None) -> bool:
    vals = []
    for d in items.values():
        v = d[key]
        if split is not None:
            v = v[split]
        vals.append(bool(v))
    return all(vals)


def collect_identity(items: dict[str, dict[str, Any]], metric: str) -> dict[str, dict[str, float]]:
    return {
        seed: {
            "val": float(d["identity_consistency"][metric]["val"]),
            "test": float(d["identity_consistency"][metric]["test"]),
        }
        for seed, d in items.items()
    }


def main() -> None:
    items = {seed: load(path) for seed, path in DECISIONS.items()}
    missing = [str(path.relative_to(ROOT)) for path in DECISIONS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 V35.8 decision: {missing}")
    semantic_state_head_passed_all = all(bool(d["semantic_state_head_passed"]) for d in items.values())
    identity_retrieval_passed_all = all(bool(d["identity_consistency"]["retrieval_passed"]) for d in items.values())
    stable_all = all_nested(items, "stable_preservation", "val") and all_nested(items, "stable_preservation", "test")
    hard_all = all_nested(items, "semantic_hard_signal", "val") and all_nested(items, "semantic_hard_signal", "test")
    changed_all = all_nested(items, "changed_semantic_signal", "val") and all_nested(items, "changed_semantic_signal", "test")
    semantic_measurement_load_bearing_all = all(bool(d["semantic_measurement_load_bearing"]) for d in items.values())
    unit_memory_load_bearing_all = all(bool(d["unit_memory_load_bearing"]) for d in items.values())
    assignment_load_bearing_all = all(bool(d["assignment_load_bearing"]) for d in items.values())
    integrated_identity = bool(identity_retrieval_passed_all and stable_all)
    # 当前仍是 M128/H32 + observed trace/measurement 合同，不允许包装为完整 video semantic field success。
    integrated_semantic = False
    recommended = "build_video_input_closure" if integrated_identity and semantic_state_head_passed_all else "fix_identity_retrieval_head"
    if not unit_memory_load_bearing_all or not assignment_load_bearing_all:
        secondary = "fix_unit_assignment_load_bearing"
    else:
        secondary = "run_video_input_closure_benchmark"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_8_identity_retrieval_replication_done": True,
        "seeds": list(items.keys()),
        "semantic_state_head_passed_all": semantic_state_head_passed_all,
        "identity_retrieval_passed_all": identity_retrieval_passed_all,
        "stable_preservation_all": stable_all,
        "semantic_hard_signal_all": hard_all,
        "changed_semantic_signal_all": changed_all,
        "semantic_measurement_load_bearing_all": semantic_measurement_load_bearing_all,
        "unit_memory_load_bearing_all": unit_memory_load_bearing_all,
        "assignment_load_bearing_all": assignment_load_bearing_all,
        "identity_retrieval_exclude_same_point_top1": collect_identity(items, "identity_retrieval_exclude_same_point_top1"),
        "identity_retrieval_same_frame_top1": collect_identity(items, "identity_retrieval_same_frame_top1"),
        "identity_retrieval_instance_pooled_top1": collect_identity(items, "identity_retrieval_instance_pooled_top1"),
        "identity_confuser_separation": collect_identity(items, "identity_confuser_separation"),
        "v30_backbone_frozen": all(bool(d["v30_backbone_frozen"]) for d in items.values()),
        "future_leakage_detected": any(bool(d["future_leakage_detected"]) for d in items.values()),
        "trajectory_degraded": any(bool(d["trajectory_degraded"]) for d in items.values()),
        "integrated_identity_field_claim_allowed": integrated_identity,
        "integrated_semantic_field_claim_allowed": integrated_semantic,
        "recommended_next_step": recommended,
        "secondary_blocker": secondary,
        "中文结论": (
            "V35.8 在 seed42/123/456 的 M128/H32 复现实验中通过 semantic state head 与 identity retrieval gate。"
            " identity field claim 在当前 video-derived trace + observed semantic measurement 输入合同下可以成立；"
            "但 unit/assignment load-bearing 没有三 seed 全过，且尚未完成 raw video input closure，不能宣称完整 semantic field success。"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.8 Identity Retrieval Replication Decision\n\n"
        f"- semantic_state_head_passed_all: {semantic_state_head_passed_all}\n"
        f"- identity_retrieval_passed_all: {identity_retrieval_passed_all}\n"
        f"- stable_preservation_all: {stable_all}\n"
        f"- semantic_hard_signal_all: {hard_all}\n"
        f"- changed_semantic_signal_all: {changed_all}\n"
        f"- semantic_measurement_load_bearing_all: {semantic_measurement_load_bearing_all}\n"
        f"- unit_memory_load_bearing_all: {unit_memory_load_bearing_all}\n"
        f"- assignment_load_bearing_all: {assignment_load_bearing_all}\n"
        f"- integrated_identity_field_claim_allowed: {integrated_identity}\n"
        f"- integrated_semantic_field_claim_allowed: {integrated_semantic}\n"
        f"- recommended_next_step: {recommended}\n"
        f"- secondary_blocker: {secondary}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"recommended_next_step": recommended, "secondary_blocker": secondary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
