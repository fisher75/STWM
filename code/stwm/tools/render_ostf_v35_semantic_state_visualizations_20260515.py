#!/usr/bin/env python3
"""V35 semantic state 可视化 manifest。若 head 未训练，则写 skipped manifest。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

setproctitle.setproctitle("python")
ROOT = Path(__file__).resolve().parents[3]
DECISION = ROOT / "reports/stwm_ostf_v35_semantic_state_target_predictability_decision_20260515.json"
MANIFEST = ROOT / "reports/stwm_ostf_v35_semantic_state_visualization_manifest_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_SEMANTIC_STATE_VISUALIZATION_20260515.md"
HEAD_EVAL = ROOT / "reports/stwm_ostf_v35_semantic_state_head_eval_summary_20260515.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    pred = read_json(DECISION)
    head = read_json(HEAD_EVAL)
    suite_ready = bool(pred.get("observed_predictable_semantic_state_suite_ready", False))
    head_ran = bool(head.get("semantic_state_head_training_ran", False) or head.get("semantic_state_head_eval_ran", False))
    cases = []
    ready = False
    reason = "semantic_state_head_not_trained"
    if suite_ready and head_ran:
        reason = "eval_arrays_not_available_in_this_minimal_renderer"
    else:
        cases = [
            {"case_type": "semantic cluster transition success/failure", "status": "not_run", "reason": reason},
            {"case_type": "semantic changed success/failure", "status": "not_run", "reason": reason},
            {"case_type": "stable preservation success/failure", "status": "not_run", "reason": reason},
            {"case_type": "evidence anchor family prediction success/failure", "status": "not_run", "reason": reason},
            {"case_type": "identity consistency success/failure", "status": "not_run", "reason": reason},
            {"case_type": "uncertainty high-risk abstain", "status": "not_run", "reason": reason},
            {"case_type": "M128 future trace + predicted semantic state overlay", "status": "not_run", "reason": reason},
        ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visualization_ready": ready,
        "semantic_state_head_training_ran": head_ran,
        "observed_predictable_semantic_state_suite_ready": suite_ready,
        "cases": cases,
        "skip_reason": None if ready else reason,
        "中文结论": "V35 当前阶段如果 target suite 未 ready，则不生成模型可视化；保留 manifest 作为 claim-boundary artifact。",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35 Semantic State Visualization\n\n"
        f"- visualization_ready: {ready}\n"
        f"- skip_reason: {manifest['skip_reason']}\n\n"
        "## 中文总结\n"
        + manifest["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print("V35 visualization manifest 已写入。", flush=True)
    print(json.dumps({"visualization_ready": ready, "skip_reason": manifest["skip_reason"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
