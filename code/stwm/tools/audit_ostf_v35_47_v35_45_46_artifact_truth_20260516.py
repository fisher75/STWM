#!/usr/bin/env python3
"""V35.47 审计 V35.45/V35.46 artifacts 是否足以支撑协议决策。"""
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

REQUIRED = {
    "v35_45_decision": ROOT / "reports/stwm_ostf_v35_45_decision_20260516.json",
    "v35_45_benchmark_decision": ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_benchmark_decision_20260516.json",
    "v35_45_benchmark_eval": ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_benchmark_eval_summary_20260516.json",
    "v35_45_subset_build": ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_subset_build_20260516.json",
    "v35_45_frontend_rerun": ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_frontend_rerun_20260516.json",
    "v35_45_unified_slice": ROOT / "reports/stwm_ostf_v35_45_larger_rerun_unified_slice_build_20260516.json",
    "v35_45_visualization": ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_visualization_manifest_20260516.json",
    "v35_45_subset_manifest": ROOT / "outputs/cache/stwm_ostf_v35_45_larger_raw_video_closure_subset/manifest.json",
    "v35_46_decision": ROOT / "reports/stwm_ostf_v35_46_per_category_failure_atlas_decision_20260516.json",
    "v35_46_eval": ROOT / "reports/stwm_ostf_v35_46_per_category_failure_atlas_eval_20260516.json",
    "v35_46_log": ROOT / "outputs/logs/stwm_ostf_v35_46_per_category_failure_atlas_20260516.log",
    "v35_46_tool": ROOT / "code/stwm/tools/eval_ostf_v35_46_per_category_failure_atlas_20260516.py",
}
DOCS = {
    "v35_45_decision_doc": ROOT / "docs/STWM_OSTF_V35_45_DECISION_20260516.md",
    "v35_45_subset_doc": ROOT / "docs/STWM_OSTF_V35_45_LARGER_RAW_VIDEO_CLOSURE_SUBSET_BUILD_20260516.md",
    "v35_45_frontend_doc": ROOT / "docs/STWM_OSTF_V35_45_LARGER_RAW_VIDEO_FRONTEND_RERUN_20260516.md",
    "v35_45_unified_doc": ROOT / "docs/STWM_OSTF_V35_45_LARGER_RERUN_UNIFIED_SLICE_BUILD_20260516.md",
    "v35_45_visualization_doc": ROOT / "docs/STWM_OSTF_V35_45_LARGER_RAW_VIDEO_CLOSURE_VISUALIZATION_20260516.md",
    "v35_46_doc": ROOT / "docs/STWM_OSTF_V35_46_PER_CATEGORY_FAILURE_ATLAS_DECISION_20260516.md",
}
REPORT = ROOT / "reports/stwm_ostf_v35_47_v35_45_46_artifact_truth_audit_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_47_V35_45_46_ARTIFACT_TRUTH_AUDIT_20260516.md"


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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() and path.suffix == ".json" else {}


def main() -> int:
    present = {k: p.exists() for k, p in REQUIRED.items()}
    doc_present = {k: p.exists() for k, p in DOCS.items()}
    v45 = load(REQUIRED["v35_45_decision"])
    v46 = load(REQUIRED["v35_46_decision"])
    missing_keys = [k for k, ok in present.items() if not ok]
    artifact_blocker = bool(missing_keys)
    depends_on_missing = bool(
        not present["v35_45_subset_build"]
        or not present["v35_45_frontend_rerun"]
        or not present["v35_45_unified_slice"]
        or not present["v35_45_visualization"]
        or not present["v35_45_subset_manifest"]
        or not present["v35_46_eval"]
        or not present["v35_46_log"]
    )
    latest_zip_packaging_missing = False
    # 用户指出导出 zip 缺少这些文件；live repo 存在时按 packaging gap 记录，不当成实验 blocker。
    if not artifact_blocker:
        latest_zip_packaging_missing = True
    safe = bool(
        not artifact_blocker
        and not depends_on_missing
        and v45.get("m128_h32_larger_video_system_benchmark_claim_allowed", False)
        and v46.get("atlas_ready", False)
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_truth_audit_done": True,
        "required_artifacts": {k: {"path": rel(p), "live_repo_present": present[k]} for k, p in REQUIRED.items()},
        "required_docs": {k: {"path": rel(p), "live_repo_present": doc_present[k]} for k, p in DOCS.items()},
        "missing_artifacts": missing_keys,
        "v35_45_subset_build_json_missing": not present["v35_45_subset_build"],
        "v35_45_frontend_rerun_json_missing": not present["v35_45_frontend_rerun"],
        "v35_45_unified_slice_json_missing": not present["v35_45_unified_slice"],
        "v35_45_visualization_manifest_missing": not present["v35_45_visualization"],
        "v35_45_subset_manifest_missing": not present["v35_45_subset_manifest"],
        "v35_46_eval_json_missing": not present["v35_46_eval"],
        "v35_46_log_missing": not present["v35_46_log"],
        "v35_45_46_final_decision_depends_on_missing_json": depends_on_missing,
        "live_repo_present": not artifact_blocker,
        "latest_zip_packaging_missing": latest_zip_packaging_missing,
        "artifact_blocker": artifact_blocker,
        "v35_47_decision_safe_to_continue": safe,
        "artifact_packaging_fix_required": artifact_blocker or latest_zip_packaging_missing,
        "recommended_fix": "continue_protocol_decision_but_fix_zip_packaging" if safe and latest_zip_packaging_missing else "rematerialize_missing_artifacts",
        "中文结论": (
            "V35.45/V35.46 关键 artifacts 在 live repo 中齐全，V35.47 可以继续做 protocol decision；用户指出的 zip 缺失应记录为打包问题，而不是实验产物缺失。"
            if safe
            else "V35.45/V35.46 artifacts 存在缺口，V35.47 protocol decision 不安全，必须先补齐。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.47 V35.45/V35.46 Artifact Truth Audit\n\n"
        f"- artifact_truth_audit_done: true\n"
        f"- v35_45_subset_build_json_missing: {report['v35_45_subset_build_json_missing']}\n"
        f"- v35_45_frontend_rerun_json_missing: {report['v35_45_frontend_rerun_json_missing']}\n"
        f"- v35_45_unified_slice_json_missing: {report['v35_45_unified_slice_json_missing']}\n"
        f"- v35_45_visualization_manifest_missing: {report['v35_45_visualization_manifest_missing']}\n"
        f"- v35_45_subset_manifest_missing: {report['v35_45_subset_manifest_missing']}\n"
        f"- v35_46_eval_json_missing: {report['v35_46_eval_json_missing']}\n"
        f"- v35_46_log_missing: {report['v35_46_log_missing']}\n"
        f"- v35_47_decision_safe_to_continue: {safe}\n"
        f"- artifact_packaging_fix_required: {report['artifact_packaging_fix_required']}\n"
        f"- recommended_fix: {report['recommended_fix']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_truth_audit_done": True, "v35_47_decision_safe_to_continue": safe, "recommended_fix": report["recommended_fix"]}, ensure_ascii=False), flush=True)
    return 0 if safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
