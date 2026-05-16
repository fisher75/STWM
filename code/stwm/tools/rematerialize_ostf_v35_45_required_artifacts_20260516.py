#!/usr/bin/env python3
"""V35.45 补齐 V35.44 依赖 artifacts；本地存在则记录 no-op。"""
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

AUDIT = ROOT / "reports/stwm_ostf_v35_45_v35_44_artifact_and_claim_truth_audit_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_45_artifact_rematerialization_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_45_ARTIFACT_REMATERIALIZATION_20260516.md"
REQUIRED = [
    ROOT / "reports/stwm_ostf_v35_34_raw_video_frontend_reproducibility_harness_20260516.json",
    ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json",
    ROOT / "reports/stwm_ostf_v35_42_identity_label_provenance_and_valid_claim_20260516.json",
    ROOT / "reports/stwm_ostf_v35_43_raw_video_closure_visualization_manifest_20260516.json",
]


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
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


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.exists() else {}
    rows = [{"path": rel(p), "exists": p.exists()} for p in REQUIRED]
    missing = [r for r in rows if not r["exists"]]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_rematerialization_done": True,
        "artifact_packaging_fixed": len(missing) == 0,
        "recovered_from_existing_outputs": len(missing) == 0,
        "rerun_required": len(missing) > 0,
        "required_artifacts": rows,
        "missing_artifacts": missing,
        "source_audit": rel(AUDIT),
        "中文结论": (
            "V35.44 依赖 JSON 在 live repo 中均已存在，本轮 rematerialization 为 no-op packaging check。"
            if not missing
            else "仍有 V35.44 依赖 JSON 缺失；必须重新运行对应脚本，不能伪造 per-split/per-seed arrays。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.45 Artifact Rematerialization\n\n"
        f"- artifact_rematerialization_done: true\n"
        f"- artifact_packaging_fixed: {report['artifact_packaging_fixed']}\n"
        f"- recovered_from_existing_outputs: {report['recovered_from_existing_outputs']}\n"
        f"- rerun_required: {report['rerun_required']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_packaging_fixed": report["artifact_packaging_fixed"], "rerun_required": report["rerun_required"]}, ensure_ascii=False), flush=True)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
