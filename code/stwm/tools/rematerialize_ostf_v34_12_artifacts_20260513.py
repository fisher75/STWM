#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


QUALITY_JSON = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_quality_probe_20260513.json"
VIS_JSON = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_causality_visualization_manifest_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_12_artifact_rematerialization_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_ARTIFACT_REMATERIALIZATION_20260513.md"


def run_tool(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main() -> int:
    missing_before = {
        "quality_probe_json_missing": not QUALITY_JSON.exists(),
        "visualization_json_missing": not VIS_JSON.exists(),
    }
    recovered = False
    reran: list[str] = []
    if not QUALITY_JSON.exists():
        run_tool("code/stwm/tools/eval_ostf_v34_11_semantic_measurement_quality_probe_20260513.py")
        reran.append("eval_ostf_v34_11_semantic_measurement_quality_probe_20260513.py")
    if not VIS_JSON.exists():
        run_tool("code/stwm/tools/render_ostf_v34_11_semantic_measurement_causality_visualizations_20260513.py")
        reran.append("render_ostf_v34_11_semantic_measurement_causality_visualizations_20260513.py")
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 artifact rematerialization 已核验 V34.11 quality/visual JSON；当前 repo 中两者均存在，不需要从 MD 伪恢复 per-split arrays。",
        "missing_before": missing_before,
        "quality_probe_json_present": QUALITY_JSON.exists(),
        "visualization_json_present": VIS_JSON.exists(),
        "artifact_packaging_fixed": bool(QUALITY_JSON.exists() and VIS_JSON.exists()),
        "recovered_from_existing_outputs": recovered,
        "rerun_scripts": reran,
        "quality_probe_json_path": str(QUALITY_JSON.relative_to(ROOT)) if QUALITY_JSON.exists() else None,
        "visualization_json_path": str(VIS_JSON.relative_to(ROOT)) if VIS_JSON.exists() else None,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.12 artifact rematerialization 中文报告", payload, ["中文结论", "missing_before", "quality_probe_json_present", "visualization_json_present", "artifact_packaging_fixed", "recovered_from_existing_outputs", "rerun_scripts"])
    print(f"已写出 V34.12 artifact rematerialization: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
