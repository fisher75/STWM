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


TARGETS = {
    "nonoracle_selector": ROOT / "reports/stwm_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.json",
    "artifact_rematerialization": ROOT / "reports/stwm_ostf_v34_12_artifact_rematerialization_20260513.json",
    "visualization_manifest": ROOT / "reports/stwm_ostf_v34_12_local_evidence_visualization_manifest_20260513.json",
}
DOCS = {
    "nonoracle_selector": ROOT / "docs/STWM_OSTF_V34_12_NONORACLE_MEASUREMENT_SELECTOR_PROBE_20260513.md",
    "artifact_rematerialization": ROOT / "docs/STWM_OSTF_V34_12_ARTIFACT_REMATERIALIZATION_20260513.md",
    "visualization_manifest": ROOT / "docs/STWM_OSTF_V34_12_LOCAL_EVIDENCE_VISUALIZATION_20260513.md",
}
SCRIPTS = {
    "nonoracle_selector": ROOT / "code/stwm/tools/eval_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.py",
    "artifact_rematerialization": ROOT / "code/stwm/tools/rematerialize_ostf_v34_12_artifacts_20260513.py",
    "visualization_manifest": ROOT / "code/stwm/tools/render_ostf_v34_12_local_evidence_visualizations_20260513.py",
}
REPORT = ROOT / "reports/stwm_ostf_v34_13_artifact_rematerialization_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_13_ARTIFACT_REMATERIALIZATION_20260513.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_script(path: Path) -> dict[str, Any]:
    cmd = [sys.executable, str(path)]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    return {
        "script": str(path.relative_to(ROOT)),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> int:
    missing_before = {k: not v.exists() for k, v in TARGETS.items()}
    doc_present = {k: DOCS[k].exists() for k in TARGETS}
    runs: list[dict[str, Any]] = []
    recovered_from_existing_outputs = False
    for key, missing in missing_before.items():
        if missing:
            runs.append(run_script(SCRIPTS[key]))
    missing_after = {k: not v.exists() for k, v in TARGETS.items()}
    artifact_packaging_fixed = not any(missing_after.values())
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.13 artifact rematerialization 已核验并补齐 V34.12 selector/rematerialization/visualization JSON；若缺失则重新运行原始脚本，不从 MD 伪造 per-split arrays。",
        "missing_before": missing_before,
        "missing_after": missing_after,
        "docs_present": doc_present,
        "artifact_packaging_fixed": artifact_packaging_fixed,
        "recovered_from_existing_outputs": recovered_from_existing_outputs,
        "rerun_scripts": runs,
        "json_payload_keys": {k: sorted(load(v).keys()) if v.exists() else [] for k, v in TARGETS.items()},
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.13 artifact rematerialization 中文报告",
        payload,
        ["中文结论", "missing_before", "missing_after", "artifact_packaging_fixed", "recovered_from_existing_outputs", "rerun_scripts"],
    )
    print(f"已写出 V34.13 artifact rematerialization 报告: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
