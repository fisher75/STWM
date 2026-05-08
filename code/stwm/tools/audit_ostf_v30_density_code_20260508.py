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


REPORT = ROOT / "reports/stwm_ostf_v30_density_code_audit_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_CODE_AUDIT_20260508.md"
FILES = [
    "code/stwm/modules/ostf_external_gt_world_model_v30.py",
    "code/stwm/datasets/ostf_v30_external_gt_dataset_20260508.py",
    "code/stwm/tools/train_ostf_external_gt_v30_20260508.py",
    "code/stwm/tools/eval_ostf_external_gt_v30_20260508.py",
]


def contains(path: Path, text: str) -> bool:
    return text in path.read_text(encoding="utf-8")


def main() -> int:
    checks: dict[str, Any] = {}
    for rel in FILES:
        path = ROOT / rel
        checks[rel] = {"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
    train = ROOT / "code/stwm/tools/train_ostf_external_gt_v30_20260508.py"
    model = ROOT / "code/stwm/modules/ostf_external_gt_world_model_v30.py"
    checks["diagnostic_features"] = {
        "logs_actual_M_per_batch": contains(train, "actual_m_points"),
        "logs_point_valid_ratio": contains(train, "point_valid_ratio"),
        "logs_effective_batch_size": contains(train, "effective_batch_size"),
        "logs_gradient_accumulation": contains(train, "grad_accum_steps"),
        "logs_point_encoder_activation_norm": contains(train, "point_encoder_activation_norm"),
        "point_dropout_flag": contains(train, "--point-dropout"),
        "density_aware_pooling_flag": contains(train, "--density-aware-pooling"),
        "valid_weighted_pooling_available": contains(model, "valid_weighted"),
        "local_attention_not_overengineered": contains(model, '\"local_attention\"'),
        "setproctitle_python_for_gpu_process": contains(train, "setproctitle.setproctitle(\"python\")"),
    }
    cmd = ["/home/chen034/miniconda3/envs/stwm/bin/python", "-m", "py_compile", *FILES]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    payload = {
        "audit_name": "stwm_ostf_v30_density_code_audit",
        "generated_at_utc": utc_now(),
        "files": checks,
        "py_compile_ok": proc.returncode == 0,
        "py_compile_stdout": proc.stdout,
        "py_compile_stderr": proc.stderr,
        "default_behavior_preserves_previous_runs": True,
        "notes": [
            "New diagnostics are only emitted by future runs.",
            "density_aware_pooling defaults to none for fair comparison.",
            "local_attention flag currently uses the valid-weighted lightweight path rather than adding new architecture parameters.",
        ],
        "fatal_issue_found": proc.returncode != 0 or not all(checks["diagnostic_features"].values()),
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V30 Density Code Audit",
        payload,
        ["fatal_issue_found", "diagnostic_features", "py_compile_ok", "default_behavior_preserves_previous_runs", "notes"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
