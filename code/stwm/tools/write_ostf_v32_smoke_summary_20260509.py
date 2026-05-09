#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


RUN_DIR = ROOT / "reports/stwm_ostf_v32_recurrent_field_runs"
OUT_JSON = ROOT / "reports/stwm_ostf_v32_recurrent_field_smoke_summary_20260509.json"
OUT_MD = ROOT / "docs/STWM_OSTF_V32_RECURRENT_FIELD_SMOKE_SUMMARY_20260509.md"
SMOKE_NAMES = [
    "v32_rf_m128_h32_seed42_smoke",
    "v32_rf_m128_h64_seed42_smoke",
    "v32_rf_m128_h96_seed42_smoke",
    "v32_rf_m512_h32_seed42_smoke",
    "v32_rf_m512_h64_seed42_smoke",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _finite_positive(payload: dict[str, Any], key: str) -> bool:
    try:
        val = float(payload.get(key))
    except (TypeError, ValueError):
        return False
    return math.isfinite(val) and val > 0


def main() -> int:
    runs: dict[str, Any] = {}
    missing = []
    failed = []
    for name in SMOKE_NAMES:
        payload = _load(RUN_DIR / f"{name}.json")
        if not payload:
            missing.append(name)
            continue
        ok = bool(payload.get("completed")) and bool(payload.get("train_loss_decreased"))
        test_all = payload.get("test_metrics", {}).get("all", {})
        shape_ok = bool(payload.get("recurrent_field_dynamics")) and bool(payload.get("field_preserving_rollout"))
        loop_ok = payload.get("train_loss_last", {}).get("recurrent_loop_steps") in (32.0, 64.0, 96.0)
        mem = payload.get("gpu_peak_memory_mib")
        row = {
            "report_path": str((RUN_DIR / f"{name}.json").relative_to(ROOT)),
            "completed": payload.get("completed"),
            "M": payload.get("m_points"),
            "H": payload.get("horizon"),
            "device": payload.get("device"),
            "cuda_visible_devices": payload.get("cuda_visible_devices"),
            "batch_size": payload.get("batch_size"),
            "effective_batch_size": payload.get("effective_batch_size"),
            "duration_seconds": payload.get("duration_seconds"),
            "gpu_peak_memory_mib": mem,
            "train_loss_decreased": payload.get("train_loss_decreased"),
            "no_nan_proxy_metrics_finite": _finite_positive(test_all, "minFDE_K"),
            "eval_item_count": test_all.get("item_count"),
            "output_contract_ok": shape_ok,
            "recurrent_loop_ran_expected_h_steps": bool(loop_ok),
            "global_motion_prior_branch": payload.get("global_motion_prior_branch"),
            "test_all": test_all,
        }
        runs[name] = row
        if not (ok and shape_ok and loop_ok):
            failed.append(name)
    smoke_passed = not missing and not failed
    payload = {
        "generated_at_utc": utc_now(),
        "expected_run_count": len(SMOKE_NAMES),
        "completed_run_count": len(runs),
        "missing_runs": missing,
        "failed_runs": failed,
        "smoke_passed": smoke_passed,
        "runs": runs,
    }
    dump_json(OUT_JSON, payload)
    write_doc(
        OUT_MD,
        "STWM OSTF V32 Recurrent Field Smoke Summary",
        payload,
        ["expected_run_count", "completed_run_count", "missing_runs", "failed_runs", "smoke_passed"],
    )
    print(OUT_JSON.relative_to(ROOT))
    return 0 if smoke_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
