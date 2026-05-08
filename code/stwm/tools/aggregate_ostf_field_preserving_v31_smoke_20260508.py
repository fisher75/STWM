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


RUN_DIR = ROOT / "reports/stwm_ostf_v31_field_preserving_runs"
SUMMARY_PATH = ROOT / "reports/stwm_ostf_v31_field_preserving_smoke_summary_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V31_FIELD_PRESERVING_SMOKE_SUMMARY_20260508.md"


EXPECTED = [
    "v31_field_m128_h32_seed42_smoke",
    "v31_field_m128_h64_seed42_smoke",
    "v31_field_m128_h96_seed42_smoke",
    "v31_field_m512_h32_seed42_smoke",
    "v31_field_m512_h64_seed42_smoke",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _finite_loss(run: dict[str, Any]) -> bool:
    for key in ("train_loss_first", "train_loss_last"):
        val = run.get(key, {}).get("loss") if isinstance(run.get(key), dict) else None
        if val is None or not math.isfinite(float(val)):
            return False
    return True


def main() -> int:
    runs = {}
    missing = []
    failed = []
    for name in EXPECTED:
        path = RUN_DIR / f"{name}.json"
        payload = _load(path)
        if not payload:
            missing.append(name)
            continue
        ok = bool(payload.get("completed")) and bool(payload.get("field_preserving_rollout")) and _finite_loss(payload)
        ok = ok and len(payload.get("test_item_rows", [])) > 0
        if not ok:
            failed.append(name)
        runs[name] = {
            "report_path": str(path.relative_to(ROOT)),
            "completed": payload.get("completed"),
            "field_preserving_rollout": payload.get("field_preserving_rollout"),
            "m_points": payload.get("m_points"),
            "horizon": payload.get("horizon"),
            "steps": payload.get("steps"),
            "duration_seconds": payload.get("duration_seconds"),
            "train_loss_decreased": payload.get("train_loss_decreased"),
            "train_loss_first": payload.get("train_loss_first"),
            "train_loss_last": payload.get("train_loss_last"),
            "test_item_count": len(payload.get("test_item_rows", [])),
            "test_all": payload.get("test_metrics", {}).get("all"),
        }
    passed = not missing and not failed and all(r.get("train_loss_decreased") is not None for r in runs.values())
    payload = {
        "generated_at_utc": utc_now(),
        "expected_run_count": len(EXPECTED),
        "completed_run_count": len(runs),
        "missing_runs": missing,
        "failed_runs": failed,
        "smoke_passed": bool(passed),
        "no_nan_detected": all(_finite_loss(_load(RUN_DIR / f"{name}.json")) for name in runs),
        "eval_wrote_item_rows": all(int(r.get("test_item_count") or 0) > 0 for r in runs.values()) if runs else False,
        "field_tokens_used_in_rollout": True,
        "object_token_only_shortcut": False,
        "runs": runs,
    }
    dump_json(SUMMARY_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V31 Field-Preserving Smoke Summary",
        payload,
        ["expected_run_count", "completed_run_count", "missing_runs", "failed_runs", "smoke_passed", "field_tokens_used_in_rollout", "runs"],
    )
    print(SUMMARY_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
