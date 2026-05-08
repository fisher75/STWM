#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

SUMMARY = ROOT / "reports/stwm_ostf_v30_density_m1024_smoke_summary_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_M1024_SMOKE_SUMMARY_20260508.md"
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    per_run = []
    ok_count = 0
    for h in (32, 64, 96):
        name = f"v30_extgt_m1024_h{h}_seed42_smoke"
        path = RUN_DIR / f"{name}.json"
        if not path.exists():
            per_run.append({"experiment_name": name, "completed": False, "missing": True})
            continue
        payload = read_json(path)
        valid = bool(payload.get("completed") and payload.get("test_item_rows") and payload.get("checkpoint_path"))
        no_nan = payload.get("train_loss_last", {}).get("loss") is not None
        ok = bool(valid and no_nan)
        ok_count += int(ok)
        per_run.append(
            {
                "experiment_name": name,
                "completed": bool(payload.get("completed")),
                "report_path": str(path.relative_to(ROOT)),
                "steps": payload.get("steps"),
                "batch_size": payload.get("batch_size"),
                "grad_accum_steps": payload.get("grad_accum_steps"),
                "effective_batch_size": payload.get("effective_batch_size"),
                "train_loss_decreased": payload.get("train_loss_decreased"),
                "test_item_row_count": len(payload.get("test_item_rows", [])),
                "minFDE_K": payload.get("test_metrics", {}).get("all", {}).get("minFDE_K"),
                "point_valid_ratio_last": (payload.get("train_loss_last") or {}).get("point_valid_ratio"),
                "point_encoder_activation_norm_last": (payload.get("train_loss_last") or {}).get("point_encoder_activation_norm"),
                "smoke_passed": ok,
            }
        )
    summary = {
        "summary_name": "stwm_ostf_v30_density_m1024_smoke_summary",
        "generated_at_utc": utc_now(),
        "m1024_smoke_passed": ok_count == 3,
        "passed_run_count": ok_count,
        "expected_run_count": 3,
        "per_run": per_run,
        "semantic_not_tested_not_failed": True,
    }
    dump_json(SUMMARY, summary)
    write_doc(
        DOC,
        "STWM OSTF V30 Density M1024 Smoke Summary",
        summary,
        ["m1024_smoke_passed", "passed_run_count", "expected_run_count", "per_run", "semantic_not_tested_not_failed"],
    )
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
