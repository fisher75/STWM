#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v30_density_pooling_smoke_summary_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_POOLING_SMOKE_SUMMARY_20260508.md"
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
MODES = ("moments", "induced_attention", "hybrid_moments_attention")
MS = (512, 1024)
HS = (32, 64, 96)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    rows = []
    for mode in MODES:
        for m in MS:
            for h in HS:
                name = f"v30_extgt_density_smoke_m{m}_h{h}_{mode}_seed42"
                path = RUN_DIR / f"{name}.json"
                report = load(path)
                rows.append(
                    {
                        "experiment_name": name,
                        "report_path": str(path.relative_to(ROOT)),
                        "exists": path.exists(),
                        "completed": bool(report.get("completed")),
                        "M": m,
                        "H": h,
                        "pooling_mode": mode,
                        "train_loss_decreased": report.get("train_loss_decreased"),
                        "minFDE_K": report.get("test_metrics", {}).get("all", {}).get("minFDE_K"),
                        "threshold_auc_endpoint_16_32_64_128": report.get("test_metrics", {}).get("all", {}).get("threshold_auc_endpoint_16_32_64_128"),
                        "point_valid_ratio": report.get("train_loss_last", {}).get("point_valid_ratio"),
                        "density_attention_entropy": report.get("train_loss_last", {}).get("density_attention_entropy"),
                        "object_token_norm": report.get("train_loss_last", {}).get("object_token_norm"),
                        "duration_seconds": report.get("duration_seconds"),
                    }
                )
    payload = {
        "summary_name": "stwm_ostf_v30_density_pooling_smoke_summary",
        "generated_at_utc": utc_now(),
        "expected_run_count": len(rows),
        "completed_run_count": sum(1 for r in rows if r["completed"]),
        "failed_or_missing": [r for r in rows if not r["completed"]],
        "smoke_passed": all(r["completed"] for r in rows),
        "rows": rows,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V30 Density Pooling Smoke Summary", payload, ["expected_run_count", "completed_run_count", "smoke_passed", "failed_or_missing"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
