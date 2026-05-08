#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import DATASET_NAMES, ROOT, audit_dataset, data_roots, manifest_roots, utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_data_root_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_DATA_ROOT_AUDIT_20260508.md"


def main() -> int:
    datasets = {name: audit_dataset(name) for name in DATASET_NAMES}
    payload = {
        "audit_name": "stwm_ostf_v30_external_gt_data_root_audit",
        "generated_at_utc": utc_now(),
        "searched_data_roots": [str(p) for p in data_roots()],
        "searched_manifest_roots": [str(p) for p in manifest_roots()],
        "datasets": datasets,
        "summary": {
            "external_gt_data_available": any(v["gt_point_trajectories_available"] for v in datasets.values()),
            "pointodyssey_complete": datasets["pointodyssey"]["completeness_status"] == "complete",
            "tapvid_complete": datasets["tapvid"]["completeness_status"] == "complete",
            "tapvid3d_complete": datasets["tapvid3d"]["completeness_status"] == "complete",
            "partial_usable": [
                k
                for k, v in datasets.items()
                if v["completeness_status"] in {"complete", "partial"} and v["gt_point_trajectories_available"]
            ],
        },
    }
    dump_json(REPORT_PATH, payload)
    doc_payload = {
        "external_gt_data_available": payload["summary"]["external_gt_data_available"],
        "pointodyssey_complete": payload["summary"]["pointodyssey_complete"],
        "tapvid_complete": payload["summary"]["tapvid_complete"],
        "tapvid3d_complete": payload["summary"]["tapvid3d_complete"],
        "partial_usable": payload["summary"]["partial_usable"],
        "searched_data_roots": payload["searched_data_roots"],
    }
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 External GT Data Root Audit",
        doc_payload,
        ["external_gt_data_available", "pointodyssey_complete", "tapvid_complete", "tapvid3d_complete", "partial_usable", "searched_data_roots"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
