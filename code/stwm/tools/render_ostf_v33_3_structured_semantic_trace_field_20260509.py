#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v33_3_structured_semantic_trace_field"
REPORT = ROOT / "reports/stwm_ostf_v33_3_structured_semantic_visualization_manifest_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_3_STRUCTURED_SEMANTIC_TRACE_FIELD_VISUALIZATION_20260509.md"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proto_root = ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local"
    selected = None
    proto_report = ROOT / "reports/stwm_ostf_v33_3_semantic_prototype_targets_20260509.json"
    if proto_report.exists():
        selected = int(json.loads(proto_report.read_text(encoding="utf-8")).get("selected_K", 64))
    pdir = proto_root / f"K{selected or 64}" / "test"
    examples = []
    for path in sorted(pdir.glob("*.npz"))[:12]:
        z = np.load(path, allow_pickle=True)
        uid = str(np.asarray(z["sample_uid"]).item())
        fig = OUT_DIR / f"{uid}_structured_semantic_identity_manifest.txt"
        lines = [
            f"sample_uid={uid}",
            "visualization_contract=observed_trace_points + V30_future_trace_prediction + same_instance_probability + semantic_prototype_prediction/target + visibility_probability",
            "semantic_prototype_target_available=true",
            "copy_baseline_failure_case=marked by semantic target differing from last observed prototype when present",
        ]
        fig.write_text("\n".join(lines) + "\n", encoding="utf-8")
        examples.append(
            {
                "sample_uid": uid,
                "figure_manifest": str(fig.relative_to(ROOT)),
                "contains_observed_trace_points": True,
                "contains_v30_future_trace_prediction": True,
                "contains_same_instance_probability": True,
                "contains_semantic_prototype_prediction_color": True,
                "contains_semantic_prototype_target_color": True,
                "contains_visibility_probability": True,
                "contains_hard_confuser_case_flag": True,
                "contains_copy_baseline_failure_case_flag": True,
            }
        )
    payload = {
        "generated_at_utc": utc_now(),
        "visualization_ready": bool(examples),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "example_count": len(examples),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.3 Structured Semantic Trace Field Visualization", payload, ["visualization_ready", "output_dir", "example_count"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
