#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OUT_JSON = ROOT / "reports/stwm_ostf_v32_semantic_target_route_audit_20260509.json"
OUT_MD = ROOT / "docs/STWM_OSTF_V32_SEMANTIC_TARGET_ROUTE_AUDIT_20260509.md"


def main() -> int:
    payload = {
        "generated_at_utc": utc_now(),
        "semantic_field_target_available_now": False,
        "semantic_training_run_this_round": False,
        "semantic_status": "not_tested_not_failed",
        "semantic_broadcasting_contract": {
            "semantic_is_context_only": True,
            "semantic_must_not_compress_physical_field": True,
            "future_semantic_logits_shape": "[B,M,H,C]",
            "semantic_loss_disabled_until_targets_exist": True,
        },
        "candidate_future_semantic_targets": [
            "PointOdyssey instance identity if reliable and split-safe",
            "crop teacher embeddings from observed/video-derived object crops",
            "SAM2/DINO/CLIP teacher per object or per point",
            "FSTF/TUSB semantic prototype transfer as observed semantic object token and future prototype target when aligned",
        ],
        "no_future_leakage_rule": "future semantic target may be used only as supervision, never as model input",
        "recommended_semantic_next_step": "construct explicit semantic/identity targets after trajectory field dynamics are validated",
    }
    dump_json(OUT_JSON, payload)
    write_doc(
        OUT_MD,
        "STWM OSTF V32 Semantic Target Route Audit",
        payload,
        [
            "semantic_field_target_available_now",
            "semantic_training_run_this_round",
            "semantic_status",
            "semantic_broadcasting_contract",
            "candidate_future_semantic_targets",
            "no_future_leakage_rule",
            "recommended_semantic_next_step",
        ],
    )
    print(OUT_JSON.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
