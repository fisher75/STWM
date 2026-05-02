#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, verify_v16_cache, write_doc


def main() -> int:
    result = verify_v16_cache()
    result["audit_name"] = "stwm_ostf_v17_cache_verification"
    result["next_step"] = "train_v17_h8" if result["cache_verification_passed"] else "fix_v16_cache"
    dump_json(ROOT / "reports/stwm_ostf_v17_cache_verification_20260502.json", result)
    write_doc(
        ROOT / "docs/STWM_OSTF_V17_CACHE_VERIFICATION_20260502.md",
        "STWM OSTF V17 Cache Verification",
        result,
        [
            "cache_verification_passed",
            "cache_exists",
            "processed_clip_count",
            "valid_point_ratio",
            "teacher_source",
            "persistent_point_identity_valid",
            "fake_dense_or_anchor_copied",
            "raw_visualization_exists",
            "future_leakage_audit_passed",
            "next_step",
        ],
    )
    print("reports/stwm_ostf_v17_cache_verification_20260502.json")
    return 0 if result["cache_verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
