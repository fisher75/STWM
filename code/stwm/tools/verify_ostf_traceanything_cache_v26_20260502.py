#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stwm.tools.ostf_traceanything_common_v26_20260502 import ROOT, write_cache_verification_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports/stwm_ostf_v26_cache_verification_20260502.json"),
    )
    parser.add_argument(
        "--doc-path",
        default=str(ROOT / "docs/STWM_OSTF_V26_CACHE_VERIFICATION_20260502.md"),
    )
    args = parser.parse_args()
    payload = write_cache_verification_outputs(Path(args.report_path), Path(args.doc_path))
    print(f"[V26][cache-verify] cache_verified={int(payload['cache_verified'])} h32_ready={int(payload['h32_ready'])} h64_ready={int(payload['h64_ready'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
