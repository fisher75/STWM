#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from stwm.tools.ostf_v17_common_20260502 import ROOT


def main() -> int:
    summary = ROOT / "reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json"
    print(summary.relative_to(ROOT) if summary.exists() else "reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
