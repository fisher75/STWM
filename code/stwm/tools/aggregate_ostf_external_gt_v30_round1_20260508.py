#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.aggregate_ostf_external_gt_v30_20260508 import main


if __name__ == "__main__":
    raise SystemExit(main())
