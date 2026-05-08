#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.aggregate_ostf_field_preserving_v31_pilot_20260508 import main as pilot_main
from stwm.tools.aggregate_ostf_field_preserving_v31_smoke_20260508 import main as smoke_main


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "pilot"), default="pilot")
    args = p.parse_args()
    return smoke_main() if args.mode == "smoke" else pilot_main()


if __name__ == "__main__":
    raise SystemExit(main())
