#!/usr/bin/env python3
from __future__ import annotations

try:
    import setproctitle  # type: ignore
except Exception:  # pragma: no cover
    setproctitle = None

if setproctitle is not None:
    try:
        setproctitle.setproctitle("python")
    except Exception:
        pass

from run_stwm_decisive_validation_20260420 import (  # noqa: E402
    DOCS,
    REPORTS,
    build_killer_baselines_assets,
    write_json,
    write_md,
)


def main() -> None:
    payload, md = build_killer_baselines_assets()
    write_json(REPORTS / "stwm_killer_baselines_20260420.json", payload)
    write_md(DOCS / "STWM_KILLER_BASELINES_20260420.md", "STWM Killer Baselines 20260420", md)


if __name__ == "__main__":
    main()
