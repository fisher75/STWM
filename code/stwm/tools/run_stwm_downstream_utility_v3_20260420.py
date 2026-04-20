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
    build_downstream_utility_v3_assets,
    write_json,
    write_md,
)


def main() -> None:
    payload, md = build_downstream_utility_v3_assets()
    write_json(REPORTS / "stwm_downstream_utility_v3_20260420.json", payload)
    write_md(DOCS / "STWM_DOWNSTREAM_UTILITY_V3_20260420.md", "STWM Downstream Utility V3 20260420", md)


if __name__ == "__main__":
    main()
