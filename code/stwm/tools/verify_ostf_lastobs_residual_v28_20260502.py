#!/usr/bin/env python3
from __future__ import annotations

from stwm.tools.ostf_lastobs_v28_common_20260502 import write_v28_cache_hardbench_verification


def main() -> int:
    payload = write_v28_cache_hardbench_verification()
    print(
        "[V28][cache-hardbench-verify] "
        f"cache_verified={int(payload['cache_verified'])} "
        f"strongest_prior={payload['strongest_nonlearned_prior']} "
        f"revised_hard_subsets={int(payload['revised_hard_subsets_exist'])} "
        f"no_future_leakage={int(payload['no_future_leakage'])}",
        flush=True,
    )
    return 0 if payload["cache_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
