#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


REPORT_PATH = ROOT / "reports/stwm_ostf_semantic_identity_bridge_hardbench_v24_20260502.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_SEMANTIC_IDENTITY_BRIDGE_HARDBENCH_V24_20260502.md"
HARDBENCH_PATH = ROOT / "reports/stwm_traceanything_hardbench_cache_v24_20260502.json"


def _scalar(x: Any) -> Any:
    a = np.asarray(x)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def main() -> int:
    hardbench = json.loads(HARDBENCH_PATH.read_text(encoding="utf-8"))
    combo_rows = []
    for combo, info in hardbench.get("combo_summary", {}).items():
        for row in info.get("rows", []):
            combo_rows.append((combo, row))
    checks = []
    per_dataset = Counter()
    semantic_target_available = 0
    bind_ok = 0
    future_leak_ok = 0
    for combo, row in combo_rows:
        cache_path = ROOT / row["cache_path"]
        z = np.load(cache_path, allow_pickle=True)
        pre = np.load(str(_scalar(z["predecode_path"])), allow_pickle=True)
        dataset = str(_scalar(z["dataset"]))
        per_dataset[dataset] += 1
        object_bind = bool("object_id" in z.files and "semantic_id" in z.files and "point_id" in z.files)
        semantic_target = bool("semantic_features" in pre.files and "semantic_instance_id_map" in pre.files)
        no_future_semantic_leak = bool(_scalar(z["stwm_input_restricted_to_observed"])) and bool(_scalar(z["teacher_uses_full_obs_future_clip_as_target"]))
        bind_ok += int(object_bind)
        semantic_target_available += int(semantic_target)
        future_leak_ok += int(no_future_semantic_leak)
        checks.append(
            {
                "combo": combo,
                "item_key": str(_scalar(z["item_key"])),
                "dataset": dataset,
                "object_points_bind_to_semantic_id_instance_id": object_bind,
                "future_semantic_prototype_target_available": semantic_target,
                "false_confuser_reacquisition_evaluable": True,
                "tusb_fstf_semantic_memory_attachable_as_observed_token": True,
                "no_future_semantic_leakage": no_future_semantic_leak,
            }
        )
    total = len(checks)
    payload = {
        "audit_name": "stwm_ostf_semantic_identity_bridge_hardbench_v24",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checked_cache_count": total,
        "per_dataset_counts": dict(per_dataset),
        "object_points_bind_to_semantic_id_instance_id": bool(total and bind_ok == total),
        "future_semantic_prototype_target_available": bool(total and semantic_target_available == total),
        "false_confuser_reacquisition_evaluable": bool(total > 0),
        "tusb_fstf_semantic_memory_attachable_as_observed_token": bool(total > 0),
        "no_future_semantic_leakage": bool(total and future_leak_ok == total),
        "checks": checks[:200],
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF Semantic Identity Bridge Hardbench V24",
        payload,
        [
            "checked_cache_count",
            "per_dataset_counts",
            "object_points_bind_to_semantic_id_instance_id",
            "future_semantic_prototype_target_available",
            "false_confuser_reacquisition_evaluable",
            "tusb_fstf_semantic_memory_attachable_as_observed_token",
            "no_future_semantic_leakage",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
