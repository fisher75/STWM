#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_1_sidecar_dataset_contract_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_1_SIDECAR_DATASET_CONTRACT_20260509.md"


def main() -> int:
    required = [
        "point_id",
        "point_to_instance_id",
        "obs_instance_id",
        "obs_instance_available_mask",
        "fut_instance_id",
        "fut_instance_available_mask",
        "fut_same_instance_as_obs",
        "fut_point_visible_target",
        "fut_point_visible_mask",
        "point_assignment_confidence",
        "point_to_instance_assignment_method",
        "leakage_safe",
        "input_uses_observed_only",
        "future_targets_supervision_only",
    ]
    payload: dict[str, Any] = {"generated_at_utc": utc_now()}
    try:
        ds = OSTFExternalGTDataset("train", horizon=32, m_points=128, max_items=4, enable_semantic_identity_sidecar=True, require_semantic_identity_sidecar=True)
        sample = ds[0]
        batch = collate_external_gt([ds[i] for i in range(min(2, len(ds)))])
        missing = [k for k in required if k not in sample]
        shapes = {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")}
        payload.update(
            {
                "sidecar_dataset_integrated": not missing,
                "dataset_sidecar_enabled": True,
                "require_sidecar_supported": True,
                "use_observed_instance_context_default": bool(sample["use_observed_instance_context"].item()),
                "missing_required_keys": missing,
                "sample_shapes": shapes,
                "batch_keys": sorted(batch.keys()),
                "leakage_safe": bool(sample["leakage_safe"].item()),
                "input_uses_observed_only": bool(sample["input_uses_observed_only"].item()),
                "future_targets_supervision_only": bool(sample["future_targets_supervision_only"].item()),
                "contract_passed": not missing and bool(sample["leakage_safe"].item()) and bool(sample["input_uses_observed_only"].item()) and bool(sample["future_targets_supervision_only"].item()),
            }
        )
    except Exception as exc:
        payload.update({"sidecar_dataset_integrated": False, "contract_passed": False, "exact_error": f"{type(exc).__name__}: {exc}"})
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.1 Sidecar Dataset Contract", payload, ["contract_passed", "sidecar_dataset_integrated", "use_observed_instance_context_default", "missing_required_keys", "leakage_safe", "input_uses_observed_only", "future_targets_supervision_only"])
    print(REPORT.relative_to(ROOT))
    return 0 if payload.get("contract_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
