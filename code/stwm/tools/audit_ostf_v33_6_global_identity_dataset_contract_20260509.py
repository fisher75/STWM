#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_6_global_identity_dataset_contract_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_6_GLOBAL_IDENTITY_DATASET_CONTRACT_20260509.md"


REQUIRED_KEYS = [
    "fut_global_instance_id",
    "fut_global_instance_available_mask",
    "obs_global_instance_id",
    "point_global_instance_id",
    "global_identity_label_available",
    "global_identity_leakage_safe",
    "future_global_labels_supervision_only",
]


def tensor_shape(batch: dict[str, Any], key: str) -> list[int] | None:
    val = batch.get(key)
    return list(val.shape) if isinstance(val, torch.Tensor) else None


def main() -> int:
    errors: list[str] = []
    batch: dict[str, Any] = {}
    try:
        ds = OSTFExternalGTDataset(
            "train",
            horizon=32,
            m_points=128,
            max_items=4,
            enable_semantic_identity_sidecar=True,
            require_semantic_identity_sidecar=True,
            enable_global_identity_labels=True,
            require_global_identity_labels=True,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_external_gt)
        batch = next(iter(loader))
    except Exception as exc:  # pragma: no cover - report exact blocker to user.
        errors.append(repr(exc))
        ds = None  # type: ignore[assignment]

    missing = [k for k in REQUIRED_KEYS if k not in batch]
    shapes = {k: tensor_shape(batch, k) for k in REQUIRED_KEYS if k in batch}
    leakage_safe = bool(batch.get("global_identity_leakage_safe", torch.tensor([False])).bool().all().item()) if batch else False
    supervision_only = bool(batch.get("future_global_labels_supervision_only", torch.tensor([False])).bool().all().item()) if batch else False
    available = bool(batch.get("global_identity_label_available", torch.tensor([False])).bool().all().item()) if batch else False
    contract_ok = not errors and not missing and leakage_safe and supervision_only and available
    payload = {
        "generated_at_utc": utc_now(),
        "dataset_contract_ok": contract_ok,
        "sidecar_dataset_integrated": contract_ok,
        "sample_count_checked": len(ds) if ds is not None else 0,
        "required_keys": REQUIRED_KEYS,
        "missing_keys": missing,
        "shapes": shapes,
        "global_identity_label_available": available,
        "leakage_safe": leakage_safe,
        "future_global_labels_supervision_only": supervision_only,
        "errors": errors,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.6 Global Identity Dataset Contract",
        payload,
        ["dataset_contract_ok", "sidecar_dataset_integrated", "global_identity_label_available", "leakage_safe", "future_global_labels_supervision_only", "missing_keys", "errors"],
    )
    print(REPORT.relative_to(ROOT))
    return 0 if contract_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
