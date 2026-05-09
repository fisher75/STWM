#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_semantic_identity_code_contract_audit_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_SEMANTIC_IDENTITY_CODE_CONTRACT_AUDIT_20260509.md"


def contains(path: str, needle: str) -> bool:
    p = ROOT / path
    return p.exists() and needle in p.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    dataset = "code/stwm/datasets/ostf_v30_external_gt_dataset_20260508.py"
    model = "code/stwm/modules/ostf_external_gt_world_model_v30.py"
    train = "code/stwm/tools/train_ostf_external_gt_v30_20260508.py"
    payload = {
        "generated_at_utc": utc_now(),
        "dataset_can_load_semantic_identity_sidecar_now": contains(dataset, "semantic_identity") or contains(dataset, "sidecar"),
        "batch_can_include_point_id": contains(dataset, "point_id"),
        "batch_can_include_point_to_instance_id": contains(dataset, "point_to_instance_id"),
        "batch_can_include_fut_same_instance_as_obs": contains(dataset, "fut_same_instance_as_obs"),
        "batch_can_include_teacher_embedding": contains(dataset, "teacher_embedding"),
        "model_has_semantic_logits": contains(model, "semantic_logits"),
        "train_has_semantic_or_identity_loss": contains(train, "semantic_loss") or contains(train, "identity") or contains(train, "same_instance"),
        "should_add_identity_embedding_head": True,
        "should_freeze_trajectory_backbone_first": True,
        "needs_trajectory_preservation_loss": True,
        "minimal_change_plan": [
            "keep V30_M128 trajectory backbone selected",
            "load V33 semantic_identity sidecar in a dedicated head dataset rather than mutating V30 cache",
            "freeze trajectory backbone for first smoke/pilot",
            "train point-persistence and same-instance heads only",
            "treat class semantic and visual teacher prototypes as future work until target cache exists",
        ],
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 Semantic Identity Code Contract Audit", payload, [
        "dataset_can_load_semantic_identity_sidecar_now",
        "model_has_semantic_logits",
        "train_has_semantic_or_identity_loss",
        "should_add_identity_embedding_head",
        "should_freeze_trajectory_backbone_first",
        "needs_trajectory_preservation_loss",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
