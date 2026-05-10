#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_6_identity_label_namespace_audit_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_6_IDENTITY_LABEL_NAMESPACE_AUDIT_20260509.md"
SIDE_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"


def main() -> int:
    label_to_samples: dict[tuple[str, int], set[str]] = defaultdict(set)
    total_files = 0
    for split in ("train", "val", "test"):
        for path in sorted((SIDE_ROOT / split).glob("*_M128_H32.npz")):
            total_files += 1
            z = np.load(path, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            dataset = str(np.asarray(z["dataset"]).item()) if "dataset" in z.files else "pointodyssey"
            labels = np.asarray(z["fut_instance_id"], dtype=np.int64)
            for lab in np.unique(labels[labels >= 0]):
                label_to_samples[(dataset, int(lab))].add(uid)
    collisions = {f"{k[0]}:{k[1]}": sorted(v)[:10] for k, v in label_to_samples.items() if len(v) > 1}
    train_text = (ROOT / "code/stwm/tools/train_ostf_v33_3_structured_semantic_identity_20260509.py").read_text(encoding="utf-8")
    contrastive_uses_local = "fut_instance_id" in train_text and "fut_global_instance_id" not in train_text
    eval_uses_sample = "sample_ids" in (ROOT / "code/stwm/tools/eval_ostf_v33_5_structured_semantic_identity_manifest_driven_20260509.py").read_text(encoding="utf-8")
    payload = {
        "generated_at_utc": utc_now(),
        "sidecar_file_count": total_files,
        "fut_instance_id_global_unique": not bool(collisions),
        "cross_sample_label_collision_detected": bool(collisions),
        "collision_label_count": len(collisions),
        "collision_examples": dict(list(collisions.items())[:20]),
        "contrastive_loss_uses_global_identity": False,
        "contrastive_loss_uses_sample_local_fut_instance_id": contrastive_uses_local,
        "identity_retrieval_eval_uses_sample_id_plus_instance_id": eval_uses_sample,
        "same_instance_BCE_affected_by_collision": False,
        "identity_embedding_retrieval_top1_collision_risk": True,
        "identity_training_label_safe": False,
        "exact_risk": "PointOdyssey fut_instance_id values are sample/video-local mask ids; V33.3 contrastive loss used them directly, so equal numeric ids across unrelated samples can become false positives.",
        "recommended_fix": "Build fut_global_instance_id keyed by dataset+split+sample_uid+fut_instance_id and train contrastive/retrieval losses on global labels.",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.6 Identity Label Namespace Audit", payload, ["fut_instance_id_global_unique", "cross_sample_label_collision_detected", "contrastive_loss_uses_global_identity", "identity_training_label_safe", "exact_risk", "recommended_fix"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
