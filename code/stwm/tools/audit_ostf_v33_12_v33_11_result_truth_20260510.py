#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_12_v33_11_result_truth_audit_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_12_V33_11_RESULT_TRUTH_AUDIT_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    dec = load("reports/stwm_ostf_v33_11_decision_20260510.json")
    eval_dec = load("reports/stwm_ostf_v33_11_identity_preserving_copy_residual_eval_decision_20260510.json")
    train = load("reports/stwm_ostf_v33_11_identity_preserving_copy_residual_train_summary_20260510.json")
    baseline = load("reports/stwm_ostf_v33_11_semantic_baseline_bank_20260510.json")
    oracle_doc = ROOT / "docs/STWM_OSTF_V33_11_ORACLE_GATE_UPPER_BOUND_20260510.md"
    oracle_script = (ROOT / "code/stwm/tools/eval_ostf_v33_11_oracle_gate_upper_bound_20260510.py").read_text(encoding="utf-8")
    model_code = (ROOT / "code/stwm/modules/ostf_v33_11_identity_preserving_copy_residual_semantic_world_model.py").read_text(encoding="utf-8")
    default_uses_v3310 = "stwm_ostf_v33_10_copy_residual_semantic_h32_m128" in oracle_script
    ckpt = ROOT / str(train.get("checkpoint_path", ""))
    teacher = "clip_vit_b32_local"
    proto_k = 32
    proto_path = ROOT / "outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K32/prototype_vocab.npz"
    if proto_path.exists():
        import numpy as np

        z = np.load(proto_path, allow_pickle=True)
        teacher = str(z["teacher_name"].item()) if "teacher_name" in z.files else teacher
        proto_k = int(z["K"].item()) if "K" in z.files else proto_k
    identity_fixed = bool(train.get("identity_path_frozen_or_distilled") and not eval_dec.get("identity_regressed_vs_v33_9"))
    stable_failed = bool(eval_dec.get("stable_preservation_not_degraded_top5") is False)
    changed_hard_failed = bool(eval_dec.get("changed_top5_beats_strongest_baseline") is False or eval_dec.get("semantic_hard_top5_beats_strongest_baseline") is False)
    suspected_target = bool(changed_hard_failed and "prototype_target_space_bottleneck" in oracle_doc.read_text(encoding="utf-8") if oracle_doc.exists() else changed_hard_failed)
    payload = {
        "generated_at_utc": utc_now(),
        "v33_11_checkpoint_fresh": bool(train.get("fresh_training_completed") and ckpt.exists()),
        "v33_11_checkpoint_path": train.get("checkpoint_path"),
        "identity_path_frozen_or_distilled": bool(train.get("identity_path_frozen_or_distilled")),
        "identity_preservation_fixed": identity_fixed,
        "stable_preservation_failed": stable_failed,
        "changed_hard_failed_vs_strongest_baseline": changed_hard_failed,
        "v33_11_oracle_not_actually_run": default_uses_v3310,
        "current_teacher_name": teacher,
        "current_prototype_K": proto_k,
        "strongest_baseline_by_subset": baseline.get("strongest_baseline_by_subset_selected_on_val", {}),
        "prototype_target_space_bottleneck_suspected": suspected_target,
        "semantic_path_uses_clip_k32": "clip_vit_b32_local/K32" in str(train),
        "recommended_fix": "run_true_v33_11_oracle_then_repair_semantic_teacher_prototype_target_space",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.12 V33.11 Result Truth Audit", payload, ["v33_11_checkpoint_fresh", "identity_preservation_fixed", "stable_preservation_failed", "changed_hard_failed_vs_strongest_baseline", "v33_11_oracle_not_actually_run", "current_teacher_name", "current_prototype_K", "strongest_baseline_by_subset", "prototype_target_space_bottleneck_suspected", "recommended_fix"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
