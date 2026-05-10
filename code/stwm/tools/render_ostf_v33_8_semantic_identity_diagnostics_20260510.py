#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OUT = ROOT / "outputs/figures/stwm_ostf_v33_8_semantic_identity_diagnostics"
REPORT = ROOT / "reports/stwm_ostf_v33_8_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_SEMANTIC_IDENTITY_DIAGNOSTIC_VISUALIZATION_20260510.md"
MASK = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic/H32_M128_seed42.json"


def load_manifest() -> dict[str, Any]:
    return json.loads(MASK.read_text(encoding="utf-8")) if MASK.exists() else {"splits": {}}


def select_cases() -> dict[str, list[dict[str, Any]]]:
    payload = load_manifest()
    cases: dict[str, list[dict[str, Any]]] = {
        "hard_identity_positive": [],
        "hard_identity_negative": [],
        "same_frame_confuser": [],
        "semantic_copy_baseline_failure": [],
        "semantic_improvement": [],
        "occlusion_or_reappearance": [],
    }
    for split in ("val", "test"):
        for entry in payload.get("splits", {}).get(split, []):
            mask_path = ROOT / entry["mask_path"]
            if not mask_path.exists():
                continue
            z = np.load(mask_path, allow_pickle=True)
            im = np.asarray(z["identity_hard_eval_mask"]).astype(bool)
            sm = np.asarray(z["semantic_hard_eval_mask"]).astype(bool)
            pos = int(entry.get("selected_identity_positives", entry.get("selected_identity_positive", 0)))
            neg = int(entry.get("selected_identity_negatives", entry.get("selected_identity_negative", 0)))
            base = {
                "split": split,
                "sample_uid": entry["sample_uid"],
                "mask_path": entry["mask_path"],
                "identity_hard_count": int(im.sum()),
                "semantic_hard_count": int(sm.sum()),
            }
            if pos > 0 and len(cases["hard_identity_positive"]) < 8:
                cases["hard_identity_positive"].append({**base, "selected_identity_positives": pos})
            if neg > 0 and len(cases["hard_identity_negative"]) < 8:
                cases["hard_identity_negative"].append({**base, "selected_identity_negatives": neg})
            if pos > 0 and neg > 0 and len(cases["same_frame_confuser"]) < 8:
                cases["same_frame_confuser"].append({**base, "reason": "same manifest item contains actual positive and negative identity labels"})
            if int(sm.sum()) > 0 and len(cases["semantic_copy_baseline_failure"]) < 8:
                cases["semantic_copy_baseline_failure"].append({**base, "reason": "semantic_hard_eval_mask nonempty; compare predicted prototype vs observed-copy baseline"})
            if int(sm.sum()) > 0 and len(cases["semantic_improvement"]) < 8:
                cases["semantic_improvement"].append({**base, "reason": "candidate slot reserved when semantic prediction beats copy in eval"})
    return cases


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = select_cases()
    blockers = []
    if not cases["hard_identity_positive"]:
        blockers.append("no hard identity positive visualization cases found")
    if not cases["hard_identity_negative"]:
        blockers.append("no hard identity negative visualization cases found")
    if not cases["semantic_copy_baseline_failure"]:
        blockers.append("no semantic hard/copy-baseline diagnostic cases found")
    # Write lightweight per-case JSON descriptors; figure rendering can consume
    # these without changing the model/eval protocol.
    for name, rows in cases.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "generated_at_utc": utc_now(),
        "output_dir": str(OUT.relative_to(ROOT)),
        "visualization_ready": len(blockers) == 0,
        "case_manifest_paths": {name: str((OUT / f"{name}.json").relative_to(ROOT)) for name in cases},
        "case_counts": {name: len(rows) for name, rows in cases.items()},
        "required_views": [
            "observed trace",
            "V30 future trace",
            "predicted identity belief",
            "semantic prototype prediction",
            "semantic target",
            "visibility",
        ],
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.8 Semantic Identity Diagnostic Visualization",
        payload,
        ["output_dir", "visualization_ready", "case_counts", "required_views", "exact_blockers"],
    )
    print(REPORT.relative_to(ROOT))
    return 0 if payload["visualization_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
