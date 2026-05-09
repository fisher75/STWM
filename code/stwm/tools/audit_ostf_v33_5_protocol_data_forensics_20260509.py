#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_5_protocol_data_forensics_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_5_PROTOCOL_DATA_FORENSICS_20260509.md"
V33_4_MANIFEST = ROOT / "manifests/ostf_v33_4_separated_hard_identity_semantic/H32_M128_seed42.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def v30_uid_map(split: str) -> dict[str, dict[str, Any]]:
    entries = load_json(ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json").get("entries", [])
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        if int(e.get("H", -1)) != 32 or int(e.get("M", -1)) != 128:
            continue
        path = ROOT / e["cache_path"]
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        uid = str(np.asarray(z["video_uid"]).item() if "video_uid" in z else path.stem)
        out[uid] = e
    return out


def component_paths(split: str, uid: str) -> dict[str, Path]:
    return {
        "v30_base": ROOT / "missing_v30_base_marker",
        "identity_sidecar": ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey" / split / f"{uid}.npz",
        "visual_teacher_sidecar": ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local" / split / f"{uid}.npz",
        "semantic_prototype_target": ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32" / split / f"{uid}.npz",
    }


def audit_split(split: str, manifest: dict[str, Any]) -> dict[str, Any]:
    entries = manifest.get("splits", {}).get(split, [])
    uid_to_v30 = v30_uid_map(split)
    rows = []
    used = 0
    unused = 0
    reasons: Counter[str] = Counter()
    used_uids: set[str] = set()
    for entry in entries:
        uid = str(entry["sample_uid"])
        missing = []
        if uid not in uid_to_v30:
            missing.append("missing_v30_base")
        for name, path in component_paths(split, uid).items():
            if name == "v30_base":
                continue
            if not path.exists():
                missing.append(f"missing_{name}")
        mask_path = ROOT / entry["mask_path"]
        if not mask_path.exists():
            missing.append("missing_hard_mask")
        if missing:
            unused += 1
            reasons.update(missing)
        else:
            used += 1
            used_uids.add(uid)
        rows.append({"sample_uid": uid, "split": split, "used": not missing, "missing_components": missing})
    # Reproduce current dataset-filtered eval universe.
    eval_uids = set()
    for uid in uid_to_v30:
        paths = component_paths(split, uid)
        if paths["identity_sidecar"].exists() and paths["visual_teacher_sidecar"].exists() and paths["semantic_prototype_target"].exists():
            eval_uids.add(uid)
    manifest_uids = {str(e["sample_uid"]) for e in entries}
    return {
        "manifest_sample_count": len(entries),
        "eval_sample_count": len(eval_uids & manifest_uids),
        "manifest_entries_used_count": used,
        "manifest_entries_unused_count": unused,
        "manifest_unused_sample_uids": [r["sample_uid"] for r in rows if not r["used"]][:50],
        "eval_dataset_extra_uids": sorted(eval_uids - manifest_uids)[:50],
        "unused_reason_breakdown": dict(reasons),
        "item_level_coverage_table": rows,
        "manifest_full_coverage_ok": unused == 0 and len(entries) > 0,
    }


def main() -> int:
    manifest = load_json(V33_4_MANIFEST)
    split_shift = load_json(ROOT / "reports/stwm_ostf_v33_4_split_shift_audit_20260509.json")
    out = {split: audit_split(split, manifest) for split in ("val", "test")}
    heuristic_unreliable = False
    if split_shift:
        val_neg = split_shift.get("splits", {}).get("val", {}).get("identity_negative_ratio")
        test_neg = split_shift.get("splits", {}).get("test", {}).get("identity_negative_ratio")
        easier = split_shift.get("which_split_is_easier")
        if val_neg is not None and test_neg is not None and test_neg > val_neg and easier == "test":
            heuristic_unreliable = True
    test = out["test"]
    payload = {
        "generated_at_utc": utc_now(),
        "manifest_full_coverage_ok": bool(test["manifest_full_coverage_ok"]),
        "manifest_entries_used_count": test["manifest_entries_used_count"],
        "manifest_entries_unused_count": test["manifest_entries_unused_count"],
        "eval_sample_count": test["eval_sample_count"],
        "unused_reason_breakdown": test["unused_reason_breakdown"],
        "split_shift_suspected": bool(split_shift.get("split_shift_suspected", False)),
        "split_shift_reason": split_shift.get("split_shift_reason", []),
        "heuristic_unreliable": heuristic_unreliable,
        "recommended_fix": "Build V33.5 manifests from fully available samples and run manifest-driven eval with all entries accounted for.",
        "splits": out,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.5 Protocol Data Forensics", payload, ["manifest_full_coverage_ok", "manifest_entries_used_count", "manifest_entries_unused_count", "eval_sample_count", "unused_reason_breakdown", "split_shift_suspected", "split_shift_reason", "heuristic_unreliable", "recommended_fix"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
