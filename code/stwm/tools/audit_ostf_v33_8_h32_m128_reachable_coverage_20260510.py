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


REPORT = ROOT / "reports/stwm_ostf_v33_8_h32_m128_reachable_coverage_audit_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_H32_M128_REACHABLE_COVERAGE_AUDIT_20260510.md"

ROOTS = {
    "v30_cache": ROOT / "outputs/cache/stwm_ostf_v30_external_gt/pointodyssey/M128_H32",
    "identity": ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey",
    "global_identity": ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey",
    "visual_teacher": ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local",
    "semantic_prototype": ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32",
}


def manifest_uids(split: str) -> tuple[set[str], dict[str, str]]:
    path = ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json"
    entries = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
    uids: set[str] = set()
    uid_to_cache: dict[str, str] = {}
    for e in entries:
        if int(e.get("H", -1)) != 32 or int(e.get("M", -1)) != 128:
            continue
        cache = ROOT / e["cache_path"]
        if not cache.exists():
            continue
        z = np.load(cache, allow_pickle=True)
        uid = str(np.asarray(z["video_uid"]).item() if "video_uid" in z.files else cache.stem)
        uids.add(uid)
        uid_to_cache[uid] = cache.stem
    return uids, uid_to_cache


def stems(root: Path, split: str) -> set[str]:
    p = root / split
    return {x.stem for x in p.glob("*.npz")} if p.exists() else set()


def main() -> int:
    by_split: dict[str, Any] = {}
    blockers: list[str] = []
    sample_uid_mismatch = False
    cache_path_stem_mismatch = False
    for split in ("train", "val", "test"):
        reachable, uid_to_cache = manifest_uids(split)
        v30_cache = stems(ROOTS["v30_cache"], split)
        identity = stems(ROOTS["identity"], split)
        global_id = stems(ROOTS["global_identity"], split)
        visual = stems(ROOTS["visual_teacher"], split)
        proto = stems(ROOTS["semantic_prototype"], split)
        complete = reachable & identity & global_id & visual & proto
        max_possible = reachable & identity & global_id
        sample_uid_mismatch = sample_uid_mismatch or bool(reachable - v30_cache)
        cache_path_stem_mismatch = cache_path_stem_mismatch or any(uid_to_cache.get(uid) != uid for uid in reachable)
        by_split[split] = {
            "v30_h32_m128_manifest_reachable": len(reachable),
            "v30_h32_m128_cache_count": len(v30_cache),
            "identity_sidecar_count": len(identity),
            "global_identity_sidecar_count": len(global_id),
            "visual_teacher_sidecar_count": len(visual),
            "semantic_prototype_target_count": len(proto),
            "current_complete_count": len(complete),
            "current_complete_coverage_ratio": float(len(complete) / max(len(reachable), 1)),
            "max_possible_complete_count": len(max_possible),
            "missing_visual_teacher_uids": sorted(list(reachable - visual))[:200],
            "missing_semantic_prototype_uids": sorted(list(reachable - proto))[:200],
            "missing_global_identity_uids": sorted(list(reachable - global_id))[:200],
        }
        if len(reachable - visual):
            blockers.append(f"{split}: missing visual teacher sidecars for {len(reachable - visual)} reachable samples")
        if len(reachable - proto):
            blockers.append(f"{split}: missing semantic prototype targets for {len(reachable - proto)} reachable samples")
    payload = {
        "generated_at_utc": utc_now(),
        "v30_h32_m128_manifest_reachable_by_split": {s: by_split[s]["v30_h32_m128_manifest_reachable"] for s in by_split},
        "v30_h32_m128_cache_count_by_split": {s: by_split[s]["v30_h32_m128_cache_count"] for s in by_split},
        "identity_sidecar_count_by_split": {s: by_split[s]["identity_sidecar_count"] for s in by_split},
        "global_identity_sidecar_count_by_split": {s: by_split[s]["global_identity_sidecar_count"] for s in by_split},
        "visual_teacher_sidecar_count_by_split": {s: by_split[s]["visual_teacher_sidecar_count"] for s in by_split},
        "semantic_prototype_target_count_by_split": {s: by_split[s]["semantic_prototype_target_count"] for s in by_split},
        "current_complete_count_by_split": {s: by_split[s]["current_complete_count"] for s in by_split},
        "current_complete_coverage_ratio_by_split": {s: by_split[s]["current_complete_coverage_ratio"] for s in by_split},
        "max_possible_complete_count_by_split": {s: by_split[s]["max_possible_complete_count"] for s in by_split},
        "missing_visual_teacher_uids_by_split": {s: by_split[s]["missing_visual_teacher_uids"] for s in by_split},
        "missing_semantic_prototype_uids_by_split": {s: by_split[s]["missing_semantic_prototype_uids"] for s in by_split},
        "missing_global_identity_uids_by_split": {s: by_split[s]["missing_global_identity_uids"] for s in by_split},
        "sample_uid_mismatch_detected": sample_uid_mismatch,
        "cache_path_stem_mismatch_detected": cache_path_stem_mismatch,
        "exact_blockers": blockers,
        "by_split": by_split,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.8 H32 M128 Reachable Coverage Audit", payload, ["v30_h32_m128_manifest_reachable_by_split", "current_complete_count_by_split", "current_complete_coverage_ratio_by_split", "max_possible_complete_count_by_split", "sample_uid_mismatch_detected", "cache_path_stem_mismatch_detected", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
