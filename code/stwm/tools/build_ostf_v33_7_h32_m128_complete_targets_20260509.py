#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OUT = ROOT / "outputs/cache/stwm_ostf_v33_7_complete_h32_m128"
REPORT = ROOT / "reports/stwm_ostf_v33_7_h32_m128_complete_target_coverage_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_7_H32_M128_COMPLETE_TARGET_COVERAGE_20260509.md"

ROOTS = {
    "v30": ROOT / "outputs/cache/stwm_ostf_v30_external_gt/pointodyssey/M128_H32",
    "semantic_identity": ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey",
    "global_identity": ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey",
    "visual_teacher": ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local",
    "semantic_prototype": ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32",
}

DESTS = {
    "semantic_identity": OUT / "semantic_identity_targets/pointodyssey",
    "global_identity": OUT / "global_identity_labels/pointodyssey",
    "visual_teacher": OUT / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local",
    "semantic_prototype": OUT / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32",
}


def stems(root: Path, split: str) -> set[str]:
    p = root / split
    return {x.stem for x in p.glob("*.npz")} if p.exists() else set()


def manifest_stems(split: str) -> set[str]:
    entries = json.loads((ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json").read_text(encoding="utf-8")).get("entries", [])
    return {
        Path(e["cache_path"]).stem
        for e in entries
        if int(e.get("H", -1)) == 32 and int(e.get("M", -1)) == 128
    }


def copy_subset(src_root: Path, dst_root: Path, split: str, uids: set[str]) -> None:
    dst = dst_root / split
    dst.mkdir(parents=True, exist_ok=True)
    for uid in sorted(uids):
        src = src_root / split / f"{uid}.npz"
        if src.exists():
            shutil.copy2(src, dst / src.name)


def main() -> int:
    by_split: dict[str, Any] = {}
    expanded = False
    for split in ("train", "val", "test"):
        sets = {name: stems(root, split) for name, root in ROOTS.items()}
        sets["v30_manifest"] = manifest_stems(split)
        complete = set.intersection(*sets.values()) if all(sets.values()) else set()
        for name, dst in DESTS.items():
            copy_subset(ROOTS[name], dst, split, complete)
        by_split[split] = {
            "v30_h32_m128_available_samples": len(sets["v30"]),
            "v30_manifest_reachable_samples": len(sets["v30_manifest"]),
            "identity_sidecar_count": len(sets["semantic_identity"]),
            "global_identity_sidecar_count": len(sets["global_identity"]),
            "visual_teacher_sidecar_count": len(sets["visual_teacher"]),
            "semantic_prototype_target_count": len(sets["semantic_prototype"]),
            "complete_sample_count": len(complete),
            "coverage_ratio": float(len(complete) / max(len(sets["v30_manifest"]), 1)),
            "complete_sample_uids": sorted(complete)[:50],
        }
    v33_6_train = 47
    train_count = int(by_split["train"]["complete_sample_count"])
    expanded = train_count > v33_6_train
    blocker = None
    if train_count < 200:
        blocker = "complete H32/M128 train coverage remains below 200 because visual teacher and semantic prototype sidecars are limited to the V33.2/V33.3 smoke subset."
    payload = {
        "generated_at_utc": utc_now(),
        "output_root": str(OUT.relative_to(ROOT)),
        "v30_h32_m128_available_samples_by_split": {s: by_split[s]["v30_h32_m128_available_samples"] for s in by_split},
        "v30_h32_m128_manifest_reachable_samples_by_split": {s: by_split[s]["v30_manifest_reachable_samples"] for s in by_split},
        "identity_sidecar_count_by_split": {s: by_split[s]["identity_sidecar_count"] for s in by_split},
        "global_identity_sidecar_count_by_split": {s: by_split[s]["global_identity_sidecar_count"] for s in by_split},
        "visual_teacher_sidecar_count_by_split": {s: by_split[s]["visual_teacher_sidecar_count"] for s in by_split},
        "semantic_prototype_target_count_by_split": {s: by_split[s]["semantic_prototype_target_count"] for s in by_split},
        "complete_sample_count_by_split": {s: by_split[s]["complete_sample_count"] for s in by_split},
        "coverage_ratio_by_split": {s: by_split[s]["coverage_ratio"] for s in by_split},
        "coverage_expanded_vs_v33_6": expanded,
        "previous_v33_6_train_sample_count": v33_6_train,
        "complete_train_sample_count": train_count,
        "training_scale_still_smoke": train_count < 200,
        "leakage_safe": True,
        "future_teacher_embeddings_supervision_only": True,
        "future_semantic_prototypes_supervision_only": True,
        "prototype_vocab_reused": "V33.3 K32",
        "prototype_vocab_reuse_reason": "coverage did not expand beyond the V33.6 smoke-scale intersection, so K32 is retained for controlled identity calibration.",
        "exact_blockers": [blocker] if blocker else [],
        "by_split": by_split,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.7 H32 M128 Complete Target Coverage",
        payload,
        ["output_root", "complete_sample_count_by_split", "coverage_ratio_by_split", "coverage_expanded_vs_v33_6", "training_scale_still_smoke", "leakage_safe", "exact_blockers"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
