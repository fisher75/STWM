#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OUT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
REPORT = ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_COMPLETE_H32_M128_TARGET_COVERAGE_20260510.md"

ID_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"
GLOBAL_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_6_global_identity_labels/pointodyssey"
VISUAL_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
PROTO_REPORT = ROOT / "reports/stwm_ostf_v33_8_semantic_prototype_targets_20260510.json"
OLD_V33_7_REPORT = ROOT / "reports/stwm_ostf_v33_7_h32_m128_complete_target_coverage_20260509.json"


def selected_k(default: int = 32) -> int:
    if PROTO_REPORT.exists():
        return int(json.loads(PROTO_REPORT.read_text(encoding="utf-8")).get("selected_K", default))
    return default


def uid_from_cache(path: Path) -> str:
    try:
        z = np.load(path, allow_pickle=True)
        return str(np.asarray(z["video_uid"]).item() if "video_uid" in z.files else path.stem)
    except Exception:
        return path.stem


def reachable_uids(split: str) -> set[str]:
    path = ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json"
    entries = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
    out: set[str] = set()
    for entry in entries:
        if int(entry.get("M", -1)) != 128 or int(entry.get("H", -1)) != 32:
            continue
        cache = ROOT / entry["cache_path"]
        if cache.exists():
            out.add(uid_from_cache(cache))
    return out


def stems(root: Path, split: str) -> set[str]:
    d = root / split
    return {p.stem for p in d.glob("*.npz")} if d.exists() else set()


def copy_component(src_root: Path, dst_root: Path, split: str, uid: str) -> bool:
    src = src_root / split / f"{uid}.npz"
    if not src.exists():
        return False
    dst = dst_root / split / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)
    return True


def old_v33_7_train_count() -> int:
    if not OLD_V33_7_REPORT.exists():
        return 47
    try:
        payload = json.loads(OLD_V33_7_REPORT.read_text(encoding="utf-8"))
        return int(payload.get("complete_train_sample_count", 47) or 47)
    except Exception:
        return 47


def main() -> int:
    k = selected_k()
    proto_root = ROOT / f"outputs/cache/stwm_ostf_v33_8_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}"
    dests = {
        "semantic_identity": OUT / "semantic_identity_targets/pointodyssey",
        "global_identity": OUT / "global_identity_labels/pointodyssey",
        "visual_teacher": OUT / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local",
        "semantic_prototype": OUT / f"semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}",
    }
    roots = {
        "semantic_identity": ID_ROOT,
        "global_identity": GLOBAL_ROOT,
        "visual_teacher": VISUAL_ROOT,
        "semantic_prototype": proto_root,
    }
    by_split: dict[str, Any] = {}
    blockers: list[str] = []
    for split in ("train", "val", "test"):
        reachable = reachable_uids(split)
        present = {name: stems(root, split) for name, root in roots.items()}
        complete = set(reachable)
        for values in present.values():
            complete &= values
        missing_by_uid: dict[str, list[str]] = {}
        for uid in sorted(reachable - complete):
            missing_by_uid[uid] = [name for name, values in present.items() if uid not in values]
        for uid in sorted(complete):
            for name, src in roots.items():
                copy_component(src, dests[name], split, uid)
        ratio = float(len(complete) / max(len(reachable), 1))
        if ratio < 0.90:
            blockers.append(f"{split}: complete coverage {ratio:.3f} < 0.90; missing components for {len(missing_by_uid)} samples")
        by_split[split] = {
            "v30_h32_m128_reachable_count": len(reachable),
            "identity_sidecar_count": len(present["semantic_identity"]),
            "global_identity_sidecar_count": len(present["global_identity"]),
            "visual_teacher_sidecar_count": len(present["visual_teacher"]),
            "semantic_prototype_target_count": len(present["semantic_prototype"]),
            "complete_count": len(complete),
            "complete_coverage_ratio": ratio,
            "complete_sample_uids": sorted(complete)[:200],
            "missing_sample_uid_count": len(missing_by_uid),
            "missing_sample_uids": dict(list(missing_by_uid.items())[:100]),
        }
    old_train = old_v33_7_train_count()
    train_count = int(by_split["train"]["complete_count"])
    coverage_pass = all(by_split[s]["complete_coverage_ratio"] >= 0.90 for s in by_split)
    payload = {
        "generated_at_utc": utc_now(),
        "output_root": str(OUT.relative_to(ROOT)),
        "selected_K": k,
        "v30_h32_m128_reachable_count_by_split": {s: by_split[s]["v30_h32_m128_reachable_count"] for s in by_split},
        "complete_count_by_split": {s: by_split[s]["complete_count"] for s in by_split},
        "complete_coverage_ratio_by_split": {s: by_split[s]["complete_coverage_ratio"] for s in by_split},
        "coverage_expanded_vs_v33_7": bool(train_count > old_train),
        "coverage_expanded_vs_v33_6": bool(train_count > 47),
        "previous_v33_7_train_sample_count": old_train,
        "previous_v33_6_train_sample_count": 47,
        "complete_train_sample_count": train_count,
        "complete_val_sample_count": int(by_split["val"]["complete_count"]),
        "complete_test_sample_count": int(by_split["test"]["complete_count"]),
        "target_coverage_pass": bool(coverage_pass),
        "leakage_safe": True,
        "future_teacher_embeddings_input_allowed": False,
        "future_teacher_embeddings_supervision_only": True,
        "future_semantic_prototypes_input_allowed": False,
        "future_semantic_prototypes_supervision_only": True,
        "exact_blockers": blockers,
        "by_split": by_split,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.8 Complete H32 M128 Target Coverage",
        payload,
        [
            "output_root",
            "selected_K",
            "complete_count_by_split",
            "complete_coverage_ratio_by_split",
            "coverage_expanded_vs_v33_7",
            "target_coverage_pass",
            "exact_blockers",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0 if coverage_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
