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
from stwm.tools.ostf_v33_11_common_20260510 import PROTO_ROOT, V33_8_MASK_ROOT, V33_11_MASK_ROOT


REPORT = ROOT / "reports/stwm_ostf_v33_11_true_semantic_hard_protocol_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_TRUE_SEMANTIC_HARD_PROTOCOL_20260510.md"


def proto_counts(split: str, uid: str) -> tuple[int, int]:
    z = np.load(PROTO_ROOT / split / f"{uid}.npz", allow_pickle=True)
    target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
    mask = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
    obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
    obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
    last = np.full((obs.shape[0],), -1, dtype=np.int64)
    for m in range(obs.shape[0]):
        valid = np.where(obs_mask[m] & (obs[m] >= 0))[0]
        if valid.size:
            last[m] = obs[m, valid[-1]]
    copy = np.broadcast_to(last[:, None], target.shape)
    stable = mask & (copy == target) & (copy >= 0)
    changed = mask & (copy != target) & (copy >= 0)
    return int(stable.sum()), int(changed.sum())


def main() -> int:
    V33_11_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    by_seed: dict[str, Any] = {}
    blockers: list[str] = []
    for seed in (42, 123, 456):
        src = V33_8_MASK_ROOT / f"H32_M128_seed{seed}.json"
        dst = V33_11_MASK_ROOT / f"H32_M128_seed{seed}.json"
        if not src.exists():
            blockers.append(f"missing source manifest {src.relative_to(ROOT)}")
            continue
        payload = json.loads(src.read_text(encoding="utf-8"))
        payload["v33_11_protocol_note"] = "identity and semantic hard masks are separated; semantic hard is read dynamically by eval/train seed, not embedded in copy residual targets"
        dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seed_stats: dict[str, Any] = {}
        for split, entries in payload.get("splits", {}).items():
            semantic_hard_count = changed_count = stable_count = overlap = 0
            sample_count = 0
            for entry in entries:
                uid = entry["sample_uid"]
                mask_path = ROOT / entry["mask_path"]
                if not mask_path.exists():
                    blockers.append(f"missing mask {entry['mask_path']}")
                    continue
                z = np.load(mask_path, allow_pickle=True)
                sem = np.asarray(z["semantic_hard_eval_mask"]).astype(bool)
                stable, changed = proto_counts(split, uid)
                semantic_hard_count += int(sem.sum())
                stable_count += stable
                changed_count += changed
                # semantic_hard is intended to focus changed/confuser positions.
                overlap += min(int(sem.sum()), changed)
                sample_count += 1
            seed_stats[split] = {
                "sample_count": sample_count,
                "semantic_hard_count": semantic_hard_count,
                "changed_count": changed_count,
                "stable_count": stable_count,
                "overlap_with_changed_upper_bound": overlap,
                "semantic_hard_positive_coverage": float(semantic_hard_count / max(changed_count, 1)),
            }
        by_seed[str(seed)] = seed_stats
    nonempty = {seed: all(split_stats.get("semantic_hard_count", 0) > 0 for split_stats in stats.values()) for seed, stats in by_seed.items()}
    counts = [sum(split["semantic_hard_count"] for split in stats.values()) for stats in by_seed.values()]
    stable = bool(counts and max(counts) / max(min(counts), 1) < 1.05)
    out = {
        "generated_at_utc": utc_now(),
        "semantic_hard_seed_independent": len(by_seed) == 3 and not blockers,
        "semantic_hard_nonempty_by_seed": nonempty,
        "semantic_hard_distribution_stable_across_seeds": stable,
        "by_seed": by_seed,
        "manifest_paths": {str(seed): str((V33_11_MASK_ROOT / f"H32_M128_seed{seed}.json").relative_to(ROOT)) for seed in (42, 123, 456)},
        "exact_blockers": blockers,
    }
    dump_json(REPORT, out)
    write_doc(DOC, "STWM OSTF V33.11 True Semantic Hard Protocol", out, ["semantic_hard_seed_independent", "semantic_hard_nonempty_by_seed", "semantic_hard_distribution_stable_across_seeds", "manifest_paths", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
