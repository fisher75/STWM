#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import BASELINE_ROOT, COPY_ROOT, PROTO_ROOT, baseline_arrays_for_sample, train_global_distribution


REPORT = ROOT / "reports/stwm_ostf_v33_11_copy_residual_semantic_target_build_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_COPY_RESIDUAL_SEMANTIC_TARGET_BUILD_20260510.md"


def main() -> int:
    k = 32
    train_global = train_global_distribution(k)
    stats: dict[str, Any] = {}
    blockers: list[str] = []
    total_valid = total_copy = 0
    for split in ("train", "val", "test"):
        (COPY_ROOT / split).mkdir(parents=True, exist_ok=True)
        file_count = stable_count = changed_count = valid_count = 0
        for path in sorted((PROTO_ROOT / split).glob("*.npz")):
            z = np.load(path, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
            mask = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
            obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
            arrays = baseline_arrays_for_sample(z, train_global)
            copy = arrays["copy_semantic_prototype_id"]
            last = arrays["last_observed_semantic_prototype_id"]
            stable = mask & (copy == target) & (copy >= 0)
            changed = mask & (copy != target) & (copy >= 0)
            sample_dist = arrays["sample_level_prototype_frequency"]
            np.savez_compressed(
                COPY_ROOT / split / f"{uid}.npz",
                sample_uid=np.asarray(uid),
                dataset=np.asarray(str(np.asarray(z["dataset"]).item()) if "dataset" in z.files else "pointodyssey"),
                split=np.asarray(split),
                semantic_prototype_id=target,
                semantic_prototype_available_mask=mask,
                obs_semantic_prototype_id=obs,
                obs_semantic_prototype_available_mask=obs_mask,
                last_observed_semantic_prototype_id=last,
                copy_semantic_prototype_id=copy,
                semantic_stable_mask=stable,
                semantic_changed_mask=changed,
                semantic_update_target=changed.astype(bool),
                semantic_update_available_mask=(stable | changed),
                copy_prior_distribution=arrays["last_observed_copy"].astype(np.float32),
                observed_frequency_prior_distribution=arrays["observed_prototype_frequency"].astype(np.float32),
                sample_level_frequency_prior_distribution=sample_dist.astype(np.float32),
                train_global_prior_distribution=arrays["train_global_prototype_frequency"].astype(np.float32),
                leakage_safe=np.asarray(True),
                future_prototypes_supervision_only=np.asarray(True),
                future_prototypes_input_allowed=np.asarray(False),
            )
            file_count += 1
            stable_count += int(stable.sum())
            changed_count += int(changed.sum())
            valid_count += int(mask.sum())
            total_valid += int(mask.sum())
            total_copy += int(((copy >= 0) & mask).sum())
            if not (BASELINE_ROOT / split / f"{uid}.npz").exists():
                blockers.append(f"missing baseline bank sidecar for {split}/{uid}")
        stats[split] = {
            "sample_count": file_count,
            "stable_count": stable_count,
            "changed_count": changed_count,
            "valid_count": valid_count,
            "stable_ratio": stable_count / max(valid_count, 1),
            "changed_ratio": changed_count / max(valid_count, 1),
        }
    payload = {
        "generated_at_utc": utc_now(),
        "copy_residual_targets_built": not blockers,
        "output_root": str(COPY_ROOT.relative_to(ROOT)),
        "semantic_hard_mask_saved": False,
        "strongest_nontrivial_baseline_id_written_as_copy": False,
        "copy_prior_coverage": float(total_copy / max(total_valid, 1)),
        "stable_count_by_split": {k2: v["stable_count"] for k2, v in stats.items()},
        "changed_count_by_split": {k2: v["changed_count"] for k2, v in stats.items()},
        "stable_ratio_by_split": {k2: v["stable_ratio"] for k2, v in stats.items()},
        "changed_ratio_by_split": {k2: v["changed_ratio"] for k2, v in stats.items()},
        "leakage_safe": True,
        "future_prototypes_input_allowed": False,
        "by_split": stats,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.11 Copy Residual Semantic Target Build", payload, ["copy_residual_targets_built", "output_root", "semantic_hard_mask_saved", "strongest_nontrivial_baseline_id_written_as_copy", "copy_prior_coverage", "stable_count_by_split", "changed_count_by_split", "leakage_safe", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
