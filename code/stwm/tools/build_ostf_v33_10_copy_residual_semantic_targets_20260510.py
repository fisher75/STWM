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
from stwm.tools.ostf_v33_9_semantic_gate_utils_20260510 import last_observed_proto


COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
SRC = COMPLETE / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"
MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"
OUT = ROOT / "outputs/cache/stwm_ostf_v33_10_copy_residual_semantic_targets/pointodyssey/clip_vit_b32_local/K32"
REPORT = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_target_build_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_10_COPY_RESIDUAL_SEMANTIC_TARGET_BUILD_20260510.md"


def load_semantic_hard(split: str, uid: str, shape: tuple[int, int]) -> np.ndarray:
    path = MASK_ROOT / "H32_M128_seed42.json"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    man = json.loads(path.read_text(encoding="utf-8"))
    for entry in man.get("splits", {}).get(split, []):
        if entry.get("sample_uid") == uid:
            z = np.load(ROOT / entry["mask_path"], allow_pickle=True)
            return np.asarray(z["semantic_hard_train_mask" if split == "train" and "semantic_hard_train_mask" in z.files else "semantic_hard_eval_mask"]).astype(bool)
    return np.zeros(shape, dtype=bool)


def onehot(ids: np.ndarray, k: int) -> np.ndarray:
    out = np.full((*ids.shape, k), 1e-4 / max(k - 1, 1), dtype=np.float16)
    valid = ids >= 0
    safe = ids.clip(0, k - 1)
    np.put_along_axis(out, safe[..., None], 1.0, axis=-1)
    out[~valid] = np.float16(1.0 / k)
    return out


def observed_freq(obs: np.ndarray, obs_mask: np.ndarray, h: int, k: int) -> np.ndarray:
    out = np.zeros((obs.shape[0], h, k), dtype=np.float16)
    for m in range(obs.shape[0]):
        valid = obs_mask[m] & (obs[m] >= 0)
        if valid.any():
            counts = np.bincount(obs[m, valid], minlength=k).astype(np.float32)
            dist = counts / max(counts.sum(), 1.0)
        else:
            dist = np.ones(k, dtype=np.float32) / k
        out[m] = dist.astype(np.float16)[None, :]
    return out


def main() -> int:
    k = 32
    stats: dict[str, Any] = {}
    total_copy = total_valid = 0
    for split in ("train", "val", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        stable_count = changed_count = hard_count = valid_count = 0
        file_count = 0
        for path in sorted((SRC / split).glob("*.npz")):
            z = np.load(path, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
            mask = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
            obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
            last = last_observed_proto(obs[None], obs_mask[None])[0]
            copy = np.broadcast_to(last[:, None], target.shape).copy()
            stable = mask & (copy == target) & (copy >= 0)
            changed = mask & (copy != target) & (copy >= 0)
            semantic_hard = load_semantic_hard(split, uid, target.shape) & mask
            update = changed.astype(bool)
            np.savez_compressed(
                OUT / split / f"{uid}.npz",
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
                semantic_hard_mask=semantic_hard,
                semantic_update_target=update,
                semantic_update_available_mask=(stable | changed),
                copy_prior_distribution=onehot(copy, k),
                observed_frequency_prior_distribution=observed_freq(obs, obs_mask, target.shape[1], k),
                strongest_nontrivial_baseline_id=copy,
                future_prototypes_supervision_only=np.asarray(True),
                future_prototypes_input_allowed=np.asarray(False),
                leakage_safe=np.asarray(True),
            )
            file_count += 1
            stable_count += int(stable.sum())
            changed_count += int(changed.sum())
            hard_count += int(semantic_hard.sum())
            valid_count += int(mask.sum())
            total_valid += int(mask.sum())
            total_copy += int(((copy >= 0) & mask).sum())
        stats[split] = {
            "sample_count": file_count,
            "stable_count": stable_count,
            "changed_count": changed_count,
            "semantic_hard_count": hard_count,
            "valid_count": valid_count,
            "stable_ratio": stable_count / max(valid_count, 1),
            "changed_ratio": changed_count / max(valid_count, 1),
        }
    payload = {
        "generated_at_utc": utc_now(),
        "copy_residual_targets_built": True,
        "output_root": str(OUT.relative_to(ROOT)),
        "stable_count_by_split": {k2: v["stable_count"] for k2, v in stats.items()},
        "changed_count_by_split": {k2: v["changed_count"] for k2, v in stats.items()},
        "semantic_hard_count_by_split": {k2: v["semantic_hard_count"] for k2, v in stats.items()},
        "stable_ratio_by_split": {k2: v["stable_ratio"] for k2, v in stats.items()},
        "changed_ratio_by_split": {k2: v["changed_ratio"] for k2, v in stats.items()},
        "copy_prior_coverage": float(total_copy / max(total_valid, 1)),
        "leakage_safe": True,
        "by_split": stats,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.10 Copy Residual Semantic Target Build", payload, ["copy_residual_targets_built", "output_root", "stable_count_by_split", "changed_count_by_split", "copy_prior_coverage", "leakage_safe"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
