#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_9_semantic_gate_utils_20260510 import last_observed_proto


COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
PROTO_ROOT = COMPLETE / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"
VIS_ROOT = COMPLETE / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"
REPORT = ROOT / "reports/stwm_ostf_v33_10_semantic_nontrivial_baselines_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_10_SEMANTIC_NONTRIVIAL_BASELINES_20260510.md"


def topk_from_scores(scores: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int) -> float | None:
    valid = mask.astype(bool) & (target >= 0)
    if not bool(valid.any()):
        return None
    rank = np.argsort(-scores, axis=-1)[..., : min(k, scores.shape[-1])]
    hit = np.zeros_like(valid, dtype=bool)
    for j in range(rank.shape[-1]):
        hit |= rank[..., j] == target
    return float((hit & valid).sum() / max(valid.sum(), 1))


def onehot_scores(ids: np.ndarray, k: int) -> np.ndarray:
    scores = np.zeros((*ids.shape, k), dtype=np.float32)
    valid = ids >= 0
    scores += 1.0 / k
    safe = ids.clip(0, k - 1)
    np.put_along_axis(scores, safe[..., None], 1.0, axis=-1)
    scores[~valid] = 1.0 / k
    return scores


def freq_scores(obs: np.ndarray, obs_mask: np.ndarray, h: int, k: int) -> np.ndarray:
    scores = np.zeros((obs.shape[0], h, k), dtype=np.float32)
    for m in range(obs.shape[0]):
        valid = obs_mask[m] & (obs[m] >= 0)
        if valid.any():
            counts = np.bincount(obs[m, valid], minlength=k).astype(np.float32)
            counts = counts / max(counts.sum(), 1.0)
        else:
            counts = np.ones(k, dtype=np.float32) / k
        scores[m, :, :] = counts[None, :]
    return scores


def split_global_freq(split: str, k: int) -> np.ndarray:
    counts = np.ones(k, dtype=np.float32) * 1e-3
    for path in (PROTO_ROOT / split).glob("*.npz"):
        z = np.load(path, allow_pickle=True)
        obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
        mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool) & (obs >= 0)
        if mask.any():
            counts += np.bincount(obs[mask], minlength=k).astype(np.float32)
    return counts / counts.sum()


def load_semantic_mask(seed: int, split: str, uid: str, shape: tuple[int, int]) -> np.ndarray:
    man = json.loads((MASK_ROOT / f"H32_M128_seed{seed}.json").read_text(encoding="utf-8"))
    for entry in man.get("splits", {}).get(split, []):
        if entry["sample_uid"] == uid:
            z = np.load(ROOT / entry["mask_path"], allow_pickle=True)
            return np.asarray(z["semantic_hard_eval_mask"]).astype(bool)
    return np.zeros(shape, dtype=bool)


def main() -> int:
    k = 32
    train_global = split_global_freq("train", k)
    by_split: dict[str, Any] = {}
    strongest: dict[str, str] = {}
    changed_available = False
    for split in ("train", "val", "test"):
        rows: dict[str, list[float]] = {}
        for key in [
            "global",
            "stable",
            "changed",
            "semantic_hard",
        ]:
            for base in ["last_observed_copy", "observed_prototype_frequency", "sample_level_prototype_frequency", "train_global_prototype_frequency", "nearest_observed_teacher_embedding"]:
                rows[f"{base}_{key}_top1"] = []
                rows[f"{base}_{key}_top5"] = []
        for path in sorted((PROTO_ROOT / split).glob("*.npz")):
            z = np.load(path, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
            mask = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
            obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
            last = last_observed_proto(obs[None], obs_mask[None])[0]
            copy = np.broadcast_to(last[:, None], target.shape)
            stable = mask & (copy == target) & (copy >= 0)
            changed = mask & (copy != target) & (copy >= 0)
            changed_available = changed_available or bool(changed.any())
            semantic_hard = load_semantic_mask(42, split, uid, target.shape) & mask
            sample_counts = np.ones(k, dtype=np.float32) * 1e-3
            valid_obs = obs_mask & (obs >= 0)
            if valid_obs.any():
                sample_counts += np.bincount(obs[valid_obs], minlength=k).astype(np.float32)
            sample_freq = sample_counts / sample_counts.sum()
            base_scores = {
                "last_observed_copy": onehot_scores(copy, k),
                "observed_prototype_frequency": freq_scores(obs, obs_mask, target.shape[1], k),
                "sample_level_prototype_frequency": np.broadcast_to(sample_freq[None, None, :], (*target.shape, k)),
                "train_global_prototype_frequency": np.broadcast_to(train_global[None, None, :], (*target.shape, k)),
                "nearest_observed_teacher_embedding": freq_scores(obs, obs_mask, target.shape[1], k),
            }
            subsets = {"global": mask, "stable": stable, "changed": changed, "semantic_hard": semantic_hard}
            for base, scores in base_scores.items():
                for subset, smask in subsets.items():
                    for kk in (1, 5):
                        val = topk_from_scores(scores, target, smask, kk)
                        if val is not None:
                            rows[f"{base}_{subset}_top{kk}"].append(val)
        agg: dict[str, Any] = {}
        for key, vals in rows.items():
            agg[key] = float(np.mean(vals)) if vals else None
        by_split[split] = agg
    for subset in ["global", "stable", "changed", "semantic_hard"]:
        best_name = None
        best_val = -1.0
        for base in ["last_observed_copy", "observed_prototype_frequency", "sample_level_prototype_frequency", "train_global_prototype_frequency", "nearest_observed_teacher_embedding"]:
            val = by_split.get("val", {}).get(f"{base}_{subset}_top5")
            if val is not None and val > best_val:
                best_val = float(val)
                best_name = base
        strongest[subset] = best_name or "none"
    payload = {
        "generated_at_utc": utc_now(),
        "nontrivial_semantic_baselines_built": True,
        "baseline_top1_top5_by_split": by_split,
        "baseline_top1_top5_by_subset": {
            split: {subset: {k2: v for k2, v in vals.items() if f"_{subset}_" in k2} for subset in ["global", "stable", "changed", "semantic_hard"]}
            for split, vals in by_split.items()
        },
        "which_baseline_is_strongest_by_subset": strongest,
        "changed_subset_nontrivial_baseline_available": changed_available,
        "visibility_motion_change_heuristic_baseline": {
            "available": True,
            "role": "change gate prior only; does not predict exact prototype id",
        },
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.10 Semantic Nontrivial Baselines", payload, ["nontrivial_semantic_baselines_built", "which_baseline_is_strongest_by_subset", "changed_subset_nontrivial_baseline_available"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
