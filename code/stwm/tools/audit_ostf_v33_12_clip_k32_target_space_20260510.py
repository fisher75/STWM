#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import PROTO_ROOT, topk_from_scores


REPORT = ROOT / "reports/stwm_ostf_v33_12_clip_k32_target_space_audit_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_12_CLIP_K32_TARGET_SPACE_AUDIT_20260510.md"
VOCAB = ROOT / "outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K32/prototype_vocab.npz"


def entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(max(len(p), 2)))


def onehot(ids: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((*ids.shape, k), dtype=np.float32)
    safe = ids.clip(0, k - 1)
    np.put_along_axis(out, safe[..., None], 1.0, axis=-1)
    out[ids < 0] = 1.0 / k
    return out


def freq(obs: np.ndarray, obs_mask: np.ndarray, h: int, k: int, sample: bool) -> np.ndarray:
    out = np.zeros((obs.shape[0], h, k), dtype=np.float32)
    if sample:
        valid = obs_mask & (obs >= 0)
        counts = np.ones(k, dtype=np.float32) * 1e-3
        if valid.any():
            counts += np.bincount(obs[valid], minlength=k).astype(np.float32)
        dist = counts / counts.sum()
        out[:] = dist[None, None, :]
        return out
    for m in range(obs.shape[0]):
        valid = obs_mask[m] & (obs[m] >= 0)
        if valid.any():
            counts = np.bincount(obs[m, valid], minlength=k).astype(np.float32)
            dist = counts / counts.sum()
        else:
            dist = np.ones(k, dtype=np.float32) / k
        out[m] = dist[None, :]
    return out


def main() -> int:
    k = 32
    global_counts = np.zeros(k, dtype=np.int64)
    by_split: dict[str, Any] = {}
    same_point_scores = []
    same_inst_scores = []
    for split in ("train", "val", "test"):
        counts = np.zeros(k, dtype=np.int64)
        stable_count = changed_count = valid_count = 0
        copy_top1 = copy_top5 = sample_top1 = sample_top5 = obs_top1 = obs_top5 = 0.0
        n_metric = 0
        for path in sorted((PROTO_ROOT / split).glob("*.npz")):
            z = np.load(path, allow_pickle=True)
            target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
            mask = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
            obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
            valid_ids = target[mask & (target >= 0)]
            if valid_ids.size:
                counts += np.bincount(valid_ids, minlength=k).astype(np.int64)
            last = np.full((obs.shape[0],), -1, dtype=np.int64)
            for m in range(obs.shape[0]):
                idx = np.where(obs_mask[m] & (obs[m] >= 0))[0]
                if idx.size:
                    last[m] = obs[m, idx[-1]]
            copy = np.broadcast_to(last[:, None], target.shape)
            stable = mask & (copy == target) & (copy >= 0)
            changed = mask & (copy != target) & (copy >= 0)
            stable_count += int(stable.sum())
            changed_count += int(changed.sum())
            valid_count += int(mask.sum())
            copy_logits = np.log(onehot(copy, k).clip(1e-8, 1.0))
            sample_logits = np.log(freq(obs, obs_mask, target.shape[1], k, True).clip(1e-8, 1.0))
            obs_logits = np.log(freq(obs, obs_mask, target.shape[1], k, False).clip(1e-8, 1.0))
            vals = [
                topk_from_scores(copy_logits, target, mask, 1),
                topk_from_scores(copy_logits, target, mask, 5),
                topk_from_scores(sample_logits, target, mask, 1),
                topk_from_scores(sample_logits, target, mask, 5),
                topk_from_scores(obs_logits, target, mask, 1),
                topk_from_scores(obs_logits, target, mask, 5),
            ]
            if all(v is not None for v in vals):
                copy_top1 += vals[0]; copy_top5 += vals[1]; sample_top1 += vals[2]; sample_top5 += vals[3]; obs_top1 += vals[4]; obs_top5 += vals[5]; n_metric += 1
            # Same point consistency: majority target over future for each point.
            for m in range(target.shape[0]):
                vals_m = target[m][mask[m] & (target[m] >= 0)]
                if vals_m.size:
                    c = np.bincount(vals_m, minlength=k)
                    same_point_scores.append(float(c.max() / vals_m.size))
            # Approximate same-instance consistency with points sharing the same last observed proto.
            for proto in np.unique(last[last >= 0]):
                group = target[last == proto]
                group_mask = mask[last == proto]
                vals_g = group[group_mask & (group >= 0)]
                if vals_g.size:
                    c = np.bincount(vals_g, minlength=k)
                    same_inst_scores.append(float(c.max() / vals_g.size))
        global_counts += counts
        by_split[split] = {
            "prototype_entropy": entropy(counts),
            "empty_cluster_count": int((counts == 0).sum()),
            "dominant_cluster_ratio": float(counts.max() / max(counts.sum(), 1)),
            "stable_ratio": float(stable_count / max(valid_count, 1)),
            "changed_ratio": float(changed_count / max(valid_count, 1)),
            "copy_baseline_top1": float(copy_top1 / max(n_metric, 1)),
            "copy_baseline_top5": float(copy_top5 / max(n_metric, 1)),
            "sample_frequency_baseline_top1": float(sample_top1 / max(n_metric, 1)),
            "sample_frequency_baseline_top5": float(sample_top5 / max(n_metric, 1)),
            "observed_frequency_baseline_top1": float(obs_top1 / max(n_metric, 1)),
            "observed_frequency_baseline_top5": float(obs_top5 / max(n_metric, 1)),
            "cluster_counts_min_median_max": [int(counts.min()), float(np.median(counts)), int(counts.max())],
        }
    sample_strong = by_split["val"]["sample_frequency_baseline_top5"] >= by_split["val"]["copy_baseline_top5"] - 0.02
    teacher_jitter = float(np.mean(same_point_scores)) < 0.75
    too_coarse = by_split["train"]["dominant_cluster_ratio"] > 0.25 or by_split["train"]["empty_cluster_count"] > 0
    sufficient = bool(not sample_strong and not teacher_jitter and not too_coarse)
    payload = {
        "generated_at_utc": utc_now(),
        "current_teacher_name": "clip_vit_b32_local",
        "current_prototype_K": 32,
        "prototype_entropy": entropy(global_counts),
        "empty_cluster_count": int((global_counts == 0).sum()),
        "dominant_cluster_ratio": float(global_counts.max() / max(global_counts.sum(), 1)),
        "stable_ratio_by_split": {s: by_split[s]["stable_ratio"] for s in by_split},
        "changed_ratio_by_split": {s: by_split[s]["changed_ratio"] for s in by_split},
        "same_instance_temporal_consistency": float(np.mean(same_inst_scores)) if same_inst_scores else None,
        "same_point_temporal_consistency": float(np.mean(same_point_scores)) if same_point_scores else None,
        "sample_frequency_baseline_too_strong": bool(sample_strong),
        "teacher_jitter_suspected": bool(teacher_jitter),
        "K32_too_coarse_suspected": bool(too_coarse),
        "clip_b32_target_space_sufficient": sufficient,
        "by_split": by_split,
        "recommended_target_repair": "evaluate_larger_K_and_stronger_teacher_or_smooth_instance_temporal_targets",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.12 CLIP K32 Target Space Audit", payload, ["prototype_entropy", "empty_cluster_count", "dominant_cluster_ratio", "stable_ratio_by_split", "changed_ratio_by_split", "same_instance_temporal_consistency", "same_point_temporal_consistency", "sample_frequency_baseline_too_strong", "teacher_jitter_suspected", "K32_too_coarse_suspected", "clip_b32_target_space_sufficient", "recommended_target_repair"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
