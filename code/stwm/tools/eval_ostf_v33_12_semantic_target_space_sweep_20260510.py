#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT, topk_from_scores


REPORT = ROOT / "reports/stwm_ostf_v33_12_semantic_target_space_sweep_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_12_SEMANTIC_TARGET_SPACE_SWEEP_20260510.md"
VIS_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
VOCAB_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local"


def assign(emb: np.ndarray, centers: np.ndarray, device: torch.device) -> np.ndarray:
    shape = emb.shape[:-1]
    x = torch.from_numpy(np.nan_to_num(emb.reshape(-1, emb.shape[-1]).astype(np.float32))).to(device)
    c = torch.from_numpy(centers.astype(np.float32)).to(device)
    x = torch.nn.functional.normalize(x, dim=-1)
    c = torch.nn.functional.normalize(c, dim=-1)
    out = []
    for chunk in torch.split(x, 8192, dim=0):
        out.append((chunk @ c.T).argmax(dim=-1).detach().cpu())
    return torch.cat(out, dim=0).numpy().reshape(shape)


def onehot(ids: np.ndarray, k: int) -> np.ndarray:
    out = np.full((*ids.shape, k), 1e-4 / max(k - 1, 1), dtype=np.float32)
    safe = ids.clip(0, k - 1)
    np.put_along_axis(out, safe[..., None], 1.0, axis=-1)
    out[ids < 0] = 1.0 / k
    return out


def sample_freq(obs: np.ndarray, obs_mask: np.ndarray, h: int, k: int) -> np.ndarray:
    counts = np.ones(k, dtype=np.float32) * 1e-3
    valid = obs_mask & (obs >= 0)
    if valid.any():
        counts += np.bincount(obs[valid], minlength=k).astype(np.float32)
    dist = counts / counts.sum()
    return np.broadcast_to(dist[None, None, :], (obs.shape[0], h, k))


def entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p /= max(p.sum(), 1.0)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(max(len(p), 2)))


def load_semantic_hard(split: str, uid: str, shape: tuple[int, int]) -> np.ndarray:
    import json

    path = V33_11_MASK_ROOT / "H32_M128_seed42.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload.get("splits", {}).get(split, []):
        if entry["sample_uid"] == uid:
            z = np.load(ROOT / entry["mask_path"], allow_pickle=True)
            return np.asarray(z["semantic_hard_eval_mask"]).astype(bool)
    return np.zeros(shape, dtype=bool)


def eval_k(k: int, device: torch.device) -> dict[str, Any]:
    centers = np.asarray(np.load(VOCAB_ROOT / f"K{k}/prototype_vocab.npz")["prototype_centers"], dtype=np.float32)
    by_split: dict[str, Any] = {}
    all_counts = np.zeros(k, dtype=np.int64)
    consistency = []
    for split in ("train", "val", "test"):
        counts = np.zeros(k, dtype=np.int64)
        vals: dict[str, list[float]] = {x: [] for x in ["global_copy_top5", "global_sample_top5", "stable_copy_top5", "changed_sample_top5", "semantic_hard_sample_top5", "changed_oracle_top5", "semantic_hard_oracle_top5"]}
        stable_count = changed_count = valid_count = hard_count = 0
        for path in sorted((VIS_ROOT / split).glob("*.npz")):
            z = np.load(path, allow_pickle=True)
            uid = str(np.asarray(z["sample_uid"]).item())
            fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
            fut_mask = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_teacher_embedding"], dtype=np.float32)
            obs_mask = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
            target = assign(fut, centers, device)
            obs_id = assign(obs, centers, device)
            valid = fut_mask & (target >= 0)
            ids = target[valid]
            if ids.size:
                counts += np.bincount(ids, minlength=k).astype(np.int64)
            last = np.full((obs_id.shape[0],), -1, dtype=np.int64)
            for m in range(obs_id.shape[0]):
                ii = np.where(obs_mask[m] & (obs_id[m] >= 0))[0]
                if ii.size:
                    last[m] = obs_id[m, ii[-1]]
                tv = target[m][valid[m]]
                if tv.size:
                    cc = np.bincount(tv, minlength=k)
                    consistency.append(float(cc.max() / tv.size))
            copy = np.broadcast_to(last[:, None], target.shape)
            stable = valid & (copy == target) & (copy >= 0)
            changed = valid & (copy != target) & (copy >= 0)
            hard = load_semantic_hard(split, uid, target.shape) & valid
            stable_count += int(stable.sum()); changed_count += int(changed.sum()); valid_count += int(valid.sum()); hard_count += int(hard.sum())
            copy_logits = np.log(onehot(copy, k).clip(1e-8, 1.0))
            sample_logits = np.log(sample_freq(obs_id, obs_mask, target.shape[1], k).clip(1e-8, 1.0))
            oracle_logits = np.log(onehot(target, k).clip(1e-8, 1.0))
            for key, logits, mask in [
                ("global_copy_top5", copy_logits, valid),
                ("global_sample_top5", sample_logits, valid),
                ("stable_copy_top5", copy_logits, stable),
                ("changed_sample_top5", sample_logits, changed),
                ("semantic_hard_sample_top5", sample_logits, hard),
                ("changed_oracle_top5", oracle_logits, changed),
                ("semantic_hard_oracle_top5", oracle_logits, hard),
            ]:
                v = topk_from_scores(logits, target, mask, 5)
                if v is not None:
                    vals[key].append(v)
        all_counts += counts
        by_split[split] = {
            "prototype_entropy": entropy(counts),
            "empty_cluster_count": int((counts == 0).sum()),
            "dominant_cluster_ratio": float(counts.max() / max(counts.sum(), 1)),
            "stable_ratio": float(stable_count / max(valid_count, 1)),
            "changed_ratio": float(changed_count / max(valid_count, 1)),
            "semantic_hard_count": hard_count,
            **{name: (float(np.mean(v)) if v else None) for name, v in vals.items()},
        }
    val = by_split["val"]
    # Oracle must be nontrivial: labels exist, target distribution is not collapsed,
    # temporal consistency is adequate, and sample-frequency does not already solve
    # the changed/hard subsets.
    oracle_pass = bool(
        val["changed_oracle_top5"] == 1.0
        and val["semantic_hard_oracle_top5"] == 1.0
        and entropy(all_counts) > 0.55
        and float(np.mean(consistency) if consistency else 0.0) > 0.60
        and (val["changed_sample_top5"] or 0.0) < 0.80
    )
    return {
        "K": k,
        "teacher": "clip_vit_b32_local",
        "aggregation": "point_local_crop_cached",
        "by_split": by_split,
        "prototype_entropy": entropy(all_counts),
        "empty_cluster_count": int((all_counts == 0).sum()),
        "target_temporal_consistency": float(np.mean(consistency)) if consistency else None,
        "target_space_oracle_passes": oracle_pass,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for k in (32, 64, 128, 256):
        if (VOCAB_ROOT / f"K{k}/prototype_vocab.npz").exists():
            rows.append(eval_k(k, device))
    best = max(rows, key=lambda r: (r["by_split"]["val"].get("changed_oracle_top5") or 0.0) - (r["by_split"]["val"].get("changed_sample_top5") or 0.0)) if rows else None
    ready = bool(best and best["target_space_oracle_passes"])
    payload = {
        "generated_at_utc": utc_now(),
        "candidates": rows,
        "best_teacher_by_val": best["teacher"] if best else None,
        "best_aggregation_by_val": best["aggregation"] if best else None,
        "best_K_by_val": best["K"] if best else None,
        "clip_k32_is_bottleneck": bool(best and best["K"] != 32),
        "target_space_oracle_passes": ready,
        "target_space_ready_for_training": ready,
        "recommended_next_step": "train_v33_12_copy_residual_on_best_target_space" if ready else "build_even_stronger_teacher_ensemble",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.12 Semantic Target Space Sweep", payload, ["best_teacher_by_val", "best_aggregation_by_val", "best_K_by_val", "clip_k32_is_bottleneck", "target_space_oracle_passes", "target_space_ready_for_training", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
