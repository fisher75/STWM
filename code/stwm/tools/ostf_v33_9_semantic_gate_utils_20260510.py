from __future__ import annotations

from typing import Any

import numpy as np


def last_observed_proto(obs_proto: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    out = np.full(obs_proto.shape[:2], -1, dtype=np.int64)
    for b in range(obs_proto.shape[0]):
        for m in range(obs_proto.shape[1]):
            ids = np.where(obs_mask[b, m].astype(bool))[0]
            if ids.size:
                out[b, m] = int(obs_proto[b, m, ids[-1]])
    return out


def topk_metric(logits: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int) -> float | None:
    valid = mask.astype(bool) & (target >= 0)
    if not bool(valid.any()):
        return None
    rank = np.argsort(-logits, axis=-1)[..., : min(k, logits.shape[-1])]
    hit = np.zeros_like(valid, dtype=bool)
    for j in range(rank.shape[-1]):
        hit |= rank[..., j] == target
    return float((hit & valid).sum() / max(valid.sum(), 1))


def copy_metric(copy: np.ndarray, target: np.ndarray, mask: np.ndarray, _k: int) -> float | None:
    valid = mask.astype(bool) & (target >= 0) & (copy >= 0)
    if not bool(valid.any()):
        return None
    return float(((copy == target) & valid).sum() / max(valid.sum(), 1))


def semantic_gate_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    full_mask: np.ndarray,
    hard_mask: np.ndarray,
    obs_proto: np.ndarray,
    obs_mask: np.ndarray,
) -> dict[str, Any]:
    full = full_mask.astype(bool) & (target >= 0)
    last = last_observed_proto(obs_proto, obs_mask)
    copy = np.broadcast_to(last[:, :, None], target.shape)
    copy_available = copy >= 0
    stable = full & copy_available & (copy == target)
    changed = full & copy_available & (copy != target)
    hard = hard_mask.astype(bool) & full

    stable_copy_top1 = copy_metric(copy, target, stable, 1)
    stable_model_top1 = topk_metric(logits, target, stable, 1)
    stable_model_top5 = topk_metric(logits, target, stable, 5)
    changed_copy_top1 = copy_metric(copy, target, changed, 1)
    changed_model_top1 = topk_metric(logits, target, changed, 1)
    changed_copy_top5 = copy_metric(copy, target, changed, 5)
    changed_model_top5 = topk_metric(logits, target, changed, 5)
    hard_copy_top1 = copy_metric(copy, target, hard, 1)
    hard_model_top1 = topk_metric(logits, target, hard, 1)
    hard_copy_top5 = copy_metric(copy, target, hard, 5)
    hard_model_top5 = topk_metric(logits, target, hard, 5)
    global_copy_top1 = copy_metric(copy, target, full, 1)
    global_model_top1 = topk_metric(logits, target, full, 1)
    global_copy_top5 = copy_metric(copy, target, full, 5)
    global_model_top5 = topk_metric(logits, target, full, 5)

    return {
        "stable_count": int(stable.sum()),
        "changed_count": int(changed.sum()),
        "semantic_hard_count": int(hard.sum()),
        "stable_copy_top1": stable_copy_top1,
        "stable_model_top1": stable_model_top1,
        "stable_model_top5": stable_model_top5,
        "stable_preservation_not_degraded": bool(stable_model_top1 is not None and stable_copy_top1 is not None and stable_model_top1 + 1e-9 >= stable_copy_top1),
        "changed_copy_top1": changed_copy_top1,
        "changed_model_top1": changed_model_top1,
        "changed_model_beats_copy": bool(changed_model_top1 is not None and changed_copy_top1 is not None and changed_model_top1 > changed_copy_top1 + 1e-9),
        "changed_copy_top5": changed_copy_top5,
        "changed_model_top5": changed_model_top5,
        "changed_top5_beats_copy": bool(changed_model_top5 is not None and changed_copy_top5 is not None and changed_model_top5 > changed_copy_top5 + 1e-9),
        "semantic_hard_copy_top1": hard_copy_top1,
        "semantic_hard_model_top1": hard_model_top1,
        "semantic_hard_copy_top5": hard_copy_top5,
        "semantic_hard_model_top5": hard_model_top5,
        "semantic_hard_model_beats_copy": bool(hard_model_top1 is not None and hard_copy_top1 is not None and hard_model_top1 > hard_copy_top1 + 1e-9),
        "semantic_hard_top5_beats_copy": bool(hard_model_top5 is not None and hard_copy_top5 is not None and hard_model_top5 > hard_copy_top5 + 1e-9),
        "global_semantic_copy_top1": global_copy_top1,
        "global_semantic_model_top1": global_model_top1,
        "global_semantic_copy_top5": global_copy_top5,
        "global_semantic_model_top5": global_model_top5,
        "global_semantic_top1_copy_beaten": bool(global_model_top1 is not None and global_copy_top1 is not None and global_model_top1 > global_copy_top1 + 1e-9),
        "global_semantic_top5_copy_beaten": bool(global_model_top5 is not None and global_copy_top5 is not None and global_model_top5 > global_copy_top5 + 1e-9),
    }


def aggregate_gate_dicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    keys = sorted({k for r in rows for k in r})
    for key in keys:
        vals = [r.get(key) for r in rows]
        clean = [float(v) for v in vals if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)]
        if clean:
            out[key] = {"mean": float(np.mean(clean)), "std": float(np.std(clean)), "worst": float(np.min(clean))}
        elif all(isinstance(v, bool) for v in vals if v is not None):
            out[key] = bool(all(vals))
        else:
            out[key] = vals[0]
    return out
