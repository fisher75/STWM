#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


FEATURE_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_features/pointodyssey"
VOCAB_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_prototypes/pointodyssey"
OUT = ROOT / "outputs/cache/stwm_ostf_v33_14_semantic_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v33_14_teacher_semantic_target_build_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_TEACHER_SEMANTIC_TARGET_BUILD_20260510.md"


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-6)


def assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    shape = x.shape[:-1]
    flat = normalize(x.reshape(-1, x.shape[-1]).astype(np.float32))
    c = normalize(centers.astype(np.float32))
    out = []
    for i in range(0, flat.shape[0], 32768):
        out.append((flat[i : i + 32768] @ c.T).argmax(axis=-1))
    return np.concatenate(out).reshape(shape)


def onehot(ids: np.ndarray, k: int, eps: float = 1e-4) -> np.ndarray:
    out = np.full((*ids.shape, k), eps / max(k - 1, 1), dtype=np.float32)
    valid = ids >= 0
    np.put_along_axis(out, ids.clip(0, k - 1)[..., None], 1.0, axis=-1)
    out[~valid] = 1.0 / k
    return out


def obs_freq(obs: np.ndarray, mask: np.ndarray, h: int, k: int) -> np.ndarray:
    out = np.zeros((obs.shape[0], h, k), dtype=np.float32)
    for m in range(obs.shape[0]):
        valid = mask[m] & (obs[m] >= 0)
        if valid.any():
            counts = np.bincount(obs[m, valid], minlength=k).astype(np.float32)
            dist = counts / counts.sum()
        else:
            dist = np.ones(k, dtype=np.float32) / k
        out[m] = dist[None, :]
    return out


def sample_freq(obs: np.ndarray, mask: np.ndarray, h: int, k: int) -> np.ndarray:
    counts = np.ones(k, dtype=np.float32) * 1e-3
    valid = mask & (obs >= 0)
    if valid.any():
        counts += np.bincount(obs[valid], minlength=k).astype(np.float32)
    dist = counts / counts.sum()
    return np.broadcast_to(dist[None, None, :], (obs.shape[0], h, k)).copy()


def build_one(feature_root: Path, vocab_path: Path, teacher: str, agg: str, k: int) -> dict[str, Any]:
    centers = np.asarray(np.load(vocab_path)["prototype_centers"], dtype=np.float32)
    train_global = np.ones(k, dtype=np.float32) * 1e-3
    cached: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        for p in sorted((feature_root / split).glob("*.npz")):
            z = np.load(p, allow_pickle=True)
            fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
            fut_mask = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_teacher_embedding"], dtype=np.float32)
            obs_mask = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
            fut_id = assign(fut, centers)
            obs_id = assign(obs, centers)
            cached[p] = (fut_id, fut_mask, obs_id, obs_mask)
            if split == "train":
                valid = obs_mask & (obs_id >= 0)
                if valid.any():
                    train_global += np.bincount(obs_id[valid], minlength=k).astype(np.float32)
    train_global /= train_global.sum()
    counts = {}
    for split in ("train", "val", "test"):
        out_dir = OUT / teacher / agg / f"K{k}" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        stable_count = changed_count = total = 0
        files = sorted((feature_root / split).glob("*.npz"))
        for p in files:
            z = np.load(p, allow_pickle=True)
            uid = str(z["sample_uid"].item() if hasattr(z["sample_uid"], "item") else z["sample_uid"])
            fut_id, fut_mask, obs_id, obs_mask = cached[p]
            last = np.full((obs_id.shape[0],), -1, dtype=np.int64)
            for m in range(obs_id.shape[0]):
                idx = np.where(obs_mask[m] & (obs_id[m] >= 0))[0]
                if idx.size:
                    last[m] = obs_id[m, idx[-1]]
            copy = np.broadcast_to(last[:, None], fut_id.shape).copy()
            valid = fut_mask & (fut_id >= 0)
            stable = valid & (copy == fut_id) & (copy >= 0)
            changed = valid & (copy != fut_id) & (copy >= 0)
            copy_dist = onehot(copy, k)
            obs_dist = obs_freq(obs_id, obs_mask, fut_id.shape[1], k)
            sample_dist = sample_freq(obs_id, obs_mask, fut_id.shape[1], k)
            global_dist = np.broadcast_to(train_global[None, None, :], (*fut_id.shape, k)).copy()
            np.savez_compressed(
                out_dir / f"{uid}.npz",
                sample_uid=uid,
                split=split,
                teacher_name=teacher,
                aggregation=agg,
                K=k,
                semantic_prototype_id=fut_id.astype(np.int64),
                semantic_prototype_available_mask=valid.astype(bool),
                obs_semantic_prototype_id=obs_id.astype(np.int64),
                obs_semantic_prototype_available_mask=obs_mask.astype(bool),
                copy_semantic_prototype_id=copy.astype(np.int64),
                semantic_stable_mask=stable.astype(bool),
                semantic_changed_mask=changed.astype(bool),
                semantic_update_target=changed.astype(np.float32),
                copy_prior_distribution=copy_dist.astype(np.float32),
                observed_frequency_prior_distribution=obs_dist.astype(np.float32),
                sample_level_frequency_prior_distribution=sample_dist.astype(np.float32),
                train_global_prior_distribution=global_dist.astype(np.float32),
                prototype_vocab_path=str(vocab_path.relative_to(ROOT)),
                leakage_safe=True,
                future_prototypes_supervision_only=True,
                future_prototypes_input_allowed=False,
            )
            stable_count += int(stable.sum())
            changed_count += int(changed.sum())
            total += int(valid.sum())
        counts[split] = {
            "sample_count": len(files),
            "stable_ratio": float(stable_count / max(total, 1)),
            "changed_ratio": float(changed_count / max(total, 1)),
        }
    return {"teacher": teacher, "aggregation": agg, "K": k, "target_root": str((OUT / teacher / agg / f"K{k}").relative_to(ROOT)), "by_split": counts}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", nargs="*", default=None)
    args = p.parse_args()
    rows = []
    blockers = []
    teachers = args.teachers or [x.name for x in FEATURE_ROOT.iterdir() if x.is_dir()] if FEATURE_ROOT.exists() else []
    for teacher in teachers:
        for agg_root in sorted((FEATURE_ROOT / teacher).glob("*")):
            for vocab in sorted((VOCAB_ROOT / teacher / agg_root.name).glob("K*/prototype_vocab.npz")):
                k = int(vocab.parent.name[1:])
                rows.append(build_one(agg_root, vocab, teacher, agg_root.name, k))
    if not rows:
        blockers.append("no feature/prototype vocabulary rows available")
    payload = {
        "generated_at_utc": utc_now(),
        "teacher_semantic_targets_built": bool(rows),
        "rows": rows,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.14 Teacher Semantic Target Build", payload, ["teacher_semantic_targets_built", "rows", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
