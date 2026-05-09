#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"
PROTO_REPORT = ROOT / "reports/stwm_ostf_v33_3_semantic_prototype_targets_20260509.json"
MANIFEST_DIR = ROOT / "manifests/ostf_v33_4_separated_hard_identity_semantic"
REPORT = ROOT / "reports/stwm_ostf_v33_4_separated_hard_subset_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_4_SEPARATED_HARD_SUBSET_20260509.md"


def selected_k(default: int = 32) -> int:
    if PROTO_REPORT.exists():
        return int(json.loads(PROTO_REPORT.read_text(encoding="utf-8")).get("selected_K", default))
    return default


def choose(pos: np.ndarray, neg: np.ndarray, *, seed: int, max_each: int = 1024) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    pi = np.argwhere(pos)
    ni = np.argwhere(neg)
    n = min(len(pi), len(ni), max_each)
    mask = np.zeros_like(pos, dtype=bool)
    if n > 0:
        ps = pi[rng.choice(len(pi), n, replace=False)]
        ns = ni[rng.choice(len(ni), n, replace=False)]
        mask[ps[:, 0], ps[:, 1]] = True
        mask[ns[:, 0], ns[:, 1]] = True
    return mask, {
        "available_identity_positive": int(len(pi)),
        "available_identity_negative": int(len(ni)),
        "selected_identity_positive": int(n),
        "selected_identity_negative": int(n),
    }


def semantic_mask(uid: str, split: str, k: int, avail_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, int]]:
    path = ROOT / f"outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}/{split}/{uid}.npz"
    if not path.exists():
        return np.zeros(avail_shape, dtype=bool), {"available_semantic_changed": 0, "selected_semantic_changed": 0}
    z = np.load(path, allow_pickle=True)
    fut = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
    fm = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
    obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
    om = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
    last = np.full((obs.shape[0],), -1, dtype=np.int64)
    for i in range(obs.shape[0]):
        idx = np.where(om[i])[0]
        if idx.size:
            last[i] = int(obs[i, idx[-1]])
    changed = fm & (fut >= 0) & (last[:, None] >= 0) & (fut != last[:, None])
    return changed, {"available_semantic_changed": int(changed.sum()), "selected_semantic_changed": int(changed.sum())}


def build_split(split: str, *, k: int, max_items: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted((IDENTITY_ROOT / split).glob("*_M128_H32.npz"))
    if max_items is not None:
        files = files[: int(max_items)]
    mask_dir = MANIFEST_DIR / "masks" / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    agg = {
        "available_identity_positive": 0,
        "available_identity_negative": 0,
        "selected_identity_positive": 0,
        "selected_identity_negative": 0,
        "available_semantic_changed": 0,
        "selected_semantic_changed": 0,
        "occlusion_reappearance_count": 0,
        "visibility_change_count": 0,
    }
    for n, path in enumerate(files):
        z = np.load(path, allow_pickle=True)
        uid = str(np.asarray(z["sample_uid"]).item())
        avail = np.asarray(z["fut_instance_available_mask"]).astype(bool)
        same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
        pos = avail & same
        neg = avail & (~same)
        identity_mask, stats = choose(pos, neg, seed=seed + n)
        sem_mask, sem_stats = semantic_mask(uid, split, k, avail.shape)
        obs_av = np.asarray(z["obs_instance_available_mask"]).astype(bool) if "obs_instance_available_mask" in z.files else np.zeros((avail.shape[0], 8), dtype=bool)
        last_obs = obs_av[:, -1:]
        agg["occlusion_reappearance_count"] += int((last_obs & (~avail)).sum())
        agg["visibility_change_count"] += int((last_obs != avail).sum())
        for key, val in {**stats, **sem_stats}.items():
            agg[key] += int(val)
        out_path = mask_dir / f"{uid}.npz"
        np.savez_compressed(
            out_path,
            identity_hard_eval_mask=identity_mask,
            semantic_hard_eval_mask=sem_mask,
            identity_positive_mask=pos & identity_mask,
            identity_negative_mask=neg & identity_mask,
            semantic_changed_mask=sem_mask,
        )
        if identity_mask.any() or sem_mask.any():
            entries.append(
                {
                    "sample_uid": uid,
                    "split": split,
                    "identity_sidecar": str(path.relative_to(ROOT)),
                    "mask_path": str(out_path.relative_to(ROOT)),
                    **stats,
                    **sem_stats,
                }
            )
    total_selected = agg["selected_identity_positive"] + agg["selected_identity_negative"]
    pos_ratio = agg["selected_identity_positive"] / max(total_selected, 1)
    neg_ratio = agg["selected_identity_negative"] / max(total_selected, 1)
    summary = {
        **agg,
        "entry_count": len(entries),
        "actual_identity_positive_ratio": float(pos_ratio),
        "actual_identity_negative_ratio": float(neg_ratio),
        "identity_hard_balanced": bool(0.35 <= pos_ratio <= 0.65 and 0.35 <= neg_ratio <= 0.65 and total_selected > 0),
        "semantic_hard_nonempty": bool(agg["selected_semantic_changed"] > 0),
    }
    return entries, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--max-items", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    k = int(args.K or selected_k())
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "name": "ostf_v33_4_separated_hard_identity_semantic_H32_M128_seed42",
        "M": 128,
        "H": 32,
        "K": k,
        "mask_schema": "identity_hard_eval_mask and semantic_hard_eval_mask are separate; semantic changes are never identity negatives",
        "splits": {},
    }
    summaries: dict[str, Any] = {}
    for split in ("val", "test"):
        entries, summary = build_split(split, k=k, max_items=args.max_items, seed=args.seed)
        manifest["splits"][split] = entries
        summaries[split] = summary
    manifest_path = MANIFEST_DIR / "H32_M128_seed42.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test = summaries.get("test", {})
    payload = {
        "generated_at_utc": utc_now(),
        "separated_hard_subset_built": bool(test.get("identity_hard_balanced", False) and test.get("semantic_hard_nonempty", False)),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "selected_K": k,
        "actual_identity_positive_ratio": test.get("actual_identity_positive_ratio"),
        "actual_identity_negative_ratio": test.get("actual_identity_negative_ratio"),
        "identity_hard_balanced": bool(test.get("identity_hard_balanced", False)),
        "semantic_hard_nonempty": bool(test.get("semantic_hard_nonempty", False)),
        "exact_blocker": None if test.get("identity_hard_balanced", False) else "insufficient actual same-instance negatives to form balanced identity hard eval",
        "splits": summaries,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.4 Separated Hard Subset", payload, ["separated_hard_subset_built", "manifest_path", "actual_identity_positive_ratio", "actual_identity_negative_ratio", "identity_hard_balanced", "semantic_hard_nonempty", "exact_blocker"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
