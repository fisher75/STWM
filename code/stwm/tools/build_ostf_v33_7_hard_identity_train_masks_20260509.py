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


OUT = ROOT / "manifests/ostf_v33_7_hard_identity_train_masks"
REPORT = ROOT / "reports/stwm_ostf_v33_7_hard_identity_train_mask_build_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_7_HARD_IDENTITY_TRAIN_MASK_BUILD_20260509.md"
COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_7_complete_h32_m128"
ID_ROOT = COMPLETE / "semantic_identity_targets/pointodyssey"
PROTO_ROOT = COMPLETE / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"


def semantic_changed(uid: str, split: str, shape: tuple[int, int]) -> np.ndarray:
    path = PROTO_ROOT / split / f"{uid}.npz"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
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
    return fm & (fut >= 0) & (last[:, None] >= 0) & (fut != last[:, None])


def choose_balanced(pos: np.ndarray, neg: np.ndarray, seed: int, max_each: int = 1024) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    pi = np.argwhere(pos)
    ni = np.argwhere(neg)
    n = min(len(pi), len(ni), max_each)
    out = np.zeros_like(pos, dtype=bool)
    if n:
        ps = pi[rng.choice(len(pi), n, replace=False)]
        ns = ni[rng.choice(len(ni), n, replace=False)]
        out[ps[:, 0], ps[:, 1]] = True
        out[ns[:, 0], ns[:, 1]] = True
    return out, int(n), int(n)


def split_uids(split: str) -> list[str]:
    return sorted(p.stem for p in (ID_ROOT / split).glob("*.npz"))


def write_manifest(seed: int) -> tuple[Path, dict[str, Any]]:
    manifest: dict[str, Any] = {"generated_at_utc": utc_now(), "seed": seed, "M": 128, "H": 32, "splits": {"train": [], "val": [], "test": []}}
    summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        mask_dir = OUT / "masks" / f"seed{seed}" / split
        mask_dir.mkdir(parents=True, exist_ok=True)
        pos_sel = neg_sel = sem_sel = empty_samples = 0
        for i, uid in enumerate(split_uids(split)):
            z = np.load(ID_ROOT / split / f"{uid}.npz", allow_pickle=True)
            avail = np.asarray(z["fut_instance_available_mask"]).astype(bool)
            same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
            identity_mask, pc, nc = choose_balanced(avail & same, avail & (~same), seed + i)
            sem_mask = semantic_changed(uid, split, avail.shape)
            out = mask_dir / f"{uid}.npz"
            np.savez_compressed(out, identity_hard_train_mask=identity_mask, identity_hard_eval_mask=identity_mask, semantic_hard_train_mask=sem_mask, semantic_hard_eval_mask=sem_mask)
            pos_sel += pc
            neg_sel += nc
            sem_sel += int(sem_mask.sum())
            empty_samples += int(pc + nc == 0)
            manifest["splits"][split].append(
                {
                    "split": split,
                    "sample_uid": uid,
                    "mask_path": str(out.relative_to(ROOT)),
                    "selected_identity_positive": pc,
                    "selected_identity_negative": nc,
                    "selected_semantic_changed_count": int(sem_mask.sum()),
                    "selection_seed": seed,
                }
            )
        total = pos_sel + neg_sel
        summary[split] = {
            "sample_count": len(split_uids(split)),
            "selected_identity_positive": pos_sel,
            "selected_identity_negative": neg_sel,
            "positive_ratio": float(pos_sel / max(total, 1)),
            "negative_ratio": float(neg_sel / max(total, 1)),
            "identity_hard_balanced": bool(total > 0 and 0.35 <= pos_sel / max(total, 1) <= 0.65),
            "semantic_hard_nonempty": bool(sem_sel > 0),
            "semantic_hard_count": sem_sel,
            "empty_identity_mask_samples": empty_samples,
        }
    path = OUT / f"H32_M128_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,123,456")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    manifests: dict[str, Any] = {}
    for seed in seeds:
        path, summary = write_manifest(seed)
        manifests[f"seed{seed}"] = {"manifest_path": str(path.relative_to(ROOT)), "summary": summary}
    first = manifests[f"seed{seeds[0]}"]["summary"]
    payload = {
        "generated_at_utc": utc_now(),
        "manifests": manifests,
        "train_positive_ratio": first["train"]["positive_ratio"],
        "train_negative_ratio": first["train"]["negative_ratio"],
        "val_positive_ratio": first["val"]["positive_ratio"],
        "val_negative_ratio": first["val"]["negative_ratio"],
        "test_positive_ratio": first["test"]["positive_ratio"],
        "test_negative_ratio": first["test"]["negative_ratio"],
        "identity_hard_train_balanced": all(manifests[f"seed{s}"]["summary"]["train"]["identity_hard_balanced"] for s in seeds),
        "identity_hard_eval_balanced": all(manifests[f"seed{s}"]["summary"]["val"]["identity_hard_balanced"] and manifests[f"seed{s}"]["summary"]["test"]["identity_hard_balanced"] for s in seeds),
        "semantic_hard_nonempty": all(manifests[f"seed{s}"]["summary"][split]["semantic_hard_nonempty"] for s in seeds for split in ("train", "val", "test")),
        "exact_blockers": [],
    }
    if not payload["identity_hard_train_balanced"]:
        payload["exact_blockers"].append("train split lacks enough actual positive/negative identity labels to balance hard train masks")
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.7 Hard Identity Train Mask Build",
        payload,
        ["train_positive_ratio", "train_negative_ratio", "val_positive_ratio", "val_negative_ratio", "test_positive_ratio", "test_negative_ratio", "identity_hard_train_balanced", "identity_hard_eval_balanced", "semantic_hard_nonempty", "exact_blockers"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
