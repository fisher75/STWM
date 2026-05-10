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


OUT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"
REPORT = ROOT / "reports/stwm_ostf_v33_8_split_matched_hard_mask_build_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_SPLIT_MATCHED_HARD_MASK_BUILD_20260510.md"
COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"


def selected_k(default: int = 32) -> int:
    report = ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json"
    if report.exists():
        return int(json.loads(report.read_text(encoding="utf-8")).get("selected_K", default))
    return default


def id_root() -> Path:
    return COMPLETE / "semantic_identity_targets/pointodyssey"


def proto_root() -> Path:
    return COMPLETE / f"semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{selected_k()}"


def split_uids(split: str) -> list[str]:
    return sorted(p.stem for p in (id_root() / split).glob("*.npz"))


def semantic_changed(uid: str, split: str, shape: tuple[int, int]) -> np.ndarray:
    path = proto_root() / split / f"{uid}.npz"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    z = np.load(path, allow_pickle=True)
    fut = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
    fm = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
    obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
    om = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
    last = np.full(obs.shape[0], -1, dtype=np.int64)
    for i in range(obs.shape[0]):
        idx = np.where(om[i])[0]
        if idx.size:
            last[i] = int(obs[i, idx[-1]])
    return fm & (fut >= 0) & (last[:, None] >= 0) & (fut != last[:, None])


def choose_balanced(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator, max_each: int = 1536) -> tuple[np.ndarray, int, int]:
    pos_idx = np.argwhere(pos)
    neg_idx = np.argwhere(neg)
    n = min(len(pos_idx), len(neg_idx), max_each)
    out = np.zeros_like(pos, dtype=bool)
    if n > 0:
        ps = pos_idx[rng.choice(len(pos_idx), n, replace=False)]
        ns = neg_idx[rng.choice(len(neg_idx), n, replace=False)]
        out[ps[:, 0], ps[:, 1]] = True
        out[ns[:, 0], ns[:, 1]] = True
    return out, int(n), int(n)


def sample_stats(uid: str, split: str) -> dict[str, Any]:
    z = np.load(id_root() / split / f"{uid}.npz", allow_pickle=True)
    avail = np.asarray(z["fut_instance_available_mask"]).astype(bool)
    same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
    sem = semantic_changed(uid, split, avail.shape)
    pos = int((avail & same).sum())
    neg = int((avail & (~same)).sum())
    vis = float(np.asarray(z["fut_point_visible_target"]).astype(bool).mean()) if "fut_point_visible_target" in z.files else float(avail.mean())
    entropy = 0.0
    pp = proto_root() / split / f"{uid}.npz"
    if pp.exists():
        pz = np.load(pp, allow_pickle=True)
        ids = np.asarray(pz["semantic_prototype_id"], dtype=np.int64)
        mask = np.asarray(pz["semantic_prototype_available_mask"]).astype(bool)
        vals = ids[mask & (ids >= 0)]
        if vals.size:
            counts = np.bincount(vals)
            prob = counts[counts > 0] / max(vals.size, 1)
            entropy = float(-(prob * np.log(prob)).sum())
    return {
        "uid": uid,
        "available_identity_positive": pos,
        "available_identity_negative": neg,
        "future_visibility_ratio": vis,
        "semantic_changed_count": int(sem.sum()),
        "semantic_prototype_entropy": entropy,
        "has_identity_negative": bool(neg > 0),
        "has_identity_positive": bool(pos > 0),
    }


def distribution_distance(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float:
    keys = ["future_visibility_ratio", "semantic_changed_count", "semantic_prototype_entropy", "available_identity_negative"]
    if not a or not b:
        return 999.0
    vals = []
    for key in keys:
        av = float(np.mean([x[key] for x in a]))
        bv = float(np.mean([x[key] for x in b]))
        scale = float(np.std([x[key] for x in a + b]) + 1e-6)
        vals.append(abs(av - bv) / scale)
    return float(np.mean(vals))


def write_seed_manifest(seed: int) -> tuple[Path, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    manifest: dict[str, Any] = {"generated_at_utc": utc_now(), "seed": seed, "M": 128, "H": 32, "splits": {"train": [], "val": [], "test": []}}
    summaries: dict[str, Any] = {}
    raw_stats = {split: [sample_stats(uid, split) for uid in split_uids(split)] for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        mask_dir = OUT / "masks" / f"seed{seed}" / split
        mask_dir.mkdir(parents=True, exist_ok=True)
        selected_pos = selected_neg = sem_count = empty = 0
        rows = raw_stats[split]
        for i, row in enumerate(rows):
            uid = row["uid"]
            z = np.load(id_root() / split / f"{uid}.npz", allow_pickle=True)
            avail = np.asarray(z["fut_instance_available_mask"]).astype(bool)
            same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
            # Train and eval masks are separate arrays so downstream code cannot
            # accidentally conflate objectives, but both obey the same labels.
            train_mask, tp, tn = choose_balanced(avail & same, avail & (~same), rng)
            eval_mask, ep, en = choose_balanced(avail & same, avail & (~same), np.random.default_rng(seed * 1009 + i))
            sem_mask = semantic_changed(uid, split, avail.shape)
            out = mask_dir / f"{uid}.npz"
            np.savez_compressed(
                out,
                identity_hard_train_mask=train_mask,
                identity_hard_eval_mask=eval_mask,
                semantic_hard_train_mask=sem_mask,
                semantic_hard_eval_mask=sem_mask,
            )
            selected_pos += ep
            selected_neg += en
            sem_count += int(sem_mask.sum())
            empty += int(ep + en == 0)
            manifest["splits"][split].append(
                {
                    "split": split,
                    "sample_uid": uid,
                    "mask_path": str(out.relative_to(ROOT)),
                    "stratum_labels": {
                        "has_identity_negative": row["has_identity_negative"],
                        "has_identity_positive": row["has_identity_positive"],
                        "semantic_changed_count": row["semantic_changed_count"],
                    },
                    "selected_identity_positives": ep,
                    "selected_identity_negatives": en,
                    "selected_semantic_changed_count": int(sem_mask.sum()),
                    "selection_seed": seed,
                }
            )
        total = selected_pos + selected_neg
        summaries[split] = {
            "sample_count": len(rows),
            "selected_identity_positive": selected_pos,
            "selected_identity_negative": selected_neg,
            "actual_identity_positive_ratio": float(selected_pos / max(total, 1)),
            "actual_identity_negative_ratio": float(selected_neg / max(total, 1)),
            "identity_hard_balanced": bool(total > 0 and 0.35 <= selected_pos / max(total, 1) <= 0.65),
            "semantic_hard_nonempty": bool(sem_count > 0),
            "semantic_hard_count": sem_count,
            "empty_identity_mask_samples": empty,
            "raw_distribution": {
                "available_identity_positive": int(sum(x["available_identity_positive"] for x in rows)),
                "available_identity_negative": int(sum(x["available_identity_negative"] for x in rows)),
                "future_visibility_ratio_mean": float(np.mean([x["future_visibility_ratio"] for x in rows])) if rows else 0.0,
                "semantic_prototype_entropy_mean": float(np.mean([x["semantic_prototype_entropy"] for x in rows])) if rows else 0.0,
            },
        }
    path = OUT / f"H32_M128_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    before = distribution_distance(raw_stats["val"], raw_stats["test"])
    selected_val = [x for x in raw_stats["val"] if x["has_identity_positive"] and x["has_identity_negative"]]
    selected_test = [x for x in raw_stats["test"] if x["has_identity_positive"] and x["has_identity_negative"]]
    after = distribution_distance(selected_val, selected_test)
    summaries["val_test_distribution_distance_before"] = before
    summaries["val_test_distribution_distance_after"] = after
    return path, summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,456")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    manifests: dict[str, Any] = {}
    for seed in seeds:
        path, summary = write_seed_manifest(seed)
        manifests[f"seed{seed}"] = {"manifest_path": str(path.relative_to(ROOT)), "summary": summary}
    first = manifests[f"seed{seeds[0]}"]["summary"]
    train_bal = all(manifests[f"seed{s}"]["summary"]["train"]["identity_hard_balanced"] for s in seeds)
    eval_bal = all(
        manifests[f"seed{s}"]["summary"][split]["identity_hard_balanced"]
        for s in seeds
        for split in ("val", "test")
    )
    sem_nonempty = all(
        manifests[f"seed{s}"]["summary"][split]["semantic_hard_nonempty"]
        for s in seeds
        for split in ("train", "val", "test")
    )
    distances = [manifests[f"seed{s}"]["summary"]["val_test_distribution_distance_after"] for s in seeds]
    payload = {
        "generated_at_utc": utc_now(),
        "manifests": manifests,
        "identity_hard_train_balanced_by_split": {
            split: all(manifests[f"seed{s}"]["summary"][split]["identity_hard_balanced"] for s in seeds)
            for split in ("train", "val", "test")
        },
        "identity_hard_eval_balanced_by_split": {
            split: all(manifests[f"seed{s}"]["summary"][split]["identity_hard_balanced"] for s in seeds)
            for split in ("val", "test")
        },
        "semantic_hard_nonempty_by_split": {
            split: all(manifests[f"seed{s}"]["summary"][split]["semantic_hard_nonempty"] for s in seeds)
            for split in ("train", "val", "test")
        },
        "val_test_distribution_distance_before": first["val_test_distribution_distance_before"],
        "val_test_distribution_distance_after": float(np.mean(distances)) if distances else None,
        "hard_subset_sampling_stable": bool(train_bal and eval_bal and sem_nonempty),
        "exact_blockers": [],
    }
    if not train_bal:
        payload["exact_blockers"].append("train identity hard masks are not balanced by actual labels")
    if not eval_bal:
        payload["exact_blockers"].append("val/test identity hard masks are not balanced by actual labels")
    if not sem_nonempty:
        payload["exact_blockers"].append("semantic hard masks are empty in at least one split")
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.8 Split-Matched Hard Mask Build",
        payload,
        [
            "identity_hard_train_balanced_by_split",
            "identity_hard_eval_balanced_by_split",
            "semantic_hard_nonempty_by_split",
            "val_test_distribution_distance_before",
            "val_test_distribution_distance_after",
            "hard_subset_sampling_stable",
            "exact_blockers",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0 if payload["hard_subset_sampling_stable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
