#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


OUT_DIR = ROOT / "manifests/ostf_v33_5_split_matched_hard_identity_semantic"
REPORT = ROOT / "reports/stwm_ostf_v33_5_split_matched_hard_subset_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_5_SPLIT_MATCHED_HARD_SUBSET_20260509.md"


def v30_uid_set(split: str) -> set[str]:
    entries = json.loads((ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json").read_text(encoding="utf-8")).get("entries", [])
    out = set()
    for e in entries:
        if int(e.get("H", -1)) != 32 or int(e.get("M", -1)) != 128:
            continue
        path = ROOT / e["cache_path"]
        if path.exists():
            z = np.load(path, allow_pickle=True)
            out.add(str(np.asarray(z["video_uid"]).item() if "video_uid" in z else path.stem))
    return out


def paths(split: str, uid: str) -> dict[str, Path]:
    return {
        "identity": ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey" / split / f"{uid}.npz",
        "visual": ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local" / split / f"{uid}.npz",
        "proto": ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32" / split / f"{uid}.npz",
    }


def semantic_changed(uid: str, split: str, shape: tuple[int, int]) -> np.ndarray:
    p = paths(split, uid)["proto"]
    if not p.exists():
        return np.zeros(shape, dtype=bool)
    z = np.load(p, allow_pickle=True)
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


def candidate_rows(split: str) -> list[dict[str, Any]]:
    v30 = v30_uid_set(split)
    ids = sorted(p.stem for p in (ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey" / split).glob("*_M128_H32.npz"))
    rows = []
    for uid in ids:
        ps = paths(split, uid)
        if uid not in v30 or not all(p.exists() for p in ps.values()):
            continue
        z = np.load(ps["identity"], allow_pickle=True)
        avail = np.asarray(z["fut_instance_available_mask"]).astype(bool)
        same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
        obs_av = np.asarray(z["obs_instance_available_mask"]).astype(bool)
        fut_vis = np.asarray(z["fut_point_visible_target"]).astype(bool)
        inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
        sem = semantic_changed(uid, split, avail.shape)
        vz = np.load(ps["visual"], allow_pickle=True)
        teacher_cov = float(np.asarray(vz["fut_teacher_available_mask"]).astype(bool).mean())
        proto_ids = np.load(ps["proto"], allow_pickle=True)["semantic_prototype_id"]
        valid_proto = proto_ids[proto_ids >= 0]
        entropy = 0.0
        if valid_proto.size:
            _, counts = np.unique(valid_proto, return_counts=True)
            p = counts / counts.sum()
            entropy = float(-(p * np.log(p)).sum() / max(np.log(max(len(counts), 2)), 1e-6))
        rows.append(
            {
                "sample_uid": uid,
                "split": split,
                "available_identity_positive": int((avail & same).sum()),
                "available_identity_negative": int((avail & (~same)).sum()),
                "identity_positive_ratio": float((avail & same).sum() / max(avail.sum(), 1)),
                "identity_negative_ratio": float((avail & (~same)).sum() / max(avail.sum(), 1)),
                "occlusion_reappearance_count": int((obs_av[:, -1:] & (~avail)).sum()),
                "future_visibility_ratio": float(fut_vis.mean()),
                "semantic_prototype_entropy": entropy,
                "teacher_embedding_coverage": teacher_cov,
                "crop_failure_ratio": float(1.0 - teacher_cov),
                "instance_count": int(len(set(inst[inst >= 0].tolist()))),
                "semantic_changed_count": int(sem.sum()),
            }
        )
    return rows


def vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            row["identity_negative_ratio"],
            row["occlusion_reappearance_count"] / 4096.0,
            row["future_visibility_ratio"],
            row["semantic_prototype_entropy"],
            row["teacher_embedding_coverage"],
            row["instance_count"] / 32.0,
            row["semantic_changed_count"] / 4096.0,
        ],
        dtype=np.float64,
    )


def distance(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float:
    if not a or not b:
        return 1e9
    return float(np.abs(np.stack([vector(x) for x in a]).mean(axis=0) - np.stack([vector(x) for x in b]).mean(axis=0)).sum())


def select_matched(val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    rng = np.random.default_rng(seed)
    n = min(len(val_rows), len(test_rows))
    best_val = val_rows if len(val_rows) == n else []
    best_test = test_rows if len(test_rows) == n else []
    best = distance(best_val, best_test)
    for _ in range(200):
        vsel = [val_rows[i] for i in (rng.choice(len(val_rows), n, replace=False) if len(val_rows) > n else np.arange(n))]
        tsel = [test_rows[i] for i in (rng.choice(len(test_rows), n, replace=False) if len(test_rows) > n else np.arange(n))]
        d = distance(vsel, tsel)
        if d < best:
            best = d
            best_val = vsel
            best_test = tsel
    return best_val, best_test, best


def choose_balanced(pos: np.ndarray, neg: np.ndarray, seed: int, max_each: int = 1024) -> tuple[np.ndarray, int, int]:
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
    return mask, int(n), int(n)


def write_manifest(seed: int, val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    manifest = {"generated_at_utc": utc_now(), "seed": seed, "M": 128, "H": 32, "splits": {"val": [], "test": []}}
    summaries = {}
    for split, rows in [("val", val_rows), ("test", test_rows)]:
        mask_dir = OUT_DIR / "masks" / f"seed{seed}" / split
        mask_dir.mkdir(parents=True, exist_ok=True)
        pos_sel = neg_sel = sem_sel = 0
        for i, row in enumerate(rows):
            uid = row["sample_uid"]
            z = np.load(paths(split, uid)["identity"], allow_pickle=True)
            avail = np.asarray(z["fut_instance_available_mask"]).astype(bool)
            same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
            identity_mask, pcount, ncount = choose_balanced(avail & same, avail & (~same), seed + i)
            sem_mask = semantic_changed(uid, split, avail.shape)
            out = mask_dir / f"{uid}.npz"
            np.savez_compressed(out, identity_hard_eval_mask=identity_mask, semantic_hard_eval_mask=sem_mask)
            pos_sel += pcount
            neg_sel += ncount
            sem_sel += int(sem_mask.sum())
            manifest["splits"][split].append(
                {
                    "split": split,
                    "sample_uid": uid,
                    "mask_path": str(out.relative_to(ROOT)),
                    "stratum_labels": {
                        "identity_negative_ratio_bin": int(min(4, row["identity_negative_ratio"] * 20)),
                        "occlusion_bin": int(min(4, row["occlusion_reappearance_count"] // 512)),
                        "semantic_entropy_bin": int(min(4, row["semantic_prototype_entropy"] * 5)),
                    },
                    "selected_identity_positive": pcount,
                    "selected_identity_negative": ncount,
                    "selected_semantic_changed_count": int(sem_mask.sum()),
                    "selection_seed": seed,
                }
            )
        total = pos_sel + neg_sel
        summaries[split] = {
            "sample_count": len(rows),
            "selected_identity_positive": pos_sel,
            "selected_identity_negative": neg_sel,
            "actual_identity_positive_ratio": float(pos_sel / max(total, 1)),
            "actual_identity_negative_ratio": float(neg_sel / max(total, 1)),
            "selected_semantic_changed": sem_sel,
            "identity_hard_balanced": bool(total > 0 and 0.35 <= pos_sel / max(total, 1) <= 0.65),
            "semantic_hard_nonempty": bool(sem_sel > 0),
        }
    path = OUT_DIR / f"H32_M128_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, summaries


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = ["identity_negative_ratio", "occlusion_reappearance_count", "future_visibility_ratio", "semantic_prototype_entropy", "teacher_embedding_coverage", "crop_failure_ratio", "instance_count", "semantic_changed_count"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,123,456")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    val_all = candidate_rows("val")
    test_all = candidate_rows("test")
    before = distance(val_all, test_all)
    manifests = {}
    afters = []
    stable = True
    for seed in seeds:
        vsel, tsel, after = select_matched(val_all, test_all, seed)
        path, summaries = write_manifest(seed, vsel, tsel)
        manifests[f"seed{seed}"] = {"manifest_path": str(path.relative_to(ROOT)), "summaries": summaries, "selected_distribution": {"val": summarize_rows(vsel), "test": summarize_rows(tsel)}, "distance_after": after}
        afters.append(after)
        stable = stable and summaries["val"]["identity_hard_balanced"] and summaries["test"]["identity_hard_balanced"] and summaries["val"]["semantic_hard_nonempty"] and summaries["test"]["semantic_hard_nonempty"]
    payload = {
        "generated_at_utc": utc_now(),
        "split_matched_hard_subset_built": bool(stable),
        "complete_candidate_counts": {"val": len(val_all), "test": len(test_all)},
        "val_test_distribution_distance_before": before,
        "val_test_distribution_distance_after": float(np.mean(afters)) if afters else None,
        "identity_hard_balanced_by_split": {seed: manifests[f"seed{seed}"]["summaries"] for seed in seeds},
        "semantic_hard_nonempty_by_split": {seed: {split: manifests[f"seed{seed}"]["summaries"][split]["semantic_hard_nonempty"] for split in ("val", "test")} for seed in seeds},
        "hard_subset_sampling_stable_ready": bool(stable and np.std(afters) < max(np.mean(afters), 1e-6)),
        "manifests": manifests,
        "exact_blocker": None if stable else "unable to build balanced identity/semantic hard subsets from complete candidates",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.5 Split Matched Hard Subset", payload, ["split_matched_hard_subset_built", "complete_candidate_counts", "val_test_distribution_distance_before", "val_test_distribution_distance_after", "hard_subset_sampling_stable_ready", "exact_blocker"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
