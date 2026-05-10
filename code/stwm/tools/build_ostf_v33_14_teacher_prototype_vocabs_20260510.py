#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


FEATURE_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_features/pointodyssey"
OUT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_prototypes/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v33_14_teacher_prototype_vocab_sweep_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_TEACHER_PROTOTYPE_VOCAB_SWEEP_20260510.md"


def entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p /= max(float(p.sum()), 1.0)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(max(len(p), 2)))


def assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-6)
    c = centers.astype(np.float32)
    c /= np.linalg.norm(c, axis=-1, keepdims=True).clip(1e-6)
    out = []
    for i in range(0, x.shape[0], 32768):
        out.append((x[i : i + 32768] @ c.T).argmax(axis=-1))
    return np.concatenate(out)


def collect(root: Path, split: str, key: str = "fut_teacher_embedding", max_points: int = 220000) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    masks = []
    for p in sorted((root / split).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        x = np.asarray(z[key], dtype=np.float32)
        m = np.asarray(z[key.replace("embedding", "available_mask")]).astype(bool)
        valid = x[m]
        if valid.size:
            xs.append(valid)
            masks.append(m.reshape(-1))
    if not xs:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=bool)
    xall = np.concatenate(xs, axis=0)
    if xall.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(xall.shape[0], size=max_points, replace=False)
        xall = xall[idx]
    return xall, np.ones((xall.shape[0],), dtype=bool)


def consistency(root: Path, centers: np.ndarray, split: str) -> tuple[float | None, float | None]:
    point_scores = []
    inst_scores = []
    for p in sorted((root / split).glob("*.npz"))[:128]:
        z = np.load(p, allow_pickle=True)
        fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
        mask = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        ids = assign(fut.reshape(-1, fut.shape[-1]), centers).reshape(fut.shape[:2])
        for m in range(ids.shape[0]):
            vals = ids[m][mask[m]]
            if vals.size:
                c = np.bincount(vals, minlength=centers.shape[0])
                point_scores.append(float(c.max() / vals.size))
        inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
        for iid in np.unique(inst[inst >= 0]):
            vals = ids[inst == iid][mask[inst == iid]]
            if vals.size:
                c = np.bincount(vals, minlength=centers.shape[0])
                inst_scores.append(float(c.max() / vals.size))
    return (float(np.mean(point_scores)) if point_scores else None, float(np.mean(inst_scores)) if inst_scores else None)


def run_one(root: Path, teacher: str, aggregation: str, k: int) -> dict[str, Any]:
    train_x, _ = collect(root, "train")
    if train_x.shape[0] < k:
        return {"teacher": teacher, "aggregation": aggregation, "K": k, "built": False, "exact_blocker": f"not enough train embeddings: {train_x.shape[0]} < {k}"}
    km = MiniBatchKMeans(n_clusters=k, batch_size=8192, random_state=42, n_init="auto", max_iter=120)
    km.fit(train_x)
    centers = km.cluster_centers_.astype(np.float32)
    out_dir = OUT / teacher / aggregation / f"K{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "prototype_vocab.npz", prototype_centers=centers, teacher_name=teacher, aggregation=aggregation, K=k)
    by_split: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        x, _ = collect(root, split, max_points=400000)
        ids = assign(x, centers) if x.size else np.zeros((0,), dtype=np.int64)
        counts = np.bincount(ids, minlength=k) if ids.size else np.zeros(k, dtype=np.int64)
        point_cons, inst_cons = consistency(root, centers, split)
        by_split[split] = {
            "cluster_entropy": entropy(counts),
            "empty_cluster_count": int((counts == 0).sum()),
            "dominant_cluster_ratio": float(counts.max() / max(counts.sum(), 1)),
            "prototype_size_min_median_max": [int(counts.min()), float(np.median(counts)), int(counts.max())],
            "same_point_temporal_consistency": point_cons,
            "same_instance_temporal_consistency": inst_cons,
        }
    return {
        "teacher": teacher,
        "aggregation": aggregation,
        "K": k,
        "built": True,
        "prototype_vocab_path": str((out_dir / "prototype_vocab.npz").relative_to(ROOT)),
        "by_split": by_split,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", nargs="*", default=None)
    p.add_argument("--ks", type=int, nargs="*", default=[64, 128, 256, 512])
    args = p.parse_args()
    rows = []
    teachers = args.teachers or [p.name for p in FEATURE_ROOT.iterdir() if p.is_dir()] if FEATURE_ROOT.exists() else []
    for teacher in teachers:
        for agg_root in sorted((FEATURE_ROOT / teacher).glob("*")):
            if not agg_root.is_dir():
                continue
            for k in args.ks:
                rows.append(run_one(agg_root, teacher, agg_root.name, k))
    built = [r for r in rows if r.get("built")]
    best = max(built, key=lambda r: r["by_split"]["val"]["cluster_entropy"] - r["by_split"]["val"]["dominant_cluster_ratio"]) if built else None
    payload = {
        "generated_at_utc": utc_now(),
        "teacher_prototype_vocab_sweep_done": bool(rows),
        "rows": rows,
        "best_teacher_by_val": best.get("teacher") if best else None,
        "best_aggregation_by_val": best.get("aggregation") if best else None,
        "best_K_by_val": best.get("K") if best else None,
        "exact_blockers": [] if built else ["no teacher feature cache available"],
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.14 Teacher Prototype Vocab Sweep", payload, ["teacher_prototype_vocab_sweep_done", "best_teacher_by_val", "best_aggregation_by_val", "best_K_by_val", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
