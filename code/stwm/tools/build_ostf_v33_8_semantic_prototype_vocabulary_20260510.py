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


SRC_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
OLD_SRC_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local"
REPORT = ROOT / "reports/stwm_ostf_v33_8_semantic_prototype_vocab_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_SEMANTIC_PROTOTYPE_VOCAB_20260510.md"


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def files(split: str) -> list[Path]:
    return sorted((SRC_ROOT / split).glob("*.npz"))


def collect(split: str) -> tuple[np.ndarray, dict[str, Any]]:
    chunks: list[np.ndarray] = []
    obs_chunks: list[np.ndarray] = []
    total = valid = 0
    fs = files(split)
    for path in fs:
        z = np.load(path, allow_pickle=True)
        fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
        fm = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        obs = np.asarray(z["obs_teacher_embedding"], dtype=np.float32)
        om = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
        total += int(fm.size)
        valid += int(fm.sum())
        if fm.any():
            chunks.append(fut[fm])
        if om.any():
            obs_chunks.append(obs[om])
    fut_emb = normalize(np.concatenate(chunks, axis=0)) if chunks else np.zeros((0, 512), dtype=np.float32)
    obs_emb = normalize(np.concatenate(obs_chunks, axis=0)) if obs_chunks else np.zeros((0, 512), dtype=np.float32)
    fit = np.concatenate([fut_emb, obs_emb], axis=0) if split == "train" and obs_emb.size else fut_emb
    return fit, {"split": split, "file_count": len(fs), "future_valid_count": int(fut_emb.shape[0]), "observed_valid_count": int(obs_emb.shape[0]), "future_embedding_coverage": float(valid / max(total, 1))}


def cluster_stats(labels: np.ndarray, k: int) -> dict[str, Any]:
    counts = np.bincount(labels, minlength=k)
    p = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / max(np.log(k), 1e-6))
    return {"cluster_entropy": ent, "empty_cluster_count": int((counts == 0).sum()), "cluster_size_min": int(counts.min()) if counts.size else 0, "cluster_size_median": float(np.median(counts)) if counts.size else 0.0, "cluster_size_max": int(counts.max()) if counts.size else 0, "cluster_size_distribution": counts.tolist()}


def copy_baseline(split: str, centers: np.ndarray) -> dict[str, Any]:
    top1 = top5 = count = 0
    c = normalize(centers)
    for path in files(split):
        z = np.load(path, allow_pickle=True)
        fut = normalize(np.asarray(z["fut_teacher_embedding"], dtype=np.float32))
        fm = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        obs = normalize(np.asarray(z["obs_teacher_embedding"], dtype=np.float32))
        om = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
        if not fm.any():
            continue
        fut_id = (fut.reshape(-1, fut.shape[-1]) @ c.T).argmax(axis=1).reshape(fut.shape[:2])
        obs_valid = om.astype(np.float32)
        obs_pooled = (obs * obs_valid[..., None]).sum(axis=1) / np.maximum(obs_valid.sum(axis=1, keepdims=True), 1.0)
        rank = np.argsort(-(obs_pooled @ c.T), axis=1)
        for i in range(fut_id.shape[0]):
            for h in range(fut_id.shape[1]):
                if not fm[i, h]:
                    continue
                count += 1
                target = int(fut_id[i, h])
                top1 += int(rank[i, 0] == target)
                top5 += int(target in rank[i, : min(5, rank.shape[1])])
    return {f"{split}_copy_top1": float(top1 / max(count, 1)), f"{split}_copy_top5": float(top5 / max(count, 1)), f"{split}_copy_eval_count": int(count)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default="32,64,128,256")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    train, train_meta = collect("train")
    old_train_count = len(list((OLD_SRC_ROOT / "train").glob("*.npz")))
    no_expansion = train_meta["file_count"] <= old_train_count
    if train.shape[0] == 0:
        payload = {"generated_at_utc": utc_now(), "prototype_vocab_built": False, "exact_blocker": "no expanded train visual teacher embeddings", "no_coverage_expansion_detected": no_expansion}
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V33.8 Semantic Prototype Vocabulary", payload, ["prototype_vocab_built", "exact_blocker", "no_coverage_expansion_detected"])
        print(REPORT.relative_to(ROOT))
        return 2
    results: dict[str, Any] = {}
    best_k = ks[0]
    best_score = -1e9
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=args.random_state, batch_size=8192, n_init=3, max_iter=120)
        labels = km.fit_predict(train)
        centers = normalize(km.cluster_centers_.astype(np.float32))
        out_dir = OUT_ROOT / f"K{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "prototype_vocab.npz", prototype_centers=centers, K=np.asarray(k), teacher_name=np.asarray("clip_vit_b32_local"))
        stats = cluster_stats(labels, k)
        stats.update(copy_baseline("val", centers))
        stats.update(copy_baseline("test", centers))
        stats["prototype_vocab_path"] = str((out_dir / "prototype_vocab.npz").relative_to(ROOT))
        stats["val_selection_score"] = float(stats["val_copy_top5"] + 0.1 * stats["cluster_entropy"] - 0.05 * stats["empty_cluster_count"] / max(k, 1))
        results[f"K{k}"] = stats
        if stats["val_selection_score"] > best_score:
            best_score = stats["val_selection_score"]
            best_k = k
    payload = {"generated_at_utc": utc_now(), "prototype_vocab_built": True, "teacher_name": "clip_vit_b32_local", "candidate_K": ks, "selected_K": best_k, "selection_uses_val_only": True, "train_meta": train_meta, "old_v33_3_train_visual_sidecar_count": old_train_count, "no_coverage_expansion_detected": no_expansion, "future_teacher_embeddings_input_allowed": False, "future_teacher_embeddings_supervision_only": True, "results": results}
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.8 Semantic Prototype Vocabulary", payload, ["prototype_vocab_built", "candidate_K", "selected_K", "selection_uses_val_only", "train_meta", "no_coverage_expansion_detected"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
