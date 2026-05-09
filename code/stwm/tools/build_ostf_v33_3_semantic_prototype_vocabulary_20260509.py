#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SRC_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_3_semantic_prototypes/pointodyssey/clip_vit_b32_local"
REPORT = ROOT / "reports/stwm_ostf_v33_3_semantic_prototype_vocab_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_3_SEMANTIC_PROTOTYPE_VOCAB_20260509.md"


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def collect_embeddings(split: str, *, max_files: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    files = sorted((SRC_ROOT / split).glob("*.npz"))
    if max_files is not None:
        files = files[: int(max_files)]
    chunks: list[np.ndarray] = []
    obs_chunks: list[np.ndarray] = []
    total = valid = 0
    for path in files:
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
    embs = normalize(np.concatenate(chunks, axis=0)) if chunks else np.zeros((0, 512), dtype=np.float32)
    obs_embs = normalize(np.concatenate(obs_chunks, axis=0)) if obs_chunks else np.zeros((0, embs.shape[-1] if embs.size else 512), dtype=np.float32)
    meta = {
        "split": split,
        "file_count": len(files),
        "future_valid_count": int(embs.shape[0]),
        "observed_valid_count": int(obs_embs.shape[0]),
        "future_embedding_coverage": float(valid / max(total, 1)),
    }
    if split == "train" and obs_embs.size:
        embs = np.concatenate([embs, obs_embs], axis=0)
        meta["fit_uses_train_observed_and_future_only"] = True
    return embs, meta


def assign(embs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    sims = normalize(embs) @ normalize(centers).T
    return sims.argmax(axis=1).astype(np.int64)


def cluster_stats(labels: np.ndarray, k: int) -> dict[str, Any]:
    counts = np.bincount(labels, minlength=k)
    p = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / max(np.log(k), 1e-6))
    return {
        "cluster_entropy_normalized": ent,
        "min_cluster_size": int(counts.min()) if counts.size else 0,
        "median_cluster_size": float(np.median(counts)) if counts.size else 0.0,
        "max_cluster_size": int(counts.max()) if counts.size else 0,
        "empty_cluster_count": int((counts == 0).sum()),
        "cluster_size_counts": counts.tolist(),
    }


def copy_baseline_for_split(split: str, centers: np.ndarray, *, max_files: int | None = None) -> dict[str, Any]:
    files = sorted((SRC_ROOT / split).glob("*.npz"))
    if max_files is not None:
        files = files[: int(max_files)]
    top1 = top5 = count = 0
    for path in files:
        z = np.load(path, allow_pickle=True)
        fut = normalize(np.asarray(z["fut_teacher_embedding"], dtype=np.float32))
        fm = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        obs = normalize(np.asarray(z["obs_teacher_embedding"], dtype=np.float32))
        om = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
        if not fm.any():
            continue
        obs_valid = om.astype(np.float32)
        obs_pooled = (obs * obs_valid[..., None]).sum(axis=1) / np.maximum(obs_valid.sum(axis=1, keepdims=True), 1.0)
        center_norm = normalize(centers)
        fut_id = (fut.reshape(-1, fut.shape[-1]) @ center_norm.T).argmax(axis=1).reshape(fut.shape[:2])
        obs_rank = np.argsort(-(obs_pooled @ center_norm.T), axis=1)
        m, h = fut_id.shape
        for i in range(m):
            for t in range(h):
                if not fm[i, t]:
                    continue
                count += 1
                target = int(fut_id[i, t])
                top1 += int(obs_rank[i, 0] == target)
                top5 += int(target in obs_rank[i, : min(5, obs_rank.shape[1])])
    return {
        f"{split}_copy_top1": float(top1 / max(count, 1)),
        f"{split}_copy_top5": float(top5 / max(count, 1)),
        f"{split}_copy_eval_count": int(count),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default="32,64,128,256")
    p.add_argument("--max-files-per-split", type=int, default=None)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    train_embs, train_meta = collect_embeddings("train", max_files=args.max_files_per_split)
    if train_embs.shape[0] == 0:
        payload = {"generated_at_utc": utc_now(), "prototype_vocab_built": False, "exact_blocker": "no train visual teacher embeddings found"}
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V33.3 Semantic Prototype Vocabulary", payload, ["prototype_vocab_built", "exact_blocker"])
        print(REPORT.relative_to(ROOT))
        return 2
    results: dict[str, Any] = {}
    best_k = None
    best_score = -1e9
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=args.random_state, batch_size=8192, n_init=3, max_iter=100)
        labels = km.fit_predict(train_embs)
        centers = normalize(km.cluster_centers_.astype(np.float32))
        out_dir = OUT_ROOT / f"K{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "prototype_vocab.npz", prototype_centers=centers, K=np.asarray(k), teacher_name=np.asarray("clip_vit_b32_local"))
        stats = cluster_stats(labels, k)
        stats.update(copy_baseline_for_split("val", centers, max_files=args.max_files_per_split))
        stats.update(copy_baseline_for_split("test", centers, max_files=args.max_files_per_split))
        stats["prototype_vocab_path"] = str((out_dir / "prototype_vocab.npz").relative_to(ROOT))
        stats["fit_split"] = "train"
        stats["val_selection_score"] = float(stats["val_copy_top5"] - 0.05 * stats["empty_cluster_count"] / max(k, 1) + 0.1 * stats["cluster_entropy_normalized"])
        results[f"K{k}"] = stats
        if stats["val_selection_score"] > best_score:
            best_score = float(stats["val_selection_score"])
            best_k = k
    payload = {
        "generated_at_utc": utc_now(),
        "prototype_vocab_built": True,
        "teacher_name": "clip_vit_b32_local",
        "teacher_embedding_dim": int(train_embs.shape[-1]),
        "train_meta": train_meta,
        "candidate_K": ks,
        "selected_K": int(best_k if best_k is not None else ks[0]),
        "selection_uses_val_only": True,
        "future_teacher_embeddings_input_allowed": False,
        "future_teacher_embeddings_supervision_only": True,
        "results": results,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.3 Semantic Prototype Vocabulary", payload, ["prototype_vocab_built", "teacher_name", "teacher_embedding_dim", "candidate_K", "selected_K", "selection_uses_val_only", "future_teacher_embeddings_input_allowed"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
