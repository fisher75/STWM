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


SRC_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
VOCAB_REPORT = ROOT / "reports/stwm_ostf_v33_8_semantic_prototype_vocab_20260510.json"
OUT_BASE = ROOT / "outputs/cache/stwm_ostf_v33_8_semantic_prototype_targets/pointodyssey/clip_vit_b32_local"
REPORT = ROOT / "reports/stwm_ostf_v33_8_semantic_prototype_targets_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_SEMANTIC_PROTOTYPE_TARGETS_20260510.md"


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def assign(emb: np.ndarray, mask: np.ndarray, centers: np.ndarray, topk: int = 8) -> tuple[np.ndarray, np.ndarray]:
    flat = emb.reshape(-1, emb.shape[-1])
    sims = normalize(flat) @ normalize(centers).T
    ids = sims.argmax(axis=1).astype(np.int64).reshape(emb.shape[:-1])
    dist = np.zeros((*emb.shape[:-1], centers.shape[0]), dtype=np.float16)
    kk = min(topk, centers.shape[0])
    part = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
    vals = np.take_along_axis(sims, part, axis=1)
    vals = np.exp(vals - vals.max(axis=1, keepdims=True))
    vals = vals / np.maximum(vals.sum(axis=1, keepdims=True), 1e-6)
    dist_flat = dist.reshape(flat.shape[0], centers.shape[0])
    rows = np.arange(flat.shape[0])[:, None]
    dist_flat[rows, part] = vals.astype(np.float16)
    ids = np.where(mask, ids, -1)
    return ids.astype(np.int64), dist


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=None)
    args = p.parse_args()
    report = json.loads(VOCAB_REPORT.read_text(encoding="utf-8")) if VOCAB_REPORT.exists() else {}
    k = int(args.K or report.get("selected_K", 32))
    vocab_path = ROOT / f"outputs/cache/stwm_ostf_v33_8_semantic_prototypes/pointodyssey/clip_vit_b32_local/K{k}/prototype_vocab.npz"
    if not vocab_path.exists():
        payload = {"generated_at_utc": utc_now(), "prototype_targets_built": False, "exact_blocker": f"missing vocab {vocab_path}"}
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V33.8 Semantic Prototype Targets", payload, ["prototype_targets_built", "exact_blocker"])
        print(REPORT.relative_to(ROOT))
        return 2
    centers = np.asarray(np.load(vocab_path)["prototype_centers"], dtype=np.float32)
    out_root = OUT_BASE / f"K{k}"
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        files = sorted((SRC_ROOT / split).glob("*.npz"))
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        fut_valid = obs_valid = 0
        for path in files:
            z = np.load(path, allow_pickle=True)
            fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
            fm = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
            obs = np.asarray(z["obs_teacher_embedding"], dtype=np.float32)
            om = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
            fut_id, fut_dist = assign(fut, fm, centers)
            obs_id, obs_dist = assign(obs, om, centers)
            uid = str(np.asarray(z["sample_uid"]).item())
            np.savez_compressed(
                out_dir / f"{uid}.npz",
                sample_uid=np.asarray(uid),
                dataset=np.asarray(str(np.asarray(z["dataset"]).item())),
                split=np.asarray(split),
                semantic_prototype_id=fut_id,
                semantic_prototype_available_mask=fm.astype(bool),
                semantic_prototype_distribution=fut_dist,
                obs_semantic_prototype_id=obs_id,
                obs_semantic_prototype_available_mask=om.astype(bool),
                obs_semantic_prototype_distribution=obs_dist,
                prototype_vocab_path=np.asarray(str(vocab_path.relative_to(ROOT))),
                teacher_name=np.asarray("clip_vit_b32_local"),
                leakage_safe=np.asarray(True),
                future_prototypes_supervision_only=np.asarray(True),
                future_prototypes_input_allowed=np.asarray(False),
            )
            fut_valid += int(fm.sum())
            obs_valid += int(om.sum())
        splits[split] = {"sample_count": len(files), "future_prototype_valid_count": fut_valid, "observed_prototype_valid_count": obs_valid, "output_dir": str(out_dir.relative_to(ROOT))}
    payload = {"generated_at_utc": utc_now(), "prototype_targets_built": True, "selected_K": k, "prototype_vocab_path": str(vocab_path.relative_to(ROOT)), "teacher_name": "clip_vit_b32_local", "leakage_safe": True, "future_prototypes_supervision_only": True, "future_prototypes_input_allowed": False, "splits": splits}
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.8 Semantic Prototype Targets", payload, ["prototype_targets_built", "selected_K", "prototype_vocab_path", "leakage_safe", "future_prototypes_supervision_only", "future_prototypes_input_allowed", "splits"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
