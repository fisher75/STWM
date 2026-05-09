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
MANIFEST_DIR = ROOT / "manifests/ostf_v33_3_balanced_hard_identity_semantic"
REPORT = ROOT / "reports/stwm_ostf_v33_3_balanced_hard_subset_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_3_BALANCED_HARD_SUBSET_20260509.md"


def selected_k(default: int = 64) -> int:
    if not PROTO_REPORT.exists():
        return default
    return int(json.loads(PROTO_REPORT.read_text(encoding="utf-8")).get("selected_K", default))


def proto_root(k: int) -> Path:
    return ROOT / f"outputs/cache/stwm_ostf_v33_3_semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K{k}"


def choose_balanced(pos: np.ndarray, neg: np.ndarray, *, seed: int, max_each: int | None = None) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    pidx = np.argwhere(pos)
    nidx = np.argwhere(neg)
    n = min(len(pidx), len(nidx))
    if max_each is not None:
        n = min(n, int(max_each))
    mask = np.zeros_like(pos, dtype=bool)
    if n > 0:
        ps = pidx[rng.choice(len(pidx), size=n, replace=False)]
        ns = nidx[rng.choice(len(nidx), size=n, replace=False)]
        mask[ps[:, 0], ps[:, 1]] = True
        mask[ns[:, 0], ns[:, 1]] = True
    return mask, {"selected_positive": int(n), "selected_negative": int(n), "available_positive": int(len(pidx)), "available_negative": int(len(nidx))}


def build_split(split: str, *, k: int, max_items: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted((IDENTITY_ROOT / split).glob("*_M128_H32.npz"))
    if max_items is not None:
        files = files[: int(max_items)]
    entries: list[dict[str, Any]] = []
    total_pos = total_neg = total_sel_pos = total_sel_neg = 0
    confuser_count = occlusion_count = semantic_confuser = visibility_changes = 0
    mask_dir = MANIFEST_DIR / "masks" / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    pr_root = proto_root(k) / split
    for idx, path in enumerate(files):
        z = np.load(path, allow_pickle=True)
        uid = str(np.asarray(z["sample_uid"]).item())
        same = np.asarray(z["fut_same_instance_as_obs"]).astype(bool)
        avail = np.asarray(z["fut_instance_available_mask"]).astype(bool)
        vis = np.asarray(z["fut_point_visible_target"]).astype(bool)
        neg = avail & (~same)
        pos = avail & same
        # Semantic confusers: future prototype differs from last observed prototype.
        proto_conf = np.zeros_like(avail, dtype=bool)
        ppath = pr_root / f"{uid}.npz"
        if ppath.exists():
            pz = np.load(ppath, allow_pickle=True)
            fut_proto = np.asarray(pz["semantic_prototype_id"], dtype=np.int64)
            obs_proto = np.asarray(pz["obs_semantic_prototype_id"], dtype=np.int64)
            obs_mask = np.asarray(pz["obs_semantic_prototype_available_mask"]).astype(bool)
            last_obs = np.full((obs_proto.shape[0],), -1, dtype=np.int64)
            for m in range(obs_proto.shape[0]):
                valid_t = np.where(obs_mask[m])[0]
                if valid_t.size:
                    last_obs[m] = int(obs_proto[m, valid_t[-1]])
            proto_conf = avail & (fut_proto >= 0) & (last_obs[:, None] >= 0) & (fut_proto != last_obs[:, None])
        vis_change = avail & (vis != np.asarray(z["obs_instance_available_mask"]).astype(bool)[:, -1:, None].squeeze(-1) if "obs_instance_available_mask" in z.files else False)
        hard_neg = neg | proto_conf
        mask, stats = choose_balanced(pos, hard_neg, seed=seed + idx, max_each=512)
        if not mask.any():
            continue
        out_path = mask_dir / f"{uid}.npz"
        np.savez_compressed(out_path, hard_eval_mask=mask, positive_mask=pos & mask, negative_mask=hard_neg & mask)
        total_pos += stats["available_positive"]
        total_neg += stats["available_negative"]
        total_sel_pos += stats["selected_positive"]
        total_sel_neg += stats["selected_negative"]
        confuser_count += int(hard_neg.sum())
        semantic_confuser += int(proto_conf.sum())
        visibility_changes += int(vis_change.sum()) if isinstance(vis_change, np.ndarray) else 0
        # In the current PointOdyssey sidecars, occlusion/reappearance requires unavailable->available future instance.
        if "fut_instance_available_mask" in z.files and "obs_instance_available_mask" in z.files:
            obs_any = np.asarray(z["obs_instance_available_mask"]).astype(bool).any(axis=1)
            fut_av = np.asarray(z["fut_instance_available_mask"]).astype(bool)
            occlusion_count += int((obs_any[:, None] & (~fut_av)).sum())
        entries.append(
            {
                "sample_uid": uid,
                "split": split,
                "identity_sidecar": str(path.relative_to(ROOT)),
                "prototype_sidecar": str(ppath.relative_to(ROOT)) if ppath.exists() else None,
                "hard_mask_path": str(out_path.relative_to(ROOT)),
                **stats,
            }
        )
    selected_total = total_sel_pos + total_sel_neg
    ratio = total_sel_pos / max(selected_total, 1)
    summary = {
        "entry_count": len(entries),
        "available_positive": int(total_pos),
        "available_negative": int(total_neg),
        "selected_positive": int(total_sel_pos),
        "selected_negative": int(total_sel_neg),
        "positive_ratio": float(ratio),
        "negative_ratio": float(1.0 - ratio if selected_total else 0.0),
        "hard_negative_count": int(confuser_count),
        "same_video_negative_count": int(total_neg),
        "teacher_semantic_confuser_count": int(semantic_confuser),
        "visibility_change_count": int(visibility_changes),
        "occlusion_reappearance_count": int(occlusion_count),
        "whether_balanced_eval_possible": bool(0.35 <= ratio <= 0.65 and selected_total > 0),
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
        "name": "ostf_v33_3_balanced_hard_identity_semantic_H32_M128_seed42",
        "M": 128,
        "H": 32,
        "K": k,
        "selection_rule": "balanced positives vs hard negatives from target sidecars; no random eval-time positive sampling",
        "splits": {},
    }
    report_splits: dict[str, Any] = {}
    for split in ("val", "test"):
        entries, summary = build_split(split, k=k, max_items=args.max_items, seed=args.seed)
        manifest["splits"][split] = entries
        report_splits[split] = summary
    manifest_path = MANIFEST_DIR / "H32_M128_seed42.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test_summary = report_splits.get("test", {})
    payload = {
        "generated_at_utc": utc_now(),
        "balanced_hard_subset_built": bool(test_summary.get("whether_balanced_eval_possible", False)),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "positive_ratio": test_summary.get("positive_ratio"),
        "negative_ratio": test_summary.get("negative_ratio"),
        "hard_negative_count": test_summary.get("hard_negative_count"),
        "occlusion_reappearance_count": test_summary.get("occlusion_reappearance_count"),
        "confuser_count": test_summary.get("hard_negative_count"),
        "same_video_negative_count": test_summary.get("same_video_negative_count"),
        "teacher_semantic_confuser_count": test_summary.get("teacher_semantic_confuser_count"),
        "whether_balanced_eval_possible": test_summary.get("whether_balanced_eval_possible", False),
        "exact_blocker": None if test_summary.get("whether_balanced_eval_possible", False) else "insufficient hard negatives to balance eval",
        "splits": report_splits,
    }
    if int(test_summary.get("occlusion_reappearance_count") or 0) == 0:
        payload["occlusion_reappearance_exact_reason"] = "No observed-visible to future-unavailable instance transitions found in selected PointOdyssey M128/H32 sidecars."
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.3 Balanced Hard Identity Semantic Subset", payload, ["balanced_hard_subset_built", "manifest_path", "positive_ratio", "negative_ratio", "hard_negative_count", "occlusion_reappearance_count", "teacher_semantic_confuser_count", "whether_balanced_eval_possible", "exact_blocker"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
