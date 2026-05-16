#!/usr/bin/env python3
"""把 V35.12 video semantic targets 修成更可观测、低维、可预测的状态目标。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

SRC_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_12_video_derived_future_semantic_state_targets/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_13_fixed_video_semantic_state_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_13_fixed_video_semantic_state_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_13_FIXED_VIDEO_SEMANTIC_STATE_TARGET_BUILD_20260515.md"
FAMILY_NAMES = ["copy_last_visible", "copy_instance_pooled", "copy_max_confidence", "changed_transition", "uncertain_abstain"]
UNCERTAIN = 4
CHANGED_TRANSITION = 3
COPY_INSTANCE = 1
COPY_LAST = 0
COPY_MAX = 2


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def list_npz(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.npz"))


def row_entropy(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((labels.shape[0],), dtype=np.float32)
    for i, row in enumerate(labels):
        vals = row[row >= 0]
        if vals.size:
            cnt = np.bincount(vals, minlength=k).astype(np.float32)
            p = cnt[cnt > 0] / max(cnt.sum(), 1.0)
            out[i] = float(-(p * np.log2(np.maximum(p, 1e-12))).sum() / max(np.log2(k), 1e-6))
    return out


def collect_risk_by_split(src_root: Path, semantic_clusters: int) -> dict[str, tuple[float, float]]:
    thresholds: dict[str, tuple[float, float]] = {}
    for split in ["train", "val", "test"]:
        vals: list[float] = []
        for p in list_npz(src_root, split):
            z = np.load(p, allow_pickle=True)
            obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
            obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
            obs_mconf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
            obs_mask = np.asarray(z["obs_semantic_measurement_mask"], dtype=np.float32)
            obs_cluster = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
            ent = row_entropy(obs_cluster, semantic_clusters)
            meas_conf = (obs_mconf * obs_mask).sum(axis=1) / np.maximum(obs_mask.sum(axis=1), 1.0)
            raw = (
                0.30 * (1.0 - obs_vis.mean(axis=1))
                + 0.25 * (1.0 - obs_conf.mean(axis=1))
                + 0.25 * ent
                + 0.20 * (1.0 - np.clip(meas_conf, 0.0, 1.0))
            )
            vals.extend(raw.tolist())
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size:
            thresholds[split] = (float(np.quantile(arr, 0.10)), float(np.quantile(arr, 0.90)))
        else:
            thresholds[split] = (0.0, 1.0)
    return thresholds


def normalized_observed_risk(z: Any, split: str, thresholds: dict[str, tuple[float, float]], semantic_clusters: int) -> np.ndarray:
    obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
    obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
    obs_mconf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
    obs_mask = np.asarray(z["obs_semantic_measurement_mask"], dtype=np.float32)
    obs_cluster = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
    ent = row_entropy(obs_cluster, semantic_clusters)
    meas_conf = (obs_mconf * obs_mask).sum(axis=1) / np.maximum(obs_mask.sum(axis=1), 1.0)
    raw = (
        0.30 * (1.0 - obs_vis.mean(axis=1))
        + 0.25 * (1.0 - obs_conf.mean(axis=1))
        + 0.25 * ent
        + 0.20 * (1.0 - np.clip(meas_conf, 0.0, 1.0))
    )
    lo, hi = thresholds[split]
    return np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default=str(SRC_ROOT))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--semantic-clusters", type=int, default=64)
    args = ap.parse_args()
    src_root = Path(args.src_root)
    if not src_root.is_absolute():
        src_root = ROOT / src_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    risk_thresholds = collect_risk_by_split(src_root, args.semantic_clusters)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    split_counts: dict[str, Counter[str]] = {}
    family_counts: Counter[int] = Counter()

    for split in ["train", "val", "test"]:
        for p in list_npz(src_root, split):
            try:
                z = np.load(p, allow_pickle=True)
                out_dir = out_root / split
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / p.name
                valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
                raw_changed = np.asarray(z["semantic_cluster_changed_mask"], dtype=bool) & valid
                target_conf = np.asarray(z["target_confidence"], dtype=np.float32)
                obs_risk_point = normalized_observed_risk(z, split, risk_thresholds, args.semantic_clusters)
                obs_risk = np.repeat(obs_risk_point[:, None], valid.shape[1], axis=1)
                # 修复：不再把任意 future CLIP KMeans cluster 当主语义目标；保留 coarse class state。
                semantic_id = np.asarray(z["semantic_id"], dtype=np.int64)
                class_cluster = np.mod(np.maximum(semantic_id, 0), args.semantic_clusters).astype(np.int64)
                target_cluster = np.repeat(class_cluster[:, None], valid.shape[1], axis=1)
                obs_cluster = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
                stable = valid & (~raw_changed | (target_conf >= 0.62))
                changed = valid & raw_changed & (target_conf >= 0.38)
                hard = valid & changed & ((target_conf < 0.72) | (obs_risk > 0.45))
                uncertainty = np.clip(0.72 * obs_risk + 0.28 * (1.0 - np.clip(target_conf, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
                family = np.full(valid.shape, COPY_INSTANCE, dtype=np.int64)
                family[stable & (obs_risk < 0.35)] = COPY_INSTANCE
                family[stable & (obs_risk >= 0.35)] = COPY_LAST
                family[stable & (target_conf > 0.80)] = COPY_MAX
                family[changed] = CHANGED_TRANSITION
                family[(uncertainty > 0.65) | (~valid)] = UNCERTAIN
                transition = np.where(valid, target_cluster, -1).astype(np.int64)
                same_instance = np.asarray(z["same_instance_as_observed_target"], dtype=bool) & valid
                identity_avail = np.asarray(z["identity_consistency_available_mask"], dtype=bool) & valid

                copy_keys = {k: z[k] for k in z.files if k not in {
                    "target_semantic_cluster_id",
                    "semantic_cluster_transition_id",
                    "semantic_cluster_changed_mask",
                    "semantic_stable_mask",
                    "semantic_changed_mask",
                    "semantic_hard_mask",
                    "evidence_anchor_family_target",
                    "semantic_uncertainty_target",
                    "same_instance_as_observed_target",
                    "identity_consistency_available_mask",
                    "future_teacher_embeddings_input_allowed",
                    "leakage_safe",
                }}
                np.savez_compressed(
                    out_path,
                    **copy_keys,
                    target_semantic_cluster_id=target_cluster.astype(np.int64),
                    semantic_cluster_transition_id=transition.astype(np.int64),
                    semantic_cluster_changed_mask=changed,
                    semantic_stable_mask=stable,
                    semantic_changed_mask=changed,
                    semantic_hard_mask=hard,
                    evidence_anchor_family_target=family.astype(np.int64),
                    semantic_uncertainty_target=uncertainty.astype(np.float32),
                    same_instance_as_observed_target=same_instance,
                    identity_consistency_available_mask=identity_avail,
                    fixed_target_version="v35_13_observed_predictable_video_state",
                    future_teacher_embeddings_input_allowed=False,
                    leakage_safe=True,
                )
                c = split_counts.setdefault(split, Counter())
                c["samples"] += 1
                c["tokens"] += int(valid.size)
                c["valid"] += int(valid.sum())
                c["stable"] += int(stable.sum())
                c["changed"] += int(changed.sum())
                c["hard"] += int(hard.sum())
                c["uncertain_high"] += int((uncertainty[valid] > 0.5).sum())
                family_counts.update([int(v) for v in family[valid].reshape(-1)])
                rows.append(
                    {
                        "source_path": str(p.relative_to(ROOT)),
                        "output_path": str(out_path.relative_to(ROOT)),
                        "split": split,
                        "valid_ratio": float(valid.mean()),
                        "changed_ratio": float(changed[valid].mean()) if valid.any() else 0.0,
                        "hard_ratio": float(hard[valid].mean()) if valid.any() else 0.0,
                        "uncertainty_high_ratio": float((uncertainty[valid] > 0.5).mean()) if valid.any() else 0.0,
                    }
                )
            except Exception as exc:
                blockers.append(f"{p}: {type(exc).__name__}: {exc}")

    split_report = {
        s: {
            "samples": int(c["samples"]),
            "valid_ratio": float(c["valid"] / max(c["tokens"], 1)),
            "stable_ratio": float(c["stable"] / max(c["valid"], 1)),
            "changed_ratio": float(c["changed"] / max(c["valid"], 1)),
            "semantic_hard_ratio": float(c["hard"] / max(c["valid"], 1)),
            "uncertainty_high_ratio": float(c["uncertain_high"] / max(c["valid"], 1)),
        }
        for s, c in sorted(split_counts.items())
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_video_semantic_state_targets_built": bool(rows),
        "src_root": str(src_root.relative_to(ROOT)),
        "out_root": str(out_root.relative_to(ROOT)),
        "sample_count": len(rows),
        "target_coverage_by_split": split_report,
        "risk_thresholds_by_split": {k: {"q10": v[0], "q90": v[1]} for k, v in risk_thresholds.items()},
        "evidence_anchor_family_distribution": {FAMILY_NAMES[k]: int(v) for k, v in sorted(family_counts.items()) if 0 <= k < len(FAMILY_NAMES)},
        "future_teacher_embeddings_input_allowed": False,
        "leakage_safe": True,
        "exact_blockers": blockers[:20],
        "recommended_next_step": "eval_fixed_video_semantic_state_target_predictability",
        "中文结论": "V35.13 已把 video semantic target 从 future CLIP KMeans 精确 cluster 修成可观测的 coarse state：changed/risk/abstain + stable copy family。下一步重跑 observed-only predictability。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.13 Fixed Video Semantic State Target Build\n\n"
        f"- fixed_video_semantic_state_targets_built: {report['fixed_video_semantic_state_targets_built']}\n"
        f"- sample_count: {len(rows)}\n"
        f"- future_teacher_embeddings_input_allowed: false\n"
        f"- leakage_safe: true\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"样本数": len(rows), "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
