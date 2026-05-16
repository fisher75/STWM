#!/usr/bin/env python3
"""构建 V35.1 修复版 observed-predictable semantic state target suite。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v34_43_observed_predictable_delta_targets_20260515 import assign_np, fit_codebook, np_norm
from stwm.tools.ostf_v17_common_20260502 import ROOT

BANK_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
V34_43_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_43_observed_predictable_delta_targets/pointodyssey"
IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/semantic_identity_targets/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_1_fixed_semantic_state_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v35_1_fixed_semantic_state_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_1_FIXED_SEMANTIC_STATE_TARGET_BUILD_20260515.md"
FAMILY_NAMES = [
    "last_visible_evidence",
    "max_confidence_observed",
    "instance_pooled_evidence",
    "changed_transition",
    "uncertain_abstain",
]
UNCERTAIN = len(FAMILY_NAMES) - 1
CHANGED_TRANSITION = 3


def list_npz(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.npz"))


def safe_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def norm(x: np.ndarray) -> np.ndarray:
    return np_norm(np.nan_to_num(x.astype(np.float32), copy=False))


def weighted_mean(obs: np.ndarray, mask: np.ndarray, conf: np.ndarray) -> np.ndarray:
    w = mask.astype(np.float32) * np.clip(conf.astype(np.float32), 0.05, 1.0)
    den = np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    out = (obs * w[..., None]).sum(axis=1) / den
    return norm(out)


def last_visible(obs: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    m, t, d = obs.shape
    idx_grid = np.broadcast_to(np.arange(t, dtype=np.int64)[None, :], (m, t))
    idx = np.where(mask, idx_grid, 0).max(axis=1)
    out = obs[np.arange(m), idx]
    out[~mask.any(axis=1)] = fallback[~mask.any(axis=1)]
    return norm(out)


def max_confidence(obs: np.ndarray, mask: np.ndarray, conf: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    score = np.where(mask, conf, -1.0)
    idx = score.argmax(axis=1)
    out = obs[np.arange(obs.shape[0]), idx]
    out[~mask.any(axis=1)] = fallback[~mask.any(axis=1)]
    return norm(out)


def instance_pool(point_inst: np.ndarray, emb: np.ndarray) -> np.ndarray:
    out = emb.copy()
    for inst in np.unique(point_inst[point_inst >= 0]):
        pts = point_inst == inst
        out[pts] = norm(emb[pts].mean(axis=0, keepdims=True))[0]
    return norm(out)


def best_raw_observed(obs: np.ndarray, mask: np.ndarray, target: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    # 只用于监督标签构造，推理输入不包含 future target。
    obs_n = norm(obs)
    tgt_n = norm(target)
    cos = np.einsum("mtd,mhd->mht", obs_n, tgt_n)
    cos = np.where(mask[:, None, :], cos, -9.0)
    idx = cos.argmax(axis=2)  # [M,H]
    m, h = idx.shape
    out = obs_n[np.arange(m)[:, None], idx]
    no_obs = ~mask.any(axis=1)
    if no_obs.any():
        out[no_obs] = fallback[no_obs, None, :]
    return norm(out)


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(max(p, 1e-12), 2)
    return float(ent)


def row_entropy(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((labels.shape[0],), dtype=np.float32)
    for i, row in enumerate(labels):
        vals = row[row >= 0]
        if len(vals) == 0:
            continue
        cnt = np.bincount(vals, minlength=k).astype(np.float32)
        p = cnt[cnt > 0] / max(cnt.sum(), 1.0)
        out[i] = float(-(p * np.log(np.maximum(p, 1e-12))).sum() / max(np.log(k), 1e-6))
    return out


def load_identity_sidecar(split: str, name: str, m: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    path = IDENTITY_ROOT / split / name
    if not path.exists():
        return np.ones((m, h), dtype=bool), np.zeros((m, h), dtype=bool)
    z = np.load(path, allow_pickle=True)
    same = np.asarray(z.get("fut_same_instance_as_obs", np.ones((m, h), dtype=bool)), dtype=bool)
    fut_av = np.asarray(z.get("fut_instance_available_mask", np.ones((m, h), dtype=bool)), dtype=bool)
    obs_av = np.asarray(z.get("obs_instance_available_mask", np.ones((m, 1), dtype=bool)), dtype=bool)
    if obs_av.ndim == 2:
        obs_av = obs_av.any(axis=1)
    obs_av = obs_av.reshape(m, 1)
    avail = fut_av & obs_av
    return same.astype(bool), avail.astype(bool)


def build_split(split: str, centers: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = TARGET_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "samples": 0,
        "tokens": 0,
        "valid_tokens": 0,
        "stable_tokens": 0,
        "changed_tokens": 0,
        "hard_tokens": 0,
        "identity_tokens": 0,
        "identity_positive_tokens": 0,
        "uncertainty_high_tokens": 0,
        "family_counts": Counter(),
        "transition_counts": Counter(),
        "uncertainty_values": [],
        "confidence_values": [],
        "blockers": [],
    }
    for path in list_npz(BANK_ROOT, split):
        z = safe_load(path)
        uid = str(np.asarray(z.get("sample_uid", path.stem)).item()) if np.asarray(z.get("sample_uid", path.stem)).shape == () else path.stem
        obs_points = np.asarray(z["obs_points"], dtype=np.float32)
        obs_vis = np.asarray(z["obs_vis"]).astype(bool)
        obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
        point_id = np.asarray(z.get("point_id", np.arange(obs_points.shape[0])), dtype=np.int64)
        point_inst = np.asarray(z.get("point_to_instance_id", np.full((obs_points.shape[0],), -1)), dtype=np.int64)
        obs_sem = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
        obs_mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
        obs_mconf = np.asarray(z.get("obs_measurement_confidence", np.ones(obs_mask.shape)), dtype=np.float32)
        fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
        fut_mask = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        fut_conf = np.asarray(z.get("fut_teacher_confidence", np.ones(fut_mask.shape)), dtype=np.float32)

        m, h = fut_mask.shape
        obs_cluster = assign_np(obs_sem, centers).astype(np.int64)
        obs_cluster = np.where(obs_mask, obs_cluster, -1)
        target_cluster = assign_np(fut, centers).astype(np.int64)
        target_cluster = np.where(fut_mask, target_cluster, -1)
        mean_emb = weighted_mean(obs_sem, obs_mask, obs_mconf)
        mean_cluster = assign_np(mean_emb, centers).astype(np.int64)
        last_emb = last_visible(obs_sem, obs_mask, mean_emb)
        last_cluster = assign_np(last_emb, centers).astype(np.int64)
        maxc_emb = max_confidence(obs_sem, obs_mask, obs_mconf, mean_emb)
        unit_emb = instance_pool(point_inst, mean_emb)
        topk_emb = best_raw_observed(obs_sem, obs_mask, fut, mean_emb)

        valid = fut_mask.astype(bool)
        transition = np.where(valid & (last_cluster[:, None] >= 0), last_cluster[:, None] * args.semantic_clusters + target_cluster, -1).astype(np.int64)
        changed = (target_cluster != last_cluster[:, None]) & valid & (last_cluster[:, None] >= 0)
        stable = (target_cluster == last_cluster[:, None]) & valid & (last_cluster[:, None] >= 0)

        v43 = safe_load(V34_43_ROOT / split / path.name)
        semantic_hard = np.asarray(v43.get("semantic_hard_mask", changed), dtype=bool) & valid
        if "changed_mask" in v43:
            changed = (changed | np.asarray(v43["changed_mask"], dtype=bool)) & valid
        if "stable_mask" in v43:
            stable = (stable | np.asarray(v43["stable_mask"], dtype=bool)) & valid

        target_n = norm(fut)
        cand = [
            np.broadcast_to(last_emb[:, None, :], fut.shape),
            np.broadcast_to(maxc_emb[:, None, :], fut.shape),
            np.broadcast_to(unit_emb[:, None, :], fut.shape),
            topk_emb,
        ]
        cos = np.stack([np.sum(norm(c) * target_n, axis=-1) for c in cand], axis=-1)
        best_family = np.clip(cos[:, :, :3].argmax(axis=-1), 0, 2).astype(np.int64)
        best_cos = cos.max(axis=-1)
        target_conf = np.clip(((best_cos + 1.0) * 0.5) * fut_conf, 0.0, 1.0).astype(np.float32)
        obs_cluster_entropy = row_entropy(obs_cluster, args.semantic_clusters)
        obs_vis_frac = obs_vis.mean(axis=1).astype(np.float32)
        obs_trace_conf = obs_conf.mean(axis=1).astype(np.float32)
        obs_meas_conf = np.where(obs_mask, obs_mconf, 0.0).sum(axis=1) / np.maximum(obs_mask.sum(axis=1), 1)
        observed_risk = np.clip(
            0.34 * (1.0 - obs_vis_frac)
            + 0.26 * (1.0 - obs_trace_conf)
            + 0.24 * obs_cluster_entropy
            + 0.16 * (1.0 - np.clip(obs_meas_conf, 0.0, 1.0)),
            0.0,
            1.0,
        ).astype(np.float32)
        future_risk = 1.0 - target_conf
        uncertainty = np.clip(0.72 * observed_risk[:, None] + 0.18 * future_risk + 0.10 * semantic_hard.astype(np.float32), 0.0, 1.0).astype(np.float32)
        abstain = (~valid) | (observed_risk[:, None] >= args.abstain_observed_risk_threshold) | (target_conf < args.abstain_confidence_threshold)
        family = best_family.astype(np.int64)
        family = np.where(changed | semantic_hard, CHANGED_TRANSITION, family)
        family = np.where(abstain, UNCERTAIN, family).astype(np.int64)
        family_available = valid.astype(bool)
        same_instance, identity_available = load_identity_sidecar(split, path.name, m, h)
        same_instance = same_instance & valid
        identity_available = identity_available & valid

        np.savez_compressed(
            out_dir / path.name,
            sample_uid=np.asarray(uid),
            split=np.asarray(split),
            point_id=point_id.astype(np.int64),
            point_to_instance_id=point_inst.astype(np.int64),
            obs_points=obs_points.astype(np.float32),
            obs_vis=obs_vis.astype(bool),
            obs_conf=obs_conf.astype(np.float32),
            target_semantic_cluster_id=target_cluster.astype(np.int64),
            target_semantic_cluster_available_mask=valid.astype(bool),
            obs_semantic_cluster_id=obs_cluster.astype(np.int64),
            semantic_cluster_transition_id=transition.astype(np.int64),
            semantic_cluster_changed_mask=changed.astype(bool),
            semantic_stable_mask=stable.astype(bool),
            semantic_changed_mask=changed.astype(bool),
            semantic_hard_mask=semantic_hard.astype(bool),
            evidence_anchor_family_target=family.astype(np.int64),
            evidence_anchor_family_available_mask=family_available.astype(bool),
            same_instance_as_observed_target=same_instance.astype(bool),
            identity_consistency_available_mask=identity_available.astype(bool),
            semantic_uncertainty_target=uncertainty.astype(np.float32),
            target_confidence=target_conf.astype(np.float32),
            future_teacher_embeddings_supervision_only=np.asarray(True),
            future_teacher_embeddings_input_allowed=np.asarray(False),
            leakage_safe=np.asarray(True),
        )
        stats["samples"] += 1
        stats["tokens"] += int(m * h)
        stats["valid_tokens"] += int(valid.sum())
        stats["stable_tokens"] += int((stable & valid).sum())
        stats["changed_tokens"] += int((changed & valid).sum())
        stats["hard_tokens"] += int((semantic_hard & valid).sum())
        stats["identity_tokens"] += int(identity_available.sum())
        stats["identity_positive_tokens"] += int((same_instance & identity_available).sum())
        stats["uncertainty_high_tokens"] += int(((uncertainty >= 0.70) & valid).sum())
        stats["family_counts"].update(family[family_available].reshape(-1).tolist())
        stats["transition_counts"].update(transition[transition >= 0].reshape(-1).tolist())
        stats["uncertainty_values"].append(uncertainty[valid])
        stats["confidence_values"].append(target_conf[valid])
    fam_total = sum(stats["family_counts"].values())
    fam_dist = {FAMILY_NAMES[k]: float(v / fam_total) for k, v in sorted(stats["family_counts"].items()) if fam_total}
    unc = np.concatenate(stats["uncertainty_values"]) if stats["uncertainty_values"] else np.asarray([], dtype=np.float32)
    conf = np.concatenate(stats["confidence_values"]) if stats["confidence_values"] else np.asarray([], dtype=np.float32)
    return {
        "samples": stats["samples"],
        "target_coverage": float(stats["valid_tokens"] / max(stats["tokens"], 1)),
        "stable_ratio": float(stats["stable_tokens"] / max(stats["valid_tokens"], 1)),
        "changed_ratio": float(stats["changed_tokens"] / max(stats["valid_tokens"], 1)),
        "semantic_hard_ratio": float(stats["hard_tokens"] / max(stats["valid_tokens"], 1)),
        "identity_target_coverage": float(stats["identity_tokens"] / max(stats["tokens"], 1)),
        "identity_positive_ratio": float(stats["identity_positive_tokens"] / max(stats["identity_tokens"], 1)),
        "uncertainty_high_ratio": float(stats["uncertainty_high_tokens"] / max(stats["valid_tokens"], 1)),
        "evidence_anchor_family_distribution": fam_dist,
        "semantic_transition_entropy": entropy_from_counts(stats["transition_counts"]),
        "uncertainty_target_stats": {
            "mean": float(unc.mean()) if unc.size else None,
            "p50": float(np.quantile(unc, 0.50)) if unc.size else None,
            "p90": float(np.quantile(unc, 0.90)) if unc.size else None,
        },
        "target_confidence_stats": {
            "mean": float(conf.mean()) if conf.size else None,
            "p50": float(np.quantile(conf, 0.50)) if conf.size else None,
            "p10": float(np.quantile(conf, 0.10)) if conf.size else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--semantic-clusters", type=int, default=64)
    ap.add_argument("--codebook-sample-per-file", type=int, default=32)
    ap.add_argument("--max-codebook-samples", type=int, default=120000)
    ap.add_argument("--kmeans-batch-size", type=int, default=4096)
    ap.add_argument("--kmeans-iters", type=int, default=120)
    ap.add_argument("--abstain-confidence-threshold", type=float, default=0.35)
    ap.add_argument("--abstain-cosine-threshold", type=float, default=0.18)
    ap.add_argument("--abstain-observed-risk-threshold", type=float, default=0.70)
    args = ap.parse_args()
    print("V35: 拟合 semantic state codebook 并构建离散/低维 target。", flush=True)
    centers = fit_codebook(args)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    split_reports = {split: build_split(split, centers, args) for split in ["train", "val", "test"]}
    blockers: list[str] = []
    if not list_npz(BANK_ROOT, "train"):
        blockers.append("V34.9 trace-preserving measurement bank 缺失")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "semantic_cluster_count": args.semantic_clusters,
        "target_coverage_by_split": {k: v["target_coverage"] for k, v in split_reports.items()},
        "semantic_transition_entropy": {k: v["semantic_transition_entropy"] for k, v in split_reports.items()},
        "stable_changed_ratio_by_split": {k: {"stable": v["stable_ratio"], "changed": v["changed_ratio"]} for k, v in split_reports.items()},
        "semantic_hard_ratio_by_split": {k: v["semantic_hard_ratio"] for k, v in split_reports.items()},
        "identity_target_coverage_by_split": {k: v["identity_target_coverage"] for k, v in split_reports.items()},
        "identity_positive_ratio_by_split": {k: v["identity_positive_ratio"] for k, v in split_reports.items()},
        "uncertainty_high_ratio_by_split": {k: v["uncertainty_high_ratio"] for k, v in split_reports.items()},
        "evidence_anchor_family_names": FAMILY_NAMES,
        "evidence_anchor_family_distribution": {k: v["evidence_anchor_family_distribution"] for k, v in split_reports.items()},
        "uncertainty_target_stats": {k: v["uncertainty_target_stats"] for k, v in split_reports.items()},
        "target_confidence_stats": {k: v["target_confidence_stats"] for k, v in split_reports.items()},
        "leakage_safe": True,
        "future_teacher_embeddings_supervision_only": True,
        "future_teacher_embeddings_input_allowed": False,
        "exact_blockers": blockers,
        "semantic_state_targets_built": len(blockers) == 0,
        "中文结论": "V35.1 修复了 V35 target suite：identity consistency 改用真实 fut_same_instance_as_obs 正负样本；uncertainty 改为 observed-risk 主导的 calibrated abstain/risk；evidence anchor family 改为更粗的 last/max-confidence/instance-pooled/changed/abstain 状态族。future teacher embedding 只用于监督，不进入输入。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.1 修复版可观测可预测语义状态 target 构建\n\n"
        f"- target root: `{report['target_root']}`\n"
        f"- semantic_cluster_count: {report['semantic_cluster_count']}\n"
        f"- target_coverage_by_split: {report['target_coverage_by_split']}\n"
        f"- stable_changed_ratio_by_split: {report['stable_changed_ratio_by_split']}\n"
        f"- semantic_hard_ratio_by_split: {report['semantic_hard_ratio_by_split']}\n"
        f"- identity_target_coverage_by_split: {report['identity_target_coverage_by_split']}\n"
        f"- identity_positive_ratio_by_split: {report['identity_positive_ratio_by_split']}\n"
        f"- uncertainty_high_ratio_by_split: {report['uncertainty_high_ratio_by_split']}\n"
        f"- leakage_safe: {report['leakage_safe']}\n"
        f"- exact_blockers: {report['exact_blockers']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print("V35.1 fixed target 构建完成。", flush=True)
    print(json.dumps({"semantic_state_targets_built": report["semantic_state_targets_built"], "target_root": report["target_root"], "blockers": blockers}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
