#!/usr/bin/env python3
"""构建 V35.2 identity-confuser / assignment hard-negative targets。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_1_fixed_semantic_state_targets/pointodyssey"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_2_identity_confuser_assignment_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v35_2_identity_confuser_assignment_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_2_IDENTITY_CONFUSER_ASSIGNMENT_TARGET_BUILD_20260515.md"


def list_npz(split: str) -> list[Path]:
    return sorted((V35_TARGET_ROOT / split).glob("*.npz"))


def last_valid_cluster(obs_cluster: np.ndarray) -> np.ndarray:
    out = np.full((obs_cluster.shape[0],), -1, dtype=np.int64)
    for i, row in enumerate(obs_cluster):
        idx = np.where(row >= 0)[0]
        if len(idx):
            out[i] = int(row[idx[-1]])
    return out


def last_visible_point(obs_points: np.ndarray, obs_vis: np.ndarray) -> np.ndarray:
    m, t, _ = obs_points.shape
    idx_grid = np.broadcast_to(np.arange(t, dtype=np.int64)[None, :], (m, t))
    idx = np.where(obs_vis, idx_grid, 0).max(axis=1)
    return obs_points[np.arange(m), idx]


def build_one(path: Path, split: str, out_path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    point_id = np.asarray(z["point_id"], dtype=np.int64)
    inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"], dtype=bool)
    obs_cluster = np.asarray(z["obs_semantic_cluster_id"], dtype=np.int64)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    same_future = np.asarray(z["same_instance_as_observed_target"], dtype=bool)
    same_avail = np.asarray(z["identity_consistency_available_mask"], dtype=bool) & valid
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    transition = np.asarray(z["semantic_cluster_transition_id"], dtype=np.int64)
    m, h = valid.shape

    last_cluster = last_valid_cluster(obs_cluster)
    last_pos = last_visible_point(obs_points, obs_vis)
    vel = obs_points[:, -1] - obs_points[:, 0]
    pair_available = (inst[:, None] >= 0) & (inst[None, :] >= 0)
    eye = np.eye(m, dtype=bool)
    pair_available &= ~eye
    same_pair = pair_available & (inst[:, None] == inst[None, :])
    diff_pair = pair_available & (inst[:, None] != inst[None, :])
    sem_same = (last_cluster[:, None] == last_cluster[None, :]) & (last_cluster[:, None] >= 0)
    dist = np.sqrt(((last_pos[:, None] - last_pos[None, :]) ** 2).sum(axis=-1))
    vdist = np.sqrt(((vel[:, None] - vel[None, :]) ** 2).sum(axis=-1))
    diff_dist = dist[diff_pair]
    diff_vdist = vdist[diff_pair]
    spatial_thr = float(np.quantile(diff_dist, 0.20)) if diff_dist.size else 0.0
    motion_thr = float(np.quantile(diff_vdist, 0.25)) if diff_vdist.size else 0.0
    spatial_close = dist <= spatial_thr
    motion_close = vdist <= motion_thr
    confuser_pair = diff_pair & ((sem_same & spatial_close) | (spatial_close & motion_close))
    if confuser_pair.sum() == 0 and diff_pair.any():
        # 保底：每个有实例的点选最近的一个不同实例点作为 hard negative。
        masked = np.where(diff_pair, dist, np.inf)
        nn = masked.argmin(axis=1)
        rows = np.arange(m)
        keep = np.isfinite(masked[rows, nn])
        confuser_pair[rows[keep], nn[keep]] = True
        confuser_pair[nn[keep], rows[keep]] = True

    point_has_confuser = confuser_pair.any(axis=1)
    identity_confuser_mask = same_avail & point_has_confuser[:, None] & (changed | hard | (~same_future))
    same_instance_hard_positive_mask = same_avail & point_has_confuser[:, None] & same_future
    same_instance_hard_negative_mask = same_avail & point_has_confuser[:, None] & (~same_future)
    transition_valid = valid & (transition >= 0)
    transition_pair_positive = np.zeros((m, m), dtype=bool)
    for tid in np.unique(transition[transition_valid]):
        pts = (transition == tid).any(axis=1)
        transition_pair_positive |= pts[:, None] & pts[None, :] & pair_available
    assignment_positive_pair_mask = same_pair | transition_pair_positive
    assignment_negative_pair_mask = confuser_pair
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sample_uid=np.asarray(path.stem),
        split=np.asarray(split),
        point_id=point_id,
        point_to_instance_id=inst,
        last_semantic_cluster_id=last_cluster.astype(np.int64),
        pair_available_mask=pair_available.astype(bool),
        same_instance_pair_mask=same_pair.astype(bool),
        identity_confuser_pair_mask=confuser_pair.astype(bool),
        assignment_positive_pair_mask=assignment_positive_pair_mask.astype(bool),
        assignment_negative_pair_mask=assignment_negative_pair_mask.astype(bool),
        identity_confuser_point_mask=identity_confuser_mask.astype(bool),
        same_instance_hard_positive_mask=same_instance_hard_positive_mask.astype(bool),
        same_instance_hard_negative_mask=same_instance_hard_negative_mask.astype(bool),
        semantic_transition_pair_positive_mask=transition_pair_positive.astype(bool),
        spatial_confuser_threshold=np.asarray(spatial_thr, dtype=np.float32),
        motion_confuser_threshold=np.asarray(motion_thr, dtype=np.float32),
        future_labels_supervision_only=np.asarray(True),
        future_teacher_embeddings_input_allowed=np.asarray(False),
        leakage_safe=np.asarray(True),
    )
    return {
        "samples": 1,
        "pair_available": int(pair_available.sum()),
        "same_pair": int(same_pair.sum()),
        "confuser_pair": int(confuser_pair.sum()),
        "assignment_positive_pair": int(assignment_positive_pair_mask.sum()),
        "assignment_negative_pair": int(assignment_negative_pair_mask.sum()),
        "identity_confuser_tokens": int(identity_confuser_mask.sum()),
        "hard_positive_tokens": int(same_instance_hard_positive_mask.sum()),
        "hard_negative_tokens": int(same_instance_hard_negative_mask.sum()),
        "valid_tokens": int(valid.sum()),
    }


def build_split(split: str) -> dict[str, Any]:
    totals = defaultdict(int)
    for path in list_npz(split):
        out_path = OUT_ROOT / split / path.name
        stats = build_one(path, split, out_path)
        for k, v in stats.items():
            totals[k] += int(v)
    return {
        "samples": totals["samples"],
        "pair_available": totals["pair_available"],
        "same_pair_ratio": totals["same_pair"] / max(totals["pair_available"], 1),
        "confuser_pair_ratio": totals["confuser_pair"] / max(totals["pair_available"], 1),
        "assignment_positive_pair_ratio": totals["assignment_positive_pair"] / max(totals["pair_available"], 1),
        "assignment_negative_pair_ratio": totals["assignment_negative_pair"] / max(totals["pair_available"], 1),
        "identity_confuser_token_ratio": totals["identity_confuser_tokens"] / max(totals["valid_tokens"], 1),
        "hard_positive_token_ratio": totals["hard_positive_tokens"] / max(totals["valid_tokens"], 1),
        "hard_negative_token_ratio": totals["hard_negative_tokens"] / max(totals["valid_tokens"], 1),
    }


def main() -> None:
    print("V35.2: 构建 identity-confuser / assignment targets。", flush=True)
    split_reports = {split: build_split(split) for split in ["train", "val", "test"]}
    blockers = []
    if any(v["confuser_pair_ratio"] <= 0.0 for v in split_reports.values()):
        blockers.append("identity confuser pair 为空")
    if any(v["hard_negative_token_ratio"] <= 0.0 for v in split_reports.values()):
        blockers.append("identity hard negative token 为空")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_root": str(OUT_ROOT.relative_to(ROOT)),
        "identity_confuser_assignment_targets_built": len(blockers) == 0,
        "split_reports": split_reports,
        "leakage_safe": True,
        "future_teacher_embeddings_input_allowed": False,
        "exact_blockers": blockers,
        "中文结论": "V35.2 target 为 identity consistency 构造 same-instance positives 与 semantic/trajectory confuser hard negatives，并为 assignment 提供正负 pair supervision。该 target 不使用 future teacher embedding 作为输入。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.2 Identity Confuser Assignment Target Build\n\n"
        f"- identity_confuser_assignment_targets_built: {report['identity_confuser_assignment_targets_built']}\n"
        f"- target_root: `{report['target_root']}`\n"
        f"- split_reports: {split_reports}\n"
        f"- exact_blockers: {blockers}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"built": report["identity_confuser_assignment_targets_built"], "target_root": report["target_root"], "blockers": blockers}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
