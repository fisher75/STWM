#!/usr/bin/env python3
"""V35.25 joint video semantic/identity closure case mining 可视化。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import setproctitle
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v35_18_vipseg_to_vspw_video_semantic_domain_shift_20260515 import (
    build_from_paths,
    choose_threshold,
    paths_for,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT

SEM_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_24_balanced_cross_dataset_changed_targets/M128_H32"
ID_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_16_video_identity_pairwise_retrieval_targets/M128_H32"
FIG_ROOT = ROOT / "figures/stwm_ostf_v35_25_joint_video_semantic_identity_closure"
MANIFEST = ROOT / "reports/stwm_ostf_v35_25_joint_video_semantic_identity_case_mining_manifest_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_25_JOINT_VIDEO_SEMANTIC_IDENTITY_CASE_MINING_20260516.md"


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def train_changed_probe() -> tuple[HistGradientBoostingClassifier, float]:
    seed = 345
    train = build_from_paths(paths_for(SEM_ROOT, "train", "VIPSEG"), 120000, seed + 1)
    val = build_from_paths(paths_for(SEM_ROOT, "val", "VSPW"), 60000, seed + 22)
    clf = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.055,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(train["x"][:, -5:], train["changed"])
    sv = clf.predict_proba(val["x"][:, -5:])[:, 1]
    threshold = choose_threshold(sv, val["changed"])
    return clf, float(threshold)


def semantic_sample_scores(path: Path, clf: HistGradientBoostingClassifier) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    future_vis = np.asarray(z["future_vis"], dtype=bool).astype(np.float32)
    future_conf = np.asarray(z["future_conf"], dtype=np.float32)
    m, h = changed.shape
    fut_disp = fut - obs[:, -1:, :]
    step = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None].repeat(m, axis=0)
    x = np.concatenate([fut_disp, future_vis[:, :, None], future_conf[:, :, None], step], axis=-1)
    score = clf.predict_proba(x.reshape(-1, 5))[:, 1].reshape(m, h)
    return {"score": score, "changed": changed, "hard": hard, "valid": valid, "obs": obs, "future": fut}


def pick_semantic_cases(clf: HistGradientBoostingClassifier, threshold: float) -> list[dict[str, Any]]:
    specs = [
        ("semantic_changed_success", lambda s, c, h, v: v & c & (s >= threshold), True),
        ("semantic_changed_failure", lambda s, c, h, v: v & c & (s < threshold), False),
        ("semantic_stable_success", lambda s, c, h, v: v & (~c) & (s < threshold), True),
        ("semantic_hard_success", lambda s, c, h, v: v & h & (s >= threshold), True),
        ("semantic_hard_failure", lambda s, c, h, v: v & h & (s < threshold), False),
    ]
    out: list[dict[str, Any]] = []
    used: set[Path] = set()
    for case_type, mask_fn, high_score in specs:
        best: dict[str, Any] | None = None
        for path in sorted((SEM_ROOT / "test").glob("VSPW__*.npz")):
            if path in used:
                continue
            d = semantic_sample_scores(path, clf)
            mask = mask_fn(d["score"], d["changed"], d["hard"], d["valid"])
            if not mask.any():
                continue
            idx = np.argwhere(mask)
            scores = d["score"][mask]
            pick = int(np.argmax(scores) if high_score else np.argmin(scores))
            point_i, horizon_i = [int(x) for x in idx[pick]]
            priority = float(abs(float(d["score"][point_i, horizon_i]) - threshold))
            row = {
                "case_type": case_type,
                "source_npz": path,
                "point_index": point_i,
                "horizon_index": horizon_i,
                "changed_score": float(d["score"][point_i, horizon_i]),
                "threshold": threshold,
                "is_changed": bool(d["changed"][point_i, horizon_i]),
                "is_hard": bool(d["hard"][point_i, horizon_i]),
                "priority": priority,
            }
            if best is None or row["priority"] > best["priority"]:
                best = row
        if best is not None:
            used.add(best["source_npz"])
            out.append(best)
    return out


def plot_semantic_case(row: dict[str, Any]) -> dict[str, Any]:
    z = np.load(row["source_npz"], allow_pickle=True)
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool)
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool)
    i = int(row["point_index"])
    h = int(row["horizon_index"])
    # 选附近运动最大的点和目标点，避免 toy 图只画一个点。
    motion = np.linalg.norm(fut[:, -1] - obs[:, -1], axis=-1)
    idx = np.argsort(-motion)[:384]
    if i not in idx:
        idx = np.concatenate([[i], idx[:383]])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    ax.scatter(obs[idx, -1, 0], obs[idx, -1, 1], s=4, c="lightgray", alpha=0.35, label="observed last")
    for j in idx[:220]:
        c = "tab:red" if changed[j, h] else ("tab:orange" if hard[j, h] else "tab:blue")
        ax.plot(fut[j, :, 0], fut[j, :, 1], color=c, linewidth=0.6, alpha=0.18)
    ax.plot(obs[i, :, 0], obs[i, :, 1], color="black", linewidth=2.0, label="target obs trace")
    ax.plot(fut[i, :, 0], fut[i, :, 1], color="crimson", linewidth=2.0, label="target future trace")
    ax.scatter(fut[i, h, 0], fut[i, h, 1], s=48, c="yellow", edgecolors="black", linewidths=0.8)
    ax.invert_yaxis()
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=7)
    ax.set_title(
        f"{row['case_type']} | score={row['changed_score']:.3f} thr={row['threshold']:.3f} "
        f"changed={row['is_changed']} hard={row['is_hard']}",
        fontsize=9,
    )
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / f"{row['case_type']}_{Path(row['source_npz']).stem}_p{i}_h{h}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {
        **{k: v for k, v in row.items() if k != "source_npz"},
        "source_npz": rel(row["source_npz"]),
        "png_path": rel(out),
        "case_selection_reason": "由 V35.24 future-trace-only changed probe 在 VSPW test 上真实挖掘",
    }


def identity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ID_ROOT.glob("*/*.npz")):
        z = np.load(path, allow_pickle=True)
        inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
        obs = np.asarray(z["obs_points"], dtype=np.float32)
        fut = np.asarray(z["future_points"], dtype=np.float32)
        conf = np.asarray(z["identity_confuser_pair_mask"], dtype=bool)
        occ = np.asarray(z["occlusion_reappear_point_mask"], dtype=bool)
        cross = np.asarray(z["trajectory_crossing_pair_mask"], dtype=bool)
        motion = np.linalg.norm(fut[:, -1] - obs[:, -1], axis=-1)
        rows.append(
            {
                "path": path,
                "sample_uid": str(np.asarray(z["sample_uid"]).item()),
                "split": path.parent.name,
                "dataset": str(np.asarray(z["dataset"]).item()),
                "instance_count": int(len(np.unique(inst[inst >= 0]))),
                "confuser_pair_count": int(conf.sum()),
                "occlusion_reappear_point_count": int(occ.sum()),
                "trajectory_crossing_pair_count": int(cross.sum()),
                "p90_motion": float(np.quantile(motion, 0.90)),
            }
        )
    return rows


def plot_identity_case(case_type: str, row: dict[str, Any]) -> dict[str, Any]:
    z = np.load(row["path"], allow_pickle=True)
    inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    conf = np.asarray(z["identity_confuser_pair_mask"], dtype=bool)
    occ = np.asarray(z["occlusion_reappear_point_mask"], dtype=bool)
    score = conf.sum(axis=1) + 500 * occ.astype(np.int64)
    idx = np.argsort(-score)[:384]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for i in idx:
        c = cmap(int(inst[i]) % 20)
        ax.plot(obs[i, :, 0], obs[i, :, 1], color=c, linewidth=0.6, alpha=0.25)
        ax.plot(fut[i, :, 0], fut[i, :, 1], color=c, linewidth=1.0, alpha=0.58)
        if occ[i]:
            ax.scatter(fut[i, -1, 0], fut[i, -1, 1], s=12, color="black", alpha=0.7)
    pairs = np.argwhere(conf[np.ix_(idx, idx)])
    for a, b in pairs[:80]:
        i, j = idx[a], idx[b]
        ax.plot([obs[i, -1, 0], obs[j, -1, 0]], [obs[i, -1, 1], obs[j, -1, 1]], color="crimson", alpha=0.08, linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(alpha=0.15)
    ax.set_title(f"{case_type} | confuser={row['confuser_pair_count']} occ={row['occlusion_reappear_point_count']} crossing={row['trajectory_crossing_pair_count']}", fontsize=9)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / f"{case_type}_{row['split']}_{row['sample_uid']}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {
        "case_type": case_type,
        "png_path": rel(out),
        "source_npz": rel(row["path"]),
        "case_selection_reason": "由 V35.16 identity pairwise target cache 按 confuser/occlusion/crossing 真实挖掘",
        **{k: v for k, v in row.items() if k != "path"},
    }


def main() -> int:
    clf, threshold = train_changed_probe()
    semantic_cases = [plot_semantic_case(row) for row in pick_semantic_cases(clf, threshold)]
    rows = identity_rows()
    identity_specs = [
        ("identity_confuser_rich", "confuser_pair_count"),
        ("identity_occlusion_reappear", "occlusion_reappear_point_count"),
        ("identity_trajectory_crossing", "trajectory_crossing_pair_count"),
    ]
    identity_cases: list[dict[str, Any]] = []
    used: set[Path] = set()
    for case_type, key in identity_specs:
        for row in sorted(rows, key=lambda r: r[key], reverse=True):
            if row["path"] not in used:
                used.add(row["path"])
                identity_cases.append(plot_identity_case(case_type, row))
                break
    cases = semantic_cases + identity_cases
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "joint_video_semantic_identity_case_mining_done": True,
        "visualization_ready": bool(cases),
        "real_images_rendered": bool(cases),
        "case_mining_used": True,
        "png_count": len(cases),
        "semantic_case_count": len(semantic_cases),
        "identity_case_count": len(identity_cases),
        "semantic_probe": "V35.24 future_trace_only HGB changed probe; VSPW val threshold",
        "identity_source": "V35.16 pairwise retrieval target cache",
        "cases": cases,
        "中文结论": "V35.25 生成 joint semantic/identity 的真实 case-mined 可视化，包含 changed/hard/stable 语义案例和 identity confuser/occlusion/crossing 案例。",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.25 Joint Video Semantic Identity Case Mining\n\n"
        f"- visualization_ready: {manifest['visualization_ready']}\n"
        f"- real_images_rendered: {manifest['real_images_rendered']}\n"
        f"- case_mining_used: {manifest['case_mining_used']}\n"
        f"- png_count: {manifest['png_count']}\n"
        f"- semantic_case_count: {manifest['semantic_case_count']}\n"
        f"- identity_case_count: {manifest['identity_case_count']}\n\n"
        "## 中文总结\n"
        + manifest["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"visualization_ready": manifest["visualization_ready"], "png_count": manifest["png_count"]}, ensure_ascii=False), flush=True)
    return 0 if cases else 2


if __name__ == "__main__":
    raise SystemExit(main())
