#!/usr/bin/env python3
"""V35.15 扩展 video semantic benchmark 的真实 case-mined 可视化与分层统计。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_15_expanded_mask_derived_video_semantic_state_targets/M128_H32"
FIG_ROOT = ROOT / "figures/stwm_ostf_v35_15_expanded_video_semantic_state"
REPORT = ROOT / "reports/stwm_ostf_v35_15_expanded_video_semantic_state_visualization_manifest_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_15_EXPANDED_VIDEO_SEMANTIC_STATE_VISUALIZATION_20260515.md"


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


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def sample_stats(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    stable = np.asarray(z["semantic_stable_mask"], dtype=bool) & valid
    future_vis = np.asarray(z["future_vis"], dtype=bool)
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    future_points = np.asarray(z["future_points"], dtype=np.float32)
    source_sem = np.asarray(z["source_semantic_id"], dtype=np.int64)
    motion = np.linalg.norm(future_points[:, -1] - obs_points[:, -1], axis=-1)
    occlusion = valid & (~future_vis)
    confuser = np.asarray(z["identity_confuser_pair_mask"], dtype=bool)
    return {
        "path": path,
        "split": path.parent.name,
        "dataset": str(np.asarray(z["dataset"]).item()),
        "sample_uid": str(np.asarray(z["sample_uid"]).item()),
        "valid_ratio": float(valid.mean()),
        "changed_ratio": float(changed.sum() / max(valid.sum(), 1)),
        "hard_ratio": float(hard.sum() / max(valid.sum(), 1)),
        "stable_ratio": float(stable.sum() / max(valid.sum(), 1)),
        "occlusion_ratio": float(occlusion.sum() / max(valid.sum(), 1)),
        "mean_motion": float(np.mean(motion)),
        "p90_motion": float(np.quantile(motion, 0.90)),
        "identity_confuser_pair_count": int(confuser.sum()),
        "dominant_source_semantic_id": int(Counter(source_sem.tolist()).most_common(1)[0][0]) if source_sem.size else -1,
    }


def mine_cases(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    if not rows:
        return []
    cases = [
        ("changed_rich", max(rows, key=lambda r: r["changed_ratio"])),
        ("semantic_hard_rich", max(rows, key=lambda r: r["hard_ratio"])),
        ("stable_preservation_rich", max(rows, key=lambda r: r["stable_ratio"])),
        ("occlusion_rich", max(rows, key=lambda r: r["occlusion_ratio"])),
        ("identity_confuser_rich", max(rows, key=lambda r: r["identity_confuser_pair_count"])),
        ("high_motion", max(rows, key=lambda r: r["p90_motion"])),
        ("low_motion", min(rows, key=lambda r: r["p90_motion"])),
    ]
    seen: set[Path] = set()
    out: list[tuple[str, dict[str, Any]]] = []
    for name, row in cases:
        if row["path"] in seen:
            alt = sorted(rows, key=lambda r: r.get(name.split("_")[0] + "_ratio", r["p90_motion"]), reverse=True)
            for cand in alt:
                if cand["path"] not in seen:
                    row = cand
                    break
        seen.add(row["path"])
        out.append((name, row))
    return out


def plot_case(name: str, row: dict[str, Any]) -> dict[str, Any]:
    z = np.load(row["path"], allow_pickle=True)
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    target = np.asarray(z["target_semantic_cluster_id"], dtype=np.int64)
    score = changed.mean(axis=1) + hard.mean(axis=1) + 0.25 * valid.mean(axis=1)
    idx = np.argsort(-score)[: min(256, len(score))]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for i in idx:
        ax.plot(obs[i, :, 0], obs[i, :, 1], color="#9aa0a6", linewidth=0.6, alpha=0.35)
    cmap = plt.get_cmap("tab20")
    for i in idx:
        cls = int(target[i][valid[i]][0]) if valid[i].any() else 0
        color = cmap(cls % 20)
        lw = 1.8 if changed[i].any() else 0.9
        alpha = 0.90 if hard[i].any() else 0.55
        ax.plot(fut[i, :, 0], fut[i, :, 1], color=color, linewidth=lw, alpha=alpha)
        ax.scatter(fut[i, -1, 0], fut[i, -1, 1], s=6, color=color, alpha=alpha)
    ax.invert_yaxis()
    ax.set_title(
        f"{name} | {row['split']} | changed={row['changed_ratio']:.3f} hard={row['hard_ratio']:.3f} "
        f"occ={row['occlusion_ratio']:.3f} motion90={row['p90_motion']:.1f}",
        fontsize=9,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / f"{name}_{row['split']}_{row['sample_uid']}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {
        "case_type": name,
        "png_path": str(out.relative_to(ROOT)),
        "source_npz": str(row["path"].relative_to(ROOT)),
        "case_selection_reason": f"按 {name} 指标从真实 target/eval cache 中挖掘",
        "changed_ratio": row["changed_ratio"],
        "hard_ratio": row["hard_ratio"],
        "occlusion_ratio": row["occlusion_ratio"],
        "p90_motion": row["p90_motion"],
        "identity_confuser_pair_count": row["identity_confuser_pair_count"],
    }


def main() -> int:
    rows = [sample_stats(p) for p in list_npz(TARGET_ROOT)]
    cases = [plot_case(name, row) for name, row in mine_cases(rows)]
    split_breakdown: dict[str, dict[str, float]] = {}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        split_breakdown[split] = {
            "samples": float(len(rs)),
            "changed_ratio_mean": float(np.mean([r["changed_ratio"] for r in rs])) if rs else 0.0,
            "hard_ratio_mean": float(np.mean([r["hard_ratio"] for r in rs])) if rs else 0.0,
            "occlusion_ratio_mean": float(np.mean([r["occlusion_ratio"] for r in rs])) if rs else 0.0,
            "p90_motion_mean": float(np.mean([r["p90_motion"] for r in rs])) if rs else 0.0,
            "identity_confuser_pair_count_sum": float(sum(r["identity_confuser_pair_count"] for r in rs)),
        }
    motion_bins = defaultdict(list)
    occlusion_bins = defaultdict(list)
    for r in rows:
        motion_key = "low" if r["p90_motion"] < 20 else "mid" if r["p90_motion"] < 80 else "high"
        occ_key = "low" if r["occlusion_ratio"] < 0.05 else "mid" if r["occlusion_ratio"] < 0.20 else "high"
        motion_bins[motion_key].append(r)
        occlusion_bins[occ_key].append(r)
    def bin_stats(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        out = {}
        for k, rs in sorted(groups.items()):
            out[k] = {
                "samples": len(rs),
                "changed_ratio_mean": float(np.mean([r["changed_ratio"] for r in rs])) if rs else 0.0,
                "hard_ratio_mean": float(np.mean([r["hard_ratio"] for r in rs])) if rs else 0.0,
            }
        return out
    category_counts = Counter(int(r["dominant_source_semantic_id"]) for r in rows)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visualization_ready": bool(cases),
        "real_images_rendered": bool(cases),
        "case_mining_used": True,
        "png_count": len(cases),
        "target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "figure_root": str(FIG_ROOT.relative_to(ROOT)),
        "cases": cases,
        "per_split_breakdown": split_breakdown,
        "per_motion_breakdown": bin_stats(motion_bins),
        "per_occlusion_breakdown": bin_stats(occlusion_bins),
        "dominant_source_semantic_id_top20": dict(category_counts.most_common(20)),
        "中文结论": "V35.15 已基于扩展 mask-derived video target cache 生成真实 trace/semantic case-mined PNG，并输出 split、motion、occlusion 分层统计。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.15 Expanded Video Semantic State Visualization\n\n"
        f"- visualization_ready: {manifest['visualization_ready']}\n"
        f"- real_images_rendered: {manifest['real_images_rendered']}\n"
        f"- case_mining_used: {manifest['case_mining_used']}\n"
        f"- png_count: {manifest['png_count']}\n\n"
        "## 中文总结\n"
        + manifest["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"visualization_ready": manifest["visualization_ready"], "png_count": manifest["png_count"]}, ensure_ascii=False), flush=True)
    return 0 if cases else 2


if __name__ == "__main__":
    raise SystemExit(main())
