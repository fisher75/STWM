#!/usr/bin/env python3
"""V35.16 video identity pairwise retrieval 的真实 case-mined 可视化。"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_16_video_identity_pairwise_retrieval_targets/M128_H32"
FIG_ROOT = ROOT / "figures/stwm_ostf_v35_16_video_identity_retrieval"
REPORT = ROOT / "reports/stwm_ostf_v35_16_video_identity_retrieval_visualization_manifest_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_16_VIDEO_IDENTITY_RETRIEVAL_VISUALIZATION_20260515.md"


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


def stats(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    conf = np.asarray(z["identity_confuser_pair_mask"], dtype=bool)
    occ = np.asarray(z["occlusion_reappear_point_mask"], dtype=bool)
    cross = np.asarray(z["trajectory_crossing_pair_mask"], dtype=bool)
    inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    motion = np.linalg.norm(fut[:, -1] - obs[:, -1], axis=-1)
    return {
        "path": path,
        "split": path.parent.name,
        "sample_uid": str(np.asarray(z["sample_uid"]).item()),
        "dataset": str(np.asarray(z["dataset"]).item()),
        "point_count": int(len(inst)),
        "instance_count": int(len(np.unique(inst[inst >= 0]))),
        "confuser_pair_count": int(conf.sum()),
        "occlusion_reappear_point_count": int(occ.sum()),
        "trajectory_crossing_pair_count": int(cross.sum()),
        "p90_motion": float(np.quantile(motion, 0.90)),
    }


def choose_cases(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    specs = [
        ("identity_confuser_rich", "confuser_pair_count", True),
        ("occlusion_reappear_rich", "occlusion_reappear_point_count", True),
        ("trajectory_crossing_rich", "trajectory_crossing_pair_count", True),
        ("high_motion_identity", "p90_motion", True),
        ("many_instance_identity", "instance_count", True),
        ("low_confuser_control", "confuser_pair_count", False),
    ]
    out: list[tuple[str, dict[str, Any]]] = []
    used: set[Path] = set()
    for name, key, desc in specs:
        sorted_rows = sorted(rows, key=lambda r: r[key], reverse=desc)
        for row in sorted_rows:
            if row["path"] not in used:
                out.append((name, row))
                used.add(row["path"])
                break
    return out


def plot_case(name: str, row: dict[str, Any]) -> dict[str, Any]:
    z = np.load(row["path"], allow_pickle=True)
    inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    conf = np.asarray(z["identity_confuser_pair_mask"], dtype=bool)
    occ = np.asarray(z["occlusion_reappear_point_mask"], dtype=bool)
    score = conf.sum(axis=1) + 1000 * occ.astype(np.int64)
    idx = np.argsort(-score)[: min(384, len(score))]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for i in idx:
        c = cmap(int(inst[i]) % 20)
        ax.plot(obs[i, :, 0], obs[i, :, 1], color=c, linewidth=0.6, alpha=0.25)
        ax.plot(fut[i, :, 0], fut[i, :, 1], color=c, linewidth=1.2, alpha=0.65)
        if occ[i]:
            ax.scatter(fut[i, -1, 0], fut[i, -1, 1], s=14, color="black", alpha=0.75)
    pairs = np.argwhere(conf[np.ix_(idx, idx)])
    for a, b in pairs[:80]:
        i, j = idx[a], idx[b]
        ax.plot([obs[i, -1, 0], obs[j, -1, 0]], [obs[i, -1, 1], obs[j, -1, 1]], color="crimson", alpha=0.08, linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(alpha=0.15)
    ax.set_title(
        f"{name} | {row['split']} | inst={row['instance_count']} confuser={row['confuser_pair_count']} "
        f"occ={row['occlusion_reappear_point_count']} crossing={row['trajectory_crossing_pair_count']}",
        fontsize=9,
    )
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / f"{name}_{row['split']}_{row['sample_uid']}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {
        "case_type": name,
        "png_path": str(out.relative_to(ROOT)),
        "source_npz": str(row["path"].relative_to(ROOT)),
        "case_selection_reason": "从 V35.16 pairwise target cache 按 hard identity 条件真实挖掘",
        **{k: v for k, v in row.items() if k != "path"},
    }


def main() -> int:
    rows = [stats(p) for p in sorted(TARGET_ROOT.glob("*/*.npz"))]
    cases = [plot_case(name, row) for name, row in choose_cases(rows)]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visualization_ready": bool(cases),
        "real_images_rendered": bool(cases),
        "case_mining_used": True,
        "png_count": len(cases),
        "cases": cases,
        "中文结论": "V35.16 已生成 video identity pairwise retrieval 的真实 case-mined 可视化，覆盖 confuser、occlusion/reappear、trajectory crossing 与多实例场景。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.16 Video Identity Retrieval Visualization\n\n"
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
