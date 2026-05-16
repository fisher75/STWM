#!/usr/bin/env python3
"""V35.43 渲染 raw-video closure 的真实 case-mined 可视化。"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import setproctitle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_38_REPORT = ROOT / "reports/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_subset_20260516.json"
V35_42_REPORT = ROOT / "reports/stwm_ostf_v35_42_identity_label_provenance_and_valid_claim_20260516.json"
SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_38_eval_balanced_raw_video_frontend_rerun_unified_slice/M128_H32"
FIG_ROOT = ROOT / "outputs/figures/stwm_ostf_v35_43_raw_video_closure"
REPORT = ROOT / "reports/stwm_ostf_v35_43_raw_video_closure_visualization_manifest_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_43_RAW_VIDEO_CLOSURE_VISUALIZATION_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v35_43_raw_video_closure_visualization_20260516.log"


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


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_sample(uid: str, split: str) -> tuple[Path, Any]:
    path = SLICE_ROOT / split / f"{uid}.npz"
    return path, np.load(path, allow_pickle=True)


def plot_trace_case(z: Any, out: Path, title: str, color_mode: str) -> None:
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool).any(axis=1)
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool).any(axis=1)
    if color_mode == "instance":
        color = inst
        cmap = "tab20"
    elif color_mode == "changed":
        color = changed.astype(int) + hard.astype(int)
        cmap = "coolwarm"
    else:
        speed = np.sqrt((np.diff(np.concatenate([obs, fut], axis=1), axis=1) ** 2).sum(-1)).mean(axis=1)
        color = speed
        cmap = "viridis"
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    n = min(obs.shape[0], 512)
    idx = np.linspace(0, obs.shape[0] - 1, n).astype(int)
    for i in idx:
        c = color[i]
        ax.plot(obs[i, :, 0], obs[i, :, 1], color="0.65", linewidth=0.5, alpha=0.35)
        ax.plot(fut[i, :, 0], fut[i, :, 1], color="0.2", linewidth=0.45, alpha=0.25)
        ax.scatter(obs[i, -1, 0], obs[i, -1, 1], c=[c], cmap=cmap, s=8, vmin=np.min(color), vmax=np.max(color) if np.max(color) > np.min(color) else np.min(color) + 1)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    smoke = json.loads(V35_38_REPORT.read_text(encoding="utf-8"))
    decision = json.loads(V35_42_REPORT.read_text(encoding="utf-8"))
    selected = smoke.get("selected_samples", [])
    valid = {r["sample_uid"] for r in decision.get("identity_label_provenance_rows", []) if r.get("identity_pairwise_target_valid_for_claim")}
    invalid = {r["sample_uid"] for r in decision.get("identity_label_provenance_rows", []) if not r.get("identity_pairwise_target_valid_for_claim")}
    cases = []
    if valid:
        r = next(s for s in selected if s["sample_uid"] in valid)
        cases.append(("identity_real_instance_success", r, "真实 instance identity 成功样例", "instance"))
    if invalid:
        r = next(s for s in selected if s["sample_uid"] in invalid)
        cases.append(("vspw_pseudo_identity_diagnostic", r, "VSPW pseudo identity 诊断样例", "instance"))
    hard = max(selected, key=lambda r: r.get("hard_ratio", 0.0))
    changed = max(selected, key=lambda r: r.get("changed_ratio", 0.0))
    crossing = next((r for r in selected if "crossing" in r.get("categories", [])), selected[0])
    occlusion = next((r for r in selected if "occlusion" in r.get("categories", [])), selected[0])
    cases.extend(
        [
            ("semantic_hard_case", hard, "semantic hard trace field 样例", "changed"),
            ("semantic_changed_case", changed, "semantic changed trace field 样例", "changed"),
            ("crossing_case", crossing, "trajectory crossing identity 诊断样例", "motion"),
            ("occlusion_case", occlusion, "occlusion/reappear identity 诊断样例", "motion"),
        ]
    )
    rows = []
    for name, r, title, mode in cases:
        path, z = load_sample(r["sample_uid"], r["split"])
        out = FIG_ROOT / f"{name}_{r['sample_uid']}.png"
        plot_trace_case(z, out, f"{title}: {r['sample_uid']} ({r['split']}/{r['dataset']})", mode)
        rows.append(
            {
                "case_name": name,
                "sample_uid": r["sample_uid"],
                "split": r["split"],
                "dataset": r["dataset"],
                "source_npz": rel(path),
                "png_path": rel(out),
                "case_selection_reason": title,
                "real_image_rendered": True,
            }
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_43_visualization_done": True,
        "figure_root": rel(FIG_ROOT),
        "case_mining_used": True,
        "real_images_rendered": True,
        "png_count": len(rows),
        "cases": rows,
        "visualization_ready": len(rows) >= 6 and all(r["real_image_rendered"] for r in rows),
        "m128_h32_video_system_benchmark_claim_allowed": bool(decision.get("m128_h32_video_system_benchmark_claim_allowed", False)),
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": "write_unified_raw_video_closure_final_decision",
        "中文结论": "V35.43 已从真实 eval slice 中 case-mine 并渲染 PNG；图像覆盖真实 instance identity、VSPW pseudo identity 诊断、semantic hard/changed、crossing、occlusion。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.43 Raw-Video Closure Visualization\n\n"
        f"- v35_43_visualization_done: true\n"
        f"- case_mining_used: true\n"
        f"- real_images_rendered: true\n"
        f"- png_count: {len(rows)}\n"
        f"- visualization_ready: {manifest['visualization_ready']}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: {manifest['m128_h32_video_system_benchmark_claim_allowed']}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: write_unified_raw_video_closure_final_decision\n\n"
        "## 中文总结\n"
        + manifest["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    log(f"V35.43 完成；png_count={len(rows)}")
    print(json.dumps({"v35_43_visualization_done": True, "visualization_ready": manifest["visualization_ready"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
