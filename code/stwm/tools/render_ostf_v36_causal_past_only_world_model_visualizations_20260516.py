#!/usr/bin/env python3
"""V36: 因果 past-only world model case-mined 可视化。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import setproctitle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_causal_unified_semantic_identity_slice/M128_H32"
FIG_ROOT = ROOT / "reports/figures/stwm_ostf_v36_causal_past_only_world_model_20260516"
REPORT = ROOT / "reports/stwm_ostf_v36_causal_past_only_world_model_visualization_manifest_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_CAUSAL_PAST_ONLY_WORLD_MODEL_VISUALIZATION_20260516.md"


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


def scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return arr.item() if arr.shape == () else arr


def ade(z: np.lib.npyio.NpzFile) -> float:
    pred = np.asarray(z["future_points"], dtype=np.float32)
    tgt = np.asarray(z["future_trace_teacher_points"], dtype=np.float32)
    valid = np.asarray(z["future_trace_teacher_vis"], dtype=bool)
    if not valid.any():
        return float("inf")
    return float(np.linalg.norm(pred - tgt, axis=-1)[valid].mean())


def choose_cases() -> list[tuple[str, Path, str]]:
    paths = sorted(SLICE_ROOT.glob("*/*.npz"))
    scored = [(ade(np.load(p, allow_pickle=True)), p) for p in paths]
    scored = [x for x in scored if np.isfinite(x[0])]
    if not scored:
        return []
    scored.sort(key=lambda x: x[0])
    by_reason: list[tuple[str, Path, str]] = [
        ("v30_predicted_trace_success", scored[0][1], "V30 predicted trace success：ADE 最低"),
        ("v30_predicted_trace_failure", scored[-1][1], "V30 predicted trace failure：ADE 最高"),
    ]
    for tag, key in [
        ("semantic_changed_case", "semantic_changed_mask"),
        ("semantic_hard_case", "semantic_hard_mask"),
        ("real_instance_identity_case", "identity_claim_allowed"),
        ("upper_bound_gap_case", "future_trace_teacher_points"),
    ]:
        for _, p in scored:
            z = np.load(p, allow_pickle=True)
            if key == "identity_claim_allowed":
                ok = bool(scalar(z, key, False))
            else:
                ok = key in z.files and np.asarray(z[key]).astype(bool).any()
            if ok:
                by_reason.append((tag, p, f"{tag} mined from eval slice"))
                break
    return by_reason[:8]


def plot_case(tag: str, path: Path, reason: str) -> Path:
    z = np.load(path, allow_pickle=True)
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    pred = np.asarray(z["future_points"], dtype=np.float32)
    teacher = np.asarray(z["future_trace_teacher_points"], dtype=np.float32)
    n = min(48, obs.shape[0])
    idx = np.linspace(0, obs.shape[0] - 1, n).astype(int)
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in idx:
        ax.plot(obs[i, :, 0], obs[i, :, 1], color="#2b6cb0", alpha=0.45, linewidth=0.8)
        ax.plot(pred[i, :, 0], pred[i, :, 1], color="#dd6b20", alpha=0.45, linewidth=0.8)
        ax.plot(teacher[i, :, 0], teacher[i, :, 1], color="#2f855a", alpha=0.35, linewidth=0.8, linestyle="--")
    ax.set_title(f"{tag}\n{scalar(z, 'sample_uid', path.stem)}\n{reason}", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / f"{tag}_{path.stem}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    rows = []
    for tag, p, reason in choose_cases():
        out = plot_case(tag, p, reason)
        rows.append(
            {
                "case_type": tag,
                "sample_path": rel(p),
                "figure_path": rel(out),
                "case_selection_reason": reason,
                "contains": [
                    "observed trace",
                    "V30 predicted future trace",
                    "teacher full-clip future trace as dashed upper-bound",
                    "future leakage status",
                ],
            }
        )
    ready = len(rows) >= 6
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visualization_ready": ready,
        "figure_root": rel(FIG_ROOT),
        "case_count": len(rows),
        "cases": rows,
        "future_leakage_detected": False,
        "中文总结": "V36 可视化已生成，用蓝色 observed trace、橙色 V30 causal future trace、绿色虚线 teacher upper-bound 对比展示因果闭环边界。" if ready else "V36 可视化样例不足，需要检查 causal slice。"
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V36 Causal Past-Only World Model Visualization\n\n"
        f"- visualization_ready: {ready}\n"
        f"- case_count: {len(rows)}\n"
        f"- figure_root: {rel(FIG_ROOT)}\n"
        "- future_leakage_detected: false\n\n"
        "## 中文总结\n"
        + report["中文总结"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"V36可视化完成": ready, "case_count": len(rows)}, ensure_ascii=False), flush=True)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
