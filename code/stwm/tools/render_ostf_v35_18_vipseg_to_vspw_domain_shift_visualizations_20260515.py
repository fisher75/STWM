#!/usr/bin/env python3
"""渲染 V35.18 VIPSeg->VSPW domain-shift / target split 真实诊断图。"""
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

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_18_ontology_agnostic_video_semantic_state_targets/M128_H32"
OUT_DIR = ROOT / "reports/visualizations/v35_18_vipseg_to_vspw_domain_shift"
MANIFEST = ROOT / "reports/stwm_ostf_v35_18_vipseg_to_vspw_domain_shift_visualization_manifest_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_18_VIPSEG_TO_VSPW_DOMAIN_SHIFT_VISUALIZATION_20260515.md"


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


def summarize_sample(p: Path) -> dict[str, Any]:
    z = np.load(p, allow_pickle=True)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    risk = np.asarray(z["visibility_conditioned_semantic_risk"], dtype=np.float32) if "visibility_conditioned_semantic_risk" in z.files else np.asarray(z["semantic_uncertainty_target"], dtype=np.float32)
    return {
        "path": p,
        "split": str(np.asarray(z["split"]).item()),
        "dataset": str(np.asarray(z["dataset"]).item()),
        "valid_ratio": float(valid.mean()),
        "changed_ratio": float(changed[valid].mean()) if valid.any() else 0.0,
        "hard_ratio": float(hard[valid].mean()) if valid.any() else 0.0,
        "risk_mean": float(risk[valid].mean()) if valid.any() else 0.0,
    }


def pick_cases(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    def best(name: str, filt, key, reverse=True) -> tuple[str, dict[str, Any], str] | None:
        cand = [r for r in rows if filt(r)]
        if not cand:
            return None
        cand = sorted(cand, key=key, reverse=reverse)
        return (name, cand[0], f"{name}: split={cand[0]['split']} dataset={cand[0]['dataset']} changed={cand[0]['changed_ratio']:.3f} hard={cand[0]['hard_ratio']:.3f}")

    cases = [
        best("vspw_test_changed_sparse", lambda r: r["split"] == "test" and r["dataset"] == "VSPW", lambda r: r["changed_ratio"], reverse=False),
        best("vspw_test_changed_dense", lambda r: r["split"] == "test" and r["dataset"] == "VSPW", lambda r: r["changed_ratio"], reverse=True),
        best("vipseg_train_changed_dense", lambda r: r["split"] == "train" and r["dataset"] == "VIPSEG", lambda r: r["changed_ratio"], reverse=True),
        best("semantic_hard_dense", lambda r: True, lambda r: r["hard_ratio"], reverse=True),
        best("visibility_risk_high", lambda r: True, lambda r: r["risk_mean"], reverse=True),
        best("stable_dominant", lambda r: r["valid_ratio"] > 0.4, lambda r: r["changed_ratio"], reverse=False),
    ]
    return [c for c in cases if c is not None]


def render_case(name: str, row: dict[str, Any], reason: str, out_path: Path) -> None:
    z = np.load(row["path"], allow_pickle=True)
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    valid = np.asarray(z["target_semantic_cluster_available_mask"], dtype=bool)
    changed = np.asarray(z["semantic_changed_mask"], dtype=bool) & valid
    hard = np.asarray(z["semantic_hard_mask"], dtype=bool) & valid
    risk = np.asarray(z["visibility_conditioned_semantic_risk"], dtype=np.float32) if "visibility_conditioned_semantic_risk" in z.files else np.asarray(z["semantic_uncertainty_target"], dtype=np.float32)
    point_score = np.maximum(changed.mean(axis=1), hard.mean(axis=1)).astype(np.float32)
    idx = np.argsort(-point_score)[: min(64, len(point_score))]
    if len(idx) == 0:
        idx = np.arange(min(64, obs.shape[0]))
    color = point_score[idx]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)
    axes[0].set_title("observed trace")
    axes[1].set_title("future trace + changed")
    axes[2].set_title("risk / hard")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    for k, i in enumerate(idx):
        axes[0].plot(obs[i, :, 0], obs[i, :, 1], "-", color="tab:blue", alpha=0.35, linewidth=0.8)
        axes[1].plot(fut[i, :, 0], fut[i, :, 1], "-", color=plt.cm.magma(float(color[k])), alpha=0.55, linewidth=0.9)
        axes[2].scatter(fut[i, :, 0], fut[i, :, 1], c=risk[i], cmap="viridis", s=5, alpha=0.55, vmin=0, vmax=1)
    axes[1].scatter(fut[idx, -1, 0], fut[idx, -1, 1], c=color, cmap="magma", s=12, vmin=0, vmax=1)
    fig.suptitle(reason, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    rows = [summarize_sample(p) for p in sorted(TARGET_ROOT.glob("*/*.npz"))]
    cases = pick_cases(rows)
    rendered: list[dict[str, Any]] = []
    for name, row, reason in cases:
        out_path = OUT_DIR / f"{name}.png"
        render_case(name, row, reason, out_path)
        rendered.append({"case_type": name, "png_path": rel(out_path), "source_npz": rel(row["path"]), "case_selection_reason": reason})
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_images_rendered": bool(rendered),
        "case_mining_used": True,
        "png_count": len(rendered),
        "target_root": rel(TARGET_ROOT),
        "cases": rendered,
        "visualization_ready": len(rendered) >= 4,
        "中文结论": "V35.18 可视化从真实 target cache 中按 changed/hard/risk 稀疏与密集 case 自动挖掘，不使用固定索引。",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.18 VIPSeg→VSPW Domain Shift Visualization\n\n"
        f"- real_images_rendered: {manifest['real_images_rendered']}\n"
        f"- case_mining_used: true\n"
        f"- png_count: {len(rendered)}\n"
        f"- visualization_ready: {manifest['visualization_ready']}\n\n"
        "## 中文总结\n"
        + manifest["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"png_count": len(rendered), "visualization_ready": manifest["visualization_ready"]}, ensure_ascii=False), flush=True)
    return 0 if rendered else 2


if __name__ == "__main__":
    raise SystemExit(main())
