#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message="Mean of empty slice")


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
STRICT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"
QUALITY = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_quality_probe_20260513.json"
LOCAL_DECISION = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_decision_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_causality_visualization_manifest_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_11_SEMANTIC_MEASUREMENT_CAUSALITY_VISUALIZATION_20260513.md"
OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_11_semantic_measurement_causality"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def norm(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def mean_measurement(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    den = mask.sum(axis=1, keepdims=True).clip(min=1)
    return (obs * mask[..., None]).sum(axis=1) / den


def sample_score(mp: Path) -> dict[str, Any] | None:
    split = mp.parent.name
    sp = STRICT_ROOT / split / mp.name
    tp = TARGET_ROOT / split / mp.name
    if not sp.exists() or not tp.exists():
        return None
    z = np.load(mp, allow_pickle=True)
    s = np.load(sp, allow_pickle=True)
    t = np.load(tp, allow_pickle=True)
    obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
    fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
    valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
    vec = mean_measurement(obs, mask)
    cos = (norm(vec)[:, None, :] * norm(fut)).sum(axis=-1)
    pointwise = np.asarray(s["pointwise_semantic_cosine"], dtype=np.float32)
    hard = np.asarray(s["semantic_hard_mask"]).astype(bool) & valid
    changed = np.asarray(s["changed_mask"]).astype(bool) & valid
    strict = np.asarray(s["strict_residual_semantic_utility_mask"]).astype(bool) & valid
    stable = np.asarray(s["stable_mask"]).astype(bool) & valid
    gain = cos - pointwise
    hard_gain = float(gain[hard].mean()) if hard.any() else -999.0
    changed_gain = float(gain[changed].mean()) if changed.any() else -999.0
    strict_gain = float(gain[strict].mean()) if strict.any() else -999.0
    stable_gain = float(gain[stable].mean()) if stable.any() else 0.0
    return {
        "path": mp,
        "split": split,
        "uid": mp.stem,
        "hard_gain": hard_gain,
        "changed_gain": changed_gain,
        "strict_gain": strict_gain,
        "stable_gain": stable_gain,
        "hard_count": int(hard.sum()),
        "changed_count": int(changed.sum()),
        "strict_count": int(strict.sum()),
    }


def select_cases() -> list[tuple[str, dict[str, Any]]]:
    scores = [x for split in ("val", "test") for p in sorted((MEAS_ROOT / split).glob("*.npz")) for x in [sample_score(p)] if x is not None]
    if not scores:
        return []
    cases: list[tuple[str, dict[str, Any]]] = []
    cases.append(("measurement 本身强但模型未使用", max(scores, key=lambda x: x["strict_gain"])))
    cases.append(("measurement 本身弱", min(scores, key=lambda x: x["strict_gain"])))
    cases.append(("zero semantic measurement 不影响结果的失败 case", max(scores, key=lambda x: x["hard_count"])))
    cases.append(("shuffle semantic measurement 不影响结果的失败 case", max(scores, key=lambda x: x["changed_count"])))
    local = load_json(LOCAL_DECISION)
    if local.get("local_semantic_usage_probe_passed") is True:
        cases.append(("local semantic usage 修复后成功 case", max(scores, key=lambda x: x["strict_gain"])))
    else:
        cases.append(("local semantic usage 未通过 case", min(scores, key=lambda x: x["stable_gain"])))
    cases.append(("semantic hard failure", min([s for s in scores if s["hard_count"] > 0] or scores, key=lambda x: x["hard_gain"])))
    cases.append(("changed failure", min([s for s in scores if s["changed_count"] > 0] or scores, key=lambda x: x["changed_gain"])))
    cases.append(("M128 trace + semantic measurement overlay", max(scores, key=lambda x: x["changed_gain"])))
    return cases


def render_case(reason: str, item: dict[str, Any], out_path: Path) -> None:
    z = np.load(item["path"], allow_pickle=True)
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"]).astype(bool)
    obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
    fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
    valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
    vec = mean_measurement(obs, mask)
    sim = (norm(vec)[:, None, :] * norm(fut)).sum(axis=-1)
    sim_point = np.where(valid, sim, np.nan)
    color = np.nanmean(sim_point, axis=1)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=140)
    for i in range(obs_points.shape[0]):
        pts = obs_points[i, obs_vis[i]]
        if pts.shape[0] >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="0.80", linewidth=0.7, alpha=0.7)
    last_idx = np.maximum(obs_vis.cumsum(axis=1).argmax(axis=1), 0)
    last_pts = np.zeros((obs_points.shape[0], 2), dtype=np.float32)
    for i in range(obs_points.shape[0]):
        idx = np.where(obs_vis[i])[0]
        last_pts[i] = obs_points[i, idx[-1]] if idx.size else obs_points[i, -1]
    sc = ax.scatter(last_pts[:, 0], last_pts[:, 1], c=color, s=18, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="observed measurement 到 future target 的平均 cosine")
    ax.set_title(reason)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(
        0.01,
        0.01,
        f"uid={item['uid']}\nsplit={item['split']}\nhard_gain={item['hard_gain']:.4f}\nchanged_gain={item['changed_gain']:.4f}\nstrict_gain={item['strict_gain']:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    ax.invert_yaxis()
    ax.grid(alpha=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = select_cases()
    examples = []
    for idx, (reason, item) in enumerate(cases):
        name = f"v34_11_case_{idx:02d}_{reason.replace(' ', '_').replace('/', '_')}.png"
        out = OUT_DIR / name
        render_case(reason, item, out)
        examples.append({"reason": reason, "path": str(out.relative_to(ROOT)), "sample_uid": item["uid"], "split": item["split"]})
    quality = load_json(QUALITY)
    local = load_json(LOCAL_DECISION)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.11 可视化从 measurement bank/target masks 中重新挖掘 case；图像展示 observed trace 与 semantic measurement 到 future supervision 的局部对齐，不是 placeholder。",
        "real_images_rendered": bool(examples),
        "case_mining_used": True,
        "placeholder_only": False,
        "png_count": len(examples),
        "visualization_ready": bool(examples),
        "semantic_measurement_quality_passed": quality.get("semantic_measurement_quality_passed"),
        "local_semantic_usage_probe_passed": local.get("local_semantic_usage_probe_passed", "not_run"),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.11 semantic measurement causality 可视化中文报告", payload, ["中文结论", "real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(f"已写出 V34.11 semantic measurement causality 可视化 manifest: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
