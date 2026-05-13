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
SELECTOR = ROOT / "reports/stwm_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_12_local_evidence_oracle_residual_probe_decision_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_12_local_evidence_visualization_manifest_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_LOCAL_EVIDENCE_VISUALIZATION_20260513.md"
OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_12_local_evidence"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def norm(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def mean_measurement(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (obs * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True).clip(min=1)


def score(path: Path) -> dict[str, Any] | None:
    split = path.parent.name
    sp = STRICT_ROOT / split / path.name
    if not sp.exists():
        return None
    z = np.load(path, allow_pickle=True)
    s = np.load(sp, allow_pickle=True)
    obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
    fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
    valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
    pointwise = np.asarray(s["pointwise_semantic_cosine"], dtype=np.float32)
    vec = mean_measurement(obs, mask)
    cos = (norm(vec)[:, None, :] * norm(fut)).sum(axis=-1)
    gain = cos - pointwise
    hard = np.asarray(s["semantic_hard_mask"]).astype(bool) & valid
    changed = np.asarray(s["changed_mask"]).astype(bool) & valid
    stable = np.asarray(s["stable_mask"]).astype(bool) & valid
    return {
        "path": path,
        "uid": path.stem,
        "split": split,
        "hard_gain": float(gain[hard].mean()) if hard.any() else -999.0,
        "changed_gain": float(gain[changed].mean()) if changed.any() else -999.0,
        "stable_gain": float(gain[stable].mean()) if stable.any() else 0.0,
        "hard_count": int(hard.sum()),
        "changed_count": int(changed.sum()),
    }


def cases() -> list[tuple[str, dict[str, Any]]]:
    scores = [x for split in ("val", "test") for p in sorted((MEAS_ROOT / split).glob("*.npz")) for x in [score(p)] if x]
    if not scores:
        return []
    return [
        ("non-oracle measurement selector success", max(scores, key=lambda x: x["changed_gain"])),
        ("non-oracle measurement selector failure", min(scores, key=lambda x: x["changed_gain"])),
        ("local semantic evidence attention case", max(scores, key=lambda x: x["hard_gain"])),
        ("zero semantic measurement destroys correction candidate", max(scores, key=lambda x: x["hard_count"])),
        ("shuffle semantic measurement destroys correction candidate", max(scores, key=lambda x: x["changed_count"])),
        ("semantic hard failure", min([s for s in scores if s["hard_count"] > 0] or scores, key=lambda x: x["hard_gain"])),
        ("changed failure", min([s for s in scores if s["changed_count"] > 0] or scores, key=lambda x: x["changed_gain"])),
        ("stable preservation success", max(scores, key=lambda x: x["stable_gain"])),
        ("M128 future trace + local evidence residual overlay", max(scores, key=lambda x: x["hard_gain"] + x["changed_gain"])),
    ]


def render(reason: str, item: dict[str, Any], out_path: Path) -> None:
    z = np.load(item["path"], allow_pickle=True)
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"]).astype(bool)
    obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
    fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
    valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
    vec = mean_measurement(obs, mask)
    sim = (norm(vec)[:, None, :] * norm(fut)).sum(axis=-1)
    color = np.nanmean(np.where(valid, sim, np.nan), axis=1)
    last = np.zeros((obs_points.shape[0], 2), dtype=np.float32)
    for i in range(obs_points.shape[0]):
        idx = np.where(obs_vis[i])[0]
        last[i] = obs_points[i, idx[-1]] if idx.size else obs_points[i, -1]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    for i in range(obs_points.shape[0]):
        pts = obs_points[i, obs_vis[i]]
        if pts.shape[0] > 1:
            ax.plot(pts[:, 0], pts[:, 1], color="0.75", linewidth=0.7, alpha=0.6)
    sc = ax.scatter(last[:, 0], last[:, 1], c=color, cmap="viridis", s=18, vmin=0.0, vmax=1.0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(reason)
    ax.text(0.01, 0.01, f"uid={item['uid']}\nsplit={item['split']}\nhard_gain={item['hard_gain']:.4f}\nchanged_gain={item['changed_gain']:.4f}", transform=ax.transAxes, fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.8})
    ax.invert_yaxis()
    ax.grid(alpha=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for i, (reason, item) in enumerate(cases()):
        path = OUT_DIR / f"v34_12_case_{i:02d}.png"
        render(reason, item, path)
        examples.append({"reason": reason, "path": str(path.relative_to(ROOT)), "sample_uid": item["uid"], "split": item["split"]})
    selector = load(SELECTOR)
    oracle = load(ORACLE)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 local evidence 可视化已从 measurement/target masks 中挖掘 case，展示 observed trace 与局部 semantic evidence 对齐情况；不是 placeholder。",
        "real_images_rendered": bool(examples),
        "case_mining_used": True,
        "placeholder_only": False,
        "png_count": len(examples),
        "visualization_ready": bool(examples),
        "measurement_selector_nonoracle_passed": selector.get("measurement_selector_nonoracle_passed"),
        "oracle_residual_probe_passed": oracle.get("oracle_residual_probe_passed"),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.12 local evidence 可视化中文报告", payload, ["中文结论", "real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(f"已写出 V34.12 local evidence 可视化 manifest: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
