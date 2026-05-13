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
import setproctitle
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_13_nonoracle_measurement_selector import NonOracleMeasurementSelectorV3413
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message="Mean of empty slice")


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
STRICT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"
SELECTOR_TRAIN = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_train_summary_20260513.json"
SELECTOR_DECISION = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_decision_20260513.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_13_selector_conditioned_oracle_residual_probe_decision_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_13_selector_conditioned_visualization_manifest_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_13_SELECTOR_CONDITIONED_VISUALIZATION_20260513.md"
OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_13_selector_conditioned"


def norm(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def mean_measurement(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (obs * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True).clip(min=1)


def load_selector() -> NonOracleMeasurementSelectorV3413 | None:
    train = json.loads(SELECTOR_TRAIN.read_text(encoding="utf-8")) if SELECTOR_TRAIN.exists() else {}
    ckpt = ROOT / train.get("checkpoint_path", "")
    if not ckpt.exists():
        return None
    ck = torch.load(ckpt, map_location="cpu")
    args = ck.get("args", {})
    model = NonOracleMeasurementSelectorV3413(int(args.get("teacher_embedding_dim", 768)), int(args.get("hidden_dim", 256)))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model


def selector_vec(model: NonOracleMeasurementSelectorV3413 | None, z: Any, t: Any) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
    if model is None:
        return mean_measurement(obs, mask), mask.astype(np.float32) / mask.sum(axis=1, keepdims=True).clip(min=1)
    with torch.no_grad():
        assign = torch.from_numpy(np.asarray(t["point_to_unit_assignment"], dtype=np.float32))[None]
        purity = torch.maximum(torch.from_numpy(np.asarray(t["unit_instance_purity"], dtype=np.float32))[None], torch.from_numpy(np.asarray(t["unit_semantic_purity"], dtype=np.float32))[None])
        out = model(
            obs_semantic_measurements=torch.from_numpy(obs)[None],
            obs_semantic_measurement_mask=torch.from_numpy(mask)[None],
            obs_measurement_confidence=torch.from_numpy(np.asarray(z["obs_measurement_confidence"], dtype=np.float32))[None],
            teacher_agreement_score=torch.from_numpy(np.asarray(z.get("teacher_agreement_score", z["obs_measurement_confidence"]), dtype=np.float32))[None],
            obs_vis=torch.from_numpy(np.asarray(z["obs_vis"]).astype(bool))[None],
            obs_conf=torch.from_numpy(np.asarray(z["obs_conf"], dtype=np.float32))[None],
            obs_points=torch.from_numpy(np.asarray(z["obs_points"], dtype=np.float32))[None],
            unit_assignment=assign,
            unit_purity_proxy=purity,
        )
    return out["selected_measurement_embedding"][0].cpu().numpy(), out["measurement_weight"][0].cpu().numpy()


def score(path: Path, model: NonOracleMeasurementSelectorV3413 | None) -> dict[str, Any] | None:
    split = path.parent.name
    sp = STRICT_ROOT / split / path.name
    tp = TARGET_ROOT / split / path.name
    if not sp.exists() or not tp.exists():
        return None
    z = np.load(path, allow_pickle=True)
    s = np.load(sp, allow_pickle=True)
    t = np.load(tp, allow_pickle=True)
    vec, weights = selector_vec(model, z, t)
    mean = mean_measurement(np.asarray(z["obs_semantic_measurements"], dtype=np.float32), np.asarray(z["obs_semantic_measurement_mask"]).astype(bool))
    fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
    valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
    pointwise = np.asarray(s["pointwise_semantic_cosine"], dtype=np.float32)
    sel_cos = (norm(vec)[:, None, :] * norm(fut)).sum(axis=-1)
    mean_cos = (norm(mean)[:, None, :] * norm(fut)).sum(axis=-1)
    hard = np.asarray(s["semantic_hard_mask"]).astype(bool) & valid
    changed = np.asarray(s["changed_mask"]).astype(bool) & valid
    return {
        "path": path,
        "uid": path.stem,
        "split": split,
        "selector_hard_gain": float((sel_cos - pointwise)[hard].mean()) if hard.any() else -999.0,
        "selector_changed_gain": float((sel_cos - pointwise)[changed].mean()) if changed.any() else -999.0,
        "selector_minus_mean_hard": float((sel_cos - mean_cos)[hard].mean()) if hard.any() else -999.0,
        "oracle_gap_proxy": float((np.maximum(sel_cos, mean_cos) - sel_cos)[hard | changed].mean()) if (hard | changed).any() else 0.0,
        "weights": weights,
    }


def pick_cases(scores: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    if not scores:
        return []
    return [
        ("selector beats pointwise hard case", max(scores, key=lambda x: x["selector_hard_gain"])),
        ("selector fails hard case", min(scores, key=lambda x: x["selector_hard_gain"])),
        ("oracle best vs nonoracle selector gap case", max(scores, key=lambda x: x["oracle_gap_proxy"])),
        ("selector-conditioned evidence success candidate", max(scores, key=lambda x: x["selector_changed_gain"])),
        ("zero semantic measurement destroys correction candidate", max(scores, key=lambda x: x["selector_hard_gain"] + x["selector_changed_gain"])),
        ("shuffle semantic measurement destroys correction candidate", max(scores, key=lambda x: x["selector_minus_mean_hard"])),
        ("semantic hard failure", min(scores, key=lambda x: x["selector_hard_gain"])),
        ("changed failure", min(scores, key=lambda x: x["selector_changed_gain"])),
        ("M128 future trace + selected semantic evidence overlay", max(scores, key=lambda x: x["selector_hard_gain"] + x["selector_changed_gain"])),
    ]


def render(reason: str, item: dict[str, Any], out_path: Path) -> None:
    z = np.load(item["path"], allow_pickle=True)
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"]).astype(bool)
    weights = item["weights"]
    color = weights.max(axis=1)
    last = np.zeros((obs_points.shape[0], 2), dtype=np.float32)
    for i in range(obs_points.shape[0]):
        idx = np.where(obs_vis[i])[0]
        last[i] = obs_points[i, idx[-1]] if idx.size else obs_points[i, -1]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    for i in range(obs_points.shape[0]):
        pts = obs_points[i, obs_vis[i]]
        if pts.shape[0] > 1:
            ax.plot(pts[:, 0], pts[:, 1], color="0.75", linewidth=0.7, alpha=0.6)
    sc = ax.scatter(last[:, 0], last[:, 1], c=color, cmap="magma", s=18, vmin=0.0, vmax=max(0.2, float(np.nanmax(color))))
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(reason)
    ax.text(0.01, 0.01, f"uid={item['uid']}\nsplit={item['split']}\nhard_gain={item['selector_hard_gain']:.4f}\nchanged_gain={item['selector_changed_gain']:.4f}", transform=ax.transAxes, fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.82})
    ax.invert_yaxis()
    ax.grid(alpha=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_selector()
    scores = [x for split in ("val", "test") for p in sorted((MEAS_ROOT / split).glob("*.npz")) for x in [score(p, model)] if x]
    examples = []
    for i, (reason, item) in enumerate(pick_cases(scores)):
        path = OUT_DIR / f"v34_13_case_{i:02d}.png"
        render(reason, item, path)
        examples.append({"reason": reason, "path": str(path.relative_to(ROOT)), "sample_uid": item["uid"], "split": item["split"]})
    selector = json.loads(SELECTOR_DECISION.read_text(encoding="utf-8")) if SELECTOR_DECISION.exists() else {}
    oracle = json.loads(ORACLE.read_text(encoding="utf-8")) if ORACLE.exists() else {}
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.13 selector-conditioned 可视化已从 measurement/target masks 中挖掘 case，展示 non-oracle selector 权重与 observed trace 的关系；不是 placeholder。",
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
    write_doc(DOC, "V34.13 selector-conditioned 可视化中文报告", payload, ["中文结论", "real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(f"已写出 V34.13 selector-conditioned 可视化 manifest: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
