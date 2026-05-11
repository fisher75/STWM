#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_2_dual_source_semantic_trace_units import DualSourceSemanticTraceUnitsV342
from stwm.modules.ostf_v34_2_pointwise_no_unit_baseline import PointwiseNoUnitBaselineV342
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import collate_v341, make_loader
from stwm.tools.train_ostf_v34_2_dual_source_semantic_trace_units_20260511 import CKPT_DIR as DUAL_CKPT_DIR, SUMMARY as DUAL_TRAIN
from stwm.tools.train_ostf_v34_2_pointwise_no_unit_baseline_20260511 import CKPT_DIR as POINT_CKPT_DIR, SUMMARY as POINT_TRAIN


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_2_dual_source_units"
REPORT = ROOT / "reports/stwm_ostf_v34_2_dual_source_visualization_manifest_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_2_DUAL_SOURCE_VISUALIZATION_20260511.md"


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def load_dual(args: argparse.Namespace, device: torch.device) -> tuple[DualSourceSemanticTraceUnitsV342, argparse.Namespace]:
    tr = json.loads(DUAL_TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / tr.get("checkpoint_path", str(DUAL_CKPT_DIR / "v34_2_dual_source_semantic_trace_units_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = DualSourceSemanticTraceUnitsV342(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs


def load_point(args: argparse.Namespace, device: torch.device) -> PointwiseNoUnitBaselineV342:
    tr = json.loads(POINT_TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / tr.get("checkpoint_path", str(POINT_CKPT_DIR / "v34_2_pointwise_no_unit_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    model = PointwiseNoUnitBaselineV342(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model


def render(path: Path, batch: dict[str, torch.Tensor], dual: dict[str, torch.Tensor], point: dict[str, torch.Tensor], category: str, idx: tuple[int, int, int], reason: str) -> dict[str, Any]:
    bi, mi, hi = idx
    obs = batch["obs_points"][bi].detach().cpu().numpy()
    fut = dual["point_pred"][bi, :, hi].detach().cpu().numpy()
    assign = dual["point_to_unit_assignment"][bi].detach().cpu().numpy()
    units = assign.argmax(axis=-1)
    target = batch["fut_teacher_embedding"][bi, :, hi].detach().cpu().numpy()
    dual_sem = dual["future_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    point_sem = point["future_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    dual_sim = (_norm(dual_sem) * _norm(target)).sum(axis=-1)
    point_sim = (_norm(point_sem) * _norm(target)).sum(axis=-1)
    ident = torch.sigmoid(dual["future_identity_belief"][bi, :, hi]).detach().cpu().numpy()
    conf = (assign * dual["unit_confidence"][bi].detach().cpu().numpy()[None]).sum(axis=-1)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), c=np.repeat(units, obs.shape[1]), s=8, cmap="tab20")
    ax[0].scatter(obs[mi, :, 0], obs[mi, :, 1], c="red", s=22)
    ax[0].set_title("observed trace / unit")
    ax[1].scatter(fut[:, 0], fut[:, 1], c=dual_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[1].scatter(fut[mi, 0], fut[mi, 1], c="red", s=36)
    ax[1].set_title("V34.2 semantic sim")
    ax[2].scatter(fut[:, 0], fut[:, 1], c=point_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[2].scatter(fut[mi, 0], fut[mi, 1], c="red", s=36)
    ax[2].set_title("pointwise semantic sim")
    ax[3].scatter(fut[:, 0], fut[:, 1], c=ident, s=16, cmap="coolwarm", vmin=0, vmax=1)
    ax[3].scatter(fut[:, 0], fut[:, 1], s=np.clip(conf * 24, 2, 32), facecolors="none", edgecolors="black", linewidths=0.3)
    ax[3].set_title("identity / unit confidence")
    for a in ax:
        a.invert_yaxis()
        a.set_aspect("equal", adjustable="box")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"category": category, "path": str(path.relative_to(ROOT)), "case_selection_reason": reason}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dual_model, ckargs = load_dual(args, device)
    point_model = load_point(args, device)
    loader = make_loader("test", ckargs, shuffle=False)
    batch = next(iter(loader))
    bd = move_batch(batch, device)
    with torch.no_grad():
        dual = dual_model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
        point = point_model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
        drop_dyn = dual_model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="drop_z_dyn")
        drop_sem = dual_model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="drop_z_sem")
    target = bd["fut_teacher_embedding"].detach().cpu().numpy()
    dsim = (_norm(dual["future_semantic_belief"].detach().cpu().numpy()) * _norm(target)).sum(axis=-1)
    psim = (_norm(point["future_semantic_belief"].detach().cpu().numpy()) * _norm(target)).sum(axis=-1)
    delta = dsim - psim
    hard = bd["semantic_hard_train_mask"].detach().cpu().numpy().astype(bool)
    stable = dsim > 0.8
    changed = dsim < 0.65
    cases = [
        ("v34_2_beats_pointwise_semantic_hard", np.argwhere(hard), delta, True),
        ("v34_2_fails_vs_pointwise_semantic_hard", np.argwhere(hard), delta, False),
        ("stable_preservation_success", np.argwhere(stable), dsim, True),
        ("stable_preservation_failure", np.argwhere(stable), dsim, False),
        ("changed_semantic_success", np.argwhere(changed), delta, True),
        ("changed_semantic_failure", np.argwhere(changed), delta, False),
    ]
    examples = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, idxs, score, high in cases:
        if idxs.size == 0:
            continue
        vals = score[tuple(idxs.T)]
        sel = tuple(map(int, idxs[int(np.argmax(vals) if high else np.argmin(vals))]))
        examples.append(render(OUT_DIR / f"{len(examples):02d}_{name}.png", bd, dual, point, name, sel, f"{name}: v34.2-pointwise delta={float(delta[sel]):.3f}"))
    for mode, out in [("drop_z_dyn_intervention_example", drop_dyn), ("drop_z_sem_intervention_example", drop_sem)]:
        sim = (_norm(out["future_semantic_belief"].detach().cpu().numpy()) * _norm(dual["future_semantic_belief"].detach().cpu().numpy())).sum(axis=-1)
        idxs = np.argwhere(np.isfinite(sim))
        sel = tuple(map(int, idxs[int(np.argmin(sim[tuple(idxs.T)]))]))
        examples.append(render(OUT_DIR / f"{len(examples):02d}_{mode}.png", bd, dual, point, mode, sel, f"{mode}: semantic delta={float(1.0 - sim[sel]):.3f}"))
    payload = {"generated_at_utc": utc_now(), "real_images_rendered": bool(examples), "case_mining_used": True, "placeholder_only": False, "png_count": len(examples), "visualization_ready": bool(examples), "examples": examples}
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34.2 Dual-Source Visualization", payload, ["real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
