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

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import make_loader
from stwm.tools.train_ostf_v34_3_pointwise_unit_residual_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_3_pointwise_unit_residual"
REPORT = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_visualization_manifest_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_3_POINTWISE_UNIT_RESIDUAL_VISUALIZATION_20260511.md"


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[PointwiseUnitResidualWorldModelV343, argparse.Namespace]:
    tr = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = ROOT / tr.get("checkpoint_path", str(CKPT_DIR / "v34_3_pointwise_unit_residual_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = PointwiseUnitResidualWorldModelV343(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs


def _copy_cos(batch: dict[str, torch.Tensor], target: np.ndarray) -> np.ndarray:
    obs = batch["obs_semantic_measurements"].detach().cpu().numpy()
    mask = batch["obs_semantic_measurement_mask"].detach().cpu().numpy().astype(bool)
    last = np.zeros((obs.shape[0], obs.shape[1], obs.shape[-1]), dtype=np.float32)
    for bi in range(obs.shape[0]):
        for mi in range(obs.shape[1]):
            idx = np.where(mask[bi, mi])[0]
            if idx.size:
                last[bi, mi] = obs[bi, mi, idx[-1]]
    copy = np.broadcast_to(last[:, :, None, :], target.shape)
    return (_norm(copy) * _norm(target)).sum(axis=-1)


def render_case(
    path: Path,
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    forced: dict[str, torch.Tensor] | None,
    category: str,
    idx: tuple[int, int, int],
    reason: str,
) -> dict[str, Any]:
    bi, mi, hi = idx
    obs = batch["obs_points"][bi].detach().cpu().numpy()
    fut = out["point_pred"][bi, :, hi].detach().cpu().numpy()
    assign = out["point_to_unit_assignment"][bi].detach().cpu().numpy()
    units = assign.argmax(axis=-1)
    target = batch["fut_teacher_embedding"][bi, :, hi].detach().cpu().numpy()
    point_sem = out["pointwise_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    residual_sem = out["unit_semantic_residual"][bi, :, hi].detach().cpu().numpy()
    final_sem = out["future_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    point_sim = (_norm(point_sem) * _norm(target)).sum(axis=-1)
    residual_sim = (_norm(residual_sem) * _norm(target)).sum(axis=-1)
    final_sim = (_norm(final_sem) * _norm(target)).sum(axis=-1)
    identity = torch.sigmoid(out["future_identity_belief"][bi, :, hi]).detach().cpu().numpy()
    gate = out["semantic_residual_gate"][bi, :, hi].detach().cpu().numpy()
    conf = (assign * out["unit_confidence"][bi].detach().cpu().numpy()[None]).sum(axis=-1)
    force_delta = None
    if forced is not None:
        fsem = forced["future_semantic_belief"][bi, :, hi].detach().cpu().numpy()
        force_delta = 1.0 - (_norm(fsem) * _norm(final_sem)).sum(axis=-1)
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), c=np.repeat(units, obs.shape[1]), s=8, cmap="tab20")
    ax[0].scatter(obs[mi, :, 0], obs[mi, :, 1], c="red", s=22)
    ax[0].set_title("observed trace / unit")
    ax[1].scatter(fut[:, 0], fut[:, 1], c=point_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[1].scatter(fut[mi, 0], fut[mi, 1], c="red", s=34)
    ax[1].set_title("pointwise semantic")
    ax[2].scatter(fut[:, 0], fut[:, 1], c=residual_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[2].scatter(fut[mi, 0], fut[mi, 1], c="red", s=34)
    ax[2].set_title("unit residual semantic")
    ax[3].scatter(fut[:, 0], fut[:, 1], c=final_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[3].scatter(fut[mi, 0], fut[mi, 1], c="red", s=34)
    ax[3].set_title("final semantic")
    color = force_delta if force_delta is not None else gate
    ax[4].scatter(fut[:, 0], fut[:, 1], c=color, s=18, cmap="magma")
    ax[4].scatter(fut[:, 0], fut[:, 1], s=np.clip(conf * 28, 3, 36), facecolors="none", edgecolors="black", linewidths=0.3)
    ax[4].scatter(fut[mi, 0], fut[mi, 1], c="cyan", s=34)
    ax[4].set_title("gate/delta + confidence")
    for a in ax:
        a.invert_yaxis()
        a.set_aspect("equal", adjustable="box")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{category}: {reason} | identity={float(identity[mi]):.3f} gate={float(gate[mi]):.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"category": category, "path": str(path.relative_to(ROOT)), "case_selection_reason": reason}


def select(idxs: np.ndarray, score: np.ndarray, high: bool) -> tuple[int, int, int] | None:
    if idxs.size == 0:
        return None
    vals = score[tuple(idxs.T)]
    pos = int(np.nanargmax(vals) if high else np.nanargmin(vals))
    return tuple(map(int, idxs[pos]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs = load_model(args, device)
    loader = make_loader("test", ckargs, shuffle=False)
    batch = next(iter(loader))
    bd = move_batch(batch, device)
    with torch.no_grad():
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
        force_zero = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="force_gate_zero")
        drop_dyn = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="drop_z_dyn")
        drop_sem = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="drop_z_sem")
    target = bd["fut_teacher_embedding"].detach().cpu().numpy()
    final_sim = (_norm(out["future_semantic_belief"].detach().cpu().numpy()) * _norm(target)).sum(axis=-1)
    point_sim = (_norm(out["pointwise_semantic_belief"].detach().cpu().numpy()) * _norm(target)).sum(axis=-1)
    copy_cos = _copy_cos(bd, target)
    sem_mask = bd["fut_teacher_available_mask"].detach().cpu().numpy().astype(bool)
    hard = sem_mask & bd["semantic_hard_train_mask"].detach().cpu().numpy().astype(bool)
    stable = sem_mask & (copy_cos >= 0.80)
    changed = sem_mask & (copy_cos < 0.65)
    delta_vs_point = final_sim - point_sim
    gate = out["semantic_residual_gate"].detach().cpu().numpy()
    id_target = bd["fut_same_instance_as_obs"].detach().cpu().numpy().astype(bool)
    id_hard = bd["identity_hard_train_mask"].detach().cpu().numpy().astype(bool)
    id_score = torch.sigmoid(out["future_identity_belief"]).detach().cpu().numpy()
    identity_success = id_hard & (np.abs(id_score - id_target.astype(np.float32)) < 0.25)
    identity_failure = id_hard & (np.abs(id_score - id_target.astype(np.float32)) > 0.45)
    case_specs = [
        ("v34_3_beats_pointwise_semantic_hard", np.argwhere(hard), delta_vs_point, True, None),
        ("v34_3_fails_vs_pointwise_semantic_hard", np.argwhere(hard), delta_vs_point, False, None),
        ("identity_confuser_residual_success", np.argwhere(identity_success), id_score, True, None),
        ("identity_confuser_residual_failure", np.argwhere(identity_failure), np.abs(id_score - id_target.astype(np.float32)), True, None),
        ("stable_gate_suppression_success", np.argwhere(stable), -gate, True, None),
        ("stable_gate_over_update_failure", np.argwhere(stable), gate, True, None),
        ("changed_semantic_residual_success", np.argwhere(changed), delta_vs_point, True, None),
        ("changed_semantic_residual_failure", np.argwhere(changed), delta_vs_point, False, None),
    ]
    examples = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, idxs, score, high, forced in case_specs:
        sel = select(idxs, score, high)
        if sel is None:
            continue
        examples.append(render_case(OUT_DIR / f"{len(examples):02d}_{name}.png", bd, out, forced, name, sel, f"score={float(score[sel]):.4f}"))
    for name, alt in [("force_gate_zero_vs_normal", force_zero), ("drop_z_dyn_intervention", drop_dyn), ("drop_z_sem_intervention", drop_sem)]:
        sim = (_norm(alt["future_semantic_belief"].detach().cpu().numpy()) * _norm(out["future_semantic_belief"].detach().cpu().numpy())).sum(axis=-1)
        idxs = np.argwhere(sem_mask)
        sel = select(idxs, 1.0 - sim, True)
        if sel is not None:
            examples.append(render_case(OUT_DIR / f"{len(examples):02d}_{name}.png", bd, out, alt, name, sel, f"intervention_delta={float(1.0 - sim[sel]):.4f}"))
    payload = {
        "generated_at_utc": utc_now(),
        "real_images_rendered": bool(examples),
        "case_mining_used": True,
        "placeholder_only": False,
        "png_count": len(examples),
        "visualization_ready": bool(examples),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34.3 Pointwise Unit Residual Visualization", payload, ["real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
