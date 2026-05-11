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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_4_oracle_residual_probe_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, ResidualUtilityDataset, collate_v344, oracle_outputs


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_4_residual_utility"
REPORT = ROOT / "reports/stwm_ostf_v34_4_residual_utility_visualization_manifest_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_RESIDUAL_UTILITY_VISUALIZATION_20260511.md"


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[PointwiseUnitResidualWorldModelV343, argparse.Namespace]:
    tr = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = ROOT / tr.get("checkpoint_path", str(CKPT_DIR / "v34_4_oracle_residual_probe_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = PointwiseUnitResidualWorldModelV343(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs


def render(
    path: Path,
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    final_sem: torch.Tensor,
    final_id: torch.Tensor,
    category: str,
    idx: tuple[int, int, int],
    reason: str,
) -> dict[str, Any]:
    bi, mi, hi = idx
    obs = batch["obs_points"][bi].detach().cpu().numpy()
    fut = out["point_pred"][bi, :, hi].detach().cpu().numpy()
    target = batch["fut_teacher_embedding"][bi, :, hi].detach().cpu().numpy()
    point_sem = out["pointwise_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    resid_sem = out["unit_semantic_residual"][bi, :, hi].detach().cpu().numpy()
    final = final_sem[bi, :, hi].detach().cpu().numpy()
    point_sim = (_norm(point_sem) * _norm(target)).sum(axis=-1)
    resid_sim = (_norm(resid_sem) * _norm(target)).sum(axis=-1)
    final_sim = (_norm(final) * _norm(target)).sum(axis=-1)
    assign = out["point_to_unit_assignment"][bi].detach().cpu().numpy()
    units = assign.argmax(axis=-1)
    utility = batch["residual_semantic_utility_mask"][bi, :, hi].detach().cpu().numpy().astype(float)
    gate = batch["residual_gate_target"][bi, :, hi].detach().cpu().numpy().astype(float)
    ident = torch.sigmoid(final_id[bi, :, hi]).detach().cpu().numpy()
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), c=np.repeat(units, obs.shape[1]), s=8, cmap="tab20")
    ax[0].scatter(obs[mi, :, 0], obs[mi, :, 1], c="red", s=20)
    ax[0].set_title("observed trace / unit")
    ax[1].scatter(fut[:, 0], fut[:, 1], c=point_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[1].scatter(fut[mi, 0], fut[mi, 1], c="red", s=34)
    ax[1].set_title("pointwise prediction")
    ax[2].scatter(fut[:, 0], fut[:, 1], c=resid_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[2].scatter(fut[mi, 0], fut[mi, 1], c="red", s=34)
    ax[2].set_title("unit residual")
    ax[3].scatter(fut[:, 0], fut[:, 1], c=final_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[3].scatter(fut[mi, 0], fut[mi, 1], c="red", s=34)
    ax[3].set_title("oracle final")
    ax[4].scatter(fut[:, 0], fut[:, 1], c=gate + utility, s=18, cmap="magma", vmin=0, vmax=2)
    ax[4].scatter(fut[:, 0], fut[:, 1], s=np.clip(ident * 24, 3, 30), facecolors="none", edgecolors="black", linewidths=0.3)
    ax[4].scatter(fut[mi, 0], fut[mi, 1], c="cyan", s=34)
    ax[4].set_title("utility/gate + identity")
    for a in ax:
        a.invert_yaxis()
        a.set_aspect("equal", adjustable="box")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{category}: {reason}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"category": category, "path": str(path.relative_to(ROOT)), "case_selection_reason": reason}


def select(mask: np.ndarray, score: np.ndarray, high: bool) -> tuple[int, int, int] | None:
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        return None
    vals = score[tuple(idxs.T)]
    return tuple(map(int, idxs[int(np.nanargmax(vals) if high else np.nanargmin(vals))]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs = load_model(args, device)
    ds = ResidualUtilityDataset("test", ckargs)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v344)))
    bd = move_batch(batch, device)
    with torch.no_grad():
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="force_gate_zero")
        final_sem, final_id = oracle_outputs(out, bd)
    target = bd["fut_teacher_embedding"].detach().cpu().numpy()
    point_sim = (_norm(out["pointwise_semantic_belief"].detach().cpu().numpy()) * _norm(target)).sum(axis=-1)
    final_sim = (_norm(final_sem.detach().cpu().numpy()) * _norm(target)).sum(axis=-1)
    gain = final_sim - point_sim
    utility = bd["residual_semantic_utility_mask"].detach().cpu().numpy().astype(bool)
    stable = bd["stable_suppress_mask"].detach().cpu().numpy().astype(bool)
    hard = bd["semantic_hard_mask"].detach().cpu().numpy().astype(bool)
    id_utility = bd["residual_identity_utility_mask"].detach().cpu().numpy().astype(bool)
    specs = [
        ("pointwise_wrong_oracle_residual_correct", utility, gain, True),
        ("pointwise_wrong_residual_also_wrong", utility, gain, False),
        ("oracle_gate_should_open", utility, gain, True),
        ("oracle_gate_target_positive_failure", utility, gain, False),
        ("stable_pointwise_correct_gate_suppressed", stable, -np.abs(gain), True),
        ("stable_pointwise_correct_over_update", stable, np.abs(gain), True),
        ("semantic_hard_residual_success", hard, gain, True),
        ("semantic_hard_residual_failure", hard, gain, False),
        ("identity_hard_residual_case", id_utility, bd["pointwise_identity_error"].detach().cpu().numpy(), True),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for name, mask, score, high in specs:
        sel = select(mask, score, high)
        if sel is None:
            continue
        examples.append(render(OUT_DIR / f"{len(examples):02d}_{name}.png", bd, out, final_sem, final_id, name, sel, f"gain={float(gain[sel]):.4f}"))
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
    write_doc(DOC, "STWM OSTF V34.4 Residual Utility Visualization", payload, ["real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
