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
import setproctitle
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_3_pointwise_unit_residual_world_model import PointwiseUnitResidualWorldModelV343
from stwm.tools.eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import _norm
from stwm.tools.eval_ostf_v34_6_residual_parameterization_sweep_20260511 import DECISION as SWEEP_DECISION
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_6_residual_parameterization_sweep_20260511 import StrictResidualUtilityDataset, collate_v345, compose_semantic


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_6_residual_parameterization"
REPORT = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_visualization_manifest_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_RESIDUAL_PARAMETERIZATION_VISUALIZATION_20260511.md"


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[PointwiseUnitResidualWorldModelV343 | None, argparse.Namespace | None, dict[str, Any]]:
    decision = json.loads(SWEEP_DECISION.read_text(encoding="utf-8")) if SWEEP_DECISION.exists() else {}
    ckpt_rel = decision.get("best_checkpoint_path")
    if not ckpt_rel:
        return None, None, decision
    ck = torch.load(ROOT / ckpt_rel, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    model = PointwiseUnitResidualWorldModelV343(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, decision


def candidate_score(gain: np.ndarray, mask: np.ndarray, high: bool) -> tuple[float, tuple[int, int, int] | None]:
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        return (-1e9 if high else 1e9), None
    vals = gain[tuple(idxs.T)]
    pos = int(np.nanargmax(vals) if high else np.nanargmin(vals))
    return float(vals[pos]), tuple(map(int, idxs[pos]))


def render(path: Path, bd: dict[str, torch.Tensor], out: dict[str, torch.Tensor], final: torch.Tensor, idx: tuple[int, int, int], title: str, reason: str) -> dict[str, Any]:
    bi, mi, hi = idx
    obs = bd["obs_points"][bi].detach().cpu().numpy()
    fut = out["point_pred"][bi, :, hi].detach().cpu().numpy()
    target = bd["fut_teacher_embedding"][bi, :, hi].detach().cpu().numpy()
    point = out["pointwise_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    resid = out["unit_semantic_residual"][bi, :, hi].detach().cpu().numpy()
    fin = final[bi, :, hi].detach().cpu().numpy()
    assign = out["point_to_unit_assignment"][bi].detach().cpu().numpy()
    units = assign.argmax(axis=-1)
    point_sim = (_norm(point) * _norm(target)).sum(axis=-1)
    resid_sim = (_norm(resid) * _norm(target)).sum(axis=-1)
    final_sim = (_norm(fin) * _norm(target)).sum(axis=-1)
    utility = bd["strict_residual_semantic_utility_mask"][bi, :, hi].detach().cpu().numpy().astype(float)
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), c=np.repeat(units, obs.shape[1]), s=8, cmap="tab20")
    ax[0].scatter(obs[mi, :, 0], obs[mi, :, 1], c="red", s=20)
    ax[0].set_title("observed/unit")
    ax[1].scatter(fut[:, 0], fut[:, 1], c=point_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[1].scatter(fut[mi, 0], fut[mi, 1], c="red", s=32)
    ax[1].set_title("pointwise")
    ax[2].scatter(fut[:, 0], fut[:, 1], c=resid_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[2].scatter(fut[mi, 0], fut[mi, 1], c="red", s=32)
    ax[2].set_title("unit residual")
    ax[3].scatter(fut[:, 0], fut[:, 1], c=final_sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[3].scatter(fut[mi, 0], fut[mi, 1], c="red", s=32)
    ax[3].set_title("final")
    ax[4].scatter(fut[:, 0], fut[:, 1], c=utility, s=18, cmap="magma", vmin=0, vmax=1)
    ax[4].scatter(fut[mi, 0], fut[mi, 1], c="cyan", s=32)
    ax[4].set_title("strict utility")
    for a in ax:
        a.invert_yaxis()
        a.set_aspect("equal", adjustable="box")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{title}: {reason}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"category": title, "path": str(path.relative_to(ROOT)), "case_selection_reason": reason}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, decision = load_model(args, device)
    if model is None or ckargs is None:
        payload = {"generated_at_utc": utc_now(), "real_images_rendered": False, "case_mining_used": False, "png_count": 0, "placeholder_only": False, "visualization_ready": False, "exact_blockers": ["missing_best_residual_checkpoint"]}
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V34.6 Residual Parameterization Visualization", payload, ["real_images_rendered", "case_mining_used", "png_count", "visualization_ready", "exact_blockers"])
        print(REPORT.relative_to(ROOT))
        return 0
    ds = StrictResidualUtilityDataset("test", ckargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v345)
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    specs = {
        "true_vector_or_best_residual_success": ("strict", True),
        "pointwise_wrong_best_residual_correct": ("strict", True),
        "pointwise_wrong_all_residuals_wrong": ("strict", False),
        "stable_suppress_success": ("stable", True),
        "semantic_hard_success": ("hard", True),
        "semantic_hard_failure": ("hard", False),
        "m128_future_trace_residual_overlay": ("strict", True),
    }
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"], intervention="force_gate_zero")
            final = compose_semantic(out, bd, ckargs, gate_mode="strict")
            target = bd["fut_teacher_embedding"].detach().cpu().numpy()
            point = out["pointwise_semantic_belief"].detach().cpu().numpy()
            fin = final.detach().cpu().numpy()
            gain = (_norm(fin) * _norm(target)).sum(axis=-1) - (_norm(point) * _norm(target)).sum(axis=-1)
            masks = {
                "strict": bd["strict_residual_semantic_utility_mask"].detach().cpu().numpy().astype(bool),
                "stable": bd["strict_stable_suppress_mask"].detach().cpu().numpy().astype(bool),
                "hard": (bd["semantic_hard_mask"] if "semantic_hard_mask" in bd else bd["semantic_hard_train_mask"]).detach().cpu().numpy().astype(bool),
            }
            for name, (mask_name, high) in specs.items():
                score, idx = candidate_score(gain, masks[mask_name], high)
                if idx is None:
                    continue
                current = best.get(name)
                better = current is None or (score > current[0] if high else score < current[0])
                if better:
                    best[name] = (score, {"bd": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in bd.items()}, "out": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in out.items() if torch.is_tensor(v)}, "final": final.detach().cpu(), "idx": idx})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for name, (score, pack) in best.items():
        examples.append(render(OUT_DIR / f"{len(examples):02d}_{name}.png", pack["bd"], pack["out"], pack["final"], pack["idx"], name, f"mined_gain={score:.4f}; best_variant={decision.get('best_residual_parameterization')}"))
    payload = {
        "generated_at_utc": utc_now(),
        "real_images_rendered": bool(examples),
        "case_mining_used": True,
        "placeholder_only": False,
        "png_count": len(examples),
        "visualization_ready": bool(examples),
        "best_residual_parameterization": decision.get("best_residual_parameterization"),
        "best_residual_init": decision.get("best_residual_init"),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34.6 Residual Parameterization Visualization", payload, ["real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "best_residual_parameterization", "best_residual_init", "examples"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
