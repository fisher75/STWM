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

from stwm.modules.ostf_v34_1_identity_bound_semantic_trace_units import IdentityBoundSemanticTraceUnitsV341
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_1_identity_bound_semantic_trace_units_20260511 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, collate_v341, make_loader


OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v34_1_unit_loadbearing"
REPORT = ROOT / "reports/stwm_ostf_v34_1_unit_loadbearing_visualization_manifest_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_1_UNIT_LOADBEARING_VISUALIZATION_20260511.md"


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def pick_cases(pred: np.ndarray, target: np.ndarray, copy: np.ndarray, hard_sem: np.ndarray, assign: np.ndarray, same: np.ndarray) -> list[tuple[str, tuple[int, int, int], str]]:
    pred_cos = (_norm(pred) * _norm(target)).sum(axis=-1)
    copy_cos = (_norm(copy) * _norm(target)).sum(axis=-1)
    stable = copy_cos >= 0.80
    changed = copy_cos < 0.65
    unit_entropy = -(assign.clip(1e-8) * np.log(assign.clip(1e-8))).sum(axis=-1)
    cases: list[tuple[str, tuple[int, int, int], str]] = []
    for name, mask, score, high in [
        ("stable_preservation_success", stable, pred_cos - copy_cos, True),
        ("stable_preservation_failure", stable, pred_cos - copy_cos, False),
        ("changed_semantic_success", changed, pred_cos - copy_cos, True),
        ("changed_semantic_failure", changed, pred_cos - copy_cos, False),
        ("semantic_hard_success", hard_sem, pred_cos - copy_cos, True),
        ("semantic_hard_failure", hard_sem, pred_cos - copy_cos, False),
    ]:
        idx = np.argwhere(mask)
        if idx.size:
            vals = score[mask]
            sel = idx[int(np.argmax(vals) if high else np.argmin(vals))]
            cases.append((name, tuple(map(int, sel)), f"{name}: delta={float(score[tuple(sel)]):.3f}"))
    flat_entropy = unit_entropy.reshape(-1)
    hi = np.unravel_index(int(np.argmax(flat_entropy)), unit_entropy.shape)
    lo = np.unravel_index(int(np.argmin(flat_entropy)), unit_entropy.shape)
    cases.append(("unit_assignment_purity_success", (int(hi[0]), int(hi[1]), 0), f"high point assignment entropy={float(unit_entropy[hi]):.3f}"))
    cases.append(("unit_assignment_collapse_failure", (int(lo[0]), int(lo[1]), 0), f"low point assignment entropy={float(unit_entropy[lo]):.3f}"))
    conf = np.abs(same - 0.5)
    hi_conf = np.unravel_index(int(np.argmax(conf)), same.shape)
    lo_conf = np.unravel_index(int(np.argmin(conf)), same.shape)
    cases.append(("identity_confuser_success", tuple(map(int, hi_conf)), f"identity belief confident score={float(same[hi_conf]):.3f}"))
    cases.append(("identity_confuser_failure", tuple(map(int, lo_conf)), f"identity belief ambiguous score={float(same[lo_conf]):.3f}"))
    return cases[:10]


def render_case(path: Path, batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor], case: tuple[str, tuple[int, int, int], str]) -> dict[str, Any]:
    category, (bi, mi, hi), reason = case
    obs = batch["obs_points"][bi].detach().cpu().numpy()
    fut = out["point_pred"][bi, :, hi].detach().cpu().numpy()
    assign = out["point_to_unit_assignment"][bi].detach().cpu().numpy()
    units = assign.argmax(axis=-1)
    sem_pred = out["future_semantic_belief"][bi, :, hi].detach().cpu().numpy()
    sem_tgt = batch["fut_teacher_embedding"][bi, :, hi].detach().cpu().numpy()
    sim = (_norm(sem_pred) * _norm(sem_tgt)).sum(axis=-1)
    conf = out["unit_confidence"][bi].detach().cpu().numpy()
    point_conf = (assign * conf[None]).sum(axis=-1)
    identity = torch.sigmoid(out["future_identity_belief"][bi, :, hi]).detach().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(obs[:, :, 0].reshape(-1), obs[:, :, 1].reshape(-1), c=np.repeat(units, obs.shape[1]), s=8, cmap="tab20")
    ax[0].scatter(obs[mi, :, 0], obs[mi, :, 1], c="red", s=20)
    ax[0].set_title("observed trace + unit color")
    ax[1].scatter(fut[:, 0], fut[:, 1], c=sim, s=16, cmap="viridis", vmin=-1, vmax=1)
    ax[1].scatter(fut[mi, 0], fut[mi, 1], c="red", s=40)
    ax[1].set_title("V30 future trace + semantic sim")
    ax[2].scatter(fut[:, 0], fut[:, 1], c=identity, s=16, cmap="coolwarm", vmin=0, vmax=1)
    ax[2].scatter(fut[:, 0], fut[:, 1], s=np.clip(point_conf * 20, 2, 30), facecolors="none", edgecolors="black", linewidths=0.3)
    ax[2].set_title("identity belief + unit confidence")
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
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_1_identity_bound_semantic_trace_units_m128_h32_seed42_best.pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = IdentityBoundSemanticTraceUnitsV341(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=True)
    loader = make_loader("test", ckargs, shuffle=False)
    batch = next(iter(loader))
    bd = move_batch(batch, device)
    model.eval()
    with torch.no_grad():
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
    obs = bd["obs_semantic_measurements"].detach().cpu().numpy()
    obs_mask = bd["obs_semantic_measurement_mask"].detach().cpu().numpy().astype(bool)
    last = np.zeros((obs.shape[0], obs.shape[1], obs.shape[-1]), dtype=np.float32)
    for bi in range(obs.shape[0]):
        for mi in range(obs.shape[1]):
            idx = np.where(obs_mask[bi, mi])[0]
            if idx.size:
                last[bi, mi] = obs[bi, mi, idx[-1]]
    copy = np.broadcast_to(last[:, :, None, :], bd["fut_teacher_embedding"].shape)
    cases = pick_cases(
        out["future_semantic_belief"].detach().cpu().numpy(),
        bd["fut_teacher_embedding"].detach().cpu().numpy(),
        copy,
        bd["semantic_hard_train_mask"].detach().cpu().numpy().astype(bool),
        out["point_to_unit_assignment"].detach().cpu().numpy(),
        torch.sigmoid(out["future_identity_belief"]).detach().cpu().numpy(),
    )
    examples = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases):
        examples.append(render_case(OUT_DIR / f"{i:02d}_{case[0]}.png", bd, out, case))
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
    write_doc(DOC, "STWM OSTF V34.1 Unit Load-Bearing Visualization", payload, ["real_images_rendered", "case_mining_used", "placeholder_only", "png_count", "visualization_ready", "examples"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
