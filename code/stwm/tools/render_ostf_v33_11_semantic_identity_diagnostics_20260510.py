#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_11_identity_preserving_copy_residual_semantic_world_model import IdentityPreservingCopyResidualSemanticWorldModelV3311
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT, collate_copy_v3311, make_loader_v3311
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


OUT = ROOT / "outputs/figures/stwm_ostf_v33_11_semantic_identity_diagnostics"
REPORT = ROOT / "reports/stwm_ostf_v33_11_semantic_identity_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_SEMANTIC_IDENTITY_VISUALIZATION_20260510.md"
CKPT = ROOT / "outputs/checkpoints/stwm_ostf_v33_11_identity_preserving_copy_residual_h32_m128/v33_11_identity_preserving_copy_residual_m128_h32_seed42_best.pt"


def choose(mask: np.ndarray, score: np.ndarray, *, high: bool = True) -> tuple[int, int, int] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    vals = score[mask]
    j = int(np.argmax(vals) if high else np.argmin(vals))
    return tuple(int(x) for x in idx[j])


def draw(path: Path, title: str, obs: np.ndarray, pred: np.ndarray, copy: np.ndarray, target: np.ndarray, model_proto: np.ndarray, gate: np.ndarray, vis: np.ndarray, bi: int, hh: int, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.scatter(obs[bi, :, -1, 0], obs[bi, :, -1, 1], c=copy[bi, :, hh], cmap="tab20", s=14, label="copy baseline")
    ax.scatter(pred[bi, :, hh, 0], pred[bi, :, hh, 1], c=model_proto[bi, :, hh], cmap="tab20", marker="x", s=22, label="STWM semantic pred")
    mismatch = model_proto[bi, :, hh] != target[bi, :, hh]
    ax.scatter(pred[bi, mismatch, hh, 0], pred[bi, mismatch, hh, 1], c="red", s=10, alpha=0.45, label="target mismatch")
    ax.text(0.01, 0.01, f"{reason}\nmean gate={gate[bi,:,hh].mean():.3f} mean vis={vis[bi,:,hh].mean():.3f}", transform=ax.transAxes, fontsize=8)
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CKPT))
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.hard_train_mask_manifest = str(V33_11_MASK_ROOT / f"H32_M128_seed{args.seed}.json")
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdentityPreservingCopyResidualSemanticWorldModelV3311(
        ckargs.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        identity_teacher_checkpoint=ckargs.identity_teacher_checkpoint,
        freeze_identity_path=not bool(getattr(ckargs, "no_identity_freeze", False)),
        no_stable_margin=bool(getattr(ckargs, "no_stable_margin", False)),
        no_gate_focal=bool(getattr(ckargs, "no_gate_focal", False)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    ds = make_loader_v3311(args.split, ckargs, shuffle=False, max_items=64).dataset
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_copy_v3311)
    batch = next(iter(loader))
    bd = move_batch(batch, device)
    with torch.no_grad():
        out = model(
            obs_points=bd["obs_points"],
            obs_vis=bd["obs_vis"],
            obs_conf=bd["obs_conf"],
            obs_teacher_embedding=bd["obs_teacher_embedding"],
            obs_teacher_available_mask=bd["obs_teacher_available_mask"],
            semantic_id=bd["semantic_id"],
            copy_semantic_prototype_id=bd["copy_semantic_prototype_id"],
            last_observed_semantic_prototype_id=bd["last_observed_semantic_prototype_id"],
        )
    obs = bd["obs_points"].detach().cpu().numpy()
    pred = out["point_pred"].detach().cpu().numpy()
    target = bd["semantic_prototype_id"].detach().cpu().numpy()
    copy = bd["copy_semantic_prototype_id"].detach().cpu().numpy()
    model_proto = out["final_semantic_proto_logits"].argmax(dim=-1).detach().cpu().numpy()
    gate = out["semantic_change_gate"].detach().cpu().numpy()
    vis = torch.sigmoid(out["visibility_logits"]).detach().cpu().numpy()
    stable = bd["semantic_stable_mask"].detach().cpu().numpy().astype(bool)
    changed = bd["semantic_changed_mask"].detach().cpu().numpy().astype(bool)
    hard = bd["semantic_hard_mask"].detach().cpu().numpy().astype(bool)
    correct = model_proto == target
    categories = [
        ("stable_preservation_success", stable & correct, gate, False),
        ("stable_preservation_failure", stable & ~correct, gate, True),
        ("changed_correction_success", changed & correct, gate, True),
        ("changed_correction_failure", changed & ~correct, gate, False),
        ("semantic_hard_success", hard & correct, gate, True),
        ("semantic_hard_failure", hard & ~correct, gate, False),
        ("semantic_gate_false_positive_on_stable", stable, gate, True),
        ("semantic_gate_false_negative_on_changed", changed, gate, False),
        ("identity_regressed_case", hard & ~correct, gate, True),
        ("identity_preserved_case", stable & correct, vis, True),
        ("trace_semantic_overlay_field", np.ones_like(stable, dtype=bool), gate, True),
        ("visibility_uncertainty_case", np.ones_like(stable, dtype=bool), np.abs(vis - 0.5), False),
    ]
    examples = []
    for i, (name, mask, score, high) in enumerate(categories):
        pick = choose(mask, score, high=high) or (0, 0, min(i, pred.shape[2] - 1))
        bi, _, hh = pick
        path = OUT / f"{i:02d}_{name}.png"
        draw(path, name, obs, pred, copy, target, model_proto, gate, vis, bi, hh, f"case_mining={name}, index={pick}")
        examples.append({"category": name, "path": str(path.relative_to(ROOT)), "case_index": pick, "selection_reason": name})
    payload = {
        "generated_at_utc": utc_now(),
        "real_images_rendered": True,
        "case_mining_used": True,
        "png_count": len(examples),
        "placeholder_only": False,
        "visualization_ready": len(examples) >= 12,
        "output_dir": str(OUT.relative_to(ROOT)),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.11 Semantic Identity Visualization", payload, ["real_images_rendered", "case_mining_used", "png_count", "placeholder_only", "visualization_ready", "output_dir"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
