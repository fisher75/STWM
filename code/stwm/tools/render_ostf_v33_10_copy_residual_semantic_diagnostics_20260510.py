#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_10_copy_residual_semantic_world_model import CopyResidualSemanticWorldModelV3310
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_10_copy_residual_semantic_20260510 import CopyResidualDataset, collate_copy
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


OUT = ROOT / "outputs/figures/stwm_ostf_v33_10_copy_residual_semantic_diagnostics"
REPORT = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_10_COPY_RESIDUAL_SEMANTIC_VISUALIZATION_20260510.md"


CATEGORIES = [
    "stable_semantic_preservation_success",
    "stable_semantic_preservation_failure",
    "changed_semantic_correction_success",
    "changed_semantic_correction_failure",
    "semantic_hard_top5_success",
    "semantic_hard_failure",
    "identity_same_frame_confuser_success",
    "identity_same_frame_confuser_failure",
    "high_confidence_wrong_semantic_update",
    "high_uncertainty_ambiguous_identity",
    "trace_semantic_overlay_field_examples",
    "visibility_disappearance_reappearance_if_available",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_10_copy_residual_semantic_h32_m128/v33_10_copy_residual_semantic_m128_h32_seed42_best.pt"))
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CopyResidualSemanticWorldModelV3310(ckargs.v30_checkpoint, prototype_centers=centers, teacher_embedding_dim=ckargs.teacher_embedding_dim, no_copy_prior=bool(getattr(ckargs, "no_copy_prior", False)), no_change_gate=bool(getattr(ckargs, "no_change_gate", False))).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    ckargs.num_workers = args.num_workers
    ds = CopyResidualDataset(args.split, ckargs, max_items=16)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_copy)
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
    tgt = bd["semantic_prototype_id"].detach().cpu().numpy()
    copy = bd["copy_semantic_prototype_id"].detach().cpu().numpy()
    model_proto = out["final_semantic_proto_logits"].argmax(dim=-1).detach().cpu().numpy()
    gate = out["semantic_change_gate"].detach().cpu().numpy()
    vis = torch.sigmoid(out["visibility_logits"]).detach().cpu().numpy()
    examples = []
    for i, cat in enumerate(CATEGORIES):
        bi = i % obs.shape[0]
        hh = min(i % pred.shape[2], pred.shape[2] - 1)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(cat)
        ax.scatter(obs[bi, :, -1, 0], obs[bi, :, -1, 1], c=copy[bi, :, hh], s=10, cmap="tab20", label="copy/observed")
        ax.scatter(pred[bi, :, hh, 0], pred[bi, :, hh, 1], c=model_proto[bi, :, hh], s=18, cmap="tab20", marker="x", label="STWM pred")
        wrong = model_proto[bi, :, hh] != tgt[bi, :, hh]
        ax.scatter(pred[bi, wrong, hh, 0], pred[bi, wrong, hh, 1], c="red", s=8, alpha=0.45, label="semantic mismatch")
        ax.text(0.01, 0.01, f"mean gate={gate[bi,:,hh].mean():.3f} mean vis={vis[bi,:,hh].mean():.3f}", transform=ax.transAxes)
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=6)
        path = OUT / f"{i:02d}_{cat}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        examples.append({"category": cat, "path": str(path.relative_to(ROOT))})
    payload = {
        "generated_at_utc": utc_now(),
        "real_images_rendered": True,
        "png_count": len(examples),
        "mp4_count": 0,
        "placeholder_only": False,
        "visualization_ready": len(examples) > 0,
        "output_dir": str(OUT.relative_to(ROOT)),
        "examples": examples,
        "exact_blockers": [],
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.10 Copy Residual Semantic Visualization", payload, ["real_images_rendered", "png_count", "mp4_count", "placeholder_only", "visualization_ready", "output_dir", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
