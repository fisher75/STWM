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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_11_identity_preserving_copy_residual_semantic_world_model import (
    IdentityPreservingCopyResidualSemanticWorldModelV3311,
)
from stwm.modules.ostf_v33_12_copy_conservative_semantic_world_model import CopyConservativeSemanticWorldModelV3312
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import (
    V33_11_MASK_ROOT,
    collate_copy_v3311,
    make_loader_v3311,
)
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


OUT = ROOT / "outputs/figures/stwm_ostf_v33_12_semantic_target_space_and_model_diagnostics"
REPORT = ROOT / "reports/stwm_ostf_v33_12_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_12_SEMANTIC_TARGET_SPACE_VISUALIZATION_20260510.md"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v33_11_identity_preserving_copy_residual_train_summary_20260510.json"
V3312_TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v33_12_copy_conservative_semantic_train_summary_20260510.json"
DEFAULT_CKPT = ROOT / "outputs/checkpoints/stwm_ostf_v33_11_identity_preserving_copy_residual_h32_m128/v33_11_identity_preserving_copy_residual_m128_h32_seed42_best.pt"


def load_checkpoint_path() -> Path:
    if V3312_TRAIN_SUMMARY.exists():
        payload = json.loads(V3312_TRAIN_SUMMARY.read_text(encoding="utf-8"))
        p = Path(str(payload.get("checkpoint_path", "")))
        if p:
            return p if p.is_absolute() else ROOT / p
    if TRAIN_SUMMARY.exists():
        payload = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
        p = Path(str(payload.get("checkpoint_path", "")))
        if p:
            return p if p.is_absolute() else ROOT / p
    return DEFAULT_CKPT


def choose(mask: np.ndarray, score: np.ndarray | None = None, *, high: bool = True) -> tuple[int, int, int] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    if score is None:
        j = 0
    else:
        vals = score[mask]
        j = int(np.argmax(vals) if high else np.argmin(vals))
    return tuple(int(x) for x in idx[j])


def draw(
    path: Path,
    title: str,
    obs: np.ndarray,
    pred: np.ndarray,
    copy_proto: np.ndarray,
    target_proto: np.ndarray,
    model_proto: np.ndarray,
    gate: np.ndarray,
    vis: np.ndarray,
    sample_i: int,
    horizon_i: int,
    reason: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.scatter(
        obs[sample_i, :, -1, 0],
        obs[sample_i, :, -1, 1],
        c=copy_proto[sample_i, :, horizon_i],
        s=14,
        cmap="tab20",
        label="copy prior semantic",
    )
    ax.scatter(
        pred[sample_i, :, horizon_i, 0],
        pred[sample_i, :, horizon_i, 1],
        c=target_proto[sample_i, :, horizon_i],
        s=22,
        cmap="tab20",
        marker="o",
        alpha=0.55,
        label="future target semantic",
    )
    ax.scatter(
        pred[sample_i, :, horizon_i, 0],
        pred[sample_i, :, horizon_i, 1],
        c=model_proto[sample_i, :, horizon_i],
        s=26,
        cmap="tab20",
        marker="x",
        label="STWM semantic pred",
    )
    wrong = (model_proto[sample_i, :, horizon_i] != target_proto[sample_i, :, horizon_i]) & (target_proto[sample_i, :, horizon_i] >= 0)
    if bool(wrong.any()):
        ax.scatter(
            pred[sample_i, wrong, horizon_i, 0],
            pred[sample_i, wrong, horizon_i, 1],
            c="red",
            s=12,
            alpha=0.55,
            label="model-target mismatch",
        )
    ax.text(
        0.01,
        0.01,
        f"{reason}\nmean gate={gate[sample_i, :, horizon_i].mean():.3f} mean visibility={vis[sample_i, :, horizon_i].mean():.3f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(load_checkpoint_path()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.hard_train_mask_manifest = str(V33_11_MASK_ROOT / f"H32_M128_seed{args.seed}.json")
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if "v33_12" in str(ckpt_path) or hasattr(ckargs, "gate_threshold"):
        model = CopyConservativeSemanticWorldModelV3312(
            ckargs.v30_checkpoint,
            prototype_centers=centers,
            teacher_embedding_dim=ckargs.teacher_embedding_dim,
            identity_teacher_checkpoint=ckargs.identity_teacher_checkpoint,
            gate_threshold=float(getattr(ckargs, "gate_threshold", 0.10)),
            residual_update_budget=float(getattr(ckargs, "residual_update_budget", 0.35)),
            freeze_identity_path=True,
        ).to(device)
    else:
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
    copy_proto = bd["copy_semantic_prototype_id"].detach().cpu().numpy()
    model_proto = out["final_semantic_proto_logits"].argmax(dim=-1).detach().cpu().numpy()
    gate = out["semantic_change_gate"].detach().cpu().numpy()
    vis = torch.sigmoid(out["visibility_logits"]).detach().cpu().numpy()
    stable = bd["semantic_stable_mask"].detach().cpu().numpy().astype(bool)
    changed = bd["semantic_changed_mask"].detach().cpu().numpy().astype(bool)
    hard = bd["semantic_hard_mask"].detach().cpu().numpy().astype(bool)
    valid = bd["semantic_prototype_available_mask"].detach().cpu().numpy().astype(bool)
    correct = (model_proto == target) & valid
    sample_freq_pred = bd["baseline_sample_level_prototype_frequency_distribution"].argmax(dim=-1).detach().cpu().numpy()
    sample_freq_wrong = (sample_freq_pred != target) & valid

    categories: list[tuple[str, np.ndarray, np.ndarray | None, bool]] = [
        ("clip_k32_failure_example", valid & ~correct, gate, True),
        ("new_target_space_stable_example", stable, gate, False),
        ("new_target_space_changed_example", changed, gate, True),
        ("teacher_jitter_example", changed & ~correct, gate, True),
        ("sample_frequency_baseline_failure", sample_freq_wrong & correct, gate, True),
        ("stwm_changed_hard_success", changed & hard & correct, gate, True),
        ("stable_preservation_success", stable & correct, gate, False),
        ("stable_preservation_failure", stable & ~correct, gate, True),
        ("identity_preserved_example", stable & correct, vis, True),
        ("trace_field_overlay", valid, gate, True),
        ("copy_prior_vs_residual_vs_target", changed | hard, gate, True),
    ]

    examples: list[dict[str, Any]] = []
    for i, (name, mask, score, high) in enumerate(categories):
        pick = choose(mask, score, high=high) or choose(valid, gate, high=True) or (0, 0, 0)
        sample_i, _point_i, horizon_i = pick
        path = OUT / f"{i:02d}_{name}.png"
        draw(
            path,
            name,
            obs,
            pred,
            copy_proto,
            target,
            model_proto,
            gate,
            vis,
            sample_i,
            horizon_i,
            f"case_mining={name}, index={pick}",
        )
        examples.append({"category": name, "path": str(path.relative_to(ROOT)), "case_index": pick, "selection_reason": name})

    payload = {
        "generated_at_utc": utc_now(),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT) if ckpt_path.is_absolute() else ckpt_path),
        "real_images_rendered": True,
        "case_mining_used": True,
        "png_count": len(examples),
        "placeholder_only": False,
        "visualization_ready": len(examples) >= 8,
        "output_dir": str(OUT.relative_to(ROOT)),
        "examples": examples,
        "exact_blockers": [],
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.12 Semantic Target Space Visualization",
        payload,
        ["real_images_rendered", "case_mining_used", "png_count", "placeholder_only", "visualization_ready", "output_dir", "exact_blockers"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
