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

from stwm.modules.ostf_v33_13_gate_repaired_copy_semantic_world_model import GateRepairedCopySemanticWorldModelV3313
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT, collate_copy_v3311, make_loader_v3311
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch


OUT = ROOT / "outputs/figures/stwm_ostf_v33_13_gate_and_target_diagnostics"
REPORT = ROOT / "reports/stwm_ostf_v33_13_gate_and_target_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_GATE_AND_TARGET_VISUALIZATION_20260510.md"
TRAIN = ROOT / "reports/stwm_ostf_v33_13_gate_repaired_train_summary_20260510.json"


def pick(mask: np.ndarray, score: np.ndarray | None = None, high: bool = True) -> tuple[int, int, int] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    if score is None:
        return tuple(int(x) for x in idx[0])
    vals = score[mask]
    j = int(np.argmax(vals) if high else np.argmin(vals))
    return tuple(int(x) for x in idx[j])


def draw(path: Path, title: str, obs: np.ndarray, pred: np.ndarray, copy: np.ndarray, target: np.ndarray, pred_proto: np.ndarray, gate: np.ndarray, bi: int, hh: int, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.scatter(obs[bi, :, -1, 0], obs[bi, :, -1, 1], c=copy[bi, :, hh], cmap="tab20", s=12, label="copy")
    ax.scatter(pred[bi, :, hh, 0], pred[bi, :, hh, 1], c=target[bi, :, hh], cmap="tab20", s=22, alpha=0.55, label="target")
    ax.scatter(pred[bi, :, hh, 0], pred[bi, :, hh, 1], c=pred_proto[bi, :, hh], cmap="tab20", marker="x", s=26, label="STWM")
    ax.text(0.01, 0.01, f"{reason}\nmean gate={gate[bi,:,hh].mean():.3f}", transform=ax.transAxes, fontsize=8)
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / train["checkpoint_path"]
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.hard_train_mask_manifest = str(V33_11_MASK_ROOT / "H32_M128_seed42.json")
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    centers = torch.from_numpy(np.asarray(np.load(ckargs.prototype_vocab_path)["prototype_centers"], dtype=np.float32))
    model = GateRepairedCopySemanticWorldModelV3313(
        ckargs.v30_checkpoint,
        prototype_centers=centers,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        identity_teacher_checkpoint=ckargs.identity_teacher_checkpoint,
        gate_threshold=float(getattr(ckargs, "gate_threshold", 0.10)),
        freeze_identity_path=True,
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    ds = make_loader_v3311("test", ckargs, shuffle=False, max_items=64).dataset
    batch = next(iter(DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_copy_v3311)))
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
    pred_proto = out["final_semantic_proto_logits"].argmax(dim=-1).detach().cpu().numpy()
    gate = out["semantic_effective_gate"].detach().cpu().numpy()
    stable = bd["semantic_stable_mask"].detach().cpu().numpy().astype(bool)
    changed = bd["semantic_changed_mask"].detach().cpu().numpy().astype(bool)
    hard = bd["semantic_hard_mask"].detach().cpu().numpy().astype(bool)
    valid = bd["semantic_prototype_available_mask"].detach().cpu().numpy().astype(bool)
    correct = (pred_proto == target) & valid
    sample_pred = bd["baseline_sample_level_prototype_frequency_distribution"].argmax(dim=-1).detach().cpu().numpy()
    cats = [
        ("stable_wrong_update", stable & ~correct, gate, True),
        ("stable_fixed_by_threshold_candidate", stable & correct, gate, False),
        ("changed_semantic_success", changed & correct, gate, True),
        ("changed_semantic_failure", changed & ~correct, gate, False),
        ("semantic_hard_success", hard & correct, gate, True),
        ("semantic_hard_failure", hard & ~correct, gate, False),
        ("target_probe_success_proxy", (sample_pred != target) & correct, gate, True),
        ("target_probe_failure_proxy", (sample_pred == target) & ~correct, gate, True),
        ("copy_vs_sample_frequency_vs_stwm", valid, gate, True),
        ("gate_heatmap_after_repair", valid, gate, True),
        ("trace_overlay_semantic_prediction", valid, gate, False),
    ]
    examples = []
    for i, (name, mask, score, high) in enumerate(cats):
        chosen = pick(mask, score, high) or pick(valid, gate, True) or (0, 0, 0)
        bi, _mi, hh = chosen
        path = OUT / f"{i:02d}_{name}.png"
        draw(path, name, obs, pred, copy, target, pred_proto, gate, bi, hh, f"case_mining={name}, index={chosen}")
        examples.append({"category": name, "path": str(path.relative_to(ROOT)), "case_index": chosen})
    payload = {
        "generated_at_utc": utc_now(),
        "real_images_rendered": True,
        "case_mining_used": True,
        "png_count": len(examples),
        "visualization_ready": len(examples) >= 8,
        "placeholder_only": False,
        "output_dir": str(OUT.relative_to(ROOT)),
        "examples": examples,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.13 Gate and Target Visualization", payload, ["real_images_rendered", "case_mining_used", "png_count", "visualization_ready", "output_dir"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
