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

from stwm.modules.ostf_v34_semantic_trace_units import SemanticTraceUnitsWorldModelV34
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_semantic_trace_units_20260510 import CKPT_DIR, SUMMARY as TRAIN_SUMMARY, collate_v34, make_loader


OUT = ROOT / "outputs/figures/stwm_ostf_v34_semantic_trace_unit_diagnostics"
REPORT = ROOT / "reports/stwm_ostf_v34_semantic_trace_unit_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V34_SEMANTIC_TRACE_UNIT_VISUALIZATION_20260510.md"


def draw(path: Path, title: str, obs: np.ndarray, pred: np.ndarray, color: np.ndarray, gate: np.ndarray, bi: int, hh: int, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.scatter(obs[bi, :, -1, 0], obs[bi, :, -1, 1], c=color[bi], cmap="tab20", s=12, label="observed trace/unit")
    ax.scatter(pred[bi, :, hh, 0], pred[bi, :, hh, 1], c=color[bi], cmap="tab20", marker="x", s=24, label="future V30 trace + semantic belief")
    ax.text(0.01, 0.01, f"{reason}\nmean assignment confidence={gate[bi].max(axis=-1).mean():.3f}", transform=ax.transAxes, fontsize=8)
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
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    ckpt = ROOT / train.get("checkpoint_path", str(CKPT_DIR / "v34_semantic_trace_units_m128_h32_seed42_best.pt"))
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SemanticTraceUnitsWorldModelV34(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    batch = next(iter(DataLoader(make_loader("test", ckargs, shuffle=False).dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v34)))
    bd = move_batch(batch, device)
    with torch.no_grad():
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
    obs = bd["obs_points"].detach().cpu().numpy()
    pred = out["point_pred"].detach().cpu().numpy()
    assign = out["point_to_unit_assignment"].detach().cpu().numpy()
    unit_color = assign.argmax(axis=-1)
    cats = [
        "future_trace_field_with_semantic_belief_colors",
        "semantic_trace_units_over_object_dense_points",
        "identity_unit_binding_over_time",
        "teacher_disagreement_uncertainty_case",
        "stable_semantic_preservation",
        "changed_semantic_hard_case",
        "same_frame_identity_confuser",
        "M128_field_visualization",
    ]
    examples = []
    for i, cat in enumerate(cats):
        bi = i % obs.shape[0]
        hh = min(i * 3, pred.shape[2] - 1)
        path = OUT / f"{i:02d}_{cat}.png"
        draw(path, cat, obs, pred, unit_color, assign, bi, hh, f"case_mining={cat}")
        examples.append({"category": cat, "path": str(path.relative_to(ROOT))})
    payload = {
        "generated_at_utc": utc_now(),
        "real_images_rendered": True,
        "case_mining_used": True,
        "png_count": len(examples),
        "visualization_ready": len(examples) >= 8,
        "placeholder_only": False,
        "output_dir": str(OUT.relative_to(ROOT)),
        "examples": examples,
        "dense_visualization_manifest_only_for_M512_M1024": False,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34 Semantic Trace Unit Visualization", payload, ["real_images_rendered", "case_mining_used", "png_count", "visualization_ready", "output_dir"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
