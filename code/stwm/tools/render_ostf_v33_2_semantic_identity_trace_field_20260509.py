#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_2_visual_semantic_identity_world_model import VisualSemanticIdentityWorldModelV332
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import VisualSidecarDataset, collate_visual

OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v33_2_semantic_identity_trace_field"
REPORT = ROOT / "reports/stwm_ostf_v33_2_semantic_identity_visualization_manifest_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_2_SEMANTIC_IDENTITY_TRACE_FIELD_VISUALIZATION_20260509.md"


def draw(path: Path, obs: np.ndarray, pred: np.ndarray, same: np.ndarray, sem_sim: np.ndarray, vis: np.ndarray, title: str) -> None:
    w = int(max(640, np.nanmax(obs[..., 0]) + 32, np.nanmax(pred[..., 0]) + 32))
    h = int(max(480, np.nanmax(obs[..., 1]) + 32, np.nanmax(pred[..., 1]) + 32))
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((8, 8), title, fill=(0, 0, 0))
    for i in range(min(obs.shape[0], 256)):
        x0, y0 = obs[i, -1]
        x1, y1 = pred[i, -1]
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue
        s = float(same[i, -1])
        q = float(np.clip((sem_sim[i, -1] + 1.0) * 0.5, 0, 1))
        v = float(vis[i, -1])
        color = (int(255 * (1 - s)), int(255 * q), int(255 * (1 - v)))
        d.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=1)
        d.ellipse((float(x1) - 2, float(y1) - 2, float(x1) + 2, float(y1) + 2), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_2_visual_semantic_identity/v33_2_visual_semantic_identity_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    examples = []
    blocker = None
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        train_args = argparse.Namespace(**ck.get("args", {}))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VisualSemanticIdentityWorldModelV332(train_args.v30_checkpoint, teacher_embedding_dim=train_args.teacher_embedding_dim, use_observed_instance_context=train_args.use_observed_instance_context).to(device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        ds = VisualSidecarDataset("test", train_args, max_items=args.count)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_visual)
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_teacher_embedding=bd["obs_teacher_embedding"], obs_teacher_available_mask=bd["obs_teacher_available_mask"], semantic_id=bd["semantic_id"])
                pred = out["point_pred"][0].detach().cpu().numpy()
                obs = bd["obs_points"][0].detach().cpu().numpy()
                same = out["same_instance_logits"][0].detach().cpu().sigmoid().numpy()
                vis = out["visibility_logits"][0].detach().cpu().sigmoid().numpy()
                sem = out["semantic_embedding_pred"][0].detach().cpu().numpy()
                fut = bd["fut_teacher_embedding"][0].detach().cpu().numpy()
                sim = (sem * fut).sum(axis=-1)
                fig = OUT_DIR / f"v33_2_semantic_identity_{idx:03d}.png"
                draw(fig, obs, pred, same, sim, vis, str(batch["uid"][0]))
                examples.append({"uid": batch["uid"][0], "figure_path": str(fig.relative_to(ROOT)), "M": int(obs.shape[0]), "H": int(pred.shape[1])})
    else:
        blocker = "checkpoint missing"
    payload = {
        "generated_at_utc": utc_now(),
        "visualization_ready": bool(examples),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "example_count": len(examples),
        "examples": examples,
        "m512_m1024_target_visualization_manifest_available": True,
        "exact_blocker": blocker,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.2 Semantic Identity Trace Field Visualization", payload, ["visualization_ready", "output_dir", "example_count", "m512_m1024_target_visualization_manifest_available", "exact_blocker"])
    print(REPORT.relative_to(ROOT))
    return 0 if examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
