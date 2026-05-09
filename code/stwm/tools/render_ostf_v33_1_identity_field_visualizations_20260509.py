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

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.modules.ostf_v33_integrated_semantic_identity_world_model import IntegratedSemanticIdentityWorldModelV331
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v33_1_identity_field"
REPORT = ROOT / "reports/stwm_ostf_v33_1_identity_field_visualization_manifest_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_1_IDENTITY_FIELD_VISUALIZATION_20260509.md"


def draw_points(path: Path, obs: np.ndarray, pred: np.ndarray, prob: np.ndarray, vis_prob: np.ndarray, title: str) -> None:
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
        p = float(prob[i, -1])
        vp = float(vis_prob[i, -1])
        color = (int(255 * (1 - p)), int(200 * p), int(255 * (1 - vp)))
        d.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=1)
        r = 2
        d.ellipse((float(x1) - r, float(y1) - r, float(x1) + r, float(y1) + r), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_1_integrated/v33_1_integrated_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    examples = []
    blocker = None
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        train_args = ck.get("args", {})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IntegratedSemanticIdentityWorldModelV331(train_args["v30_checkpoint"], identity_dim=int(train_args.get("identity_dim", 64)), use_observed_instance_context=bool(train_args.get("use_observed_instance_context", False))).to(device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        ds = OSTFExternalGTDataset("test", horizon=int(train_args.get("horizon", 32)), m_points=int(train_args.get("m_points", 128)), max_items=args.count, enable_semantic_identity_sidecar=True, semantic_identity_sidecar_root=train_args.get("semantic_identity_sidecar_root"), require_semantic_identity_sidecar=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_external_gt)
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], semantic_id=bd["semantic_id"])
                obs = bd["obs_points"][0].detach().cpu().numpy()
                pred = out["point_pred"][0].detach().cpu().numpy()
                prob = out["same_instance_logits"][0].detach().cpu().sigmoid().numpy()
                vprob = out["visibility_logits"][0].detach().cpu().sigmoid().numpy()
                out_path = OUT_DIR / f"identity_field_{idx:03d}.png"
                draw_points(out_path, obs, pred, prob, vprob, f"{batch['uid'][0]} same-instance/visibility")
                examples.append({"uid": batch["uid"][0], "figure_path": str(out_path.relative_to(ROOT)), "M": int(obs.shape[0]), "H": int(pred.shape[1])})
    else:
        blocker = "integrated smoke checkpoint missing"
    payload = {
        "generated_at_utc": utc_now(),
        "visualization_ready": bool(examples),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)) if ckpt_path.exists() else str(ckpt_path),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "example_count": len(examples),
        "examples": examples,
        "m512_m1024_visualization_manifest_prepared": True,
        "exact_blocker": blocker,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.1 Identity Field Visualization", payload, ["visualization_ready", "checkpoint_path", "output_dir", "example_count", "m512_m1024_visualization_manifest_prepared", "exact_blocker"])
    print(REPORT.relative_to(ROOT))
    return 0 if examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
