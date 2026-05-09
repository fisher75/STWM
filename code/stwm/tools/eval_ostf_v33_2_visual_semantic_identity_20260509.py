#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_v33_2_visual_semantic_identity_world_model import VisualSemanticIdentityWorldModelV332
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import VisualSidecarDataset, collate_visual, evaluate_model

REPORT = ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_eval_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_2_VISUAL_SEMANTIC_IDENTITY_EVAL_20260509.md"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_2_visual_semantic_identity/v33_2_visual_semantic_identity_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--split", default="test")
    p.add_argument("--max-items", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        payload = {"generated_at_utc": utc_now(), "checkpoint_exists": False, "exact_blocker": "checkpoint missing", "checkpoint_path": str(ckpt_path)}
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V33.2 Visual Semantic Identity Eval", payload, ["checkpoint_exists", "exact_blocker"])
        print(REPORT.relative_to(ROOT))
        return 2
    ck = torch.load(ckpt_path, map_location="cpu")
    train_args = argparse.Namespace(**ck.get("args", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisualSemanticIdentityWorldModelV332(train_args.v30_checkpoint, teacher_embedding_dim=train_args.teacher_embedding_dim, use_observed_instance_context=train_args.use_observed_instance_context).to(device)
    model.load_state_dict(ck["model"], strict=True)
    ds = VisualSidecarDataset(args.split, train_args, max_items=args.max_items)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_visual)
    metrics = evaluate_model(model, loader, device)
    payload = {"generated_at_utc": utc_now(), "checkpoint_exists": True, "checkpoint_path": str(ckpt_path.relative_to(ROOT)), "split": args.split, "item_count": len(ds), "metrics": metrics}
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.2 Visual Semantic Identity Eval", payload, ["checkpoint_exists", "split", "item_count", "metrics"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
