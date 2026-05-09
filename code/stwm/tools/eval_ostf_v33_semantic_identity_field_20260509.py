#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_semantic_identity_heads_v33 import OSTFSemanticIdentityHeadV33
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_semantic_identity_head_20260509 import SidecarDataset, collate, evaluate

REPORT = ROOT / "reports/stwm_ostf_v33_semantic_identity_eval_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_SEMANTIC_IDENTITY_EVAL_20260509.md"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_semantic_identity/v33_semantic_identity_m128_h32_seed42_smoke.pt"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-items", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        payload: dict[str, Any] = {
            "generated_at_utc": utc_now(),
            "eval_stub_detected": False,
            "checkpoint_exists": False,
            "checkpoint_path": str(ckpt_path),
            "exact_blocker": "checkpoint missing",
        }
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V33 Semantic Identity Eval", payload, ["checkpoint_exists", "exact_blocker"])
        print(REPORT.relative_to(ROOT))
        return 2
    ck = torch.load(ckpt_path, map_location="cpu")
    train_args = ck.get("args", {}) if isinstance(ck, dict) else {}
    m_points = int(train_args.get("m_points", 128))
    horizon = int(train_args.get("horizon", 32))
    ds = SidecarDataset(args.split, m_points=m_points, horizon=horizon, max_items=args.max_items)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OSTFSemanticIdentityHeadV33().to(device)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    metrics = evaluate(model, loader, device)
    payload = {
        "generated_at_utc": utc_now(),
        "eval_stub_detected": False,
        "checkpoint_exists": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "split": args.split,
        "M": m_points,
        "H": horizon,
        "item_count": len(ds),
        "metrics": metrics,
        "trajectory_minFDE_delta_vs_frozen_V30": None,
        "trajectory_degraded": "not_applicable",
        "semantic_top1": metrics.get("semantic_top1"),
        "prototype_retrieval_top1": metrics.get("prototype_retrieval_top1"),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 Semantic Identity Eval", payload, ["checkpoint_exists", "split", "M", "H", "item_count", "metrics", "trajectory_degraded"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
