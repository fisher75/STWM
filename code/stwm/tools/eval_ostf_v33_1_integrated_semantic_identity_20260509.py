#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import OSTFExternalGTDataset, collate_external_gt
from stwm.modules.ostf_v33_integrated_semantic_identity_world_model import IntegratedSemanticIdentityWorldModelV331
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_1_integrated_semantic_identity_20260509 import evaluate_model

REPORT = ROOT / "reports/stwm_ostf_v33_1_integrated_eval_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_1_INTEGRATED_EVAL_20260509.md"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/stwm_ostf_v33_1_integrated/v33_1_integrated_m128_h32_seed42_smoke_best.pt"))
    p.add_argument("--split", default="test")
    p.add_argument("--max-items", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        payload = {"generated_at_utc": utc_now(), "checkpoint_exists": False, "checkpoint_path": str(ckpt_path), "exact_blocker": "missing checkpoint"}
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V33.1 Integrated Eval", payload, ["checkpoint_exists", "exact_blocker"])
        print(REPORT.relative_to(ROOT))
        return 2
    ck = torch.load(ckpt_path, map_location="cpu")
    train_args = ck.get("args", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntegratedSemanticIdentityWorldModelV331(
        train_args["v30_checkpoint"],
        identity_dim=int(train_args.get("identity_dim", 64)),
        use_observed_instance_context=bool(train_args.get("use_observed_instance_context", False)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)
    ds = OSTFExternalGTDataset(
        args.split,
        horizon=int(train_args.get("horizon", 32)),
        m_points=int(train_args.get("m_points", 128)),
        max_items=args.max_items,
        enable_semantic_identity_sidecar=True,
        semantic_identity_sidecar_root=train_args.get("semantic_identity_sidecar_root"),
        require_semantic_identity_sidecar=True,
        use_observed_instance_context=bool(train_args.get("use_observed_instance_context", False)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_external_gt)
    metrics, rows = evaluate_model(model, loader, device)
    payload = {
        "generated_at_utc": utc_now(),
        "checkpoint_exists": True,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "split": args.split,
        "item_count": len(ds),
        "integrated_v30_backbone_used": True,
        "v30_checkpoint_loaded": True,
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "observed_instance_context_used": bool(train_args.get("use_observed_instance_context", False)),
        "metrics": metrics,
        "item_rows": rows[:256],
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.1 Integrated Eval", payload, ["checkpoint_exists", "split", "item_count", "integrated_v30_backbone_used", "v30_checkpoint_loaded", "v30_backbone_frozen", "observed_instance_context_used", "metrics"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
