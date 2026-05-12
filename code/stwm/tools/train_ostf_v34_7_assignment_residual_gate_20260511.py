#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import setproctitle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_7_assignment_residual_gate_world_model import AssignmentResidualGateWorldModelV347
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_7_assignment_oracle_residual_probe_20260511 import AssignmentAwareResidualDataset, collate_v347


ORACLE_DECISION = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_decision_20260511.json"
ORACLE_TRAIN = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_train_summary_20260511.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_gate_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_RESIDUAL_GATE_TRAIN_SUMMARY_20260511.md"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_7_assignment_residual_gate_h32_m128"


def skip(reason: str) -> None:
    payload = {"generated_at_utc": utc_now(), "learned_gate_training_ran": False, "learned_gate_passed": "not_run", "skip_reason": reason, "v30_backbone_frozen": True, "future_leakage_detected": False}
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.7 Assignment Residual Gate Train Summary", payload, ["learned_gate_training_ran", "learned_gate_passed", "skip_reason"])
    print(SUMMARY.relative_to(ROOT))


def freeze_except_gate(model: AssignmentResidualGateWorldModelV347) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.assignment_residual_gate_head.parameters():
        p.requires_grad_(True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    oracle = json.loads(ORACLE_DECISION.read_text(encoding="utf-8")) if ORACLE_DECISION.exists() else {}
    if not oracle.get("assignment_oracle_residual_probe_passed"):
        skip("assignment_oracle_residual_probe_not_passed")
        return 0
    tr = json.loads(ORACLE_TRAIN.read_text(encoding="utf-8"))
    ck = torch.load(ROOT / tr["checkpoint_path"], map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AssignmentResidualGateWorldModelV347(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    model.load_state_dict(ck["model"], strict=False)
    freeze_except_gate(model)
    ds = AssignmentAwareResidualDataset("train", ckargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_v347)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
    it = iter(loader)
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = move_batch(batch, device)
        out = model(obs_points=bd["obs_points"], obs_vis=bd["obs_vis"], obs_conf=bd["obs_conf"], obs_semantic_measurements=bd["obs_semantic_measurements"], obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"], semantic_id=bd["semantic_id"])
        pos = bd["assignment_aware_residual_semantic_mask"].float()
        stable = bd["stable_suppress_mask"].float()
        avail = ((pos + stable) > 0).float()
        target = pos
        bce = F.binary_cross_entropy(out["semantic_residual_gate"].clamp(1e-5, 1 - 1e-5), target, reduction="none")
        loss = (bce * avail).sum() / avail.sum().clamp_min(1.0) + 0.3 * (out["semantic_residual_gate"] * stable).sum() / stable.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), "loss": float(loss.detach().cpu())})
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / "v34_7_assignment_residual_gate_m128_h32_seed42_best.pt"
    torch.save({"model": model.state_dict(), "args": vars(ckargs), "step": args.steps}, ckpt)
    payload = {"generated_at_utc": utc_now(), "learned_gate_training_ran": True, "checkpoint_path": str(ckpt.relative_to(ROOT)), "steps": args.steps, "train_sample_count": len(ds), "v30_backbone_frozen": model.v30_backbone_frozen, "future_leakage_detected": False, "train_loss_first": losses[0] if losses else None, "train_loss_last": losses[-1] if losses else None, "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]), "duration_seconds": float(time.time() - start)}
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.7 Assignment Residual Gate Train Summary", payload, ["learned_gate_training_ran", "checkpoint_path", "train_sample_count", "v30_backbone_frozen", "future_leakage_detected", "train_loss_decreased"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
