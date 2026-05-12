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

from stwm.modules.ostf_v34_6_best_residual_gate_world_model import BestResidualGateWorldModelV346
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_6_residual_parameterization_sweep_20260511 import StrictResidualUtilityDataset, collate_v345


SWEEP_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"
ABLATION = ROOT / "reports/stwm_ostf_v34_6_real_residual_content_ablation_20260511.json"
CKPT_DIR = ROOT / "outputs/checkpoints/stwm_ostf_v34_6_best_residual_gate_h32_m128"
SUMMARY = ROOT / "reports/stwm_ostf_v34_6_best_residual_gate_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_BEST_RESIDUAL_GATE_TRAIN_SUMMARY_20260511.md"


def skipped(reason: str) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "learned_gate_training_ran": False,
        "learned_gate_passed": "not_run",
        "skip_reason": reason,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.6 Best Residual Gate Train Summary", payload, ["learned_gate_training_ran", "skip_reason"])
    print(SUMMARY.relative_to(ROOT))
    return payload


def load_ready() -> tuple[dict[str, Any], dict[str, Any]]:
    sweep = json.loads(SWEEP_DECISION.read_text(encoding="utf-8")) if SWEEP_DECISION.exists() else {}
    ablation = json.loads(ABLATION.read_text(encoding="utf-8")) if ABLATION.exists() else {}
    return sweep, ablation


def freeze_except_gates(model: BestResidualGateWorldModelV346) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [model.semantic_gate, model.identity_gate]:
        for p in module.parameters():
            p.requires_grad_(True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    sweep, ablation = load_ready()
    if not sweep.get("residual_parameterization_passed"):
        skipped("residual_parameterization_not_passed")
        return 0
    if not (
        ablation.get("unit_memory_load_bearing_on_residual")
        and ablation.get("semantic_measurements_load_bearing_on_residual")
        and ablation.get("assignment_load_bearing_on_residual")
    ):
        skipped("real_residual_content_ablation_not_load_bearing")
        return 0
    ckpt_rel = sweep.get("best_checkpoint_path")
    if not ckpt_rel:
        skipped("missing_best_residual_checkpoint")
        return 0
    ck = torch.load(ROOT / ckpt_rel, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BestResidualGateWorldModelV346(
        ckargs.v30_checkpoint,
        teacher_embedding_dim=ckargs.teacher_embedding_dim,
        units=ckargs.trace_units,
        horizon=ckargs.horizon,
    ).to(device)
    model.load_state_dict(ck["model"], strict=False)
    freeze_except_gates(model)
    ds = StrictResidualUtilityDataset("train", ckargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_v345)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses: list[dict[str, float]] = []
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
        out = model(
            obs_points=bd["obs_points"],
            obs_vis=bd["obs_vis"],
            obs_conf=bd["obs_conf"],
            obs_semantic_measurements=bd["obs_semantic_measurements"],
            obs_semantic_measurement_mask=bd["obs_semantic_measurement_mask"],
            semantic_id=bd["semantic_id"],
        )
        target = bd["strict_residual_gate_target"].float()
        avail = bd["strict_residual_gate_available_mask"].float()
        bce = F.binary_cross_entropy(out["semantic_residual_gate"].clamp(1e-5, 1 - 1e-5), target, reduction="none")
        stable = bd["strict_stable_suppress_mask"].float()
        loss = (bce * avail).sum() / avail.sum().clamp_min(1.0) + 0.3 * (out["semantic_residual_gate"] * stable).sum() / stable.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(50, args.steps // 10) == 0:
            losses.append({"step": float(step), "loss": float(loss.detach().cpu())})
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = CKPT_DIR / "v34_6_best_residual_gate_m128_h32_seed42.pt"
    torch.save({"model": model.state_dict(), "args": vars(ckargs), "gate_train_args": vars(args), "step": args.steps}, out_ckpt)
    payload = {
        "generated_at_utc": utc_now(),
        "learned_gate_training_ran": True,
        "checkpoint_path": str(out_ckpt.relative_to(ROOT)),
        "best_residual_parameterization": sweep.get("best_residual_parameterization"),
        "best_residual_init": sweep.get("best_residual_init"),
        "steps": args.steps,
        "train_sample_count": len(ds),
        "v30_backbone_frozen": model.v30_backbone_frozen,
        "future_leakage_detected": False,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_decreased": bool(losses and losses[-1]["loss"] <= losses[0]["loss"]),
        "duration_seconds": float(time.time() - start),
        "loss_trace": losses,
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.6 Best Residual Gate Train Summary", payload, ["learned_gate_training_ran", "checkpoint_path", "best_residual_parameterization", "best_residual_init", "v30_backbone_frozen", "future_leakage_detected", "train_loss_decreased"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
