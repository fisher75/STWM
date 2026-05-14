#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.modules.ostf_v34_8_causal_assignment_bound_residual_memory import CausalAssignmentBoundResidualMemoryV348
from stwm.modules.ostf_v34_14_horizon_conditioned_measurement_selector import HorizonConditionedMeasurementSelectorV3414
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_14_horizon_conditioned_measurement_selector_20260513 import selector_inputs


V3414_TRAIN = ROOT / "reports/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_train_summary_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_17_topk_evidence_selector_ablation_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_17_TOPK_EVIDENCE_SELECTOR_ABLATION_20260513.md"


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, val: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(val[m].sum().detach().cpu())
            self.count[key] = self.count.get(key, 0) + int(m.sum().detach().cpu())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def load_model(device: torch.device) -> tuple[HorizonConditionedMeasurementSelectorV3414, CausalAssignmentBoundResidualMemoryV348, argparse.Namespace]:
    train = json.loads(V3414_TRAIN.read_text(encoding="utf-8"))
    ckpt = ROOT / train["checkpoint_path"]
    ck = torch.load(ckpt, map_location="cpu")
    args = argparse.Namespace(**ck["args"])
    base = CausalAssignmentBoundResidualMemoryV348(args.v30_checkpoint, teacher_embedding_dim=args.teacher_embedding_dim, units=args.trace_units, horizon=args.horizon).to(device)
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    selector = HorizonConditionedMeasurementSelectorV3414(int(ck["trace_hidden_dim"]), args.teacher_embedding_dim, args.hidden_dim).to(device)
    selector.load_state_dict(ck["model"], strict=True)
    selector.eval()
    return selector, base, args


def make_loader(split: str, args: argparse.Namespace, batch_size: int, num_workers: int) -> DataLoader:
    args.batch_size = batch_size
    args.num_workers = num_workers
    return DataLoader(TraceContractResidualDataset(split, args), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_v3410)


def cos(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (F.normalize(pred, dim=-1) * F.normalize(torch.nan_to_num(target.float()), dim=-1)).sum(dim=-1)


def topk_evidence(out: dict[str, torch.Tensor], obs: torch.Tensor, k: int) -> torch.Tensor:
    weight = out["measurement_weight"]
    k = min(k, weight.shape[-1])
    vals, idx = torch.topk(weight, k=k, dim=-1)
    sem = torch.gather(obs[:, :, None, :, :].expand(-1, -1, weight.shape[2], -1, -1), 3, idx[..., None].expand(-1, -1, -1, -1, obs.shape[-1]))
    vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.normalize((sem * vals[..., None]).sum(dim=3), dim=-1)


def split_eval(split: str, selector: HorizonConditionedMeasurementSelectorV3414, base: CausalAssignmentBoundResidualMemoryV348, ckargs: argparse.Namespace, args: argparse.Namespace) -> dict[str, Any]:
    loader = make_loader(split, ckargs, args.batch_size, args.num_workers)
    acc = Acc()
    topks = [1, 2, 3, 4, 8]
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, next(selector.parameters()).device)
            out = selector(**selector_inputs(selector, base, bd))
            obs = torch.nan_to_num(bd["obs_semantic_measurements"].float())
            target = bd["fut_teacher_embedding"]
            valid = bd["fut_teacher_available_mask"].bool()
            masks = {
                "valid": valid,
                "hard": bd["semantic_hard_mask"].bool() & valid,
                "changed": bd["changed_mask"].bool() & valid,
                "strict": bd["strict_residual_semantic_utility_mask"].bool() & valid,
                "causal": bd["causal_assignment_residual_semantic_mask"].bool() & valid,
            }
            pointwise = bd["pointwise_semantic_cosine"].float()
            obs_cos = torch.einsum("bmtd,bmhd->bmht", F.normalize(obs, dim=-1), F.normalize(torch.nan_to_num(target.float()), dim=-1))
            obs_cos = obs_cos.masked_fill(~bd["obs_semantic_measurement_mask"].bool()[:, :, None, :], -1e4)
            oracle = obs_cos.max(dim=-1).values
            for k in topks:
                pred = topk_evidence(out, obs, k)
                c = cos(pred, target)
                for name, mask in masks.items():
                    acc.add(f"k{k}:{name}", c, mask)
                    acc.add(f"k{k}_minus_pointwise:{name}", c - pointwise, mask)
                    acc.add(f"oracle_minus_k{k}:{name}", oracle - c, mask)
    per = {}
    for k in topks:
        per[f"top{k}"] = {
            "cosine": {name: acc.mean(f"k{k}:{name}") for name in ["valid", "hard", "changed", "strict", "causal"]},
            "minus_pointwise": {name: acc.mean(f"k{k}_minus_pointwise:{name}") for name in ["valid", "hard", "changed", "strict", "causal"]},
            "oracle_gap": {name: acc.mean(f"oracle_minus_k{k}:{name}") for name in ["valid", "hard", "changed", "strict", "causal"]},
        }
    return per


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    selector, base, ckargs = load_model(device)
    per = {split: split_eval(split, selector, base, ckargs, args) for split in ("val", "test")}
    # Select by val hard+changed oracle gap, then confirm on test.
    best_name = min(per["val"], key=lambda k: (per["val"][k]["oracle_gap"]["hard"] or 999.0) + (per["val"][k]["oracle_gap"]["changed"] or 999.0))
    best = {
        "name": best_name,
        "val": per["val"][best_name],
        "test": per["test"][best_name],
    }
    improves_hard_changed = bool(
        (best["val"]["minus_pointwise"]["hard"] or 0.0) > 0.002
        and (best["test"]["minus_pointwise"]["hard"] or 0.0) > 0.002
        and (best["val"]["minus_pointwise"]["changed"] or 0.0) > 0.002
        and (best["test"]["minus_pointwise"]["changed"] or 0.0) > 0.002
    )
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.17 top-k evidence selector ablation 已完成；不训练新模型，仅用 V34.14 horizon-conditioned 权重评估 top-k memory set 是否比全 soft sum 更合理。",
        "topk_ablation_done": True,
        "base_selector": "v34_14_horizon_conditioned_soft_reader",
        "per_split": per,
        "best_topk_by_val": best,
        "best_topk_improves_hard_changed_vs_pointwise": improves_hard_changed,
        "future_teacher_embedding_input_allowed": False,
        "recommended_fix": "如果后续进入 residual，应让 residual 读取 top-k evidence set，而不是单个 selected vector 或 top-1 timestep label。",
        "recommended_next_step": "build_topk_evidence_conditioned_residual_probe" if improves_hard_changed else "fix_nonoracle_measurement_selector",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.17 top-k evidence selector ablation 中文报告", payload, ["中文结论", "topk_ablation_done", "base_selector", "best_topk_by_val", "best_topk_improves_hard_changed_vs_pointwise", "recommended_fix", "recommended_next_step"])
    print(f"已写出 V34.17 top-k evidence selector ablation: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
