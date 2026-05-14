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
from stwm.tools.train_ostf_v34_14_horizon_conditioned_measurement_selector_20260513 import SUMMARY as TRAIN_SUMMARY, selector_inputs


EVAL = ROOT / "reports/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_eval_summary_20260513.json"
DECISION = ROOT / "reports/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_14_HORIZON_CONDITIONED_MEASUREMENT_SELECTOR_DECISION_20260513.md"
V3413_DECISION = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_decision_20260513.json"


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        if bool(m.any()):
            self.sum[key] = self.sum.get(key, 0.0) + float(value[m].sum().detach().cpu())
            self.count[key] = self.count.get(key, 0) + int(m.sum().detach().cpu())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def make_loader(split: str, args: argparse.Namespace) -> DataLoader:
    return DataLoader(TraceContractResidualDataset(split, args), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_v3410)


def mean_pool(obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (obs * mask.float()[..., None]).sum(dim=2) / mask.float().sum(dim=2, keepdim=True).clamp_min(1.0)


def last_pool(obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    idx = (mask.long() * torch.arange(mask.shape[-1], device=mask.device)[None, None, :]).argmax(dim=-1)
    out = obs[torch.arange(obs.shape[0], device=obs.device)[:, None], torch.arange(obs.shape[1], device=obs.device)[None, :], idx]
    return out * mask.any(dim=-1, keepdim=True).float()


def max_conf_pool(obs: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    score = conf.masked_fill(~mask.bool(), -1.0)
    idx = score.argmax(dim=-1)
    out = obs[torch.arange(obs.shape[0], device=obs.device)[:, None], torch.arange(obs.shape[1], device=obs.device)[None, :], idx]
    return out * (score.max(dim=-1).values >= 0).float()[..., None]


def weighted_pool(obs: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor, agree: torch.Tensor) -> torch.Tensor:
    w = mask.float() * conf.float().clamp(0.0, 1.0) * agree.float().clamp(0.0, 1.0)
    return (obs * w[..., None]).sum(dim=2) / w.sum(dim=2, keepdim=True).clamp_min(1e-6)


def cos_future(vec: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
    return (F.normalize(vec, dim=-1) * F.normalize(torch.nan_to_num(fut.float()), dim=-1)).sum(dim=-1)


def repeat_h(vec: torch.Tensor, h: int) -> torch.Tensor:
    return vec[:, :, None, :].expand(-1, -1, h, -1)


def load_model(device: torch.device) -> tuple[HorizonConditionedMeasurementSelectorV3414 | None, CausalAssignmentBoundResidualMemoryV348 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt_rel = train.get("checkpoint_path")
    if not ckpt_rel:
        return None, None, None, train
    ckpt = ROOT / ckpt_rel
    if not ckpt.exists():
        return None, None, None, train
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    base_model = CausalAssignmentBoundResidualMemoryV348(ckargs.v30_checkpoint, teacher_embedding_dim=ckargs.teacher_embedding_dim, units=ckargs.trace_units, horizon=ckargs.horizon).to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    selector = HorizonConditionedMeasurementSelectorV3414(int(ck["trace_hidden_dim"]), ckargs.teacher_embedding_dim, ckargs.hidden_dim).to(device)
    selector.load_state_dict(ck["model"], strict=True)
    selector.eval()
    return selector, base_model, ckargs, train


def eval_split(split: str, selector: HorizonConditionedMeasurementSelectorV3414, base_model: CausalAssignmentBoundResidualMemoryV348, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    loader = make_loader(split, ckargs)
    acc = Acc()
    entropy_vals: list[float] = []
    maxw_vals: list[float] = []
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = selector(**selector_inputs(selector, base_model, bd))
            obs = torch.nan_to_num(bd["obs_semantic_measurements"].float())
            mask = bd["obs_semantic_measurement_mask"].bool()
            conf = bd["semantic_measurement_confidence"].float()
            agree = bd.get("teacher_agreement_score", bd.get("semantic_measurement_agreement", bd["semantic_measurement_confidence"])).float()
            if conf.dim() == 2:
                conf = conf[:, :, None].expand_as(mask.float())
            if agree.dim() == 2:
                agree = agree[:, :, None].expand_as(mask.float())
            h = bd["fut_teacher_embedding"].shape[2]
            variants = {
                "horizon_conditioned_selector": out["selected_evidence"],
                "fixed_mean": repeat_h(F.normalize(mean_pool(obs, mask), dim=-1), h),
                "fixed_last": repeat_h(F.normalize(last_pool(obs, mask), dim=-1), h),
                "fixed_max_confidence": repeat_h(F.normalize(max_conf_pool(obs, mask, conf), dim=-1), h),
                "fixed_agreement_weighted": repeat_h(F.normalize(weighted_pool(obs, mask, conf, agree), dim=-1), h),
            }
            variants["random_shuffled"] = variants["fixed_mean"][:, torch.randperm(variants["fixed_mean"].shape[1], device=device)]
            cos = {k: cos_future(v, bd["fut_teacher_embedding"]) for k, v in variants.items()}
            obs_cos = torch.einsum("bmtd,bmhd->bmht", F.normalize(obs, dim=-1), F.normalize(torch.nan_to_num(bd["fut_teacher_embedding"].float()), dim=-1))
            obs_cos = obs_cos.masked_fill(~mask[:, :, None, :], -1e4)
            cos["oracle_timestep_best"] = obs_cos.max(dim=-1).values
            stack = torch.stack([cos[k] for k in ["fixed_mean", "fixed_last", "fixed_max_confidence", "fixed_agreement_weighted"]], dim=0)
            cos["oracle_fixed_variant_best"] = stack.max(dim=0).values
            cos["pointwise_base"] = bd["pointwise_semantic_cosine"].float()
            valid = bd["fut_teacher_available_mask"].bool()
            masks = {
                "valid": valid,
                "hard": bd["semantic_hard_mask"].bool() & valid,
                "changed": bd["changed_mask"].bool() & valid,
                "strict": bd["strict_residual_semantic_utility_mask"].bool() & valid,
                "causal": bd["causal_assignment_residual_semantic_mask"].bool() & valid,
            }
            for name, val in cos.items():
                for subset, smask in masks.items():
                    acc.add(f"{name}:{subset}", val, smask)
            for subset, smask in masks.items():
                acc.add(f"selector_minus_random:{subset}", cos["horizon_conditioned_selector"] - cos["random_shuffled"], smask)
                acc.add(f"selector_minus_pointwise:{subset}", cos["horizon_conditioned_selector"] - cos["pointwise_base"], smask)
                acc.add(f"selector_minus_mean:{subset}", cos["horizon_conditioned_selector"] - cos["fixed_mean"], smask)
                acc.add(f"selector_minus_fixed_oracle:{subset}", cos["horizon_conditioned_selector"] - cos["oracle_fixed_variant_best"], smask)
                acc.add(f"oracle_fixed_minus_selector:{subset}", cos["oracle_fixed_variant_best"] - cos["horizon_conditioned_selector"], smask)
                acc.add(f"oracle_timestep_minus_selector:{subset}", cos["oracle_timestep_best"] - cos["horizon_conditioned_selector"], smask)
            entropy_vals.append(float(out["selector_entropy"].mean().detach().cpu()))
            maxw_vals.append(float(out["selector_max_weight"].mean().detach().cpu()))
    deltas = {
        "selector_minus_random_valid": acc.mean("selector_minus_random:valid"),
        "selector_minus_pointwise_hard": acc.mean("selector_minus_pointwise:hard"),
        "selector_minus_pointwise_changed": acc.mean("selector_minus_pointwise:changed"),
        "selector_minus_pointwise_causal": acc.mean("selector_minus_pointwise:causal"),
        "selector_minus_mean_hard": acc.mean("selector_minus_mean:hard"),
        "selector_minus_fixed_oracle_hard": acc.mean("selector_minus_fixed_oracle:hard"),
        "oracle_gap_to_selector_hard": acc.mean("oracle_fixed_minus_selector:hard"),
        "oracle_gap_to_selector_changed": acc.mean("oracle_fixed_minus_selector:changed"),
        "oracle_timestep_gap_to_selector_hard": acc.mean("oracle_timestep_minus_selector:hard"),
        "oracle_timestep_gap_to_selector_changed": acc.mean("oracle_timestep_minus_selector:changed"),
    }
    return {
        "sample_count": len(loader.dataset),
        "variant_cosine_by_subset": {name: {subset: acc.mean(f"{name}:{subset}") for subset in ["valid", "hard", "changed", "strict", "causal"]} for name in ["horizon_conditioned_selector", "fixed_mean", "fixed_last", "fixed_max_confidence", "fixed_agreement_weighted", "pointwise_base", "random_shuffled", "oracle_fixed_variant_best", "oracle_timestep_best"]},
        "selector_deltas": deltas,
        "selector_entropy": float(np.mean(entropy_vals)) if entropy_vals else None,
        "selector_max_weight": float(np.mean(maxw_vals)) if maxw_vals else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    selector, base_model, ckargs, train = load_model(device)
    if selector is None or base_model is None or ckargs is None:
        decision = {
            "generated_at_utc": utc_now(),
            "中文结论": "V34.14 horizon-conditioned selector 未找到 checkpoint，评估跳过。",
            "horizon_conditioned_selector_built": False,
            "measurement_selector_nonoracle_passed": False,
            "recommended_next_step": "fix_nonoracle_measurement_selector",
        }
        dump_json(EVAL, {"generated_at_utc": utc_now(), "train_summary": train, "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "V34.14 horizon-conditioned selector 决策中文报告", decision, ["中文结论", "horizon_conditioned_selector_built", "measurement_selector_nonoracle_passed", "recommended_next_step"])
        print(f"已写出 V34.14 selector 评估跳过报告: {DECISION.relative_to(ROOT)}")
        return 0
    per = {split: eval_split(split, selector, base_model, ckargs, args, device) for split in ("val", "test")}
    val, test = per["val"]["selector_deltas"], per["test"]["selector_deltas"]
    beats_random = {"val": bool((val["selector_minus_random_valid"] or 0.0) > 0.01), "test": bool((test["selector_minus_random_valid"] or 0.0) > 0.01)}
    beats_hard = {"val": bool((val["selector_minus_pointwise_hard"] or 0.0) > 0.002), "test": bool((test["selector_minus_pointwise_hard"] or 0.0) > 0.002)}
    beats_changed = {"val": bool((val["selector_minus_pointwise_changed"] or 0.0) > 0.002), "test": bool((test["selector_minus_pointwise_changed"] or 0.0) > 0.002)}
    hard_gap = {"val": val["oracle_gap_to_selector_hard"], "test": test["oracle_gap_to_selector_hard"]}
    changed_gap = {"val": val["oracle_gap_to_selector_changed"], "test": test["oracle_gap_to_selector_changed"]}
    max_gap = max(float(hard_gap["val"] or 0.0), float(hard_gap["test"] or 0.0), float(changed_gap["val"] or 0.0), float(changed_gap["test"] or 0.0))
    passed = bool(all(beats_random.values()) and all(beats_hard.values()) and all(beats_changed.values()) and max_gap <= 0.08)
    v3413 = json.loads(V3413_DECISION.read_text(encoding="utf-8")) if V3413_DECISION.exists() else {}
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.14 horizon-conditioned selector 已评估；它用 future trace hidden 逐 horizon 读取 observed semantic memory，不使用 future teacher 作为输入。",
        "horizon_conditioned_selector_built": True,
        "selector_was_trained": True,
        "measurement_selector_nonoracle_passed": passed,
        "selector_beats_random": beats_random,
        "selector_beats_pointwise_on_hard": beats_hard,
        "selector_beats_pointwise_on_changed": beats_changed,
        "selector_beats_v34_13_on_oracle_gap": bool(max_gap < max(float(v3413.get("oracle_gap_to_selector_hard", {}).get("val") or 999), float(v3413.get("oracle_gap_to_selector_hard", {}).get("test") or 999), float(v3413.get("oracle_gap_to_selector_changed", {}).get("val") or 999), float(v3413.get("oracle_gap_to_selector_changed", {}).get("test") or 999))),
        "oracle_gap_to_selector_hard": hard_gap,
        "oracle_gap_to_selector_changed": changed_gap,
        "oracle_timestep_gap_to_selector_hard": {"val": val["oracle_timestep_gap_to_selector_hard"], "test": test["oracle_timestep_gap_to_selector_hard"]},
        "oracle_timestep_gap_to_selector_changed": {"val": val["oracle_timestep_gap_to_selector_changed"], "test": test["oracle_timestep_gap_to_selector_changed"]},
        "selector_entropy": {"val": per["val"]["selector_entropy"], "test": per["test"]["selector_entropy"]},
        "selector_max_weight": {"val": per["val"]["selector_max_weight"], "test": per["test"]["selector_max_weight"]},
        "measurement_weight_shape": "B,M,H,Tobs",
        "selected_evidence_shape": "B,M,H,D",
        "future_teacher_embedding_input_allowed": False,
        "v30_backbone_frozen": True,
        "recommended_next_step": "run_selector_conditioned_oracle_probe" if passed else "fix_nonoracle_measurement_selector",
    }
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "v34_13_reference": v3413, "per_split": per, "decision": decision}
    dump_json(EVAL, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "V34.14 horizon-conditioned selector 决策中文报告", decision, ["中文结论", "horizon_conditioned_selector_built", "selector_was_trained", "measurement_selector_nonoracle_passed", "selector_beats_random", "selector_beats_pointwise_on_hard", "selector_beats_pointwise_on_changed", "selector_beats_v34_13_on_oracle_gap", "oracle_gap_to_selector_hard", "oracle_gap_to_selector_changed", "selector_entropy", "selector_max_weight", "recommended_next_step"])
    print(f"已写出 V34.14 horizon-conditioned selector 评估摘要: {EVAL.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
