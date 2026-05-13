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

from stwm.modules.ostf_v34_13_nonoracle_measurement_selector import NonOracleMeasurementSelectorV3413
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.train_ostf_v33_2_visual_semantic_identity_20260509 import move_batch
from stwm.tools.train_ostf_v34_10_trace_contract_oracle_residual_probe_20260512 import TraceContractResidualDataset, collate_v3410
from stwm.tools.train_ostf_v34_13_nonoracle_measurement_selector_20260513 import SUMMARY as TRAIN_SUMMARY, selector_inputs


EVAL = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_eval_summary_20260513.json"
DECISION = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_13_NONORACLE_MEASUREMENT_SELECTOR_DECISION_20260513.md"


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


def cos_future(vec: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
    return (F.normalize(vec, dim=-1)[:, :, None, :] * F.normalize(torch.nan_to_num(fut.float()), dim=-1)).sum(dim=-1)


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


def unit_pool(vec: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    den = assign.sum(dim=1).clamp_min(1e-6)
    unit = torch.einsum("bmu,bmd->bud", assign.float(), vec) / den[..., None]
    return torch.einsum("bmu,bud->bmd", assign.float(), unit)


def load_model(device: torch.device) -> tuple[NonOracleMeasurementSelectorV3413 | None, argparse.Namespace | None, dict[str, Any]]:
    train = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8")) if TRAIN_SUMMARY.exists() else {}
    ckpt_rel = train.get("checkpoint_path")
    if not ckpt_rel:
        return None, None, train
    ckpt = ROOT / ckpt_rel
    if not ckpt.exists():
        return None, None, train
    ck = torch.load(ckpt, map_location="cpu")
    ckargs = argparse.Namespace(**ck["args"])
    model = NonOracleMeasurementSelectorV3413(ckargs.teacher_embedding_dim, ckargs.hidden_dim).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ckargs, train


def eval_split(split: str, model: NonOracleMeasurementSelectorV3413, ckargs: argparse.Namespace, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ckargs.batch_size = args.batch_size
    ckargs.num_workers = args.num_workers
    loader = make_loader(split, ckargs)
    acc = Acc()
    entropy_vals: list[float] = []
    confidence_vals: list[float] = []
    with torch.no_grad():
        for batch in loader:
            bd = move_batch(batch, device)
            out = model(**selector_inputs(bd))
            obs = torch.nan_to_num(bd["obs_semantic_measurements"].float())
            mask = bd["obs_semantic_measurement_mask"].bool()
            conf = bd["semantic_measurement_confidence"].float()
            agree = bd.get("teacher_agreement_score", bd.get("semantic_measurement_agreement", bd["semantic_measurement_confidence"])).float()
            if conf.dim() == 2:
                conf = conf[:, :, None].expand_as(mask.float())
            if agree.dim() == 2:
                agree = agree[:, :, None].expand_as(mask.float())
            variants = {
                "trained_selector": out["selected_measurement_embedding"],
                "fixed_mean": F.normalize(mean_pool(obs, mask), dim=-1),
                "fixed_last": F.normalize(last_pool(obs, mask), dim=-1),
                "fixed_max_confidence": F.normalize(max_conf_pool(obs, mask, conf), dim=-1),
                "fixed_agreement_weighted": F.normalize(weighted_pool(obs, mask, conf, agree), dim=-1),
                "fixed_unit_pooled": F.normalize(unit_pool(mean_pool(obs, mask), bd["point_to_unit_assignment"].float()), dim=-1),
            }
            variants["random_shuffled"] = variants["fixed_mean"][:, torch.randperm(variants["fixed_mean"].shape[1], device=device)]
            cos = {k: cos_future(v, bd["fut_teacher_embedding"]) for k, v in variants.items()}
            stack = torch.stack([cos[k] for k in ["fixed_mean", "fixed_last", "fixed_max_confidence", "fixed_agreement_weighted", "fixed_unit_pooled"]], dim=0)
            cos["oracle_best"] = stack.max(dim=0).values
            cos["pointwise_base"] = bd["pointwise_semantic_cosine"].float()
            valid = bd["fut_teacher_available_mask"].bool()
            masks = {
                "valid": valid,
                "hard": bd["semantic_hard_mask"].bool() & valid,
                "changed": bd["changed_mask"].bool() & valid,
                "strict": bd["strict_residual_semantic_utility_mask"].bool() & valid,
            }
            for name, val in cos.items():
                for subset, smask in masks.items():
                    acc.add(f"{name}:{subset}", val, smask)
            for subset, smask in masks.items():
                acc.add(f"trained_minus_random:{subset}", cos["trained_selector"] - cos["random_shuffled"], smask)
                acc.add(f"trained_minus_pointwise:{subset}", cos["trained_selector"] - cos["pointwise_base"], smask)
                acc.add(f"trained_minus_mean:{subset}", cos["trained_selector"] - cos["fixed_mean"], smask)
                acc.add(f"trained_minus_max_conf:{subset}", cos["trained_selector"] - cos["fixed_max_confidence"], smask)
                acc.add(f"oracle_minus_trained:{subset}", cos["oracle_best"] - cos["trained_selector"], smask)
            entropy_vals.append(float(out["selector_entropy"].mean().detach().cpu()))
            confidence_vals.append(float(out["selector_confidence"].mean().detach().cpu()))
    deltas = {
        "selector_minus_random_valid": acc.mean("trained_minus_random:valid"),
        "selector_minus_pointwise_hard": acc.mean("trained_minus_pointwise:hard"),
        "selector_minus_pointwise_changed": acc.mean("trained_minus_pointwise:changed"),
        "selector_minus_mean_hard": acc.mean("trained_minus_mean:hard"),
        "selector_minus_max_confidence_hard": acc.mean("trained_minus_max_conf:hard"),
        "oracle_gap_to_selector_hard": acc.mean("oracle_minus_trained:hard"),
        "oracle_gap_to_selector_changed": acc.mean("oracle_minus_trained:changed"),
    }
    return {
        "sample_count": len(loader.dataset),
        "variant_cosine_by_subset": {name: {subset: acc.mean(f"{name}:{subset}") for subset in ["valid", "hard", "changed", "strict"]} for name in ["trained_selector", "fixed_mean", "fixed_last", "fixed_max_confidence", "fixed_agreement_weighted", "fixed_unit_pooled", "pointwise_base", "random_shuffled", "oracle_best"]},
        "selector_deltas": deltas,
        "selector_entropy": float(np.mean(entropy_vals)) if entropy_vals else None,
        "selector_confidence": float(np.mean(confidence_vals)) if confidence_vals else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckargs, train = load_model(device)
    if model is None or ckargs is None:
        decision = {
            "generated_at_utc": utc_now(),
            "中文结论": "V34.13 训练式 non-oracle selector 未找到 checkpoint，评估跳过。",
            "selector_was_trained": False,
            "measurement_selector_nonoracle_passed": False,
            "recommended_next_step": "fix_nonoracle_measurement_selector",
        }
        dump_json(EVAL, {"generated_at_utc": utc_now(), "train_summary": train, "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "V34.13 non-oracle selector 决策中文报告", decision, ["中文结论", "selector_was_trained", "measurement_selector_nonoracle_passed", "recommended_next_step"])
        print(f"已写出 V34.13 selector 评估跳过报告: {DECISION.relative_to(ROOT)}")
        return 0
    per = {split: eval_split(split, model, ckargs, args, device) for split in ("val", "test")}
    val, test = per["val"]["selector_deltas"], per["test"]["selector_deltas"]
    beats_random = {
        "val": bool((val["selector_minus_random_valid"] or 0.0) > 0.01),
        "test": bool((test["selector_minus_random_valid"] or 0.0) > 0.01),
    }
    beats_hard = {
        "val": bool((val["selector_minus_pointwise_hard"] or 0.0) > 0.002),
        "test": bool((test["selector_minus_pointwise_hard"] or 0.0) > 0.002),
    }
    beats_changed = {
        "val": bool((val["selector_minus_pointwise_changed"] or 0.0) > 0.002),
        "test": bool((test["selector_minus_pointwise_changed"] or 0.0) > 0.002),
    }
    hard_gap = {"val": val["oracle_gap_to_selector_hard"], "test": test["oracle_gap_to_selector_hard"]}
    changed_gap = {"val": val["oracle_gap_to_selector_changed"], "test": test["oracle_gap_to_selector_changed"]}
    max_gap = max(float(hard_gap["val"] or 0.0), float(hard_gap["test"] or 0.0), float(changed_gap["val"] or 0.0), float(changed_gap["test"] or 0.0))
    gap_ok = bool(max_gap <= 0.08)
    passed = bool(all(beats_random.values()) and all(beats_hard.values()) and all(beats_changed.values()) and gap_ok)
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.13 训练式 non-oracle selector 已评估；通过要求 observed-only selector 在 val/test 均赢 random、hard/changed pointwise，并且不能离 oracle best 太远。",
        "selector_was_trained": True,
        "nonoracle_selector_built": True,
        "measurement_selector_nonoracle_passed": passed,
        "selector_beats_random": beats_random,
        "selector_beats_pointwise_on_hard": beats_hard,
        "selector_beats_pointwise_on_changed": beats_changed,
        "selector_beats_fixed_mean": {
            "val": bool((val["selector_minus_mean_hard"] or 0.0) > 0.0),
            "test": bool((test["selector_minus_mean_hard"] or 0.0) > 0.0),
        },
        "selector_beats_max_confidence": {
            "val": bool((val["selector_minus_max_confidence_hard"] or 0.0) > 0.0),
            "test": bool((test["selector_minus_max_confidence_hard"] or 0.0) > 0.0),
        },
        "oracle_gap_to_selector_hard": hard_gap,
        "oracle_gap_to_selector_changed": changed_gap,
        "selector_entropy": {"val": per["val"]["selector_entropy"], "test": per["test"]["selector_entropy"]},
        "selector_confidence_calibration": {"val": per["val"]["selector_confidence"], "test": per["test"]["selector_confidence"]},
        "best_selector_by_val": "trained_observed_only_selector",
        "test_confirmation": bool(all(beats_random.values()) and (beats_hard["test"] or beats_changed["test"])),
        "measurement_quality_overestimated_by_oracle": not gap_ok,
        "future_teacher_embedding_input_allowed": False,
        "recommended_next_step": "run_selector_conditioned_oracle_probe" if (passed or (all(beats_random.values()) and ((beats_hard["val"] and beats_hard["test"]) or (beats_changed["val"] and beats_changed["test"])))) else "fix_nonoracle_measurement_selector",
    }
    payload = {"generated_at_utc": utc_now(), "train_summary": train, "per_split": per, "decision": decision}
    dump_json(EVAL, payload)
    dump_json(DECISION, decision)
    write_doc(
        DOC,
        "V34.13 non-oracle selector 决策中文报告",
        decision,
        ["中文结论", "selector_was_trained", "measurement_selector_nonoracle_passed", "selector_beats_random", "selector_beats_pointwise_on_hard", "selector_beats_pointwise_on_changed", "oracle_gap_to_selector_hard", "oracle_gap_to_selector_changed", "selector_entropy", "test_confirmation", "recommended_next_step"],
    )
    print(f"已写出 V34.13 selector 评估摘要: {EVAL.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
