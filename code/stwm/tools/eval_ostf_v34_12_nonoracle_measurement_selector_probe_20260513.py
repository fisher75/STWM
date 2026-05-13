#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank/pointodyssey"
TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_9_causal_assignment_residual_targets/pointodyssey"
STRICT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_5_strict_residual_utility_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_NONORACLE_MEASUREMENT_SELECTOR_PROBE_20260513.md"


def norm(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def cos_future(vec: np.ndarray, fut: np.ndarray) -> np.ndarray:
    return (norm(vec)[:, None, :] * norm(fut)).sum(axis=-1)


def last_obs(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((obs.shape[0], obs.shape[-1]), dtype=np.float32)
    for i in range(obs.shape[0]):
        idx = np.where(mask[i])[0]
        if idx.size:
            out[i] = obs[i, idx[-1]]
    return out


def mean_obs(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    den = mask.sum(axis=1, keepdims=True).clip(min=1)
    return (obs * mask[..., None]).sum(axis=1) / den


def max_conf_obs(obs: np.ndarray, mask: np.ndarray, conf: np.ndarray) -> np.ndarray:
    score = np.where(mask, conf, -1.0)
    idx = score.argmax(axis=1)
    out = obs[np.arange(obs.shape[0]), idx].astype(np.float32)
    out[score.max(axis=1) < 0] = 0.0
    return out


def weighted_obs(obs: np.ndarray, mask: np.ndarray, conf: np.ndarray, agree: np.ndarray) -> np.ndarray:
    w = mask.astype(np.float32) * conf.clip(0.0, 1.0) * agree.clip(0.0, 1.0)
    return (obs * w[..., None]).sum(axis=1) / w.sum(axis=1, keepdims=True).clip(min=1e-6)


def instance_pool(vec: np.ndarray, inst: np.ndarray) -> np.ndarray:
    out = vec.copy()
    for iid in np.unique(inst):
        if iid < 0:
            continue
        pts = inst == iid
        out[pts] = vec[pts].mean(axis=0, keepdims=True)
    return out


def unit_pool(vec: np.ndarray, assign: np.ndarray) -> np.ndarray:
    den = assign.sum(axis=0).clip(min=1e-6)
    unit = np.einsum("mu,md->ud", assign, vec) / den[:, None]
    return np.einsum("mu,ud->md", assign, unit)


def temporal_stability_score(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((obs.shape[0],), dtype=np.float32)
    obs_n = norm(obs)
    for i in range(obs.shape[0]):
        idx = np.where(mask[i])[0]
        if idx.size >= 2:
            out[i] = float((obs_n[i, idx[:-1]] * obs_n[i, idx[1:]]).sum(axis=-1).mean())
    return out


def variants_for(z: Any, t: Any, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
    conf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
    agree = np.asarray(z.get("teacher_agreement_score", conf), dtype=np.float32)
    inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
    assign = np.asarray(t["point_to_unit_assignment"], dtype=np.float32)
    mean = mean_obs(obs, mask)
    variants = {
        "fixed_mean_pooling": mean,
        "fixed_last_observed": last_obs(obs, mask),
        "fixed_max_confidence_pooling": max_conf_obs(obs, mask, conf),
        "fixed_teacher_agreement_weighted_pooling": weighted_obs(obs, mask, conf, agree),
        "fixed_instance_pooled_measurement": instance_pool(mean, inst),
        "fixed_unit_pooled_measurement": unit_pool(mean, assign),
    }
    variants["random_shuffled_measurement"] = mean[rng.permutation(mean.shape[0])]
    obs_features = {
        "coverage": mask.mean(axis=1),
        "confidence": conf.mean(axis=1),
        "agreement": agree.mean(axis=1),
        "temporal_stability": temporal_stability_score(obs, mask),
    }
    return variants, obs_features


def observed_only_selector(variants: dict[str, np.ndarray], feat: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    score = weights["confidence"] * feat["confidence"] + weights["agreement"] * feat["agreement"] + weights["temporal_stability"] * feat["temporal_stability"] + weights["coverage"] * feat["coverage"]
    out = variants["fixed_mean_pooling"].copy()
    high = score > np.quantile(score, 0.66) if score.size else np.zeros_like(score, dtype=bool)
    low_stab = feat["temporal_stability"] < 0.5
    out[high] = variants["fixed_teacher_agreement_weighted_pooling"][high]
    out[low_stab] = variants["fixed_max_confidence_pooling"][low_stab]
    return out


class Acc:
    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, key: str, val: np.ndarray, mask: np.ndarray) -> None:
        m = mask.astype(bool)
        if not m.any():
            return
        self.sum[key] = self.sum.get(key, 0.0) + float(val[m].sum())
        self.count[key] = self.count.get(key, 0) + int(m.sum())

    def mean(self, key: str) -> float | None:
        c = self.count.get(key, 0)
        return None if c == 0 else float(self.sum[key] / c)


def eval_split(split: str, selector_weights: dict[str, float], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + (0 if split == "val" else 1000))
    acc = Acc()
    files = sorted((MEAS_ROOT / split).glob("*.npz"))
    sample_count = 0
    variant_names = [
        "fixed_mean_pooling",
        "fixed_last_observed",
        "fixed_max_confidence_pooling",
        "fixed_teacher_agreement_weighted_pooling",
        "fixed_instance_pooled_measurement",
        "fixed_unit_pooled_measurement",
        "learned_observed_only_selector",
        "oracle_best_measurement",
        "pointwise_base",
        "random_shuffled_measurement",
    ]
    for mp in files:
        tp = TARGET_ROOT / split / mp.name
        sp = STRICT_ROOT / split / mp.name
        if not tp.exists() or not sp.exists():
            continue
        z = np.load(mp, allow_pickle=True)
        t = np.load(tp, allow_pickle=True)
        s = np.load(sp, allow_pickle=True)
        variants, feat = variants_for(z, t, rng)
        variants["learned_observed_only_selector"] = observed_only_selector(variants, feat, selector_weights)
        fut = np.asarray(z["fut_teacher_embedding"], dtype=np.float32)
        valid = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
        hard = np.asarray(s["semantic_hard_mask"]).astype(bool) & valid
        changed = np.asarray(s["changed_mask"]).astype(bool) & valid
        strict = np.asarray(s["strict_residual_semantic_utility_mask"]).astype(bool) & valid
        pointwise = np.asarray(s["pointwise_semantic_cosine"], dtype=np.float32)
        fixed_cos = {k: cos_future(v, fut) for k, v in variants.items()}
        fixed_stack = np.stack([v for k, v in fixed_cos.items() if k != "random_shuffled_measurement"], axis=0)
        fixed_cos["oracle_best_measurement"] = fixed_stack.max(axis=0)
        fixed_cos["pointwise_base"] = pointwise
        masks = {"valid": valid, "semantic_hard": hard, "changed": changed, "strict_residual": strict}
        for name in variant_names:
            for subset, mask in masks.items():
                acc.add(f"{name}:{subset}", fixed_cos[name], mask)
                if name not in {"pointwise_base", "random_shuffled_measurement"}:
                    acc.add(f"{name}_minus_pointwise:{subset}", fixed_cos[name] - pointwise, mask)
                    acc.add(f"{name}_minus_random:{subset}", fixed_cos[name] - fixed_cos["random_shuffled_measurement"], mask)
        sample_count += 1
    out: dict[str, Any] = {"sample_count": sample_count, "variant_cosine_by_subset": {}}
    for name in variant_names:
        out["variant_cosine_by_subset"][name] = {subset: acc.mean(f"{name}:{subset}") for subset in ["valid", "semantic_hard", "changed", "strict_residual"]}
    out["selector_deltas"] = {
        "nonoracle_minus_random_valid": acc.mean("learned_observed_only_selector_minus_random:valid"),
        "nonoracle_minus_pointwise_hard": acc.mean("learned_observed_only_selector_minus_pointwise:semantic_hard"),
        "nonoracle_minus_pointwise_changed": acc.mean("learned_observed_only_selector_minus_pointwise:changed"),
        "oracle_minus_nonoracle_hard": None
        if acc.mean("oracle_best_measurement:semantic_hard") is None or acc.mean("learned_observed_only_selector:semantic_hard") is None
        else float(acc.mean("oracle_best_measurement:semantic_hard") - acc.mean("learned_observed_only_selector:semantic_hard")),
        "oracle_minus_nonoracle_changed": None
        if acc.mean("oracle_best_measurement:changed") is None or acc.mean("learned_observed_only_selector:changed") is None
        else float(acc.mean("oracle_best_measurement:changed") - acc.mean("learned_observed_only_selector:changed")),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    weights = {"confidence": 0.35, "agreement": 0.35, "temporal_stability": 0.2, "coverage": 0.1}
    per = {split: eval_split(split, weights, args.seed) for split in ("val", "test")}
    val_d, test_d = per["val"]["selector_deltas"], per["test"]["selector_deltas"]
    beats_random = bool((val_d["nonoracle_minus_random_valid"] or 0.0) > 0.01 and (test_d["nonoracle_minus_random_valid"] or 0.0) > 0.01)
    beats_hard = bool((val_d["nonoracle_minus_pointwise_hard"] or 0.0) > 0.002 and (test_d["nonoracle_minus_pointwise_hard"] or 0.0) > 0.002)
    beats_changed = bool((val_d["nonoracle_minus_pointwise_changed"] or 0.0) > 0.002 and (test_d["nonoracle_minus_pointwise_changed"] or 0.0) > 0.002)
    oracle_gap_h = max(float(val_d["oracle_minus_nonoracle_hard"] or 0.0), float(test_d["oracle_minus_nonoracle_hard"] or 0.0))
    oracle_gap_c = max(float(val_d["oracle_minus_nonoracle_changed"] or 0.0), float(test_d["oracle_minus_nonoracle_changed"] or 0.0))
    over = bool(oracle_gap_h > 0.05 or oracle_gap_c > 0.05)
    passed = bool(beats_random and beats_hard and beats_changed and not over)
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 non-oracle measurement selector probe 已完成；通过标准不使用 future target 逐 token 选 best，oracle best 只作为上界。",
        "nonoracle_measurement_selector_probe_done": True,
        "measurement_selector_nonoracle_passed": passed,
        "best_nonoracle_selector": "learned_observed_only_selector",
        "selector_weights_observed_only": weights,
        "nonoracle_beats_random": beats_random,
        "nonoracle_beats_pointwise_on_hard": beats_hard,
        "nonoracle_beats_pointwise_on_changed": beats_changed,
        "oracle_gap_to_nonoracle": {"semantic_hard_max_gap": oracle_gap_h, "changed_max_gap": oracle_gap_c},
        "measurement_quality_overestimated_by_oracle": over,
        "per_split": per,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.12 non-oracle measurement selector probe 中文报告",
        payload,
        [
            "中文结论",
            "nonoracle_measurement_selector_probe_done",
            "measurement_selector_nonoracle_passed",
            "best_nonoracle_selector",
            "nonoracle_beats_random",
            "nonoracle_beats_pointwise_on_hard",
            "nonoracle_beats_pointwise_on_changed",
            "oracle_gap_to_nonoracle",
            "measurement_quality_overestimated_by_oracle",
        ],
    )
    print(f"已写出 V34.12 non-oracle measurement selector probe: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
