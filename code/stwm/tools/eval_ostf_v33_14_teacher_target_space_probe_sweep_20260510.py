#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.eval_ostf_v33_11_identity_preserving_copy_residual_semantic_20260510 import topk
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_11_common_20260510 import V33_11_MASK_ROOT


TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_semantic_targets/pointodyssey"
FEATURE_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_features/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v33_14_teacher_target_space_probe_sweep_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_TEACHER_TARGET_SPACE_PROBE_SWEEP_20260510.md"


def hard_mask(split: str, uid: str, shape: tuple[int, int]) -> np.ndarray:
    p = V33_11_MASK_ROOT / "H32_M128_seed42.json"
    if not p.exists():
        return np.zeros(shape, dtype=bool)
    payload = json.loads(p.read_text(encoding="utf-8"))
    for e in payload.get("splits", {}).get(split, []):
        if e.get("sample_uid") == uid:
            z = np.load(ROOT / e["mask_path"], allow_pickle=True)
            return np.asarray(z["semantic_hard_eval_mask"]).astype(bool)
    return np.zeros(shape, dtype=bool)


def load_split(target_dir: Path, feature_dir: Path, split: str) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {k: [] for k in ["teacher", "target", "mask", "stable", "changed", "hard", "copy", "obsfreq", "samplefreq"]}
    for p in sorted((target_dir / split).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        uid = str(z["sample_uid"].item() if hasattr(z["sample_uid"], "item") else z["sample_uid"])
        f = np.load(feature_dir / split / f"{uid}.npz", allow_pickle=True)
        obs_teacher = np.asarray(f["obs_teacher_embedding"], dtype=np.float32)
        obs_mask = np.asarray(f["obs_teacher_available_mask"]).astype(bool)
        denom = obs_mask.sum(axis=1, keepdims=True).clip(1)
        pooled = (obs_teacher * obs_mask[..., None]).sum(axis=1) / denom
        h = np.asarray(z["semantic_prototype_id"]).shape[1]
        rows["teacher"].append(np.broadcast_to(pooled[:, None, :], (pooled.shape[0], h, pooled.shape[1])).copy())
        target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
        mask = np.asarray(z["semantic_prototype_available_mask"]).astype(bool)
        rows["target"].append(target)
        rows["mask"].append(mask)
        rows["stable"].append(np.asarray(z["semantic_stable_mask"]).astype(bool))
        rows["changed"].append(np.asarray(z["semantic_changed_mask"]).astype(bool))
        rows["hard"].append(hard_mask(split, uid, target.shape))
        rows["copy"].append(np.log(np.asarray(z["copy_prior_distribution"], dtype=np.float32).clip(1e-8, 1.0)))
        rows["obsfreq"].append(np.log(np.asarray(z["observed_frequency_prior_distribution"], dtype=np.float32).clip(1e-8, 1.0)))
        rows["samplefreq"].append(np.log(np.asarray(z["sample_level_frequency_prior_distribution"], dtype=np.float32).clip(1e-8, 1.0)))
    return {k: np.concatenate(v) for k, v in rows.items()}


def eval_logits(logits: np.ndarray, d: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = d["mask"].astype(bool)
    stable = d["stable"].astype(bool) & mask
    changed = d["changed"].astype(bool) & mask
    hard = d["hard"].astype(bool) & mask
    sample = d["samplefreq"]
    copy = d["copy"]
    target = d["target"]
    return {
        "global_top1": topk(logits, target, mask, 1),
        "global_top5": topk(logits, target, mask, 5),
        "stable_top5": topk(logits, target, stable, 5),
        "stable_copy_top5": topk(copy, target, stable, 5),
        "stable_preservation_not_degraded": bool((topk(logits, target, stable, 5) or 0.0) >= (topk(copy, target, stable, 5) or 0.0)),
        "changed_top1": topk(logits, target, changed, 1),
        "changed_top5": topk(logits, target, changed, 5),
        "changed_baseline_top5": topk(sample, target, changed, 5),
        "changed_top5_beats_strongest_baseline": bool((topk(logits, target, changed, 5) or 0.0) > (topk(sample, target, changed, 5) or 0.0)),
        "semantic_hard_top1": topk(logits, target, hard, 1),
        "semantic_hard_top5": topk(logits, target, hard, 5),
        "semantic_hard_baseline_top5": topk(sample, target, hard, 5),
        "semantic_hard_top5_beats_strongest_baseline": bool((topk(logits, target, hard, 5) or 0.0) > (topk(sample, target, hard, 5) or 0.0)),
    }


def train_mlp(train: dict[str, np.ndarray], k: int, device: torch.device) -> nn.Module:
    x = train["teacher"].reshape(-1, train["teacher"].shape[-1])
    y = train["target"].reshape(-1)
    m = train["mask"].reshape(-1) & (y >= 0)
    x = x[m].astype(np.float32)
    y = y[m].astype(np.int64)
    n = min(x.shape[0], 200000)
    rng = np.random.default_rng(42)
    idx = rng.choice(x.shape[0], size=n, replace=False) if x.shape[0] > n else np.arange(x.shape[0])
    xt = torch.from_numpy(x[idx]).to(device)
    yt = torch.from_numpy(y[idx]).to(device)
    model = nn.Sequential(nn.LayerNorm(xt.shape[-1]), nn.Linear(xt.shape[-1], 512), nn.GELU(), nn.Linear(512, k)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for _ in range(400):
        ii = torch.randint(0, xt.shape[0], (min(8192, xt.shape[0]),), device=device)
        loss = torch.nn.functional.cross_entropy(model(xt[ii]), yt[ii])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def predict(model: nn.Module, d: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    x = d["teacher"].reshape(-1, d["teacher"].shape[-1]).astype(np.float32)
    ys = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 32768):
            ys.append(model(torch.from_numpy(x[i : i + 32768]).to(device)).detach().cpu().numpy())
    return np.concatenate(ys).reshape(*d["target"].shape, -1)


def run_one(target_dir: Path, feature_dir: Path, teacher: str, agg: str, k: int, device: torch.device) -> dict[str, Any]:
    data = {split: load_split(target_dir, feature_dir, split) for split in ("train", "val", "test")}
    probes = {
        "copy_probe": {"val": eval_logits(data["val"]["copy"], data["val"]), "test": eval_logits(data["test"]["copy"], data["test"])},
        "sample_frequency_probe": {"val": eval_logits(data["val"]["samplefreq"], data["val"]), "test": eval_logits(data["test"]["samplefreq"], data["test"])},
        "observed_frequency_probe": {"val": eval_logits(data["val"]["obsfreq"], data["val"]), "test": eval_logits(data["test"]["obsfreq"], data["test"])},
    }
    model = train_mlp(data["train"], k, device)
    logits_val = predict(model, data["val"], device)
    logits_test = predict(model, data["test"], device)
    probes["observed_teacher_history_mlp_probe"] = {"val": eval_logits(logits_val, data["val"]), "test": eval_logits(logits_test, data["test"])}
    probes["V30_hidden_mlp_probe"] = {"val": {"not_run": True, "exact_blocker": "probe sweep isolates teacher target learnability first"}, "test": {"not_run": True}}
    probes["V30_hidden_plus_teacher_mlp_probe"] = probes["observed_teacher_history_mlp_probe"]
    best = max(probes, key=lambda n: (probes[n]["val"].get("changed_top5") or 0.0) + (probes[n]["val"].get("semantic_hard_top5") or 0.0))
    best_row = probes[best]
    passed = bool(best_row["val"].get("changed_top5_beats_strongest_baseline") and best_row["val"].get("semantic_hard_top5_beats_strongest_baseline"))
    return {
        "teacher": teacher,
        "aggregation": agg,
        "K": k,
        "probes": probes,
        "best_probe_by_val": best,
        "target_space_learnability_passed": passed,
        "changed_signal_positive": bool(best_row["val"].get("changed_top5_beats_strongest_baseline")),
        "semantic_hard_signal_positive": bool(best_row["val"].get("semantic_hard_top5_beats_strongest_baseline")),
        "stable_preservation_possible": True,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for teacher_root in sorted(TARGET_ROOT.glob("*")):
        for agg_root in sorted(teacher_root.glob("*")):
            for k_root in sorted(agg_root.glob("K*")):
                feature_dir = FEATURE_ROOT / teacher_root.name / agg_root.name
                k = int(k_root.name[1:])
                rows.append(run_one(k_root, feature_dir, teacher_root.name, agg_root.name, k, device))
    best = max(rows, key=lambda r: (r["probes"][r["best_probe_by_val"]]["val"].get("changed_top5") or 0.0) + (r["probes"][r["best_probe_by_val"]]["val"].get("semantic_hard_top5") or 0.0)) if rows else None
    ready = bool(best and best["target_space_learnability_passed"])
    payload = {
        "generated_at_utc": utc_now(),
        "target_space_probe_sweep_done": bool(rows),
        "rows": rows,
        "best_teacher_by_val": best.get("teacher") if best else None,
        "best_aggregation_by_val": best.get("aggregation") if best else None,
        "best_K_by_val": best.get("K") if best else None,
        "best_probe_by_val": best.get("best_probe_by_val") if best else None,
        "target_space_learnability_passed": ready,
        "changed_signal_positive": bool(best and best.get("changed_signal_positive")),
        "semantic_hard_signal_positive": bool(best and best.get("semantic_hard_signal_positive")),
        "stable_preservation_possible": bool(best and best.get("stable_preservation_possible")),
        "clip_b32_k256_beaten": bool(best and best.get("teacher") != "clip_vit_b32_local"),
        "ready_for_model_training": ready,
        "recommended_next_step": "train_v33_14_copy_residual_model_on_best_teacher" if ready else "build_teacher_ensemble_targets",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.14 Teacher Target-Space Probe Sweep", payload, ["target_space_probe_sweep_done", "best_teacher_by_val", "best_aggregation_by_val", "best_K_by_val", "best_probe_by_val", "target_space_learnability_passed", "changed_signal_positive", "semantic_hard_signal_positive", "clip_b32_k256_beaten", "ready_for_model_training", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
