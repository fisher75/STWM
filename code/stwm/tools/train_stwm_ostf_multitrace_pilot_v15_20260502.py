#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from stwm.modules.dense_to_semantic_trace_unit_v15 import OSTFMultiTracePilot


def _apply_process_title() -> None:
    if str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).lower() == "off":
        return
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM OSTF MultiTrace Pilot V15", ""]
    for key in ["M128_beats_M1", "M512_beats_M1", "dense_points_load_bearing", "semantic_unit_compression_helpful", "proceed_to_full_ostf"]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    for name, row in payload.get("runs", {}).items():
        lines.append(f"## {name}")
        for metric in ["point_l1", "endpoint_error", "pck_0_05", "constant_velocity_point_l1", "checkpoint_path"]:
            lines.append(f"- {metric}: `{row.get(metric)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_cache(m: int) -> dict[str, Any]:
    path = Path(f"outputs/cache/stwm_object_dense_trace_v15/M{m}/object_dense_trace_cache.npz")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files} | {"_path": str(path)}


def _indices(cache: dict[str, Any], split: str) -> list[tuple[int, int]]:
    splits = [str(x) for x in cache["splits"].tolist()]
    valid = cache["object_valid_mask"].astype(bool)
    out = []
    for i, s in enumerate(splits):
        if s != split:
            continue
        for j in np.where(valid[i])[0].tolist():
            out.append((i, int(j)))
    return out


def _batch(cache: dict[str, Any], pairs: list[tuple[int, int]], scale: float, device: torch.device) -> tuple[torch.Tensor, ...]:
    pts = cache["points_xy"]
    valid = cache["valid_mask"]
    rel = cache["object_relative_xy"]
    obs = []
    fut = []
    ov = []
    fv = []
    rr = []
    for i, j in pairs:
        p = np.asarray(pts[i, j], dtype=np.float32) / scale
        v = np.asarray(valid[i, j], dtype=bool)
        obs.append(p[:8].transpose(1, 0, 2))
        fut.append(p[8:16].transpose(1, 0, 2))
        ov.append(v[:8].transpose(1, 0))
        fv.append(v[8:16].transpose(1, 0))
        rr.append(np.asarray(rel[i, j], dtype=np.float32))
    return (
        torch.tensor(np.stack(obs), device=device, dtype=torch.float32),
        torch.tensor(np.stack(ov), device=device, dtype=torch.bool),
        torch.tensor(np.stack(rr), device=device, dtype=torch.float32),
        torch.tensor(np.stack(fut), device=device, dtype=torch.float32),
        torch.tensor(np.stack(fv), device=device, dtype=torch.bool),
    )


def _cv_metrics(cache: dict[str, Any], pairs: list[tuple[int, int]], scale: float) -> dict[str, float]:
    vals = []
    end_vals = []
    pcks = []
    for i, j in pairs:
        pts = np.asarray(cache["points_xy"][i, j], dtype=np.float32) / scale
        valid = np.asarray(cache["valid_mask"][i, j], dtype=bool)
        obs = pts[:8]
        fut = pts[8:16]
        vel = obs[-1] - obs[-2]
        pred = np.stack([obs[-1] + vel * float(h + 1) for h in range(8)], axis=0)
        mask = valid[8:16]
        err = np.abs(pred - fut).sum(axis=-1)
        if mask.any():
            vals.append(float(err[mask].mean()))
            end_vals.append(float(err[-1][mask[-1]].mean()) if mask[-1].any() else float(err[mask].mean()))
            pcks.append(float((err[mask] < 0.05).mean()))
    return {
        "constant_velocity_point_l1": float(np.mean(vals)) if vals else 0.0,
        "constant_velocity_endpoint_error": float(np.mean(end_vals)) if end_vals else 0.0,
        "constant_velocity_pck_0_05": float(np.mean(pcks)) if pcks else 0.0,
    }


def _eval_model(model: OSTFMultiTracePilot, cache: dict[str, Any], pairs: list[tuple[int, int]], scale: float, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    vals = []
    end_vals = []
    pcks = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            obs, ov, rel, fut, fv = _batch(cache, pairs[start : start + batch_size], scale, device)
            pred = model(obs, ov, rel).permute(0, 2, 1, 3).contiguous()
            err = torch.abs(pred - fut).sum(dim=-1)
            mask = fv
            if bool(mask.any()):
                vals.append(float(err[mask].mean().detach().cpu().item()))
                end_mask = mask[:, :, -1]
                end_err = err[:, :, -1]
                end_vals.append(float(end_err[end_mask].mean().detach().cpu().item()) if bool(end_mask.any()) else vals[-1])
                pcks.append(float((err[mask] < 0.05).float().mean().detach().cpu().item()))
    return {
        "point_l1": float(np.mean(vals)) if vals else 0.0,
        "endpoint_error": float(np.mean(end_vals)) if end_vals else 0.0,
        "pck_0_05": float(np.mean(pcks)) if pcks else 0.0,
    }


def _run_m(m: int, seed: int, steps: int, device: torch.device) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cache = _load_cache(m)
    train_pairs = _indices(cache, "train")
    test_pairs = _indices(cache, "test")
    scale = float(np.nanmax(cache["points_xy"])) if np.nanmax(cache["points_xy"]) > 0 else 4096.0
    model = OSTFMultiTracePilot(obs_len=8, horizon=8, unit_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 32 if m >= 128 else 128
    losses = []
    for step in range(steps):
        pairs = random.sample(train_pairs, k=min(batch_size, len(train_pairs)))
        obs, ov, rel, fut, fv = _batch(cache, pairs, scale, device)
        pred = model(obs, ov, rel).permute(0, 2, 1, 3).contiguous()
        loss = F.smooth_l1_loss(pred[fv], fut[fv]) if bool(fv.any()) else pred.sum() * 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            losses.append(float(loss.detach().cpu().item()))
    metrics = _eval_model(model, cache, test_pairs[: min(len(test_pairs), 2048)], scale, device, batch_size=batch_size)
    metrics.update(_cv_metrics(cache, test_pairs[: min(len(test_pairs), 2048)], scale))
    ckpt = Path(f"outputs/checkpoints/stwm_ostf_multitrace_pilot_v15/M{m}_seed{seed}.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "M": m, "seed": seed, "scale": scale, "metrics": metrics}, ckpt)
    metrics.update(
        {
            "M": m,
            "seed": seed,
            "train_object_count": len(train_pairs),
            "test_object_count": len(test_pairs),
            "checkpoint_path": str(ckpt),
            "loss_curve": losses,
            "semantic_changed_top5": "not_trained_in_phase1_pilot",
            "stable_preservation": "not_trained_in_phase1_pilot",
            "visibility_F1": "visibility_target_from_pseudo_valid_mask_not_claimed",
        }
    )
    return metrics


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,128,512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    t0 = time.time()
    runs: dict[str, Any] = {}
    for m in [int(x) for x in args.m_values.split(",") if x.strip()]:
        runs[f"M{m}_seed{args.seed}"] = _run_m(m, args.seed, args.steps, device)
    m1 = runs.get(f"M1_seed{args.seed}", {})
    m128 = runs.get(f"M128_seed{args.seed}", {})
    m512 = runs.get(f"M512_seed{args.seed}", {})
    payload = {
        "audit_name": "stwm_ostf_multitrace_pilot_v15",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "runtime_seconds": float(time.time() - t0),
        "runs": runs,
        "baselines": {
            "M1_entity_anchor_current_STWM": "represented by M1 pilot/cache and existing sparse STWM evidence",
            "constant_velocity_point_rollout_semantic_copy": "constant_velocity_* metrics in each run",
            "object_dense_transformer_without_semantic_unit_compression": "not yet trained in Phase-1; requires full OSTF follow-up",
        },
        "M128_beats_M1": bool(m128 and m1 and m128["point_l1"] < m1["point_l1"]),
        "M512_beats_M1": bool(m512 and m1 and m512["point_l1"] < m1["point_l1"]),
        "dense_points_load_bearing": bool(m128 and m1 and m128["point_l1"] < m1["point_l1"]),
        "semantic_unit_compression_helpful": "inconclusive_without_uncompressed_dense_transformer",
        "proceed_to_full_ostf": bool(m128 and m1 and m128["point_l1"] < m1["point_l1"]),
        "claim_boundary": "Phase-1 supports object-internal pseudo point trace foundation, not physical dense optical trajectories.",
    }
    _dump(Path("reports/stwm_ostf_multitrace_pilot_v15_20260502.json"), payload)
    _write_doc(Path("docs/STWM_OSTF_MULTITRACE_PILOT_V15_20260502.md"), payload)
    print("reports/stwm_ostf_multitrace_pilot_v15_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
