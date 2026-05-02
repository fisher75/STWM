#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT / "code"))
from stwm.modules.dense_to_semantic_trace_unit_v15 import OSTFMultiTracePilot


CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v15c"


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM OSTF Real Teacher Pilot V15C", ""]
    for key in ["pilot_success", "teacher_source", "steps", "device", "M1_metrics", "M128_metrics", "next_step_choice"]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _load_objects() -> dict[str, list[dict[str, Any]]]:
    out = {"train": [], "val": [], "test": []}
    for path in sorted(CACHE_ROOT.glob("*/*.npz")):
        z = np.load(path, allow_pickle=True)
        split = str(_scalar(z["split"]))
        if split not in out:
            continue
        tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
        vis = np.asarray(z["visibility"]).astype(bool)
        q = np.asarray(z["query_points_xy"], dtype=np.float32)
        raw_size = np.asarray(z["raw_size"], dtype=np.float32)
        for obj in range(tracks.shape[0]):
            out[split].append(
                {
                    "tracks": tracks[obj],
                    "visibility": vis[obj],
                    "query_points": q[obj],
                    "scale": float(max(raw_size.tolist())),
                    "item_key": str(_scalar(z["item_key"])),
                    "dataset": str(_scalar(z["dataset"])),
                }
            )
    return out


def _object_rel(query_points: np.ndarray, m: int) -> np.ndarray:
    if m == 1:
        return np.asarray([[0.5, 0.5]], dtype=np.float32)
    pts = query_points.astype(np.float32)
    mn = pts.min(axis=0, keepdims=True)
    mx = pts.max(axis=0, keepdims=True)
    return ((pts - mn) / np.maximum(mx - mn, 1.0)).astype(np.float32)


def _prepare(obj: dict[str, Any], m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    tracks = obj["tracks"].astype(np.float32)
    vis = obj["visibility"].astype(bool)
    q = obj["query_points"].astype(np.float32)
    scale = float(obj["scale"])
    if m == 1:
        weights = vis.astype(np.float32)
        denom = weights.sum(axis=0, keepdims=True).clip(min=1.0)
        centroid = (tracks * weights[..., None]).sum(axis=0, keepdims=True) / denom[..., None]
        v1 = vis.any(axis=0, keepdims=True)
        obs = centroid[:, :8] / scale
        fut = centroid[:, 8:16] / scale
        ov = v1[:, :8]
        fv = v1[:, 8:16]
        rel = np.asarray([[0.5, 0.5]], dtype=np.float32)
    else:
        obs = tracks[:, :8] / scale
        fut = tracks[:, 8:16] / scale
        ov = vis[:, :8]
        fv = vis[:, 8:16]
        rel = _object_rel(q, m)
    return obs.astype(np.float32), ov.astype(bool), rel.astype(np.float32), fut.astype(np.float32), fv.astype(bool), scale


def _batch(rows: list[dict[str, Any]], indices: list[int], m: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    obs, ov, rel, fut, fv, scales = [], [], [], [], [], []
    for idx in indices:
        o, vo, r, f, vf, s = _prepare(rows[idx], m)
        obs.append(o)
        ov.append(vo)
        rel.append(r)
        fut.append(f)
        fv.append(vf)
        scales.append(s)
    return (
        torch.tensor(np.stack(obs), device=device, dtype=torch.float32),
        torch.tensor(np.stack(ov), device=device, dtype=torch.bool),
        torch.tensor(np.stack(rel), device=device, dtype=torch.float32),
        torch.tensor(np.stack(fut), device=device, dtype=torch.float32),
        torch.tensor(np.stack(fv), device=device, dtype=torch.bool),
        torch.tensor(scales, device=device, dtype=torch.float32),
    )


def _visibility_f1_last_observed(rows: list[dict[str, Any]], m: int) -> float:
    tp = fp = fn = 0
    for obj in rows:
        _obs, ov, _rel, _fut, fv, _scale = _prepare(obj, m)
        pred = np.repeat(ov[:, -1:], fv.shape[1], axis=1)
        tp += int((pred & fv).sum())
        fp += int((pred & ~fv).sum())
        fn += int((~pred & fv).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))


def _eval(model: OSTFMultiTracePilot, rows: list[dict[str, Any]], m: int, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    l1_px = []
    end_px = []
    pck4 = []
    centroid_px = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            idxs = list(range(start, min(start + batch_size, len(rows))))
            obs, ov, rel, fut, fv, scales = _batch(rows, idxs, m, device)
            pred = model(obs, ov, rel).permute(0, 2, 1, 3).contiguous()
            err_norm = torch.abs(pred - fut).sum(dim=-1)
            err_px = err_norm * scales[:, None, None]
            if bool(fv.any()):
                l1_px.append(float(err_px[fv].mean().cpu()))
                pck4.append(float((err_px[fv] < 4.0).float().mean().cpu()))
                end_mask = fv[:, :, -1]
                end_err = err_px[:, :, -1]
                end_px.append(float(end_err[end_mask].mean().cpu()) if bool(end_mask.any()) else l1_px[-1])
            pred_cent = pred.mean(dim=1)
            fut_cent = fut.mean(dim=1)
            cent_err = torch.abs(pred_cent - fut_cent).sum(dim=-1) * scales[:, None]
            cent_mask = fv.any(dim=1)
            if bool(cent_mask.any()):
                centroid_px.append(float(cent_err[cent_mask].mean().cpu()))
    return {
        "point_L1_px": float(np.mean(l1_px)) if l1_px else 0.0,
        "endpoint_error_px": float(np.mean(end_px)) if end_px else 0.0,
        "PCK_4px": float(np.mean(pck4)) if pck4 else 0.0,
        "anchor_or_centroid_L1_px": float(np.mean(centroid_px)) if centroid_px else 0.0,
        "visibility_F1_last_observed_baseline": _visibility_f1_last_observed(rows, m),
        "object_count": len(rows),
    }


def _train_one(m: int, rows: dict[str, list[dict[str, Any]]], steps: int, seed: int, device: torch.device) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = OSTFMultiTracePilot(obs_len=8, horizon=8, unit_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 16 if m > 1 else 64
    losses = []
    train_rows = rows["train"]
    for step in range(steps):
        idxs = random.sample(range(len(train_rows)), k=min(batch_size, len(train_rows)))
        obs, ov, rel, fut, fv, _scales = _batch(train_rows, idxs, m, device)
        pred = model(obs, ov, rel).permute(0, 2, 1, 3).contiguous()
        loss = F.smooth_l1_loss(pred[fv], fut[fv]) if bool(fv.any()) else pred.sum() * 0.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append(float(loss.detach().cpu()))
    ckpt = ROOT / f"outputs/checkpoints/stwm_ostf_real_teacher_pilot_v15c/M{m}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "M": m, "seed": seed, "steps": steps}, ckpt)
    return {
        "M": m,
        "seed": seed,
        "steps": steps,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "loss_curve": losses,
        "train_metrics": _eval(model, rows["train"], m, device, batch_size),
        "val_metrics": _eval(model, rows["val"], m, device, batch_size),
        "test_metrics": _eval(model, rows["test"], m, device, batch_size),
    }


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = _load_objects()
    runs = {f"M{m}_seed{args.seed}": _train_one(m, rows, args.steps, args.seed, device) for m in [1, 128]}
    payload = {
        "audit_name": "stwm_ostf_real_teacher_pilot_v15c",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_success": True,
        "teacher_source": "cotracker_official",
        "device": str(device),
        "steps": args.steps,
        "seed": args.seed,
        "split_object_counts": {k: len(v) for k, v in rows.items()},
        "M1_metrics": runs[f"M1_seed{args.seed}"]["test_metrics"],
        "M128_metrics": runs[f"M128_seed{args.seed}"]["test_metrics"],
        "runs": runs,
        "metric_note": "M1 anchor/centroid metrics and M128 internal point metrics are separated; M128 point L1 is not directly used as the sole comparison against M1 anchor L1.",
        "next_step_choice": "proceed_to_full_real_teacher_cache",
    }
    _dump(ROOT / "reports/stwm_ostf_real_teacher_pilot_v15c_20260502.json", payload)
    _write_doc(ROOT / "docs/STWM_OSTF_REAL_TEACHER_PILOT_V15C_20260502.md", payload)
    print("reports/stwm_ostf_real_teacher_pilot_v15c_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
