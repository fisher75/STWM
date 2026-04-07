#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import itertools
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from stwm.tracewm.datasets.stage1_kubric import Stage1KubricDataset
from stwm.tracewm.datasets.stage1_pointodyssey import Stage1PointOdysseyDataset
from stwm.tracewm.datasets.stage1_tapvid import Stage1TapVidDataset
from stwm.tracewm.datasets.stage1_tapvid3d import Stage1TapVid3DDataset
from stwm.tracewm.datasets.stage1_unified import Stage1UnifiedDataset, load_stage1_minisplits, stage1_collate_fn

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_IMPORT_ERR = str(exc)
else:
    _MPL_IMPORT_ERR = ""

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    Image = None
    _PIL_IMPORT_ERR = str(exc)
else:
    _PIL_IMPORT_ERR = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_item(batch: Dict[str, Any], key: str, idx: int) -> Any:
    v = batch.get(key)
    if isinstance(v, list):
        return v[idx]
    if isinstance(v, torch.Tensor):
        return v[idx]
    return None


def _reduce_tracks(tracks: Any) -> torch.Tensor | None:
    if tracks is None:
        return None
    if not isinstance(tracks, torch.Tensor):
        return None

    t = tracks.detach().to(torch.float32)
    if t.ndim == 3:
        return t.mean(dim=1)
    if t.ndim == 2:
        return t
    return None


def build_state_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = int(batch.get("batch_size", 0))
    obs_states: List[torch.Tensor] = []
    fut_states: List[torch.Tensor] = []

    for i in range(bs):
        obs2d = _reduce_tracks(_batch_item(batch, "obs_tracks_2d", i))
        fut2d = _reduce_tracks(_batch_item(batch, "fut_tracks_2d", i))
        obs3d = _reduce_tracks(_batch_item(batch, "obs_tracks_3d", i))
        fut3d = _reduce_tracks(_batch_item(batch, "fut_tracks_3d", i))

        if obs2d is None and obs3d is None:
            obs_len = 8
            obs_valid = _batch_item(batch, "obs_valid", i)
            if isinstance(obs_valid, torch.Tensor):
                obs_len = int(obs_valid.shape[0])
            obs_state = torch.zeros((obs_len, 5), dtype=torch.float32)
        else:
            obs_len = obs2d.shape[0] if obs2d is not None else obs3d.shape[0]
            obs_state = torch.zeros((obs_len, 5), dtype=torch.float32)
            if obs2d is not None:
                obs_state[:, 0:2] = obs2d[:, 0:2]
            if obs3d is not None:
                obs_state[:, 2:5] = obs3d[:, 0:3]

        if fut2d is None and fut3d is None:
            fut_len = 8
            fut_valid = _batch_item(batch, "fut_valid", i)
            if isinstance(fut_valid, torch.Tensor):
                fut_len = int(fut_valid.shape[0])
            fut_state = torch.zeros((fut_len, 5), dtype=torch.float32)
        else:
            fut_len = fut2d.shape[0] if fut2d is not None else fut3d.shape[0]
            fut_state = torch.zeros((fut_len, 5), dtype=torch.float32)
            if fut2d is not None:
                fut_state[:, 0:2] = fut2d[:, 0:2]
            if fut3d is not None:
                fut_state[:, 2:5] = fut3d[:, 0:3]

        obs_states.append(obs_state)
        fut_states.append(fut_state)

    obs_batch = torch.stack(obs_states, dim=0).to(device)
    fut_batch = torch.stack(fut_states, dim=0).to(device)
    return obs_batch, fut_batch


class Iter1TraceModel(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_encoder = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.roll_cell = nn.GRUCell(input_size=state_dim, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, state_dim)

    def rollout_teacher_forced(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, h = self.obs_encoder(obs)
        h_t = h[-1]
        fut_len = target.shape[1]
        preds: List[torch.Tensor] = []

        for t in range(fut_len):
            if t == 0:
                x_t = obs[:, -1, :]
            else:
                x_t = target[:, t - 1, :]
            h_t = self.roll_cell(x_t, h_t)
            preds.append(self.head(h_t))

        return torch.stack(preds, dim=1)

    def rollout_free(self, obs: torch.Tensor, fut_len: int) -> torch.Tensor:
        _, h = self.obs_encoder(obs)
        h_t = h[-1]
        prev = obs[:, -1, :]
        preds: List[torch.Tensor] = []

        for _ in range(fut_len):
            h_t = self.roll_cell(prev, h_t)
            pred = self.head(h_t)
            preds.append(pred)
            prev = pred

        return torch.stack(preds, dim=1)


def _records_for(minisplits: Dict[str, Any], dataset: str, split: str) -> List[Dict[str, Any]]:
    datasets = minisplits.get("datasets", {}) if isinstance(minisplits, dict) else {}
    ds = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    out = ds.get(split, []) if isinstance(ds, dict) else []
    return [x for x in out if isinstance(x, dict)]


def dataset_mix_info(ds: Stage1UnifiedDataset) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for idx in ds.index_map:
        counter[idx.dataset_name] += 1
    return {k: int(v) for k, v in sorted(counter.items())}


def evaluate_loader(
    model: Iter1TraceModel,
    loader: DataLoader,
    device: torch.device,
    free_loss_weight: float,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    mse = nn.MSELoss(reduction="mean")

    teacher_vals: List[float] = []
    free_vals: List[float] = []
    total_vals: List[float] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            obs, fut = build_state_batch(batch, device)
            pred_t = model.rollout_teacher_forced(obs, fut)
            pred_f = model.rollout_free(obs, fut.shape[1])

            loss_t = mse(pred_t, fut)
            loss_f = mse(pred_f, fut)
            loss_total = loss_t + free_loss_weight * loss_f

            teacher_vals.append(float(loss_t.item()))
            free_vals.append(float(loss_f.item()))
            total_vals.append(float(loss_total.item()))

    def _mean(x: List[float]) -> float:
        return float(sum(x) / max(len(x), 1))

    return {
        "val_teacher_forced_loss": _mean(teacher_vals),
        "val_free_rollout_loss": _mean(free_vals),
        "val_total_loss": _mean(total_vals),
        "val_batches": len(teacher_vals),
    }


def _eval_dataset_state_space(
    model: Iter1TraceModel,
    dataset: Dataset,
    state_dims: Tuple[int, int],
    device: torch.device,
    max_samples: int,
) -> Dict[str, float]:
    mse = nn.MSELoss(reduction="mean")
    d0, d1 = state_dims

    teacher_mse_vals: List[float] = []
    free_mse_vals: List[float] = []
    teacher_endpoint_vals: List[float] = []
    free_endpoint_vals: List[float] = []

    model.eval()
    with torch.no_grad():
        n = min(len(dataset), max_samples)
        for i in range(n):
            sample = dataset[i]
            batch = stage1_collate_fn([sample])
            obs, fut = build_state_batch(batch, device)
            pred_t = model.rollout_teacher_forced(obs, fut)
            pred_f = model.rollout_free(obs, fut.shape[1])

            target = fut[..., d0:d1]
            p_teacher = pred_t[..., d0:d1]
            p_free = pred_f[..., d0:d1]

            teacher_mse_vals.append(float(mse(p_teacher, target).item()))
            free_mse_vals.append(float(mse(p_free, target).item()))

            teacher_endpoint = torch.linalg.norm(p_teacher[:, -1, :] - target[:, -1, :], dim=-1).mean().item()
            free_endpoint = torch.linalg.norm(p_free[:, -1, :] - target[:, -1, :], dim=-1).mean().item()
            teacher_endpoint_vals.append(float(teacher_endpoint))
            free_endpoint_vals.append(float(free_endpoint))

    def _mean(x: List[float]) -> float:
        return float(sum(x) / max(len(x), 1))

    return {
        "samples": min(len(dataset), max_samples),
        "teacher_forced_mse": _mean(teacher_mse_vals),
        "free_rollout_mse": _mean(free_mse_vals),
        "teacher_forced_endpoint_l2": _mean(teacher_endpoint_vals),
        "free_rollout_endpoint_l2": _mean(free_endpoint_vals),
    }


def evaluate_external(
    model: Iter1TraceModel,
    splits_path: str | Path,
    data_root: str | Path,
    device: torch.device,
    max_tapvid_samples: int,
    max_tapvid3d_samples: int,
) -> Dict[str, Any]:
    minisplits = load_stage1_minisplits(splits_path)

    tapvid_ds = Stage1TapVidDataset(
        split="eval_mini",
        minisplit_records=_records_for(minisplits, "tapvid", "eval_mini"),
    )
    tapvid3d_ds = Stage1TapVid3DDataset(
        data_root=data_root,
        split="eval_mini",
        minisplit_records=_records_for(minisplits, "tapvid3d", "eval_mini"),
    )

    tapvid_metrics = _eval_dataset_state_space(
        model=model,
        dataset=tapvid_ds,
        state_dims=(0, 2),
        device=device,
        max_samples=max_tapvid_samples,
    )
    tapvid3d_metrics = _eval_dataset_state_space(
        model=model,
        dataset=tapvid3d_ds,
        state_dims=(2, 5),
        device=device,
        max_samples=max_tapvid3d_samples,
    )

    tapvid_metrics["main_eval_ready"] = tapvid_metrics["samples"] > 0
    tapvid3d_metrics["limited_eval_ready"] = tapvid3d_metrics["samples"] > 0
    tapvid3d_metrics["full_eval_ready"] = False

    return {
        "tapvid": tapvid_metrics,
        "tapvid3d": tapvid3d_metrics,
    }


def _first_image_or_blank(obs_frames: Any, size: int = 512) -> np.ndarray:
    blank = np.ones((size, size, 3), dtype=np.uint8) * 245
    if not isinstance(obs_frames, list) or not obs_frames:
        return blank
    if Image is None:
        return blank

    first = str(obs_frames[0])
    p = Path(first)
    if not p.exists() or not p.is_file():
        return blank

    try:
        img = Image.open(p).convert("RGB").resize((size, size))
        return np.asarray(img)
    except Exception:
        return blank


def _project_tracks3d(tracks3d: torch.Tensor) -> torch.Tensor:
    xy = tracks3d[..., :2]
    mn = xy.amin(dim=(0, 1), keepdim=True)
    mx = xy.amax(dim=(0, 1), keepdim=True)
    denom = torch.clamp(mx - mn, min=1e-6)
    return (xy - mn) / denom


def _plot_sample_2d(ax: Any, bg: np.ndarray, tracks2d: torch.Tensor, obs_len: int, title: str) -> None:
    ax.imshow(bg)
    ax.set_title(title)
    ax.axis("off")

    t_all = tracks2d.shape[0]
    n_points = tracks2d.shape[1]
    k = min(16, n_points)

    for pid in range(k):
        traj = tracks2d[:, pid, :].detach().cpu().numpy()
        x = traj[:, 0] * (bg.shape[1] - 1)
        y = traj[:, 1] * (bg.shape[0] - 1)
        ax.plot(x[:obs_len], y[:obs_len], color="#1f77b4", linewidth=1.1)
        ax.plot(x[obs_len - 1 : t_all], y[obs_len - 1 : t_all], color="#d62728", linewidth=1.1)


def _render_qualitative_sample(sample: Dict[str, Any], title: str, out_path: Path) -> Tuple[bool, str]:
    if plt is None:
        return False, f"matplotlib_unavailable:{_MPL_IMPORT_ERR}"

    obs_len = int(sample["obs_valid"].shape[0]) if isinstance(sample.get("obs_valid"), torch.Tensor) else 8
    tracks2d = sample.get("obs_tracks_2d")
    fut2d = sample.get("fut_tracks_2d")
    tracks3d = sample.get("obs_tracks_3d")
    fut3d = sample.get("fut_tracks_3d")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tracks2d, torch.Tensor) and isinstance(fut2d, torch.Tensor):
        all_tracks = torch.cat([tracks2d, fut2d], dim=0)
        bg = _first_image_or_blank(sample.get("obs_frames"), size=512)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _plot_sample_2d(ax, bg, all_tracks, obs_len=obs_len, title=title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return True, "2d_overlay_done"

    if isinstance(tracks3d, torch.Tensor) and isinstance(fut3d, torch.Tensor):
        all_tracks3d = torch.cat([tracks3d, fut3d], dim=0)
        proj = _project_tracks3d(all_tracks3d)
        bg = np.ones((512, 512, 3), dtype=np.uint8) * 245
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _plot_sample_2d(ax, bg, proj, obs_len=obs_len, title=title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return True, "3d_projection_done"

    return False, "no_tracks_available"


def render_qualitative_sets(
    out_dir: Path,
    data_root: str | Path,
    splits_path: str | Path,
) -> List[Dict[str, Any]]:
    minisplits = load_stage1_minisplits(splits_path)

    point_train = Stage1PointOdysseyDataset(
        data_root=data_root,
        split="train_iter1_pointodyssey",
        minisplit_records=_records_for(minisplits, "pointodyssey", "train_iter1_pointodyssey"),
    )
    point_val = Stage1PointOdysseyDataset(
        data_root=data_root,
        split="val_iter1_pointodyssey",
        minisplit_records=_records_for(minisplits, "pointodyssey", "val_iter1_pointodyssey"),
    )
    kubric_train = Stage1KubricDataset(
        data_root=data_root,
        split="train_iter1_kubric",
        minisplit_records=_records_for(minisplits, "kubric", "train_iter1_kubric"),
    )
    kubric_val = Stage1KubricDataset(
        data_root=data_root,
        split="val_iter1_kubric",
        minisplit_records=_records_for(minisplits, "kubric", "val_iter1_kubric"),
    )
    tapvid_eval = Stage1TapVidDataset(
        split="eval_mini",
        minisplit_records=_records_for(minisplits, "tapvid", "eval_mini"),
    )

    rows = [
        ("pointodyssey_train", point_train, "PointOdyssey train sample"),
        ("pointodyssey_val", point_val, "PointOdyssey val sample"),
        ("kubric_train", kubric_train, "Kubric train sample"),
        ("kubric_val", kubric_val, "Kubric val sample"),
        ("tapvid_eval", tapvid_eval, "TAP-Vid eval sample"),
    ]

    out: List[Dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, ds, title in rows:
        sample = ds[0]
        out_path = out_dir / f"{key}.png"
        ok, note = _render_qualitative_sample(sample, title, out_path)
        out.append({
            "name": key,
            "ok": ok,
            "note": note,
            "output": str(out_path),
        })

    return out


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="TraceWM Stage1 iteration-1 trainer")
    p.add_argument("--dataset-choice", required=True, choices=["pointodyssey_only", "kubric_only", "joint_po_kubric"])
    p.add_argument("--run-name", default="")
    p.add_argument("--data-root", default="/home/chen034/workspace/data")
    p.add_argument("--splits-path", default="/home/chen034/workspace/data/_manifests/stage1_iter1_splits_20260408.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--seed", type=int, default=20260408)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--steps-per-epoch", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--free-loss-weight", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--eval-max-batches", type=int, default=0)
    p.add_argument("--eval-max-tapvid-samples", type=int, default=6)
    p.add_argument("--eval-max-tapvid3d-samples", type=int, default=12)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--qualitative-dir", default="")
    p.set_defaults(auto_resume=True)
    p.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    return p


def _choose_splits(dataset_choice: str) -> Tuple[List[str], str, str]:
    if dataset_choice == "pointodyssey_only":
        return ["pointodyssey"], "train_iter1_pointodyssey", "val_iter1_pointodyssey"
    if dataset_choice == "kubric_only":
        return ["kubric"], "train_iter1_kubric", "val_iter1_kubric"
    return ["pointodyssey", "kubric"], "train_iter1_joint", "val_iter1_joint"


def _checkpoint_payload(
    model: Iter1TraceModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    history: List[Dict[str, Any]],
    best_val_total_loss: float,
    args: Any,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "history": history,
        "best_val_total_loss": best_val_total_loss,
        "args": vars(args),
    }


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    run_name = args.run_name.strip() or f"tracewm_stage1_iter1_{args.dataset_choice}"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_names, train_split, val_split = _choose_splits(args.dataset_choice)

    train_dataset = Stage1UnifiedDataset(
        dataset_names=dataset_names,
        split=train_split,
        data_root=args.data_root,
        minisplit_path=args.splits_path,
        obs_len=8,
        fut_len=8,
    )
    val_dataset = Stage1UnifiedDataset(
        dataset_names=dataset_names,
        split=val_split,
        data_root=args.data_root,
        minisplit_path=args.splits_path,
        obs_len=8,
        fut_len=8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=stage1_collate_fn,
    )

    model = Iter1TraceModel(state_dim=5, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss(reduction="mean")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    history: List[Dict[str, Any]] = []
    best_val_total_loss = float("inf")

    latest_ckpt = ckpt_dir / "latest.pt"
    resume_path = Path(args.resume) if args.resume.strip() else None
    if (resume_path is not None and resume_path.exists()):
        payload = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
        best_val_total_loss = float(payload.get("best_val_total_loss", float("inf")))
        print(f"[iter1] resumed from --resume: {resume_path}")
    elif args.auto_resume and latest_ckpt.exists():
        payload = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
        best_val_total_loss = float(payload.get("best_val_total_loss", float("inf")))
        print(f"[iter1] resumed from latest: {latest_ckpt}")

    train_iter = itertools.cycle(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_teacher_vals: List[float] = []
        train_free_vals: List[float] = []
        train_total_vals: List[float] = []

        for _ in range(args.steps_per_epoch):
            batch = next(train_iter)
            obs, fut = build_state_batch(batch, device)

            pred_teacher = model.rollout_teacher_forced(obs, fut)
            pred_free = model.rollout_free(obs, fut.shape[1])

            loss_teacher = mse(pred_teacher, fut)
            loss_free = mse(pred_free, fut)
            loss_total = loss_teacher + args.free_loss_weight * loss_free

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_teacher_vals.append(float(loss_teacher.item()))
            train_free_vals.append(float(loss_free.item()))
            train_total_vals.append(float(loss_total.item()))
            global_step += 1

        train_metrics = {
            "train_teacher_forced_loss": float(sum(train_teacher_vals) / max(len(train_teacher_vals), 1)),
            "train_free_rollout_loss": float(sum(train_free_vals) / max(len(train_free_vals), 1)),
            "train_total_loss": float(sum(train_total_vals) / max(len(train_total_vals), 1)),
            "steps_this_epoch": args.steps_per_epoch,
        }

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            free_loss_weight=args.free_loss_weight,
            max_batches=args.eval_max_batches,
        )
        external_metrics = evaluate_external(
            model=model,
            splits_path=args.splits_path,
            data_root=args.data_root,
            device=device,
            max_tapvid_samples=args.eval_max_tapvid_samples,
            max_tapvid3d_samples=args.eval_max_tapvid3d_samples,
        )

        epoch_metrics: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            **train_metrics,
            **val_metrics,
            "tapvid": external_metrics["tapvid"],
            "tapvid3d": external_metrics["tapvid3d"],
        }
        history.append(epoch_metrics)

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            history=history,
            best_val_total_loss=best_val_total_loss,
            args=args,
        )

        epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, epoch_ckpt)
        torch.save(payload, latest_ckpt)

        current_val_total = float(val_metrics["val_total_loss"])
        if current_val_total < best_val_total_loss:
            best_val_total_loss = current_val_total
            payload_best = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                history=history,
                best_val_total_loss=best_val_total_loss,
                args=args,
            )
            torch.save(payload_best, ckpt_dir / "best.pt")

        print(
            f"[iter1][{run_name}] epoch={epoch} step={global_step} "
            f"train_total={train_metrics['train_total_loss']:.6f} "
            f"val_total={val_metrics['val_total_loss']:.6f} "
            f"tapvid_free_ep={external_metrics['tapvid']['free_rollout_endpoint_l2']:.6f} "
            f"tapvid3d_free_ep={external_metrics['tapvid3d']['free_rollout_endpoint_l2']:.6f}"
        )

    final_metrics = history[-1] if history else {}

    qualitative_dir = Path(args.qualitative_dir) if args.qualitative_dir else (output_dir / "qualitative")
    qualitative_results = render_qualitative_sets(
        out_dir=qualitative_dir,
        data_root=args.data_root,
        splits_path=args.splits_path,
    )

    summary = {
        "generated_at_utc": now_iso(),
        "task": "trace_only_future_trace_state_generation",
        "iteration_round": "stage1_iter1",
        "run_name": run_name,
        "dataset_choice": args.dataset_choice,
        "train_split": train_split,
        "val_split": val_split,
        "train_datasets": dataset_names,
        "teacher_forced_supported": True,
        "free_rollout_supported": True,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "total_steps": int(global_step),
        "batch_size": args.batch_size,
        "model": {
            "state_dim": 5,
            "hidden_dim": args.hidden_dim,
        },
        "loss_family": {
            "teacher_forced_loss": "mse",
            "free_rollout_loss": "mse",
            "total_loss": f"teacher_forced + {args.free_loss_weight} * free_rollout",
            "free_loss_weight": args.free_loss_weight,
        },
        "dataset_mix_info": {
            "train": dataset_mix_info(train_dataset),
            "val": dataset_mix_info(val_dataset),
        },
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epoch_history": history,
        "final_metrics": final_metrics,
        "best_val_total_loss": best_val_total_loss,
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_best": str(ckpt_dir / "best.pt"),
        "checkpoint_latest": str(latest_ckpt),
        "summary_json": str(Path(args.summary_json)),
        "splits_path": str(Path(args.splits_path)),
        "qualitative": qualitative_results,
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[iter1][{run_name}] wrote summary: {summary_path}")
    print(f"[iter1][{run_name}] checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
