#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader

from stwm.tracewm_v2.constants import STATE_DIM
from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset, stage1_v2_collate_fn
from stwm.tracewm_v2.losses.structured_trace_loss import StructuredTraceLoss, StructuredTraceLossConfig
from stwm.tracewm_v2.models.causal_trace_transformer import TraceCausalTransformer, build_tracewm_v2_config


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    parser = ArgumentParser(description="Torch profiler short-window run for Stage1-v2")
    parser.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    parser.add_argument("--preflight-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_preflight_20260408.json")
    parser.add_argument("--output-root", default="/home/chen034/workspace/stwm/outputs/profiler/stage1_v2_torch_profiler_20260408")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--debug-batch-size", type=int, default=2)
    parser.add_argument("--prototype-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples-per-dataset", type=int, default=64)
    return parser.parse_args()


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    non_block = bool(device.type == "cuda")
    return {
        "obs_state": batch["obs_state"].to(device, non_blocking=non_block),
        "fut_state": batch["fut_state"].to(device, non_blocking=non_block),
        "obs_valid": batch["obs_valid"].to(device, non_blocking=non_block),
        "fut_valid": batch["fut_valid"].to(device, non_blocking=non_block),
        "token_mask": batch["token_mask"].to(device, non_blocking=non_block),
    }


def _future_pred(pred: Dict[str, torch.Tensor], obs_len: int) -> Dict[str, torch.Tensor]:
    out = {
        "coord": pred["coord"][:, obs_len:],
        "vis_logit": pred["vis_logit"][:, obs_len:],
        "residual": pred["residual"][:, obs_len:],
        "velocity": pred["velocity"][:, obs_len:],
    }
    if "endpoint" in pred:
        out["endpoint"] = pred["endpoint"]
    return out


def _run_profile_one(
    preset: str,
    batch_size: int,
    args: Any,
    out_dir: Path,
) -> Dict[str, Any]:
    dataset = Stage1V2UnifiedDataset(
        dataset_names=["pointodyssey", "kubric"],
        split="train",
        contract_path=str(args.contract_path),
        obs_len=8,
        fut_len=8,
        max_tokens=64,
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=stage1_v2_collate_fn,
        pin_memory=True,
    )

    cfg = build_tracewm_v2_config(str(preset))
    if cfg.state_dim != STATE_DIM:
        raise ValueError("state_dim mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceCausalTransformer(cfg).to(device)
    loss_cfg = StructuredTraceLossConfig(
        coord_weight=1.0,
        visibility_weight=0.5,
        residual_weight=0.25,
        velocity_weight=0.25,
        endpoint_weight=0.1,
        enable_visibility=True,
        enable_residual=True,
        enable_velocity=True,
        enable_endpoint=False,
    )
    criterion = StructuredTraceLoss(loss_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    out_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_handler = tensorboard_trace_handler(str(out_dir), worker_name=f"stage1_v2_{preset}")

    status = "pass"
    error = ""
    done_steps = 0

    try:
        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for step, batch in enumerate(loader):
                data = _to_device(batch, device)
                full_state = torch.cat([data["obs_state"], data["fut_state"]], dim=1)
                full_valid = torch.cat([data["obs_valid"], data["fut_valid"]], dim=1)

                shifted = torch.zeros_like(full_state)
                shifted[:, 0] = full_state[:, 0]
                shifted[:, 1:] = full_state[:, :-1]

                pred = model(shifted, token_mask=data["token_mask"])
                losses = criterion(
                    pred=_future_pred(pred, obs_len=8),
                    target_state=full_state[:, 8:],
                    valid_mask=full_valid[:, 8:],
                    token_mask=data["token_mask"],
                )

                optimizer.zero_grad(set_to_none=True)
                losses["total_loss"].backward()
                optimizer.step()

                prof.step()
                done_steps = step + 1
                if done_steps >= int(args.max_steps):
                    break
    except Exception as exc:
        status = "fail"
        error = str(exc)

    return {
        "preset": str(preset),
        "status": status,
        "error": error,
        "steps": int(done_steps),
        "trace_dir": str(out_dir),
        "batch_size": int(batch_size),
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    preflight = json.loads(Path(args.preflight_json).read_text(encoding="utf-8"))
    preflight_pass = bool(preflight.get("preflight_pass", False))

    runs: List[Dict[str, Any]] = []
    runs.append(
        _run_profile_one(
            preset="debug_small",
            batch_size=int(args.debug_batch_size),
            args=args,
            out_dir=out_root / "debug_small",
        )
    )

    if preflight_pass:
        runs.append(
            _run_profile_one(
                preset="prototype_220m",
                batch_size=int(args.prototype_batch_size),
                args=args,
                out_dir=out_root / "prototype_220m",
            )
        )
    else:
        runs.append(
            {
                "preset": "prototype_220m",
                "status": "skipped",
                "error": "preflight_failed",
                "steps": 0,
                "trace_dir": str(out_root / "prototype_220m"),
                "batch_size": int(args.prototype_batch_size),
            }
        )

    summary = {
        "generated_at_utc": now_iso(),
        "output_root": str(out_root),
        "preflight_pass": preflight_pass,
        "runs": runs,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[stage1-v2-profiler] output_root={out_root}")
    print(f"[stage1-v2-profiler] summary={out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
