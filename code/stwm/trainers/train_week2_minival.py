from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json
import random
from typing import Any

import numpy as np
import torch

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.evaluators.eval_mini_val import evaluate_model
from stwm.models.stwm_1b import STWM1B, load_model_config
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.tokenizer import SemanticTrajectoryTokenizer
from stwm.modules.trace_adapter import TraceAdapter
from stwm.utils.week2_protocol import (
    ablation_from_args,
    build_supervision_targets,
    build_tokens_for_sample,
    compute_training_losses,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Week-2 short training with integrated mini-val")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/training/week2_minival/full")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--obs-steps", type=int, default=8)
    parser.add_argument("--pred-steps", type=int, default=8)
    parser.add_argument("--val-max-clips", type=int, default=20)
    parser.add_argument("--train-max-clips", type=int, default=32)
    parser.add_argument("--val-clip-ids-path", default="")
    parser.add_argument("--protocol-version", default="v1", choices=["v1", "v2", "v2_1", "v2_2", "v2_3"])
    parser.add_argument("--query-candidates", type=int, default=5)
    parser.add_argument("--query-hit-radius", type=float, default=0.08)
    parser.add_argument("--query-topk", type=int, default=1)
    parser.add_argument("--identity-hit-radius", type=float, default=0.03)
    parser.add_argument("--occlusion-recovery-window", type=int, default=3)
    parser.add_argument("--query-hard-negative-jitter", type=float, default=0.03)
    parser.add_argument("--identity-target-overlap-min", type=float, default=0.02)
    parser.add_argument("--identity-other-overlap-min", type=float, default=0.02)
    parser.add_argument("--occlusion-min-disappear-frames", type=int, default=2)
    parser.add_argument("--query-near-negative-count", type=int, default=1)
    parser.add_argument("--identity-consistency-window", type=int, default=3)
    parser.add_argument("--query-min-plausible-same-class", type=int, default=2)
    parser.add_argument("--occlusion-reconnect-distance", type=float, default=0.18)
    parser.add_argument("--occlusion-reconnect-target-overlap-min", type=float, default=0.01)

    parser.add_argument("--model-preset", default="prototype_220m")
    parser.add_argument("--preset-file", default="/home/chen034/workspace/stwm/code/stwm/configs/model_presets.json")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--disable-semantics", action="store_true")
    parser.add_argument("--disable-trajectory", action="store_true")
    parser.add_argument("--disable-identity-memory", action="store_true")
    parser.add_argument("--identity-memory-dim", type=int, default=8)
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _eligible_samples(dataset: STWMDataset, obs_steps: int, pred_steps: int) -> list[Any]:
    needed = obs_steps + pred_steps
    out = []
    for sample in dataset.samples:
        if len(sample.frame_paths) < needed:
            continue
        mask_paths = sample.metadata.get("mask_paths", [])
        if not isinstance(mask_paths, list) or len(mask_paths) < needed:
            continue
        out.append(sample)
    return out


def _split_train_val(
    samples: list[Any],
    val_max_clips: int,
    train_max_clips: int,
    explicit_val_ids: set[str] | None = None,
) -> tuple[list[Any], list[Any]]:
    vspw_samples = sorted(
        [sample for sample in samples if sample.metadata.get("dataset", "").lower() == "vspw"],
        key=lambda item: item.clip_id,
    )
    if explicit_val_ids:
        val_samples = [sample for sample in vspw_samples if sample.clip_id in explicit_val_ids]
        val_samples = sorted(val_samples, key=lambda item: item.clip_id)
        if val_max_clips > 0:
            val_samples = val_samples[: max(1, val_max_clips)]
    else:
        val_samples = vspw_samples[: max(1, val_max_clips)]
    val_ids = {sample.clip_id for sample in val_samples}

    train_candidates = [sample for sample in samples if sample.clip_id not in val_ids]
    if not train_candidates:
        # Keep pipeline alive if split is too small.
        train_candidates = list(vspw_samples[val_max_clips:])
    if not train_candidates:
        train_candidates = list(samples)

    if train_max_clips > 0:
        train_candidates = train_candidates[:train_max_clips]

    return train_candidates, val_samples


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)

    device = _resolve_device(args.device)
    ablation = ablation_from_args(args)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "eval"
    case_dir_root = eval_dir / "cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    case_dir_root.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "run_name": args.run_name,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "eval_interval": int(args.eval_interval),
        "save_interval": int(args.save_interval),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "obs_steps": int(args.obs_steps),
        "pred_steps": int(args.pred_steps),
        "val_max_clips": int(args.val_max_clips),
        "train_max_clips": int(args.train_max_clips),
        "val_clip_ids_path": args.val_clip_ids_path,
        "protocol_version": args.protocol_version,
        "query_candidates": int(args.query_candidates),
        "query_hit_radius": float(args.query_hit_radius),
        "query_topk": int(args.query_topk),
        "identity_hit_radius": float(args.identity_hit_radius),
        "occlusion_recovery_window": int(args.occlusion_recovery_window),
        "query_hard_negative_jitter": float(args.query_hard_negative_jitter),
        "identity_target_overlap_min": float(args.identity_target_overlap_min),
        "identity_other_overlap_min": float(args.identity_other_overlap_min),
        "occlusion_min_disappear_frames": int(args.occlusion_min_disappear_frames),
        "query_near_negative_count": int(args.query_near_negative_count),
        "identity_consistency_window": int(args.identity_consistency_window),
        "query_min_plausible_same_class": int(args.query_min_plausible_same_class),
        "occlusion_reconnect_distance": float(args.occlusion_reconnect_distance),
        "occlusion_reconnect_target_overlap_min": float(args.occlusion_reconnect_target_overlap_min),
        "model_preset": args.model_preset,
        "preset_file": args.preset_file,
        "device": str(device),
        "ablation": asdict(ablation),
        "loss_formula": "trajectory + 0.5*visibility + 0.2*semantic + 0.1*temporal_consistency",
    }

    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=None)
    eligible = _eligible_samples(dataset, obs_steps=args.obs_steps, pred_steps=args.pred_steps)
    if not eligible:
        raise RuntimeError("No eligible samples for short training")

    explicit_val_ids: set[str] | None = None
    if args.val_clip_ids_path:
        val_ids_payload = json.loads(Path(args.val_clip_ids_path).read_text())
        if not isinstance(val_ids_payload, list):
            raise RuntimeError("val clip ids file must be a JSON list")
        explicit_val_ids = {str(x) for x in val_ids_payload if str(x).strip()}

    train_samples, val_samples = _split_train_val(
        eligible,
        val_max_clips=args.val_max_clips,
        train_max_clips=args.train_max_clips,
        explicit_val_ids=explicit_val_ids,
    )
    if not train_samples:
        raise RuntimeError("Training split is empty")
    if not val_samples:
        raise RuntimeError("Validation split is empty")

    config_snapshot["num_train_samples"] = len(train_samples)
    config_snapshot["num_val_samples"] = len(val_samples)
    config_snapshot["val_clip_ids"] = [sample.clip_id for sample in val_samples]
    (output_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2))

    trace_adapter = TraceAdapter()
    semantic_adapter = SemanticAdapter()
    tokenizer = SemanticTrajectoryTokenizer()

    warmup = build_tokens_for_sample(
        train_samples[0],
        trace_adapter,
        semantic_adapter,
        tokenizer,
        ablation,
        device,
    )
    model_config = load_model_config(
        preset=args.model_preset,
        input_dim=warmup.tokens.shape[-1],
        preset_path=args.preset_file,
    )
    model = STWM1B(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    train_log_path = output_dir / "train_log.jsonl"
    final_summary_path = eval_dir / "mini_val_summary_last.json"

    def run_eval(step: int) -> dict[str, Any]:
        step_case_dir = case_dir_root / f"step_{step:05d}"
        summary = evaluate_model(
            model,
            val_samples,
            ablation=ablation,
            obs_steps=args.obs_steps,
            pred_steps=args.pred_steps,
            device=device,
            run_name=args.run_name,
            trace_adapter=trace_adapter,
            semantic_adapter=semantic_adapter,
            tokenizer=tokenizer,
            case_output_dir=step_case_dir,
            save_case_limit=args.val_max_clips,
            protocol_version=args.protocol_version,
            query_candidates=int(args.query_candidates),
            query_hit_radius=float(args.query_hit_radius),
            query_topk=int(args.query_topk),
            identity_hit_radius=float(args.identity_hit_radius),
            occlusion_recovery_window=int(args.occlusion_recovery_window),
            query_hard_negative_jitter=float(args.query_hard_negative_jitter),
            identity_target_overlap_min=float(args.identity_target_overlap_min),
            identity_other_overlap_min=float(args.identity_other_overlap_min),
            occlusion_min_disappear_frames=int(args.occlusion_min_disappear_frames),
            query_near_negative_count=int(args.query_near_negative_count),
            identity_consistency_window=int(args.identity_consistency_window),
            query_min_plausible_same_class=int(args.query_min_plausible_same_class),
            occlusion_reconnect_distance=float(args.occlusion_reconnect_distance),
            occlusion_reconnect_target_overlap_min=float(args.occlusion_reconnect_target_overlap_min),
        )
        summary["step"] = int(step)
        summary["model_config"] = {
            "input_dim": int(model.config.input_dim),
            "hidden_size": int(model.config.hidden_size),
            "num_layers": int(model.config.num_layers),
            "num_heads": int(model.config.num_heads),
            "semantic_dim": int(model.config.semantic_dim),
        }
        summary["train_split_size"] = len(train_samples)
        summary["val_split_size"] = len(val_samples)

        step_summary_path = eval_dir / f"mini_val_summary_step_{step:05d}.json"
        step_summary_path.write_text(json.dumps(summary, indent=2))
        final_summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    best_mask_iou = -1.0
    best_step = 0
    last_eval_summary: dict[str, Any] | None = None

    for step in range(1, int(args.steps) + 1):
        model.train()
        sample = train_samples[(step - 1) % len(train_samples)]
        token_result = build_tokens_for_sample(
            sample,
            trace_adapter,
            semantic_adapter,
            tokenizer,
            ablation,
            device,
        )

        outputs = model(token_result.tokens)
        targets = build_supervision_targets(
            outputs,
            token_result.trace_summary,
            token_result.semantic_summary,
            device,
        )
        total_loss, loss_items = compute_training_losses(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
        optimizer.step()

        log_entry = {
            "step": int(step),
            "clip_id": sample.clip_id,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **loss_items,
        }
        with train_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_entry) + "\n")

        should_eval = (step % int(args.eval_interval) == 0) or (step == int(args.steps))
        should_save = (step % int(args.save_interval) == 0) or should_eval

        if should_eval:
            last_eval_summary = run_eval(step)
            current_iou = float(last_eval_summary["metrics"]["future_mask_iou"])
            if current_iou >= best_mask_iou:
                best_mask_iou = current_iou
                best_step = step

        if should_save:
            checkpoint = {
                "step": int(step),
                "run_name": args.run_name,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": asdict(model.config),
                "ablation": asdict(ablation),
                "token_layout": token_result.token_layout,
                "protocol": {
                    "obs_steps": int(args.obs_steps),
                    "pred_steps": int(args.pred_steps),
                    "protocol_version": args.protocol_version,
                    "query_candidates": int(args.query_candidates),
                    "query_hit_radius": float(args.query_hit_radius),
                    "query_topk": int(args.query_topk),
                    "identity_hit_radius": float(args.identity_hit_radius),
                    "occlusion_recovery_window": int(args.occlusion_recovery_window),
                    "query_hard_negative_jitter": float(args.query_hard_negative_jitter),
                    "identity_target_overlap_min": float(args.identity_target_overlap_min),
                    "identity_other_overlap_min": float(args.identity_other_overlap_min),
                    "occlusion_min_disappear_frames": int(args.occlusion_min_disappear_frames),
                    "query_near_negative_count": int(args.query_near_negative_count),
                    "identity_consistency_window": int(args.identity_consistency_window),
                    "query_min_plausible_same_class": int(args.query_min_plausible_same_class),
                    "occlusion_reconnect_distance": float(args.occlusion_reconnect_distance),
                    "occlusion_reconnect_target_overlap_min": float(args.occlusion_reconnect_target_overlap_min),
                },
            }
            checkpoint_path = checkpoint_dir / f"step_{step:05d}.pt"
            torch.save(checkpoint, checkpoint_path)

    if last_eval_summary is None:
        last_eval_summary = run_eval(int(args.steps))

    report = {
        "run_name": args.run_name,
        "best_step_by_future_mask_iou": int(best_step),
        "best_future_mask_iou": float(best_mask_iou),
        "final_eval_summary_path": str(final_summary_path),
        "final_metrics": last_eval_summary.get("metrics", {}),
        "checkpoint_dir": str(checkpoint_dir),
        "train_log": str(train_log_path),
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
