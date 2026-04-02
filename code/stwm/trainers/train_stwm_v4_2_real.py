from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import cv2

    cv2.setNumThreads(0)
except Exception:
    # OpenCV may be absent in some environments; training should still run.
    cv2 = None

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json
import math
import random
import shutil
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.models.stwm_v4_2 import STWMV42, estimate_v4_2_parameter_budget, load_model_config_v4_2
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.trace_adapter import TraceAdapter


class _SampleDataset(Dataset):
    def __init__(self, samples: list[Any]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        return self.samples[index]


def _collate_samples(batch: list[Any]) -> list[Any]:
    return batch


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train/Eval STWM V4.2 real-budget pipeline")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real/full_v4_2")
    parser.add_argument("--run-name", default="full_v4_2")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--target-epochs", type=float, default=0.0)
    parser.add_argument("--min-optimizer-steps", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--sample-limit", type=int, default=18)
    parser.add_argument("--micro-batch-per-gpu", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--model-preset", default="prototype_220m_v4_2")
    parser.add_argument("--preset-file", default="/home/chen034/workspace/stwm/code/stwm/configs/model_presets_v4_2.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")

    parser.add_argument("--disable-semantics", action="store_true")
    parser.add_argument("--disable-identity-memory", action="store_true")
    parser.add_argument("--neutralize-object-bias", action="store_true")
    parser.add_argument("--use-teacher-priors", action="store_true")
    parser.add_argument("--enable-reconnect-loss", action="store_true")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--reconnect-window", type=int, default=3)
    parser.add_argument("--reconnect-threshold", type=float, default=0.20)

    parser.add_argument("--summary-name", default="mini_val_summary.json")
    parser.add_argument("--log-name", default="train_log.jsonl")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir-name", default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--milestone-interval", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--min-free-disk-gb", type=float, default=50.0)
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


def _read_mask_ratio(mask_path: str, target_label_id: int | None) -> float:
    p = Path(mask_path)
    if not p.exists():
        return 0.0
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if target_label_id is None:
        return float((arr > 0).mean())
    tgt = arr == int(target_label_id)
    if tgt.any():
        return float(tgt.mean())
    return float((arr > 0).mean())


def _build_features_for_sample(
    sample: Any,
    *,
    trace_adapter: TraceAdapter,
    semantic_adapter: SemanticAdapter,
    device: torch.device,
    disable_semantics: bool,
    use_teacher_priors: bool,
) -> dict[str, torch.Tensor | str | int]:
    trace_summary = trace_adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
    semantic_summary = semantic_adapter.encode(
        sample.text_labels,
        len(sample.frame_paths),
        metadata=sample.metadata,
        clip_id=sample.clip_id,
    )

    seq_len = int(min(trace_summary.centers.shape[0], semantic_summary.class_scores.shape[0]))
    if seq_len <= 1:
        raise RuntimeError(f"clip {sample.clip_id} has insufficient sequence length")

    centers = trace_summary.centers[:seq_len].to(device=device, dtype=torch.float32)
    velocities = trace_summary.velocities[:seq_len].to(device=device, dtype=torch.float32)
    visibility = trace_summary.visibility[:seq_len].to(device=device, dtype=torch.float32)
    trace_features = torch.cat([centers, velocities, visibility], dim=-1)

    sem_text = semantic_summary.text_embeddings[:seq_len].mean(dim=1).to(device=device, dtype=torch.float32)
    sem_scores = semantic_summary.class_scores[:seq_len].mean(dim=1).to(device=device, dtype=torch.float32)
    semantic_features = torch.cat([sem_text, sem_scores], dim=-1)
    if disable_semantics:
        semantic_features = torch.zeros_like(semantic_features)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    mask_paths = metadata.get("mask_paths") if isinstance(metadata.get("mask_paths"), list) else []
    target_label_id = metadata.get("target_label_id")
    try:
        target_label_id = int(target_label_id) if target_label_id is not None else None
    except (TypeError, ValueError):
        target_label_id = None

    mask_ratios: list[float] = []
    for i in range(seq_len):
        if i < len(mask_paths):
            mask_ratios.append(_read_mask_ratio(str(mask_paths[i]), target_label_id))
        else:
            mask_ratios.append(mask_ratios[-1] if mask_ratios else 0.0)
    mask_ratio_t = torch.tensor(mask_ratios, device=device, dtype=torch.float32)

    speed = torch.norm(velocities, dim=-1)
    semantic_conf = sem_scores.max(dim=-1).values
    prior_features = torch.stack(
        [
            mask_ratio_t.clamp(0.0, 1.0),
            visibility[:, 0].clamp(0.0, 1.0),
            speed.clamp(0.0, 1.0),
            semantic_conf.clamp(0.0, 1.0),
        ],
        dim=-1,
    )

    if use_teacher_priors:
        teacher_objectness = (0.6 * mask_ratio_t + 0.4 * visibility[:, 0]).clamp(0.0, 1.0)
    else:
        teacher_objectness = visibility[:, 0].clamp(0.0, 1.0)

    return {
        "clip_id": sample.clip_id,
        "seq_len": seq_len,
        "trace_features": trace_features.unsqueeze(0),
        "semantic_features": semantic_features.unsqueeze(0),
        "prior_features": prior_features.unsqueeze(0),
        "teacher_objectness": teacher_objectness.unsqueeze(0),
        "target_trajectory": centers.unsqueeze(0),
        "target_visibility": visibility.unsqueeze(0),
        "target_semantic_probs": sem_scores.unsqueeze(0),
    }


def _safe_zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def _detach_memory_state(memory_state: Any) -> Any:
    if memory_state is None:
        return None
    return type(memory_state)(
        keys=memory_state.keys.detach(),
        values=memory_state.values.detach(),
        valid_mask=memory_state.valid_mask.detach(),
    )


def _reappearance_indices(visibility: torch.Tensor, min_gap: int = 1) -> list[int]:
    flags = (visibility >= 0.5).to(dtype=torch.int32).detach().cpu().tolist()
    out: list[int] = []
    min_gap = max(1, int(min_gap))
    for i in range(1, len(flags)):
        if flags[i] != 1:
            continue
        j = i - 1
        gap = 0
        while j >= 0 and flags[j] == 0:
            gap += 1
            j -= 1
        had_visible_before = j >= 0 and flags[j] == 1
        if had_visible_before and gap >= min_gap:
            out.append(i)
    return out


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / float(1024**3)


def _estimate_checkpoint_budget_gb(model_parameters: int, train_mode: bool, total_steps: int, milestone_interval: int) -> tuple[float, float]:
    model_gb = float(model_parameters) * 4.0 / float(1024**3)
    optim_gb = (float(model_parameters) * 8.0 / float(1024**3)) if train_mode else 0.0
    estimated_per_checkpoint = model_gb + optim_gb + 0.25
    milestone_count = 0
    if milestone_interval > 0 and total_steps > 0:
        milestone_count = total_steps // milestone_interval
    estimated_retained = 2 + milestone_count
    estimated_max = estimated_per_checkpoint * float(estimated_retained)
    return estimated_per_checkpoint, estimated_max


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)

    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = output_dir / str(args.log_name)
    summary_path = output_dir / str(args.summary_name)
    checkpoint_dir = output_dir / str(args.checkpoint_dir_name)
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"

    sample_limit = int(args.sample_limit) if int(args.sample_limit) > 0 else None
    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=sample_limit)
    samples = [sample for sample in dataset.samples if len(sample.frame_paths) >= 4]
    if not samples:
        raise RuntimeError("No eligible samples for STWM V4.2 real training")

    micro_batch = max(1, int(args.micro_batch_per_gpu))
    grad_accum = max(1, int(args.grad_accum))
    effective_batch = micro_batch * grad_accum
    steps_per_epoch = int(math.ceil(len(samples) / float(max(1, effective_batch))))

    if bool(args.eval_only):
        total_steps = max(1, int(args.steps))
        epoch_steps = 0
    else:
        requested_steps = max(0, int(args.steps))
        epoch_steps = int(math.ceil(max(0.0, float(args.target_epochs)) * float(steps_per_epoch))) if float(args.target_epochs) > 0 else 0
        min_steps = max(requested_steps, int(args.min_optimizer_steps), epoch_steps)
        total_steps = min_steps
        if int(args.max_optimizer_steps) > 0:
            total_steps = min(total_steps, int(args.max_optimizer_steps))
        if total_steps <= 0:
            raise RuntimeError("total optimizer steps resolved to 0; provide positive --steps or budget constraints")

    sample_dataset = _SampleDataset(samples)
    loader_kwargs: dict[str, Any] = {
        "batch_size": micro_batch,
        "shuffle": True,
        "num_workers": max(0, int(args.num_workers)),
        "drop_last": False,
        "collate_fn": _collate_samples,
        "pin_memory": bool(args.pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)

    train_loader = DataLoader(sample_dataset, **loader_kwargs)
    data_iter = iter(train_loader)

    def next_micro_batch() -> list[Any]:
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    trace_adapter = TraceAdapter()
    semantic_adapter = SemanticAdapter()

    warmup = _build_features_for_sample(
        samples[0],
        trace_adapter=trace_adapter,
        semantic_adapter=semantic_adapter,
        device=device,
        disable_semantics=bool(args.disable_semantics),
        use_teacher_priors=bool(args.use_teacher_priors),
    )

    config = load_model_config_v4_2(
        args.model_preset,
        trace_dim=int(warmup["trace_features"].shape[-1]),
        semantic_dim=int(warmup["semantic_features"].shape[-1]),
        prior_dim=int(warmup["prior_features"].shape[-1]),
        preset_path=args.preset_file,
    )
    config.activation_checkpointing = bool(args.activation_checkpointing)
    model = STWMV42(config).to(device)

    optimizer = None
    if not bool(args.eval_only):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

    model_parameters = int(sum(param.numel() for param in model.parameters()))
    rough_budget = int(estimate_v4_2_parameter_budget(config))

    start_step = 0
    best_total_loss = float("inf")
    best_step = 0
    resolved_resume_path = ""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if str(args.resume_checkpoint).strip():
        resolved_resume_path = str(Path(str(args.resume_checkpoint)).expanduser())
    elif bool(args.auto_resume) and latest_checkpoint_path.exists():
        resolved_resume_path = str(latest_checkpoint_path)

    if resolved_resume_path:
        resume_path = Path(resolved_resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location=device)
        if isinstance(payload, dict) and "model_state" in payload:
            model.load_state_dict(payload["model_state"], strict=True)
            start_step = int(payload.get("step", 0))
            best_total_loss = float(payload.get("best_total_loss", float("inf")))
            best_step = int(payload.get("best_step", 0))
            if optimizer is not None and "optimizer_state" in payload and payload["optimizer_state"] is not None:
                optimizer.load_state_dict(payload["optimizer_state"])
        elif isinstance(payload, dict):
            model.load_state_dict(payload, strict=True)
            start_step = 0
        else:
            raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)}")

    if start_step >= total_steps:
        print(f"[stwm-v4.2-real] resume step {start_step} already reaches target steps {total_steps}; writing summary only")

    est_ckpt_gb, est_max_retained_gb = _estimate_checkpoint_budget_gb(
        model_parameters=model_parameters,
        train_mode=not bool(args.eval_only),
        total_steps=total_steps,
        milestone_interval=max(0, int(args.milestone_interval)),
    )

    checkpoint_interval = max(0, int(args.checkpoint_interval))
    milestone_interval = max(0, int(args.milestone_interval))

    if checkpoint_interval > 0:
        retention_text = f"latest_every_{checkpoint_interval}+best"
    else:
        retention_text = "latest_on_final+best"
    if milestone_interval > 0:
        retention_text = f"{retention_text}+milestone_every_{milestone_interval}"

    print(f"[stwm-v4.2-real] checkpoint_dir={checkpoint_dir}")
    print(f"[stwm-v4.2-real] retention_policy={retention_text}")
    print(f"[stwm-v4.2-real] checkpoint_interval={checkpoint_interval} milestone_interval={milestone_interval}")
    print(f"[stwm-v4.2-real] est_checkpoint_each_gb={est_ckpt_gb:.2f} est_max_retained_gb={est_max_retained_gb:.2f}")

    prev_identity_cache: dict[str, torch.Tensor] = {}
    memory_state = None
    log_rows: list[dict[str, float | int | str]] = []
    step_times: list[float] = []
    data_times: list[float] = []
    data_ratios: list[float] = []
    max_memory_gb_history: list[float] = []
    samples_processed = 0

    amp_enabled = bool(args.bf16 and device.type == "cuda")

    def maybe_save_checkpoint(step: int, step_total_loss: float, force_final: bool = False) -> None:
        nonlocal best_total_loss
        nonlocal best_step

        if not bool(args.save_checkpoint):
            return

        should_periodic = checkpoint_interval > 0 and step % checkpoint_interval == 0
        should_save = force_final or should_periodic
        if not should_save:
            return

        free_gb = _disk_free_gb(output_dir)
        if free_gb < float(args.min_free_disk_gb):
            print(
                f"[stwm-v4.2-real] skip checkpoint at step={step}: free_gb={free_gb:.2f} < min_free_disk_gb={float(args.min_free_disk_gb):.2f}"
            )
            return

        payload = {
            "step": int(step),
            "run_name": args.run_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "model_config": asdict(config),
            "args": vars(args),
            "best_total_loss": float(best_total_loss),
            "best_step": int(best_step),
        }

        torch.save(payload, latest_checkpoint_path)

        if step_total_loss <= best_total_loss:
            best_total_loss = float(step_total_loss)
            best_step = int(step)
            payload["best_total_loss"] = float(best_total_loss)
            payload["best_step"] = int(best_step)
            torch.save(payload, best_checkpoint_path)

        if milestone_interval > 0 and step % milestone_interval == 0:
            milestone_path = checkpoint_dir / f"milestone_step_{step:06d}.pt"
            torch.save(payload, milestone_path)

        if str(args.checkpoint_name).strip() and force_final:
            alias_path = output_dir / str(args.checkpoint_name)
            torch.save(payload, alias_path)

    if start_step < total_steps:
        for step in range(int(start_step) + 1, int(total_steps) + 1):
            step_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            if bool(args.eval_only):
                model.eval()
            else:
                model.train()

            data_time_s = 0.0
            metric_sum: dict[str, float] = {}
            metric_count = 0
            first_clip_id = ""

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            for _ in range(grad_accum):
                fetch_t0 = time.perf_counter()
                micro_samples = next_micro_batch()
                data_time_s += time.perf_counter() - fetch_t0

                if not isinstance(micro_samples, list):
                    micro_samples = list(micro_samples)

                micro_divisor = max(1, len(micro_samples))
                for sample in micro_samples:
                    if not first_clip_id:
                        first_clip_id = str(sample.clip_id)

                    feature_t0 = time.perf_counter()
                    batch = _build_features_for_sample(
                        sample,
                        trace_adapter=trace_adapter,
                        semantic_adapter=semantic_adapter,
                        device=device,
                        disable_semantics=bool(args.disable_semantics),
                        use_teacher_priors=bool(args.use_teacher_priors),
                    )
                    data_time_s += time.perf_counter() - feature_t0

                    use_memory = not bool(args.disable_identity_memory)
                    model_prior_features = batch["prior_features"]
                    model_teacher_objectness = batch["teacher_objectness"]
                    if bool(args.neutralize_object_bias):
                        model_prior_features = torch.zeros_like(model_prior_features)
                        model_teacher_objectness = torch.full_like(model_teacher_objectness, 0.5)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                        outputs = model(
                            trace_features=batch["trace_features"],
                            semantic_features=batch["semantic_features"],
                            prior_features=model_prior_features,
                            teacher_objectness=model_teacher_objectness,
                            memory_state=memory_state,
                            use_memory=use_memory,
                            update_memory=use_memory,
                        )

                        if use_memory:
                            memory_state = _detach_memory_state(outputs["memory_state"])
                        else:
                            memory_state = None

                        pred_traj = torch.sigmoid(outputs["trajectory"])
                        traj_loss = F.smooth_l1_loss(pred_traj, batch["target_trajectory"])
                        vis_loss = F.binary_cross_entropy_with_logits(outputs["visibility"], batch["target_visibility"])

                        frame_l1 = torch.abs(pred_traj - batch["target_trajectory"]).mean(dim=-1)
                        trajectory_l1 = float(frame_l1.mean().detach().float().cpu())

                        if bool(args.disable_semantics):
                            sem_loss = _safe_zero(device)
                        else:
                            token_attn = outputs["token_time_attention"]
                            token_targets = torch.einsum("bnt,btc->bnc", token_attn, batch["target_semantic_probs"])
                            token_targets = token_targets / token_targets.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                            sem_log_probs = F.log_softmax(outputs["semantic_logits"], dim=-1)
                            sem_loss = -(token_targets * sem_log_probs).sum(dim=-1).mean()

                        if bool(args.disable_identity_memory):
                            reid_loss = _safe_zero(device)
                        else:
                            clip_id = str(batch["clip_id"])
                            curr = outputs["identity_embeddings"][0]
                            prev = prev_identity_cache.get(clip_id)
                            if prev is not None:
                                prev = prev.to(device=device, dtype=curr.dtype)
                                n = min(curr.shape[0], prev.shape[0])
                                logits = torch.matmul(curr[:n], prev[:n].transpose(0, 1)) / float(args.contrastive_temperature)
                                labels = torch.arange(n, device=device)
                                reid_loss = F.cross_entropy(logits, labels)
                            else:
                                reid_loss = _safe_zero(device)
                            prev_identity_cache[clip_id] = curr.detach().float().cpu()

                        token_attn = outputs["token_time_attention"]
                        q_idx = torch.argmax(batch["teacher_objectness"], dim=-1)
                        gather = torch.gather(
                            token_attn,
                            dim=-1,
                            index=q_idx.view(-1, 1, 1).expand(token_attn.shape[0], token_attn.shape[1], 1),
                        ).squeeze(-1)
                        query_loss = -torch.log(gather.max(dim=1).values.clamp(min=1e-6)).mean()

                        query_token_index = int(torch.argmax(outputs["query_token_logits"][0]).item())
                        query_frame_scores = token_attn[0, query_token_index]
                        query_frame_idx = int(torch.argmax(query_frame_scores).item())
                        query_frame_idx = max(0, min(query_frame_idx, frame_l1.shape[1] - 1))
                        query_localization_error = float(frame_l1[0, query_frame_idx].detach().float().cpu())

                        q_pred = pred_traj[0, query_frame_idx].detach().float().cpu()
                        q_gt = batch["target_trajectory"][0, query_frame_idx].detach().float().cpu()

                        reconnect_loss = _safe_zero(device)
                        reconnect_success = 0.0
                        reconnect_min_error = 0.0
                        reappearance_count = 0
                        has_reappearance_event = 0.0

                        reappear = _reappearance_indices(batch["target_visibility"][0, :, 0], min_gap=1)
                        if reappear:
                            has_reappearance_event = 1.0
                            reappearance_count = int(len(reappear))
                            mins: list[float] = []
                            for idx in reappear:
                                end = min(frame_l1.shape[1], int(idx) + max(1, int(args.reconnect_window)))
                                w = frame_l1[0, int(idx):end]
                                if w.numel() > 0:
                                    mins.append(float(torch.min(w).detach().float().cpu()))
                            if mins:
                                reconnect_min_error = float(min(mins))
                                reconnect_success = float(any(x <= float(args.reconnect_threshold) for x in mins))

                        if bool(args.enable_reconnect_loss) and use_memory:
                            gate_mean = float(outputs["memory_diagnostics"].get("memory_gate_mean", 0.0))
                            low_vis = (batch["target_visibility"][..., 0] < 0.5).float().mean()
                            reconnect_target = low_vis.detach()
                            reconnect_loss = (reconnect_target - float(gate_mean)) ** 2

                        total_loss = (
                            traj_loss
                            + 0.25 * vis_loss
                            + 0.5 * sem_loss
                            + 0.25 * reid_loss
                            + 0.25 * query_loss
                            + 0.1 * reconnect_loss
                        )

                    loss_div = float(grad_accum * micro_divisor)
                    loss_scaled = total_loss / max(1.0, loss_div)
                    if optimizer is not None:
                        loss_scaled.backward()

                    attn_mean = token_attn.mean(dim=1)
                    teacher_obj = batch["teacher_objectness"]
                    low_obj = (teacher_obj < 0.3).float()
                    high_obj = (teacher_obj >= 0.3).float()
                    bg_attn = float((attn_mean * low_obj).sum().detach().float().cpu() / low_obj.sum().clamp(min=1.0).detach().float().cpu())
                    fg_attn = float((attn_mean * high_obj).sum().detach().float().cpu() / high_obj.sum().clamp(min=1.0).detach().float().cpu())

                    tdiag = outputs["tokenizer_diagnostics"]
                    mdiag = outputs["memory_diagnostics"]

                    row_values = {
                        "total_loss": float(total_loss.detach().float().cpu()),
                        "trajectory_loss": float(traj_loss.detach().float().cpu()),
                        "trajectory_l1": float(trajectory_l1),
                        "visibility_loss": float(vis_loss.detach().float().cpu()),
                        "semantic_loss": float(sem_loss.detach().float().cpu()),
                        "reid_loss": float(reid_loss.detach().float().cpu()),
                        "query_loss": float(query_loss.detach().float().cpu()),
                        "query_localization_error": float(query_localization_error),
                        "query_frame_idx": float(query_frame_idx),
                        "query_token_index": float(query_token_index),
                        "query_pred_x": float(q_pred[0].item()),
                        "query_pred_y": float(q_pred[1].item()),
                        "query_gt_x": float(q_gt[0].item()),
                        "query_gt_y": float(q_gt[1].item()),
                        "query_traj_gap": float(query_localization_error - trajectory_l1),
                        "reconnect_loss": float(reconnect_loss.detach().float().cpu()),
                        "has_reappearance_event": float(has_reappearance_event),
                        "reappearance_count": float(reappearance_count),
                        "reconnect_success": float(reconnect_success),
                        "reconnect_min_error": float(reconnect_min_error),
                        "assignment_entropy": float(tdiag.get("assignment_entropy", 0.0)),
                        "token_usage_entropy": float(tdiag.get("token_usage_entropy", 0.0)),
                        "objectness_mean": float(tdiag.get("objectness_mean", 0.0)),
                        "memory_gate_mean": float(mdiag.get("memory_gate_mean", 0.0)),
                        "bg_fg_attention_ratio": float(bg_attn / max(fg_attn, 1e-6)),
                    }
                    for key, value in row_values.items():
                        metric_sum[key] = metric_sum.get(key, 0.0) + float(value)
                    metric_count += 1
                    samples_processed += 1

            grad_norm = 0.0
            if optimizer is not None:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                grad_norm = float(grad_norm_t.detach().float().cpu())
                optimizer.step()

            step_time_s = float(time.perf_counter() - step_start)
            data_wait_ratio = float(data_time_s / step_time_s) if step_time_s > 0 else 0.0
            if device.type == "cuda":
                peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / float(1024**3))
            else:
                peak_memory_gb = 0.0
            disk_free_gb = _disk_free_gb(output_dir)

            metric_div = float(max(1, metric_count))
            row = {
                "step": int(step),
                "clip_id": str(first_clip_id or samples[(step - 1) % len(samples)].clip_id),
                "effective_batch": int(effective_batch),
                "micro_batch_per_gpu": int(micro_batch),
                "grad_accum": int(grad_accum),
                "num_workers": int(loader_kwargs["num_workers"]),
                "prefetch_factor": int(loader_kwargs.get("prefetch_factor", 0)),
                "persistent_workers": int(bool(loader_kwargs.get("persistent_workers", False))),
                "pin_memory": int(bool(loader_kwargs.get("pin_memory", False))),
                "bf16": int(bool(args.bf16)),
                "activation_checkpointing": int(bool(args.activation_checkpointing)),
                "step_time_s": float(step_time_s),
                "data_time_s": float(data_time_s),
                "data_wait_ratio": float(data_wait_ratio),
                "gpu_peak_memory_gb": float(peak_memory_gb),
                "disk_free_gb": float(disk_free_gb),
                "grad_norm": float(grad_norm),
            }
            for key, value in metric_sum.items():
                row[key] = float(value / metric_div)

            log_rows.append(row)
            step_times.append(float(step_time_s))
            data_times.append(float(data_time_s))
            data_ratios.append(float(data_wait_ratio))
            max_memory_gb_history.append(float(peak_memory_gb))

            with train_log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row) + "\n")

            maybe_save_checkpoint(step=step, step_total_loss=float(row.get("total_loss", 0.0)), force_final=(step == total_steps))

    if bool(args.save_checkpoint) and latest_checkpoint_path.exists():
        if not best_checkpoint_path.exists():
            payload = torch.load(latest_checkpoint_path, map_location="cpu")
            torch.save(payload, best_checkpoint_path)

    def _avg(key: str) -> float:
        vals = [float(x.get(key, 0.0)) for x in log_rows]
        return float(sum(vals) / max(1, len(vals)))

    def _pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    first = log_rows[0] if log_rows else {}
    last = log_rows[-1] if log_rows else {}

    summary = {
        "run_name": args.run_name,
        "mode": "eval" if bool(args.eval_only) else "train",
        "steps": int(total_steps),
        "model_preset": args.model_preset,
        "model_parameters": model_parameters,
        "rough_parameter_budget": rough_budget,
        "ablation": {
            "disable_semantics": bool(args.disable_semantics),
            "disable_identity_memory": bool(args.disable_identity_memory),
            "neutralize_object_bias": bool(args.neutralize_object_bias),
            "use_teacher_priors": bool(args.use_teacher_priors),
            "enable_reconnect_loss": bool(args.enable_reconnect_loss),
        },
        "average_losses": {
            "total": _avg("total_loss"),
            "trajectory": _avg("trajectory_loss"),
            "trajectory_l1": _avg("trajectory_l1"),
            "visibility": _avg("visibility_loss"),
            "semantic": _avg("semantic_loss"),
            "reid": _avg("reid_loss"),
            "query": _avg("query_loss"),
            "query_localization_error": _avg("query_localization_error"),
            "query_traj_gap": _avg("query_traj_gap"),
        },
        "diagnostics": {
            "assignment_entropy": _avg("assignment_entropy"),
            "token_usage_entropy": _avg("token_usage_entropy"),
            "objectness_mean": _avg("objectness_mean"),
            "memory_gate_mean": _avg("memory_gate_mean"),
            "bg_fg_attention_ratio": _avg("bg_fg_attention_ratio"),
            "reappearance_event_ratio": _avg("has_reappearance_event"),
            "reconnect_success_rate": _avg("reconnect_success"),
            "reconnect_min_error": _avg("reconnect_min_error"),
            "gpu_peak_memory_gb": _avg("gpu_peak_memory_gb"),
            "data_wait_ratio": _avg("data_wait_ratio"),
        },
        "risk_flags": {
            "tokenizer_collapse_risk": bool(_avg("token_usage_entropy") < 0.25 or _avg("assignment_entropy") < 0.25),
            "background_bias_risk": bool(_avg("bg_fg_attention_ratio") > 1.1),
            "memory_inactive_risk": bool((not args.disable_identity_memory) and _avg("memory_gate_mean") < 0.1),
            "semantic_decorative_risk": bool((not args.disable_semantics) and float(last.get("semantic_loss", 0.0)) >= float(first.get("semantic_loss", 0.0))),
            "identity_decorative_risk": bool((not args.disable_identity_memory) and float(last.get("reid_loss", 0.0)) >= float(first.get("reid_loss", 0.0))),
        },
        "budget": {
            "sample_count": int(len(samples)),
            "steps_per_epoch": int(steps_per_epoch),
            "target_epochs": float(args.target_epochs),
            "steps_for_target_epochs": int(epoch_steps),
            "min_optimizer_steps": int(args.min_optimizer_steps),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "requested_steps": int(args.steps),
            "resolved_optimizer_steps": int(total_steps),
            "micro_batch_per_gpu": int(micro_batch),
            "grad_accum": int(grad_accum),
            "effective_batch": int(effective_batch),
        },
        "precision": {
            "bf16": bool(args.bf16),
            "activation_checkpointing": bool(args.activation_checkpointing),
            "device": str(device),
        },
        "dataloader": {
            "num_workers": int(loader_kwargs["num_workers"]),
            "prefetch_factor": int(loader_kwargs.get("prefetch_factor", 0)),
            "persistent_workers": bool(loader_kwargs.get("persistent_workers", False)),
            "pin_memory": bool(loader_kwargs.get("pin_memory", False)),
        },
        "runtime": {
            "step_time_p50_s": _pct(step_times, 50.0),
            "step_time_p95_s": _pct(step_times, 95.0),
            "data_time_p50_s": _pct(data_times, 50.0),
            "data_wait_ratio_p50": _pct(data_ratios, 50.0),
            "data_wait_ratio_p95": _pct(data_ratios, 95.0),
            "gpu_peak_memory_gb_max": float(max(max_memory_gb_history) if max_memory_gb_history else 0.0),
            "samples_processed": int(samples_processed),
        },
        "checkpoint_policy": {
            "checkpoint_dir": str(checkpoint_dir),
            "retention": retention_text,
            "checkpoint_interval": int(checkpoint_interval),
            "milestone_interval": int(milestone_interval),
            "latest": str(latest_checkpoint_path) if latest_checkpoint_path.exists() else "",
            "best": str(best_checkpoint_path) if best_checkpoint_path.exists() else "",
            "min_free_disk_gb": float(args.min_free_disk_gb),
            "estimated_checkpoint_each_gb": float(est_ckpt_gb),
            "estimated_max_retained_gb": float(est_max_retained_gb),
        },
        "paths": {
            "train_log": str(train_log_path),
            "summary": str(summary_path),
            "checkpoint": str(latest_checkpoint_path) if latest_checkpoint_path.exists() else "",
        },
        "resume": {
            "requested_resume_checkpoint": str(args.resume_checkpoint),
            "auto_resume": bool(args.auto_resume),
            "resolved_resume_checkpoint": str(resolved_resume_path),
            "start_step": int(start_step),
            "best_step": int(best_step),
            "best_total_loss": float(best_total_loss if math.isfinite(best_total_loss) else _avg("total_loss")),
            "eval_only": bool(args.eval_only),
            "log_name": str(args.log_name),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
