from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.models.stwm_v4_2 import STWMV42, estimate_v4_2_parameter_budget, load_model_config_v4_2
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.trace_adapter import TraceAdapter


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train/Eval STWM V4.2 minimal pipeline")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_smoke/full_v4_2")
    parser.add_argument("--run-name", default="full_v4_2")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--sample-limit", type=int, default=18)
    parser.add_argument("--model-preset", default="prototype_220m_v4_2")
    parser.add_argument("--preset-file", default="/home/chen034/workspace/stwm/code/stwm/configs/model_presets_v4_2.json")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--disable-semantics", action="store_true")
    parser.add_argument("--disable-identity-memory", action="store_true")
    parser.add_argument("--neutralize-object-bias", action="store_true")
    parser.add_argument("--use-teacher-priors", action="store_true")
    parser.add_argument("--enable-reconnect-loss", action="store_true")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--reconnect-window", type=int, default=3)
    parser.add_argument("--reconnect-threshold", type=float, default=0.20)
    parser.add_argument("--summary-name", default="smoke_summary.json")
    parser.add_argument("--log-name", default="train_log.jsonl")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-name", default="final_model.pt")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
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


def _reappearance_indices(visibility: torch.Tensor, min_gap: int = 1) -> list[int]:
    """Return local indices where target reappears after a visibility gap."""
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


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / str(args.log_name)
    summary_path = output_dir / str(args.summary_name)
    checkpoint_path = output_dir / str(args.checkpoint_name)

    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=args.sample_limit)
    samples = [s for s in dataset.samples if len(s.frame_paths) >= 4]
    if not samples:
        raise RuntimeError("No eligible samples for V4.2 smoke")

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
    model = STWMV42(config).to(device)
    if str(args.resume_checkpoint).strip():
        ckpt_path = Path(str(args.resume_checkpoint)).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        if isinstance(payload, dict) and "model_state" in payload:
            state = payload["model_state"]
        elif isinstance(payload, dict):
            state = payload
        else:
            raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)}")
        model.load_state_dict(state, strict=True)

    optimizer = None
    if not bool(args.eval_only):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

    model_parameters = int(sum(param.numel() for param in model.parameters()))
    rough_budget = int(estimate_v4_2_parameter_budget(config))

    prev_identity_cache: dict[str, torch.Tensor] = {}
    memory_state = None
    log_rows: list[dict[str, float | int | str]] = []

    for step in range(1, int(args.steps) + 1):
        if bool(args.eval_only):
            model.eval()
        else:
            model.train()
        sample = samples[(step - 1) % len(samples)]
        batch = _build_features_for_sample(
            sample,
            trace_adapter=trace_adapter,
            semantic_adapter=semantic_adapter,
            device=device,
            disable_semantics=bool(args.disable_semantics),
            use_teacher_priors=bool(args.use_teacher_priors),
        )

        use_memory = not bool(args.disable_identity_memory)
        model_prior_features = batch["prior_features"]
        model_teacher_objectness = batch["teacher_objectness"]
        if bool(args.neutralize_object_bias):
            # Keep parameter budget and heads unchanged; only neutralize tokenizer bias inputs.
            model_prior_features = torch.zeros_like(model_prior_features)
            model_teacher_objectness = torch.full_like(model_teacher_objectness, 0.5)

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
            mstate = outputs["memory_state"]
            memory_state = type(mstate)(
                keys=mstate.keys.detach(),
                values=mstate.values.detach(),
                valid_mask=mstate.valid_mask.detach(),
            )
        else:
            memory_state = None

        pred_traj = torch.sigmoid(outputs["trajectory"])
        traj_loss = F.smooth_l1_loss(pred_traj, batch["target_trajectory"])
        vis_loss = F.binary_cross_entropy_with_logits(outputs["visibility"], batch["target_visibility"])

        frame_l1 = torch.abs(pred_traj - batch["target_trajectory"]).mean(dim=-1)
        trajectory_l1 = float(frame_l1.mean().detach().cpu())

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
            prev_identity_cache[clip_id] = curr.detach().cpu()

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
        query_localization_error = float(frame_l1[0, query_frame_idx].detach().cpu())

        q_pred = pred_traj[0, query_frame_idx].detach().cpu()
        q_gt = batch["target_trajectory"][0, query_frame_idx].detach().cpu()

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
                    mins.append(float(torch.min(w).detach().cpu()))
            if mins:
                reconnect_min_error = float(min(mins))
                reconnect_success = float(any(x <= float(args.reconnect_threshold) for x in mins))

        if bool(args.enable_reconnect_loss) and use_memory:
            gate_mean = float(outputs["memory_diagnostics"].get("memory_gate_mean", 0.0))
            low_vis = (batch["target_visibility"][..., 0] < 0.5).float().mean()
            reconnect_target = low_vis.detach()
            reconnect_loss = torch.tensor((gate_mean - float(reconnect_target.cpu())) ** 2, device=device, dtype=torch.float32)

        total_loss = traj_loss + 0.25 * vis_loss + 0.5 * sem_loss + 0.25 * reid_loss + 0.25 * query_loss + 0.1 * reconnect_loss

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()

        attn_mean = token_attn.mean(dim=1)
        teacher_obj = batch["teacher_objectness"]
        low_obj = (teacher_obj < 0.3).float()
        high_obj = (teacher_obj >= 0.3).float()
        bg_attn = float((attn_mean * low_obj).sum().detach().cpu() / low_obj.sum().clamp(min=1.0).detach().cpu())
        fg_attn = float((attn_mean * high_obj).sum().detach().cpu() / high_obj.sum().clamp(min=1.0).detach().cpu())

        tdiag = outputs["tokenizer_diagnostics"]
        mdiag = outputs["memory_diagnostics"]
        row = {
            "step": int(step),
            "clip_id": str(batch["clip_id"]),
            "total_loss": float(total_loss.detach().cpu()),
            "trajectory_loss": float(traj_loss.detach().cpu()),
            "trajectory_l1": float(trajectory_l1),
            "visibility_loss": float(vis_loss.detach().cpu()),
            "semantic_loss": float(sem_loss.detach().cpu()),
            "reid_loss": float(reid_loss.detach().cpu()),
            "query_loss": float(query_loss.detach().cpu()),
            "query_localization_error": float(query_localization_error),
            "query_frame_idx": int(query_frame_idx),
            "query_token_index": int(query_token_index),
            "query_pred_x": float(q_pred[0].item()),
            "query_pred_y": float(q_pred[1].item()),
            "query_gt_x": float(q_gt[0].item()),
            "query_gt_y": float(q_gt[1].item()),
            "query_traj_gap": float(query_localization_error - trajectory_l1),
            "reconnect_loss": float(reconnect_loss.detach().cpu()),
            "has_reappearance_event": float(has_reappearance_event),
            "reappearance_count": int(reappearance_count),
            "reconnect_success": float(reconnect_success),
            "reconnect_min_error": float(reconnect_min_error),
            "assignment_entropy": float(tdiag.get("assignment_entropy", 0.0)),
            "token_usage_entropy": float(tdiag.get("token_usage_entropy", 0.0)),
            "objectness_mean": float(tdiag.get("objectness_mean", 0.0)),
            "memory_gate_mean": float(mdiag.get("memory_gate_mean", 0.0)),
            "bg_fg_attention_ratio": float(bg_attn / max(fg_attn, 1e-6)),
        }
        log_rows.append(row)
        with train_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")

    def _avg(key: str) -> float:
        vals = [float(x[key]) for x in log_rows]
        return float(sum(vals) / max(1, len(vals)))

    first = log_rows[0] if log_rows else {}
    last = log_rows[-1] if log_rows else {}
    summary = {
        "run_name": args.run_name,
        "mode": "eval" if bool(args.eval_only) else "train",
        "steps": int(args.steps),
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
        },
        "risk_flags": {
            "tokenizer_collapse_risk": bool(_avg("token_usage_entropy") < 0.25 or _avg("assignment_entropy") < 0.25),
            "background_bias_risk": bool(_avg("bg_fg_attention_ratio") > 1.1),
            "memory_inactive_risk": bool((not args.disable_identity_memory) and _avg("memory_gate_mean") < 0.1),
            "semantic_decorative_risk": bool((not args.disable_semantics) and float(last.get("semantic_loss", 0.0)) >= float(first.get("semantic_loss", 0.0))),
            "identity_decorative_risk": bool((not args.disable_identity_memory) and float(last.get("reid_loss", 0.0)) >= float(first.get("reid_loss", 0.0))),
        },
        "paths": {
            "train_log": str(train_log_path),
            "summary": str(summary_path),
            "checkpoint": str(checkpoint_path) if bool(args.save_checkpoint) else "",
        },
        "resume": {
            "resume_checkpoint": str(args.resume_checkpoint),
            "eval_only": bool(args.eval_only),
            "log_name": str(args.log_name),
        },
    }

    if bool(args.save_checkpoint):
        torch.save(
            {
                "run_name": args.run_name,
                "model_state": model.state_dict(),
                "model_config": asdict(config),
                "args": vars(args),
                "summary": summary,
            },
            checkpoint_path,
        )

    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
