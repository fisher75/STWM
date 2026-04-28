#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from stwm.tracewm_v2.constants import STATE_DIM
from stwm.tracewm_v2.models.causal_trace_transformer import (
    TraceCausalTransformer,
    build_tracewm_v2_config,
)
from stwm.tracewm_v2.tools.run_stage1_v2_scientific_revalidation import _load_runtime_config
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    stage2_semantic_collate_fn,
)
from stwm.tracewm_v2_stage2.models.semantic_encoder import SemanticEncoder, SemanticEncoderConfig
from stwm.tracewm_v2_stage2.models.semantic_fusion import SemanticFusion, SemanticFusionConfig
from stwm.tracewm_v2_stage2.models.trace_unit_broadcast import (
    TraceUnitBroadcast,
    TraceUnitBroadcastConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_factorized_state import (
    TraceUnitFactorizedState,
    TraceUnitFactorizedStateConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_handshake import (
    TraceUnitHandshake,
    TraceUnitHandshakeConfig,
)
from stwm.tracewm_v2_stage2.models.trace_unit_tokenizer import (
    TraceUnitTokenizer,
    TraceUnitTokenizerConfig,
)
from stwm.tracewm_v2_stage2.models.semantic_trace_world_head import (
    FutureSemanticStateLossConfig,
    SemanticTraceStateHead,
    SemanticTraceStateHeadConfig,
    compute_future_semantic_state_losses,
)
from stwm.tracewm_v2_stage2.models.semantic_state_feedback import (
    SemanticStateFeedbackAdapter,
    SemanticStateFeedbackConfig,
)
from stwm.tracewm_v2_stage2.utils.visibility_reappearance_targets import build_future_visibility_reappearance_targets


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _apply_process_title_normalization() -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode != "generic":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", "python")).strip() or "python"
    lowered = title.lower()
    if "stwm" in lowered or "tracewm" in lowered or "/home/" in lowered:
        title = "python"
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 small-train trainer on frozen Stage1 backbone")
    p.add_argument(
        "--stage2-contract-path",
        default="/home/chen034/workspace/stwm/reports/stage2_bootstrap_data_contract_20260408.json",
    )
    p.add_argument(
        "--recommended-runtime-json",
        default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json",
    )
    p.add_argument("--use-recommended-runtime", action="store_true")
    p.add_argument("--predecode-cache-path", default="")

    p.add_argument(
        "--stage1-backbone-checkpoint",
        default="/home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt",
    )
    p.add_argument("--stage1-model-preset", default="prototype_220m")

    p.add_argument("--dataset-names", nargs="*", default=["vspw", "vipseg"])
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-samples-train", type=int, default=24)
    p.add_argument("--max-samples-val", type=int, default=12)
    p.add_argument("--semantic-patch-radius", type=int, default=12)
    p.add_argument("--stage1-partial-unfreeze-mode", default="none", choices=["none", "topblock"])
    p.add_argument("--stage1-partial-unfreeze-layer-count", type=int, default=1)
    p.add_argument("--stage1-partial-unfreeze-lr-scale", type=float, default=0.10)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)

    p.add_argument("--train-steps", type=int, default=240)
    p.add_argument("--eval-interval", type=int, default=40)
    p.add_argument("--eval-max-batches", type=int, default=6)
    p.add_argument("--save-every-n-steps", type=int, default=1000)

    p.add_argument("--semantic-hidden-dim", type=int, default=256)
    p.add_argument("--semantic-embed-dim", type=int, default=256)
    p.add_argument("--semantic-source-mainline", default="crop_visual_encoder")
    p.add_argument("--legacy-semantic-source", default="hand_crafted_stats")
    p.add_argument("--semantic-crop-size", type=int, default=64)
    p.add_argument("--local-temporal-window", type=int, default=1)
    p.add_argument("--local-temporal-fuse-weight", type=float, default=0.5)
    p.add_argument("--max-entities-per-sample", type=int, default=8)
    p.add_argument("--teacher-semantic-cache-path", default="")
    p.add_argument(
        "--stage2-structure-mode",
        default="calibration_only",
        choices=["calibration_only", "trace_unit_semantic_binding"],
    )
    p.add_argument("--trace-unit-count", type=int, default=16)
    p.add_argument("--trace-unit-dim", type=int, default=384)
    p.add_argument("--trace-unit-slot-iters", type=int, default=3)
    p.add_argument("--trace-unit-assignment-topk", type=int, default=2)
    p.add_argument("--trace-unit-assignment-temperature", type=float, default=0.7)
    p.add_argument("--trace-unit-use-instance-prior-bias", action="store_true")
    p.add_argument("--trace-unit-disable-instance-path", action="store_true")
    p.add_argument("--trace-unit-teacher-prior-dim", type=int, default=512)
    p.add_argument("--trace-unit-dyn-update", default="gru", choices=["gru"])
    p.add_argument("--trace-unit-sem-update", default="gated_ema", choices=["gated_ema"])
    p.add_argument("--trace-unit-sem-alpha-min", type=float, default=0.02)
    p.add_argument("--trace-unit-sem-alpha-max", type=float, default=0.12)
    p.add_argument("--trace-unit-handshake-type", default="lowrank_cross_attn", choices=["lowrank_cross_attn"])
    p.add_argument("--trace-unit-handshake-dim", type=int, default=128)
    p.add_argument("--trace-unit-handshake-layers", type=int, default=1)
    p.add_argument("--trace-unit-handshake-writeback", default="dyn_only", choices=["dyn_only"])
    p.add_argument("--trace-unit-broadcast-residual-weight", type=float, default=0.35)
    p.add_argument("--trace-unit-broadcast-stopgrad-semantic", action="store_true")
    p.add_argument("--trace-unit-assignment-sparsity-weight", type=float, default=0.02)
    p.add_argument("--trace-unit-assignment-temporal-consistency-weight", type=float, default=0.05)
    p.add_argument("--trace-unit-semantic-inertia-weight", type=float, default=0.05)
    p.add_argument("--trace-unit-instance-consistency-weight", type=float, default=0.10)
    p.add_argument("--trace-unit-instance-binding-weight", type=float, default=0.10)
    p.add_argument("--trace-unit-interinstance-repulse-weight", type=float, default=0.05)
    p.add_argument("--trace-unit-unit-purity-weight", type=float, default=0.03)
    p.add_argument("--trace-unit-instance-id-source", default="dominant_id", choices=["dominant_id", "temporal_overlap"])
    p.add_argument("--trace-unit-instance-conf-threshold", type=float, default=0.6)
    p.add_argument("--trace-unit-dynsem-decorrelation-weight", type=float, default=0.005)
    p.add_argument("--trace-unit-utilization-weight", type=float, default=0.03)
    p.add_argument("--trace-unit-min-active-target", type=float, default=4.0)
    p.add_argument("--trace-unit-diversity-weight", type=float, default=0.03)
    p.add_argument("--trace-unit-top2-floor-weight", type=float, default=0.02)
    p.add_argument("--trace-unit-top2-mass-floor", type=float, default=0.10)
    p.add_argument("--trace-unit-ambiguity-repulse-weight", type=float, default=0.0)
    p.add_argument("--trace-unit-ambiguity-risk-threshold", type=float, default=0.35)
    p.add_argument("--trace-unit-ambiguity-min-dist-weight", type=float, default=0.40)
    p.add_argument("--trace-unit-ambiguity-iou-weight", type=float, default=0.35)
    p.add_argument("--trace-unit-ambiguity-motion-cross-weight", type=float, default=0.25)
    p.add_argument("--trace-unit-confuser-separation-weight", type=float, default=0.0)
    p.add_argument("--trace-unit-confuser-risk-threshold", type=float, default=0.45)
    p.add_argument("--trace-unit-confuser-appearance-weight", type=float, default=0.35)
    p.add_argument("--trace-unit-confuser-motion-weight", type=float, default=0.25)
    p.add_argument("--trace-unit-confuser-overlap-weight", type=float, default=0.40)
    p.add_argument("--trace-unit-appearance-refine-weight", type=float, default=0.0)
    p.add_argument("--trace-unit-appearance-high-threshold", type=float, default=0.18)
    p.add_argument("--trace-unit-appearance-high-quantile", type=float, default=0.80)
    p.add_argument("--trace-unit-hardsubset-curriculum-weight", type=float, default=0.0)
    p.add_argument("--trace-unit-hardsubset-ambiguity-weight", type=float, default=1.0)
    p.add_argument("--trace-unit-hardsubset-appearance-weight", type=float, default=1.0)
    p.add_argument("--trace-unit-hardsubset-occlusion-weight", type=float, default=0.75)
    p.add_argument("--trace-unit-hardsubset-longgap-weight", type=float, default=0.75)
    p.add_argument(
        "--semantic-rescue-mode",
        default="none",
        choices=[
            "none",
            "align",
            "querypersist",
            "bootstrapplabel",
            "v2readoutalign",
            "v2readoutpersist",
            "v2readouthard",
            "v3confalign",
            "v3confpersist",
            "v3confpersistdelay",
            "v3confhardsidecar",
            "v4sparse",
            "v5sparse",
            "v6sparse",
            "v7alignonly",
            "v7alignpersist",
        ],
        help="Optional Stage2 semantic objective rescue pilot; default keeps the Wave1/Wave2 objective unchanged.",
    )
    p.add_argument("--semantic-rescue-weight", type=float, default=0.0)
    p.add_argument("--semantic-bootstrap-cache-path", default="")
    p.add_argument("--semantic-bootstrap-target-dim", type=int, default=10)
    p.add_argument("--semantic-alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--query-persistence-consistency-loss-weight", type=float, default=0.0)
    p.add_argument("--semantic-hard-curriculum-weight", type=float, default=0.0)
    p.add_argument("--readout-semantic-alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--persistence-contrastive-ranking-loss-weight", type=float, default=0.0)
    p.add_argument("--semantic-aux-subset-weighting-strength", type=float, default=0.0)
    p.add_argument("--confidence-gated-alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--sparse-persistence-contrastive-loss-weight", type=float, default=0.0)
    p.add_argument("--confidence-gating-margin-threshold", type=float, default=0.10)
    p.add_argument("--confidence-gating-temperature", type=float, default=0.05)
    p.add_argument("--semantic-hard-score-threshold", type=float, default=0.25)
    p.add_argument("--aux-loss-delay-steps", type=int, default=0)
    p.add_argument("--aux-loss-ramp-steps", type=int, default=0)
    p.add_argument(
        "--v4-sparse-gating-family",
        default="quantile_sparse_gating",
        choices=["quantile_sparse_gating", "topk_query_gating"],
    )
    p.add_argument("--v4-gating-quantile", type=float, default=0.85)
    p.add_argument("--v4-topk-token-ratio", type=float, default=0.20)
    p.add_argument("--v4-topk-min-tokens", type=int, default=2)
    p.add_argument("--v4-gate-min-strength", type=float, default=0.05)
    p.add_argument("--v4-persistence-value-quantile", type=float, default=0.70)
    p.add_argument("--v4-persistence-max-pairs", type=int, default=0)
    p.add_argument("--v4-persistence-margin", type=float, default=0.10)
    p.add_argument(
        "--v5-gating-family",
        default="hard_topk_query_gating_v2",
        choices=["hard_topk_query_gating_v2", "capped_quantile_sparse_gating_v2"],
    )
    p.add_argument("--v5-topk-query-k", type=int, default=1)
    p.add_argument("--v5-capped-quantile", type=float, default=0.85)
    p.add_argument("--v5-max-affected-ratio", type=float, default=0.20)
    p.add_argument("--v5-gate-min-strength", type=float, default=0.05)
    p.add_argument("--v5-max-pairs-per-sample", type=int, default=2)
    p.add_argument("--v5-hard-negative-cap", type=int, default=4)
    p.add_argument("--v5-pair-sampling-temperature", type=float, default=0.35)
    p.add_argument(
        "--v6-gating-family",
        default="hard_topk_query_gating_v2",
        choices=["hard_topk_query_gating_v2", "capped_quantile_sparse_gating_v2"],
    )
    p.add_argument("--v6-topk-query-k", type=int, default=1)
    p.add_argument("--v6-capped-quantile", type=float, default=0.85)
    p.add_argument("--v6-max-affected-ratio", type=float, default=0.20)
    p.add_argument("--v6-gate-min-strength", type=float, default=0.05)
    p.add_argument("--v6-strict-max-pairs-per-sample", type=int, default=2)
    p.add_argument("--v6-hard-negative-cap", type=int, default=6)
    p.add_argument("--v6-pair-sampling-temperature", type=float, default=0.35)
    p.add_argument("--v6-guaranteed-min-pairs-per-sample", type=int, default=0)
    p.add_argument("--v6-two-level-pair-mining-enabled", action="store_true")
    p.add_argument("--no-v6-two-level-pair-mining-enabled", action="store_false", dest="v6_two_level_pair_mining_enabled")
    p.add_argument("--v6-relaxed-motion-threshold", type=float, default=0.20)
    p.add_argument("--v6-relaxed-area-jump-threshold", type=float, default=0.18)
    p.add_argument("--v6-relaxed-small-query-threshold", type=float, default=0.45)
    p.add_argument("--v6-relaxed-appearance-shift-threshold", type=float, default=0.10)
    p.add_argument("--v6-relaxed-center-interaction-threshold", type=float, default=0.20)
    p.add_argument("--semantic-hard-sidecar-enabled", action="store_true")
    p.add_argument(
        "--semantic-hard-manifest-path",
        default="/home/chen034/workspace/stwm/manifests/protocol_v2/stage2_semantic_hard_subsets_20260410.json",
    )
    p.add_argument("--enable-future-semantic-state-head", action="store_true")
    p.add_argument("--future-semantic-embedding-dim", type=int, default=256)
    p.add_argument("--future-semantic-loss-weight", type=float, default=0.0)
    p.add_argument("--future-visibility-loss-weight", type=float, default=0.0)
    p.add_argument("--future-reappearance-loss-weight", type=float, default=0.0)
    p.add_argument("--future-reappearance-event-loss-weight", type=float, default=0.0)
    p.add_argument("--future-reappearance-pos-weight", default="auto")
    p.add_argument("--future-reappearance-pos-weight-max", type=float, default=50.0)
    p.add_argument("--future-reappearance-mask-policy", default="at_risk_only", choices=["at_risk_only", "all_slots"])
    p.add_argument("--reappearance-positive-oversample", action="store_true")
    p.add_argument("--reappearance-positive-min-batch-ratio", type=float, default=0.30)
    p.add_argument("--reappearance-positive-index-cache", default="")
    p.add_argument("--future-identity-belief-loss-weight", type=float, default=0.0)
    p.add_argument("--future-uncertainty-loss-weight", type=float, default=0.0)
    p.add_argument("--future-hypothesis-count", type=int, default=1)
    p.add_argument("--future-hypothesis-loss-weight", type=float, default=0.0)
    p.add_argument("--enable-future-extent-head", action="store_true")
    p.add_argument("--enable-future-multihypothesis-head", action="store_true")
    p.add_argument("--future-semantic-head-only-warmup", action="store_true")
    p.add_argument("--future-semantic-head-only-warmup-steps", type=int, default=0)
    p.add_argument("--freeze-non-future-semantic-head-during-warmup", action="store_true")
    p.add_argument("--future-semantic-controlled-joint", action="store_true")
    p.add_argument("--future-semantic-joint-train-semantic-fusion-proj", action="store_true")
    p.add_argument("--future-semantic-joint-train-readout-head", action="store_true")
    p.add_argument("--enable-semantic-state-feedback", action="store_true")
    p.add_argument("--semantic-state-feedback-alpha", type=float, default=0.05)
    p.add_argument("--semantic-state-feedback-stopgrad-state", action="store_true", default=True)
    p.add_argument("--no-semantic-state-feedback-stopgrad-state", action="store_false", dest="semantic_state_feedback_stopgrad_state")
    p.add_argument("--semantic-state-feedback-mode", default="readout_only", choices=["readout_only", "hidden_residual"])

    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-from", default="")
    p.add_argument("--auto-resume-latest", action="store_true")
    p.add_argument(
        "--skip-resume-optimizer",
        action="store_true",
        help="Load module weights from --resume-from but start with a fresh optimizer state.",
    )

    p.add_argument("--run-name", required=True)
    p.add_argument("--run-summary-json", required=True)
    p.add_argument("--progress-json", default="")
    p.add_argument("--seed", type=int, default=20260408)
    p.set_defaults(v6_two_level_pair_mining_enabled=True)
    return p.parse_args()


def _safe_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _safe_load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    try:
        payload = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)}")
    return payload


def _load_bootstrap_cache(path_value: str) -> Dict[str, torch.Tensor]:
    target = str(path_value).strip()
    if not target:
        return {}
    p = Path(target)
    if not p.exists():
        return {}

    cache: Dict[str, torch.Tensor] = {}
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            key = f"{str(item.get('dataset', '')).upper()}::{str(item.get('clip_id', ''))}"
            target_values = item.get("feature_target", [])
            if key != "::" and isinstance(target_values, list):
                try:
                    cache[key] = torch.tensor([float(x) for x in target_values], dtype=torch.float32)
                except Exception:
                    continue
        return cache

    try:
        payload = _safe_json(p)
    except Exception:
        return {}
    for item in payload.get("items", []) if isinstance(payload.get("items", []), list) else []:
        if not isinstance(item, dict):
            continue
        key = f"{str(item.get('dataset', '')).upper()}::{str(item.get('clip_id', ''))}"
        target_values = item.get("feature_target", [])
        if key != "::" and isinstance(target_values, list):
            try:
                cache[key] = torch.tensor([float(x) for x in target_values], dtype=torch.float32)
            except Exception:
                continue
    return cache


def _norm_name(name: str) -> str:
    return str(name).strip().upper()


def _extract_binding(contract: Dict[str, Any]) -> Dict[str, Any]:
    binding = contract.get("stage2_bootstrap_binding", {}) if isinstance(contract.get("stage2_bootstrap_binding", {}), dict) else {}
    core = [str(x).strip() for x in binding.get("core", [])] if isinstance(binding.get("core", []), list) else []
    optional = [str(x).strip() for x in binding.get("optional_extension", [])] if isinstance(binding.get("optional_extension", []), list) else []
    if not core:
        core = ["VSPW", "VIPSeg"]

    usage: Dict[str, Dict[str, bool]] = {}
    for ds in contract.get("datasets", []) if isinstance(contract.get("datasets", []), list) else []:
        if not isinstance(ds, dict):
            continue
        name = _norm_name(str(ds.get("dataset_name", "")))
        if not name:
            continue
        usage[name] = {
            "train": bool(ds.get("used_in_bootstrap_train", False)),
            "eval": bool(ds.get("used_in_bootstrap_eval", False)),
        }

    excluded = [
        {
            "dataset_name": str(x.get("dataset_name", "")),
            "reason": str(x.get("reason", "")),
        }
        for x in contract.get("excluded_datasets", [])
        if isinstance(x, dict)
    ]

    return {
        "core": core,
        "optional_extension": optional,
        "usage": usage,
        "excluded": excluded,
    }


def _summary_count(summary: Dict[str, Dict[str, Any]], name: str) -> int:
    target = _norm_name(name)
    for key, meta in summary.items():
        if _norm_name(str(key)) == target and isinstance(meta, dict):
            return int(meta.get("sample_count", 0))
    return 0


def _core_dataset_ready(
    train_summary: Dict[str, Dict[str, Any]],
    val_summary: Dict[str, Dict[str, Any]],
    core_names: List[str],
    usage: Dict[str, Dict[str, bool]],
) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {}
    ready = True
    for name in core_names:
        key = _norm_name(name)
        use_train = bool(usage.get(key, {}).get("train", True))
        use_eval = bool(usage.get(key, {}).get("eval", True))

        train_count = _summary_count(train_summary, key)
        val_count = _summary_count(val_summary, key)

        train_ok = (train_count > 0) if use_train else True
        val_ok = (val_count > 0) if use_eval else True
        item_ready = bool(train_ok and val_ok)
        ready = bool(ready and item_ready)

        details[key] = {
            "train_required": use_train,
            "eval_required": use_eval,
            "train_sample_count": int(train_count),
            "val_sample_count": int(val_count),
            "ready": bool(item_ready),
        }
    return bool(ready), details


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _load_frozen_stage1_backbone(args: Any, device: torch.device) -> Tuple[TraceCausalTransformer, Dict[str, Any]]:
    ckpt = _safe_load_checkpoint(args.stage1_backbone_checkpoint, device=device)
    cfg_payload = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    preset = str(cfg_payload.get("model_preset", args.stage1_model_preset))
    cfg = build_tracewm_v2_config(preset)

    model = TraceCausalTransformer(cfg).to(device)
    state_dict = ckpt.get("model_state_dict") if isinstance(ckpt.get("model_state_dict"), dict) else None
    if state_dict is None:
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected stage1 checkpoint keys: {unexpected[:8]}")
    if len(missing) > 16:
        raise RuntimeError(f"too many missing stage1 checkpoint keys: {len(missing)}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    partial_mode = str(getattr(args, "stage1_partial_unfreeze_mode", "none")).strip().lower()
    partial_layers = max(int(getattr(args, "stage1_partial_unfreeze_layer_count", 1) or 1), 1)
    unfrozen_param_names: List[str] = []
    unfrozen_layer_indices: List[int] = []
    if partial_mode == "topblock":
        layer_ids = sorted(
            {
                int(match.group(1))
                for name, _ in model.named_parameters()
                for match in [re.search(r"backbone\.layers\.(\d+)\.", str(name))]
                if match is not None
            }
        )
        if layer_ids:
            unfrozen_layer_indices = layer_ids[-partial_layers:]
            prefixes = tuple([f"backbone.layers.{idx}." for idx in unfrozen_layer_indices] + ["norm."])
            for name, param in model.named_parameters():
                if str(name).startswith(prefixes):
                    param.requires_grad = True
                    unfrozen_param_names.append(str(name))

    meta = {
        "checkpoint_path": str(args.stage1_backbone_checkpoint),
        "model_preset": preset,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "parameter_count": int(sum(x.numel() for x in model.parameters())),
        "trainable_parameter_count": int(sum(x.numel() for x in model.parameters() if x.requires_grad)),
        "partial_unfreeze_mode": partial_mode,
        "partial_unfreeze_layer_count": int(partial_layers),
        "partial_unfreeze_layer_indices": unfrozen_layer_indices,
        "partial_unfreeze_param_names": unfrozen_param_names,
    }
    return model, meta


def _resolve_resume_path(resume_from: str, auto_resume_latest: bool, latest_path: Path) -> str:
    direct = str(resume_from).strip()
    if direct:
        return str(Path(direct).expanduser())
    if bool(auto_resume_latest) and latest_path.exists():
        return str(latest_path)
    return ""


def _to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool) -> Dict[str, Any]:
    out = dict(batch)
    for k in [
        "obs_state",
        "fut_state",
        "obs_valid",
        "fut_valid",
        "token_mask",
        "semantic_features",
        "semantic_mask",
        "semantic_rgb_crop",
        "semantic_mask_crop",
        "semantic_crop_valid",
        "semantic_mask_crop_valid",
        "semantic_rgb_crop_temporal",
        "semantic_mask_crop_temporal",
        "semantic_temporal_valid",
        "semantic_instance_id_crop",
        "semantic_instance_id_temporal",
        "semantic_instance_valid",
        "semantic_objectness_score",
        "semantic_entity_dominant_instance_id",
        "semantic_entity_instance_overlap_score_over_time",
        "semantic_entity_true_instance_confidence",
        "semantic_teacher_prior",
    ]:
        out[k] = batch[k].to(device, non_blocking=non_blocking)
    return out


def _prepare_shifted(full_state: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(full_state)
    shifted[:, 0] = full_state[:, 0]
    shifted[:, 1:] = full_state[:, :-1]
    return shifted


def _masked_mse_coord(pred_coord: torch.Tensor, target_coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    sq = ((pred_coord - target_coord) ** 2).sum(dim=-1)
    mask_f = valid_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (sq * mask_f).sum() / denom


def _masked_mean_l2(pred_coord: torch.Tensor, target_coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    l2 = torch.sqrt(((pred_coord - target_coord) ** 2).sum(dim=-1).clamp_min(1e-12))
    mask_f = valid_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (l2 * mask_f).sum() / denom


def _masked_endpoint_l2(pred_coord: torch.Tensor, target_coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    l2 = torch.sqrt(((pred_coord[:, -1] - target_coord[:, -1]) ** 2).sum(dim=-1).clamp_min(1e-12))
    mask_f = valid_mask[:, -1].float()
    denom = mask_f.sum().clamp_min(1.0)
    return (l2 * mask_f).sum() / denom


def _teacher_forced_predict(
    *,
    stage1_model: TraceCausalTransformer,
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    future_semantic_state_head: SemanticTraceStateHead | None,
    structure_mode: str,
    trace_unit_tokenizer: TraceUnitTokenizer | None,
    trace_unit_factorized_state: TraceUnitFactorizedState | None,
    trace_unit_handshake: TraceUnitHandshake | None,
    trace_unit_broadcast: TraceUnitBroadcast | None,
    trace_unit_disable_instance_path: bool,
    batch: Dict[str, Any],
    obs_len: int,
    semantic_source_mainline: str,
    allow_stage1_grad: bool = False,
    semantic_state_feedback_adapter: SemanticStateFeedbackAdapter | None = None,
    semantic_state_feedback_enabled: bool = False,
    semantic_state_feedback_alpha: float = 0.05,
    semantic_state_feedback_stopgrad_state: bool = True,
    semantic_state_feedback_mode: str = "readout_only",
) -> Dict[str, Any]:
    full_state = torch.cat([batch["obs_state"], batch["fut_state"]], dim=1)
    shifted = _prepare_shifted(full_state)
    token_mask = batch["token_mask"]

    grad_ctx = nullcontext() if bool(allow_stage1_grad) else torch.no_grad()
    with grad_ctx:
        stage1_out = stage1_model(shifted, token_mask=token_mask)

    sem_enc = semantic_encoder(
        batch.get("semantic_features"),
        semantic_rgb_crop=batch.get("semantic_rgb_crop"),
        semantic_mask_crop=batch.get("semantic_mask_crop"),
        semantic_rgb_crop_temporal=batch.get("semantic_rgb_crop_temporal"),
        semantic_mask_crop_temporal=batch.get("semantic_mask_crop_temporal"),
        semantic_temporal_valid=batch.get("semantic_temporal_valid"),
        source_mode=str(semantic_source_mainline),
    )
    fused_hidden, aux = semantic_fusion(stage1_out["hidden"], sem_enc, token_mask=token_mask)
    enhanced_hidden, trace_unit_aux = _apply_trace_unit_binding(
        structure_mode=str(structure_mode),
        tokenizer=trace_unit_tokenizer,
        factorized_state=trace_unit_factorized_state,
        handshake=trace_unit_handshake,
        broadcast=trace_unit_broadcast,
        fused_hidden=fused_hidden,
        state_seq=full_state,
        semantic_tokens=sem_enc,
        batch=batch,
        obs_len=int(obs_len),
        disable_instance_path=bool(trace_unit_disable_instance_path),
    )
    pred_coord = readout_head(enhanced_hidden[:, int(obs_len) :])
    future_hidden = enhanced_hidden[:, int(obs_len) :]
    future_semantic_trace_state = None
    feedback_info: Dict[str, Any] = {
        "semantic_state_feedback_enabled": False,
        "semantic_state_feedback_mode": str(semantic_state_feedback_mode),
        "semantic_state_feedback_alpha": float(semantic_state_feedback_alpha),
        "feedback_gate_mean": 0.0,
        "feedback_gate_std": 0.0,
        "feedback_gate_saturation_ratio": 0.0,
        "feedback_delta_norm": 0.0,
    }
    if future_semantic_state_head is not None:
        future_semantic_trace_state = future_semantic_state_head(future_hidden, future_trace_coord=pred_coord)
        if bool(semantic_state_feedback_enabled) and semantic_state_feedback_adapter is not None:
            fb = semantic_state_feedback_adapter(
                future_hidden,
                future_semantic_trace_state,
                alpha=float(semantic_state_feedback_alpha),
                stopgrad_state=bool(semantic_state_feedback_stopgrad_state),
            )
            future_hidden = fb.enhanced_future_hidden
            mode = str(semantic_state_feedback_mode).strip().lower()
            if mode == "hidden_residual":
                pred_coord = readout_head(future_hidden)
            future_semantic_trace_state = future_semantic_state_head(future_hidden, future_trace_coord=pred_coord)
            feedback_info = dict(fb.feedback_info)
            feedback_info["semantic_state_feedback_mode"] = mode

    target_coord = batch["fut_state"][..., 0:2]
    valid_mask = batch["fut_valid"] & token_mask[:, None, :]

    return {
        "pred_coord": pred_coord,
        "target_coord": target_coord,
        "valid_mask": valid_mask,
        "semantic_tokens": sem_enc,
        "future_fused_hidden": future_hidden,
        "future_semantic_trace_state": future_semantic_trace_state,
        "semantic_state_feedback_info": feedback_info,
        "gate_mean": float(aux.get("gate_mean", 0.0)),
        "gate_std": float(aux.get("gate_std", 0.0)),
        "semantic_input_nonempty": bool((batch["semantic_mask"] & token_mask).any().item()),
        "trace_unit_aux": trace_unit_aux,
    }


def _free_rollout_predict(
    *,
    stage1_model: TraceCausalTransformer,
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    future_semantic_state_head: SemanticTraceStateHead | None,
    structure_mode: str,
    trace_unit_tokenizer: TraceUnitTokenizer | None,
    trace_unit_factorized_state: TraceUnitFactorizedState | None,
    trace_unit_handshake: TraceUnitHandshake | None,
    trace_unit_broadcast: TraceUnitBroadcast | None,
    trace_unit_disable_instance_path: bool,
    batch: Dict[str, Any],
    obs_len: int,
    fut_len: int,
    semantic_source_mainline: str,
    allow_stage1_grad: bool = False,
    semantic_state_feedback_adapter: SemanticStateFeedbackAdapter | None = None,
    semantic_state_feedback_enabled: bool = False,
    semantic_state_feedback_alpha: float = 0.05,
    semantic_state_feedback_stopgrad_state: bool = True,
    semantic_state_feedback_mode: str = "readout_only",
) -> Dict[str, Any]:
    token_mask = batch["token_mask"]
    obs_state = batch["obs_state"]

    bsz, _, k_len, d_state = obs_state.shape
    total_len = int(obs_len) + int(fut_len)

    state_seq = torch.zeros((bsz, total_len, k_len, d_state), device=obs_state.device, dtype=obs_state.dtype)
    state_seq[:, : int(obs_len)] = obs_state

    sem_enc = semantic_encoder(
        batch.get("semantic_features"),
        semantic_rgb_crop=batch.get("semantic_rgb_crop"),
        semantic_mask_crop=batch.get("semantic_mask_crop"),
        semantic_rgb_crop_temporal=batch.get("semantic_rgb_crop_temporal"),
        semantic_mask_crop_temporal=batch.get("semantic_mask_crop_temporal"),
        semantic_temporal_valid=batch.get("semantic_temporal_valid"),
        source_mode=str(semantic_source_mainline),
    )
    gate_vals: List[float] = []

    for step in range(int(fut_len)):
        shifted = _prepare_shifted(state_seq)
        grad_ctx = nullcontext() if bool(allow_stage1_grad) else torch.no_grad()
        with grad_ctx:
            stage1_out = stage1_model(shifted, token_mask=token_mask)

        fused_hidden, aux = semantic_fusion(stage1_out["hidden"], sem_enc, token_mask=token_mask)
        enhanced_hidden, _ = _apply_trace_unit_binding(
            structure_mode=str(structure_mode),
            tokenizer=trace_unit_tokenizer,
            factorized_state=trace_unit_factorized_state,
            handshake=trace_unit_handshake,
            broadcast=trace_unit_broadcast,
            fused_hidden=fused_hidden,
            state_seq=state_seq,
            semantic_tokens=sem_enc,
            batch=batch,
            obs_len=int(obs_len),
            disable_instance_path=bool(trace_unit_disable_instance_path),
        )
        gate_vals.append(float(aux.get("gate_mean", 0.0)))

        pred_coord_all = readout_head(enhanced_hidden)
        time_idx = int(obs_len) + int(step)
        pred_coord_t = pred_coord_all[:, time_idx : time_idx + 1]

        pred_state_t = stage1_out["pred_state"][:, time_idx : time_idx + 1].detach().clone()
        pred_state_t[..., 0:2] = pred_coord_t.detach()
        state_seq[:, time_idx : time_idx + 1] = pred_state_t

    pred_future = state_seq[:, int(obs_len) :, :, 0:2]
    target_coord = batch["fut_state"][..., 0:2]
    valid_mask = batch["fut_valid"] & token_mask[:, None, :]
    future_hidden = enhanced_hidden[:, int(obs_len) :]
    future_semantic_trace_state = None
    feedback_info: Dict[str, Any] = {
        "semantic_state_feedback_enabled": False,
        "semantic_state_feedback_mode": str(semantic_state_feedback_mode),
        "semantic_state_feedback_alpha": float(semantic_state_feedback_alpha),
        "feedback_gate_mean": 0.0,
        "feedback_gate_std": 0.0,
        "feedback_gate_saturation_ratio": 0.0,
        "feedback_delta_norm": 0.0,
    }
    future_semantic_state_validation = {"valid": False, "shapes": {}, "errors": ["future_semantic_state_head_disabled"]}
    if future_semantic_state_head is not None:
        future_semantic_trace_state = future_semantic_state_head(future_hidden, future_trace_coord=pred_future)
        if bool(semantic_state_feedback_enabled) and semantic_state_feedback_adapter is not None:
            fb = semantic_state_feedback_adapter(
                future_hidden,
                future_semantic_trace_state,
                alpha=float(semantic_state_feedback_alpha),
                stopgrad_state=bool(semantic_state_feedback_stopgrad_state),
            )
            future_hidden = fb.enhanced_future_hidden
            mode = str(semantic_state_feedback_mode).strip().lower()
            if mode == "hidden_residual":
                pred_future = readout_head(future_hidden)
            future_semantic_trace_state = future_semantic_state_head(future_hidden, future_trace_coord=pred_future)
            feedback_info = dict(fb.feedback_info)
            feedback_info["semantic_state_feedback_mode"] = mode
        future_semantic_state_validation = future_semantic_trace_state.validate(strict=False)

    return {
        "pred_coord": pred_future,
        "target_coord": target_coord,
        "valid_mask": valid_mask,
        "gate_mean": float(sum(gate_vals) / max(len(gate_vals), 1)),
        "future_hidden": future_hidden,
        "future_semantic_trace_state": future_semantic_trace_state,
        "semantic_state_feedback_info": feedback_info,
        "future_semantic_state_shapes": future_semantic_state_validation.get("shapes", {}),
        "future_semantic_state_output_valid": bool(future_semantic_state_validation.get("valid", False)),
        "semantic_state_feedback_in_free_rollout": False,
        "free_rollout_semantic_state_output": bool(future_semantic_trace_state is not None),
    }


def _apply_trace_unit_binding(
    *,
    structure_mode: str,
    tokenizer: TraceUnitTokenizer | None,
    factorized_state: TraceUnitFactorizedState | None,
    handshake: TraceUnitHandshake | None,
    broadcast: TraceUnitBroadcast | None,
    fused_hidden: torch.Tensor,
    state_seq: torch.Tensor,
    semantic_tokens: torch.Tensor,
    batch: Dict[str, Any],
    obs_len: int,
    disable_instance_path: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if str(structure_mode).strip().lower() != "trace_unit_semantic_binding":
        return fused_hidden, {}
    if tokenizer is None or factorized_state is None or handshake is None or broadcast is None:
        return fused_hidden, {}

    semantic_instance_valid = batch.get("semantic_instance_valid")
    if bool(disable_instance_path) and isinstance(semantic_instance_valid, torch.Tensor):
        semantic_instance_valid = torch.zeros_like(semantic_instance_valid)
    semantic_entity_dominant_instance_id = batch.get("semantic_entity_dominant_instance_id")
    semantic_entity_true_instance_confidence = batch.get("semantic_entity_true_instance_confidence")
    if bool(disable_instance_path) and isinstance(semantic_entity_dominant_instance_id, torch.Tensor):
        semantic_entity_dominant_instance_id = torch.zeros_like(semantic_entity_dominant_instance_id)
    if bool(disable_instance_path) and isinstance(semantic_entity_true_instance_confidence, torch.Tensor):
        semantic_entity_true_instance_confidence = torch.zeros_like(semantic_entity_true_instance_confidence)
    state_valid_mask = None
    obs_valid = batch.get("obs_valid")
    fut_valid = batch.get("fut_valid")
    if isinstance(obs_valid, torch.Tensor) and obs_valid.ndim == 3:
        if isinstance(fut_valid, torch.Tensor) and fut_valid.ndim == 3:
            full_valid = torch.cat([obs_valid, fut_valid], dim=1)
        else:
            future_steps = max(int(state_seq.shape[1]) - int(obs_valid.shape[1]), 0)
            full_valid = torch.cat([obs_valid, batch["token_mask"][:, None, :].expand(obs_valid.shape[0], future_steps, obs_valid.shape[-1])], dim=1)
        state_valid_mask = full_valid[:, : state_seq.shape[1]]
    tokenized = tokenizer(
        backbone_hidden=fused_hidden,
        state_seq=state_seq,
        semantic_tokens=semantic_tokens,
        token_mask=batch["token_mask"],
        state_valid_mask=state_valid_mask,
        semantic_objectness_score=batch.get("semantic_objectness_score"),
        semantic_instance_valid=semantic_instance_valid,
        semantic_entity_dominant_instance_id=semantic_entity_dominant_instance_id,
        semantic_entity_true_instance_confidence=semantic_entity_true_instance_confidence,
        semantic_teacher_prior=batch.get("semantic_teacher_prior"),
    )
    fact = factorized_state(
        token_features=tokenized["token_features"],
        assignment=tokenized["assignment"],
    )
    shaken = handshake(
        z_dyn=fact["z_dyn"],
        z_sem=fact["z_sem"],
        unit_presence=fact["unit_presence"],
    )
    bcast = broadcast(
        backbone_hidden=fused_hidden,
        assignment=tokenized["assignment"],
        z_dyn=shaken["z_dyn"],
        z_sem=fact["z_sem"],
    )
    return bcast["enhanced_hidden"], {
        "assignment": tokenized["assignment"],
        "token_valid": tokenized["token_valid"],
        "z_dyn": shaken["z_dyn"],
        "z_sem": fact["z_sem"],
        "unit_presence": fact["unit_presence"],
        "obs_len": int(obs_len),
        "tokenizer_metrics": tokenized["metrics"],
        "factorized_metrics": fact["metrics"],
        "handshake_metrics": shaken["metrics"],
        "broadcast_metrics": bcast["metrics"],
    }


def _trace_unit_regularization_loss(
    *,
    structure_mode: str,
    trace_unit_aux: Dict[str, Any],
    batch: Dict[str, Any],
    device: torch.device,
    assignment_sparsity_weight: float,
    assignment_temporal_consistency_weight: float,
    semantic_inertia_weight: float,
    instance_consistency_weight: float,
    instance_binding_weight: float,
    interinstance_repulse_weight: float,
    unit_purity_weight: float,
    instance_conf_threshold: float,
    ambiguity_repulse_weight: float,
    ambiguity_risk_threshold: float,
    ambiguity_min_dist_weight: float,
    ambiguity_iou_weight: float,
    ambiguity_motion_cross_weight: float,
    confuser_separation_weight: float,
    confuser_risk_threshold: float,
    confuser_appearance_weight: float,
    confuser_motion_weight: float,
    confuser_overlap_weight: float,
    appearance_refine_weight: float,
    appearance_high_threshold: float,
    appearance_high_quantile: float,
    dynsem_decorrelation_weight: float,
    utilization_weight: float,
    min_active_target: float,
    diversity_weight: float,
    top2_floor_weight: float,
    top2_mass_floor: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = batch["fut_state"].sum() * 0.0
    if str(structure_mode).strip().lower() != "trace_unit_semantic_binding" or not trace_unit_aux:
        return zero, {
            "trace_unit_loss": 0.0,
            "assignment_entropy_mean": 0.0,
            "actual_top2_assignment_ratio": 0.0,
            "active_unit_count_mean": 0.0,
            "z_dyn_drift_mean": 0.0,
            "z_sem_drift_mean": 0.0,
            "z_sem_to_z_dyn_drift_ratio": 0.0,
            "same_instance_within_unit_consistency": 0.0,
            "different_instance_between_unit_separation": 0.0,
            "same_instance_dominant_unit_match_rate": 0.0,
            "same_instance_assignment_cosine": 0.0,
            "different_instance_dominant_unit_collision_rate": 0.0,
            "ambiguity_highrisk_pair_collision_rate": 0.0,
            "confuser_highrisk_pair_collision_rate": 0.0,
            "appearance_drift_highrisk_same_instance_match_rate": 0.0,
            "appearance_drift_high_ratio": 0.0,
            "unit_purity_by_instance_id": 0.0,
            "unit_track_stability_over_time": 0.0,
            "target_entity_to_dominant_unit_consistency": 0.0,
            "unit_semantic_stability_over_time": 0.0,
            "broadcast_residual_norm_mean": 0.0,
            "unit_utilization_mean": 0.0,
            "true_instance_ratio_per_batch": 0.0,
        }

    assignment = trace_unit_aux["assignment"]
    token_valid = trace_unit_aux["token_valid"]
    z_dyn = trace_unit_aux["z_dyn"]
    z_sem = trace_unit_aux["z_sem"]
    obs_len = min(int(trace_unit_aux.get("obs_len", z_dyn.shape[1])), int(z_dyn.shape[1]))
    obs_assignment = assignment[:, :obs_len]
    obs_valid = token_valid[:, :obs_len]
    obs_z_dyn = z_dyn[:, :obs_len]
    obs_z_sem = z_sem[:, :obs_len]

    entropy = -(obs_assignment.clamp_min(1e-8) * obs_assignment.clamp_min(1e-8).log()).sum(dim=-1)
    entropy = torch.where(obs_valid, entropy, torch.zeros_like(entropy))
    entropy_loss = entropy.sum() / obs_valid.float().sum().clamp_min(1.0)
    token_count = obs_valid.float().sum(dim=(1, 2)).clamp_min(1.0)
    per_sample_unit_mass = obs_assignment.sum(dim=(1, 2)) / token_count[:, None]
    active_unit_count = (per_sample_unit_mass > 1e-3).float().sum(dim=-1)
    utilization_mean = float(active_unit_count.detach().mean().cpu().item())
    unit_mass_norm = per_sample_unit_mass / per_sample_unit_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    unit_util_entropy = -(unit_mass_norm.clamp_min(1e-8) * unit_mass_norm.clamp_min(1e-8).log()).sum(dim=-1)
    unit_utilization_loss = -unit_util_entropy.mean()
    min_active_loss = torch.relu(torch.as_tensor(float(min_active_target), device=device, dtype=active_unit_count.dtype) - active_unit_count).mean()

    temporal_consistency_loss = zero
    if obs_assignment.shape[1] >= 2:
        valid_pairs = obs_valid[:, 1:] & obs_valid[:, :-1]
        pair_mask = valid_pairs[..., None].float()
        temporal_delta = ((obs_assignment[:, 1:] - obs_assignment[:, :-1]) ** 2).sum(dim=-1)
        temporal_consistency_loss = (temporal_delta * valid_pairs.float()).sum() / valid_pairs.float().sum().clamp_min(1.0)

    sem_delta = torch.linalg.norm(obs_z_sem[:, 1:] - obs_z_sem[:, :-1], dim=-1) if obs_z_sem.shape[1] >= 2 else torch.zeros_like(obs_z_sem[..., 0])
    sem_presence = trace_unit_aux["unit_presence"][:, 1:obs_len] & trace_unit_aux["unit_presence"][:, : max(obs_len - 1, 0)]
    semantic_inertia_loss = (
        (sem_delta * sem_presence.float()).sum() / sem_presence.float().sum().clamp_min(1.0)
        if obs_z_sem.shape[1] >= 2
        else zero
    )

    dyn_norm = torch.nn.functional.normalize(obs_z_dyn, dim=-1)
    sem_norm = torch.nn.functional.normalize(obs_z_sem, dim=-1)
    decorrelation = ((dyn_norm * sem_norm).sum(dim=-1) ** 2).mean()

    same_instance_match_rate = 0.0
    same_instance_assignment_cosine = 0.0
    diff_instance_collision_rate = 0.0
    unit_purity_metric = 0.0
    unit_track_stability = 0.0
    target_entity_unit_consistency = 0.0
    true_instance_ratio_per_batch = 0.0
    instance_consistency_loss = zero
    interinstance_repulse_loss = zero
    unit_purity_loss = zero
    diff_instance_separation = 0.0
    instance_consistency_metric = 0.0
    ambiguity_repulse_loss = zero
    ambiguity_highrisk_pair_collision_rate = 0.0
    confuser_separation_loss = zero
    confuser_highrisk_pair_collision_rate = 0.0
    appearance_refine_loss = zero
    appearance_drift_highrisk_same_instance_match_rate = 0.0
    appearance_drift_high_ratio = 0.0
    appearance_signal_valid_count = 0.0
    appearance_drift_high_count = 0.0
    appearance_refine_loss_nonzero = 0.0

    dominant_ids = batch.get("semantic_entity_dominant_instance_id")
    instance_conf = batch.get("semantic_entity_true_instance_confidence")
    obs_entity_valid = batch.get("obs_valid")
    if isinstance(dominant_ids, torch.Tensor) and isinstance(instance_conf, torch.Tensor) and isinstance(obs_entity_valid, torch.Tensor):
        dominant_ids = dominant_ids.to(device=device, dtype=torch.long)
        instance_conf = instance_conf.to(device=device, dtype=obs_assignment.dtype)
        obs_entity_valid = obs_entity_valid[:, :obs_len].to(device=device, dtype=torch.bool)
        confident_entities = (dominant_ids > 0) & (instance_conf >= float(instance_conf_threshold))
        appearance_drift_by_entity, appearance_drift_entity_valid = _entity_temporal_appearance_drift(batch, device)
        local_appearance_delta_by_entity, local_appearance_valid = _entity_temporal_local_appearance_delta(batch, device)
        appearance_signal = torch.maximum(
            appearance_drift_by_entity,
            0.60 * appearance_drift_by_entity + 0.40 * (local_appearance_delta_by_entity / 0.25).clamp(0.0, 2.0),
        )
        appearance_signal_valid = appearance_drift_entity_valid | local_appearance_valid
        appearance_drift_high, _ = _appearance_high_mask(
            appearance_signal=appearance_signal,
            valid_mask=appearance_signal_valid,
            min_threshold=float(appearance_high_threshold),
            high_quantile=float(appearance_high_quantile),
        )
        if bool(appearance_signal_valid.any().item()):
            appearance_signal_valid_count = float(appearance_signal_valid.float().sum().detach().cpu().item())
            appearance_drift_high_count = float(appearance_drift_high.float().sum().detach().cpu().item())
            appearance_drift_high_ratio = float(
                appearance_drift_high.float().sum().detach().cpu().item()
                / appearance_signal_valid.float().sum().clamp_min(1.0).detach().cpu().item()
            )
            if not bool(appearance_drift_high.any().item()):
                flat_signal = appearance_signal.masked_fill(~appearance_signal_valid, float("-inf")).reshape(-1)
                top_idx = int(torch.argmax(flat_signal).detach().cpu().item())
                flat_mask = appearance_drift_high.reshape(-1)
                if 0 <= top_idx < int(flat_mask.numel()):
                    flat_mask[top_idx] = True
                    appearance_drift_high = flat_mask.reshape_as(appearance_drift_high)
                    appearance_drift_high_count = float(appearance_drift_high.float().sum().detach().cpu().item())
                    appearance_drift_high_ratio = float(
                        appearance_drift_high.float().sum().detach().cpu().item()
                        / appearance_signal_valid.float().sum().clamp_min(1.0).detach().cpu().item()
                    )
        appearance_signature, appearance_signature_valid = _entity_temporal_appearance_signature(batch, device)
        true_instance_ratio_per_batch = float(confident_entities.float().mean().detach().cpu().item())
        dominant_units = obs_assignment.argmax(dim=-1)

        same_pair_mask = (
            confident_entities[:, None, :]
            & obs_entity_valid
            & torch.cat([obs_entity_valid[:, 1:], torch.zeros_like(obs_entity_valid[:, :1])], dim=1)
        )[:, :-1, :]
        if bool(same_pair_mask.any().item()):
            assign_t = obs_assignment[:, 1:obs_len]
            assign_prev = obs_assignment[:, : obs_len - 1]
            cos = torch.nn.functional.cosine_similarity(assign_t, assign_prev, dim=-1)
            instance_consistency_loss = (1.0 - cos).mul(same_pair_mask.float()).sum() / same_pair_mask.float().sum().clamp_min(1.0)
            same_instance_assignment_cosine = float(
                (cos * same_pair_mask.float()).sum().detach().cpu().item() / same_pair_mask.float().sum().clamp_min(1.0).detach().cpu().item()
            )
            match = (dominant_units[:, 1:obs_len] == dominant_units[:, : obs_len - 1]).float()
            same_instance_match_rate = float(
                (match * same_pair_mask.float()).sum().detach().cpu().item() / same_pair_mask.float().sum().clamp_min(1.0).detach().cpu().item()
            )
            high_drift_same_mask = same_pair_mask & appearance_drift_high[:, None, :]
            if bool(high_drift_same_mask.any().item()):
                appearance_refine_loss = (1.0 - cos).mul(high_drift_same_mask.float()).sum() / high_drift_same_mask.float().sum().clamp_min(1.0)
                appearance_refine_loss_nonzero = 1.0
                appearance_drift_highrisk_same_instance_match_rate = float(
                    (match * high_drift_same_mask.float()).sum().detach().cpu().item()
                    / high_drift_same_mask.float().sum().clamp_min(1.0).detach().cpu().item()
                )
        instance_consistency_metric = same_instance_assignment_cosine

        pair_cos_terms = []
        pair_collisions = []
        ambiguity_loss_terms = []
        ambiguity_pair_collisions = []
        confuser_loss_terms = []
        confuser_pair_collisions = []
        purity_vals = []
        entity_consistency_vals = []
        obs_state_ctx = batch["obs_state"].to(device=device, dtype=torch.float32)
        for b_idx in range(int(obs_assignment.shape[0])):
            valid_entity_ids = dominant_ids[b_idx]
            for t_idx in range(int(obs_assignment.shape[1])):
                valid_here = obs_entity_valid[b_idx, t_idx] & confident_entities[b_idx]
                active_idx = torch.nonzero(valid_here, as_tuple=False).flatten()
                if active_idx.numel() >= 2:
                    for i_pos in range(int(active_idx.numel())):
                        for j_pos in range(i_pos + 1, int(active_idx.numel())):
                            i = int(active_idx[i_pos].item())
                            j = int(active_idx[j_pos].item())
                            if int(valid_entity_ids[i].item()) == int(valid_entity_ids[j].item()):
                                continue
                            ai = obs_assignment[b_idx, t_idx, i]
                            aj = obs_assignment[b_idx, t_idx, j]
                            cos_ij = torch.nn.functional.cosine_similarity(ai[None, :], aj[None, :], dim=-1)[0]
                            pair_cos_terms.append(cos_ij)
                            collided = float(int(dominant_units[b_idx, t_idx, i].item()) == int(dominant_units[b_idx, t_idx, j].item()))
                            pair_collisions.append(collided)
                            center_i = obs_state_ctx[b_idx, t_idx, i, 0:2]
                            center_j = obs_state_ctx[b_idx, t_idx, j, 0:2]
                            wh_i = obs_state_ctx[b_idx, t_idx, i, 6:8].clamp(0.0, 1.0)
                            wh_j = obs_state_ctx[b_idx, t_idx, j, 6:8].clamp(0.0, 1.0)
                            dist = float(torch.linalg.norm(center_i - center_j).detach().cpu().item())
                            dist_risk = max(0.0, 1.0 - dist / 0.25)
                            iou = _pair_iou_xywh(torch.cat([center_i, wh_i], dim=0), torch.cat([center_j, wh_j], dim=0))
                            motion_cross = 0.0
                            if (
                                t_idx + 1 < int(obs_assignment.shape[1])
                                and bool(obs_entity_valid[b_idx, t_idx + 1, i].item())
                                and bool(obs_entity_valid[b_idx, t_idx + 1, j].item())
                            ):
                                next_center_i = obs_state_ctx[b_idx, t_idx + 1, i, 0:2]
                                next_center_j = obs_state_ctx[b_idx, t_idx + 1, j, 0:2]
                                move_i = next_center_i - center_i
                                move_j = next_center_j - center_j
                                move_i_norm = float(torch.linalg.norm(move_i).detach().cpu().item())
                                move_j_norm = float(torch.linalg.norm(move_j).detach().cpu().item())
                                if move_i_norm > 1e-6 and move_j_norm > 1e-6:
                                    move_cos = float(
                                        torch.nn.functional.cosine_similarity(move_i[None, :], move_j[None, :], dim=-1)[0]
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    motion_cross = max(0.0, (1.0 - move_cos) * 0.5) * max(0.0, 1.0 - dist / 0.35)
                            ambiguity_risk = (
                                float(ambiguity_min_dist_weight) * dist_risk
                                + float(ambiguity_iou_weight) * iou
                                + float(ambiguity_motion_cross_weight) * motion_cross
                            ) / max(
                                float(ambiguity_min_dist_weight) + float(ambiguity_iou_weight) + float(ambiguity_motion_cross_weight),
                                1e-6,
                            )
                            if ambiguity_risk >= float(ambiguity_risk_threshold):
                                risk_scale = max(min((ambiguity_risk - float(ambiguity_risk_threshold)) / max(1.0 - float(ambiguity_risk_threshold), 1e-6), 1.0), 0.0)
                                ambiguity_loss_terms.append((1.0 + risk_scale) * torch.relu(cos_ij - 0.20))
                                ambiguity_pair_collisions.append(collided)
                            appearance_confuser = 0.0
                            if bool(appearance_signature_valid[b_idx, i].item()) and bool(appearance_signature_valid[b_idx, j].item()):
                                sig_i = appearance_signature[b_idx, i]
                                sig_j = appearance_signature[b_idx, j]
                                appearance_confuser = max(
                                    0.0,
                                    float(
                                        torch.nn.functional.cosine_similarity(sig_i[None, :], sig_j[None, :], dim=-1)[0]
                                        .detach()
                                        .cpu()
                                        .item()
                                    ),
                                )
                            confuser_risk = (
                                float(confuser_overlap_weight) * iou
                                + float(confuser_motion_weight) * motion_cross
                                + float(confuser_appearance_weight) * appearance_confuser
                                + float(ambiguity_min_dist_weight) * dist_risk
                            ) / max(
                                float(confuser_overlap_weight)
                                + float(confuser_motion_weight)
                                + float(confuser_appearance_weight)
                                + float(ambiguity_min_dist_weight),
                                1e-6,
                            )
                            if confuser_risk >= float(confuser_risk_threshold):
                                confuser_scale = max(
                                    min(
                                        (confuser_risk - float(confuser_risk_threshold))
                                        / max(1.0 - float(confuser_risk_threshold), 1e-6),
                                        1.0,
                                    ),
                                    0.0,
                                )
                                confuser_loss_terms.append((1.0 + 1.5 * confuser_scale) * torch.relu(cos_ij - 0.10))
                                confuser_pair_collisions.append(collided)
                    mass = obs_assignment[b_idx, t_idx, active_idx]
                    ids = valid_entity_ids[active_idx]
                    for unit_idx in range(int(mass.shape[-1])):
                        weights = mass[:, unit_idx]
                        denom = float(weights.sum().detach().cpu().item())
                        if denom <= 1e-6:
                            continue
                        unique_ids = torch.unique(ids)
                        best_mass = 0.0
                        for instance_id in unique_ids.tolist():
                            best_mass = max(best_mass, float(weights[(ids == int(instance_id))].sum().detach().cpu().item()))
                        purity_vals.append(best_mass / max(denom, 1e-6))
            for ent_idx in range(int(obs_assignment.shape[2])):
                if not bool(confident_entities[b_idx, ent_idx].item()):
                    continue
                valid_steps = torch.nonzero(obs_entity_valid[b_idx, :, ent_idx], as_tuple=False).flatten()
                if valid_steps.numel() < 2:
                    continue
                ent_units = dominant_units[b_idx, valid_steps, ent_idx]
                values, counts = torch.unique(ent_units, return_counts=True)
                if counts.numel() > 0:
                    entity_consistency_vals.append(float(counts.max().detach().cpu().item() / max(int(valid_steps.numel()), 1)))

        if pair_cos_terms:
            pair_cos_tensor = torch.stack(pair_cos_terms)
            interinstance_repulse_loss = torch.relu(pair_cos_tensor - 0.35).mean()
            diff_instance_separation = float((1.0 - pair_cos_tensor).mean().detach().cpu().item())
            diff_instance_collision_rate = float(sum(pair_collisions) / max(len(pair_collisions), 1))
        if ambiguity_loss_terms:
            ambiguity_repulse_loss = torch.stack(ambiguity_loss_terms).mean()
            ambiguity_highrisk_pair_collision_rate = float(
                sum(ambiguity_pair_collisions) / max(len(ambiguity_pair_collisions), 1)
            )
        if confuser_loss_terms:
            confuser_separation_loss = torch.stack(confuser_loss_terms).mean()
            confuser_highrisk_pair_collision_rate = float(
                sum(confuser_pair_collisions) / max(len(confuser_pair_collisions), 1)
            )
        if purity_vals:
            unit_purity_metric = float(sum(purity_vals) / max(len(purity_vals), 1))
            unit_purity_loss = torch.as_tensor(1.0 - unit_purity_metric, device=device, dtype=obs_assignment.dtype)
        if entity_consistency_vals:
            target_entity_unit_consistency = float(sum(entity_consistency_vals) / max(len(entity_consistency_vals), 1))
            unit_track_stability = float(target_entity_unit_consistency)

    global_sem = torch.nn.functional.normalize(obs_z_sem.mean(dim=(1, 2)), dim=-1)
    if global_sem.shape[0] >= 2:
        pair_i, pair_j = torch.triu_indices(global_sem.shape[0], global_sem.shape[0], offset=1, device=device)
        separation = 1.0 - (global_sem[pair_i] * global_sem[pair_j]).sum(dim=-1)
        diff_instance_separation = max(float(diff_instance_separation), float(separation.mean().detach().cpu().item()))

    weighted_terms = []
    weight_sum = 0.0
    if float(assignment_sparsity_weight) > 0.0:
        weighted_terms.append(float(assignment_sparsity_weight) * entropy_loss)
        weight_sum += float(assignment_sparsity_weight)
    if float(assignment_temporal_consistency_weight) > 0.0:
        weighted_terms.append(float(assignment_temporal_consistency_weight) * temporal_consistency_loss)
        weight_sum += float(assignment_temporal_consistency_weight)
    if float(semantic_inertia_weight) > 0.0:
        weighted_terms.append(float(semantic_inertia_weight) * semantic_inertia_loss)
        weight_sum += float(semantic_inertia_weight)
    effective_instance_binding_weight = float(instance_binding_weight) if float(instance_binding_weight) > 0.0 else float(instance_consistency_weight)
    if float(effective_instance_binding_weight) > 0.0:
        weighted_terms.append(float(effective_instance_binding_weight) * instance_consistency_loss)
        weight_sum += float(effective_instance_binding_weight)
    if float(interinstance_repulse_weight) > 0.0:
        weighted_terms.append(float(interinstance_repulse_weight) * interinstance_repulse_loss)
        weight_sum += float(interinstance_repulse_weight)
    if float(ambiguity_repulse_weight) > 0.0:
        weighted_terms.append(float(ambiguity_repulse_weight) * ambiguity_repulse_loss)
        weight_sum += float(ambiguity_repulse_weight)
    if float(confuser_separation_weight) > 0.0:
        weighted_terms.append(float(confuser_separation_weight) * confuser_separation_loss)
        weight_sum += float(confuser_separation_weight)
    if float(appearance_refine_weight) > 0.0:
        weighted_terms.append(float(appearance_refine_weight) * appearance_refine_loss)
        weight_sum += float(appearance_refine_weight)
    if float(unit_purity_weight) > 0.0:
        weighted_terms.append(float(unit_purity_weight) * unit_purity_loss)
        weight_sum += float(unit_purity_weight)
    if float(dynsem_decorrelation_weight) > 0.0:
        weighted_terms.append(float(dynsem_decorrelation_weight) * decorrelation)
        weight_sum += float(dynsem_decorrelation_weight)
    if float(utilization_weight) > 0.0:
        weighted_terms.append(float(utilization_weight) * (unit_utilization_loss + min_active_loss))
        weight_sum += float(utilization_weight)
    if float(diversity_weight) > 0.0:
        weighted_terms.append(float(diversity_weight) * min_active_loss)
        weight_sum += float(diversity_weight)
    top2_vals = torch.topk(obs_assignment, k=min(2, obs_assignment.shape[-1]), dim=-1, largest=True, sorted=True).values
    secondary_mass = top2_vals[..., 1] if top2_vals.shape[-1] > 1 else torch.zeros_like(entropy)
    top2_floor_loss = torch.relu(torch.as_tensor(float(top2_mass_floor), device=device, dtype=secondary_mass.dtype) - secondary_mass)
    top2_floor_loss = (top2_floor_loss * obs_valid.float()).sum() / obs_valid.float().sum().clamp_min(1.0)
    if float(top2_floor_weight) > 0.0:
        weighted_terms.append(float(top2_floor_weight) * top2_floor_loss)
        weight_sum += float(top2_floor_weight)
    total = sum(weighted_terms) / max(float(weight_sum), 1e-6) if weighted_terms else zero

    tokenizer_metrics = trace_unit_aux.get("tokenizer_metrics", {}) if isinstance(trace_unit_aux.get("tokenizer_metrics", {}), dict) else {}
    factorized_metrics = trace_unit_aux.get("factorized_metrics", {}) if isinstance(trace_unit_aux.get("factorized_metrics", {}), dict) else {}
    broadcast_metrics = trace_unit_aux.get("broadcast_metrics", {}) if isinstance(trace_unit_aux.get("broadcast_metrics", {}), dict) else {}
    return total, {
        "trace_unit_loss": float(total.detach().cpu().item()),
        "assignment_entropy_mean": float(tokenizer_metrics.get("assignment_entropy_mean", 0.0)),
        "actual_top2_assignment_ratio": float(tokenizer_metrics.get("actual_top2_assignment_ratio", 0.0)),
        "active_unit_count_mean": float(tokenizer_metrics.get("active_unit_count_mean", 0.0)),
        "z_dyn_drift_mean": float(factorized_metrics.get("z_dyn_drift_mean", 0.0)),
        "z_sem_drift_mean": float(factorized_metrics.get("z_sem_drift_mean", 0.0)),
        "z_sem_to_z_dyn_drift_ratio": float(factorized_metrics.get("z_sem_to_z_dyn_drift_ratio", 0.0)),
        "same_instance_within_unit_consistency": float(instance_consistency_metric),
        "same_instance_dominant_unit_match_rate": float(same_instance_match_rate),
        "same_instance_assignment_cosine": float(same_instance_assignment_cosine),
        "different_instance_dominant_unit_collision_rate": float(diff_instance_collision_rate),
        "ambiguity_highrisk_pair_collision_rate": float(ambiguity_highrisk_pair_collision_rate),
        "confuser_highrisk_pair_collision_rate": float(confuser_highrisk_pair_collision_rate),
        "appearance_drift_highrisk_same_instance_match_rate": float(appearance_drift_highrisk_same_instance_match_rate),
        "appearance_drift_high_ratio": float(appearance_drift_high_ratio),
        "batch_appearance_drift_high_ratio": float(appearance_drift_high_ratio),
        "step_appearance_drift_high_count": float(appearance_drift_high_count),
        "appearance_signal_valid_count": float(appearance_signal_valid_count),
        "appearance_refine_loss_nonzero": float(appearance_refine_loss_nonzero),
        "appearance_refine_loss_value": float(appearance_refine_loss.detach().cpu().item()),
        "unit_purity_by_instance_id": float(unit_purity_metric),
        "unit_track_stability_over_time": float(unit_track_stability),
        "target_entity_to_dominant_unit_consistency": float(target_entity_unit_consistency),
        "different_instance_between_unit_separation": float(diff_instance_separation),
        "unit_semantic_stability_over_time": float(factorized_metrics.get("unit_semantic_stability_over_time", 0.0)),
        "broadcast_residual_norm_mean": float(broadcast_metrics.get("broadcast_residual_norm_mean", 0.0)),
        "unit_utilization_mean": float(utilization_mean),
        "true_instance_ratio_per_batch": float(true_instance_ratio_per_batch),
    }


def _evaluate(
    *,
    stage1_model: TraceCausalTransformer,
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    future_semantic_state_head: SemanticTraceStateHead | None,
    structure_mode: str,
    trace_unit_tokenizer: TraceUnitTokenizer | None,
    trace_unit_factorized_state: TraceUnitFactorizedState | None,
    trace_unit_handshake: TraceUnitHandshake | None,
    trace_unit_broadcast: TraceUnitBroadcast | None,
    trace_unit_disable_instance_path: bool,
    loader: DataLoader,
    device: torch.device,
    pin_memory: bool,
    obs_len: int,
    fut_len: int,
    max_batches: int,
    semantic_source_mainline: str,
    semantic_state_feedback_adapter: SemanticStateFeedbackAdapter | None = None,
    semantic_state_feedback_enabled: bool = False,
    semantic_state_feedback_alpha: float = 0.05,
    semantic_state_feedback_stopgrad_state: bool = True,
    semantic_state_feedback_mode: str = "readout_only",
) -> Dict[str, Any]:
    semantic_encoder.eval()
    semantic_fusion.eval()
    readout_head.eval()
    if future_semantic_state_head is not None:
        future_semantic_state_head.eval()
    if semantic_state_feedback_adapter is not None:
        semantic_state_feedback_adapter.eval()

    tf_sse = 0.0
    tf_count = 0.0
    free_l2_sum = 0.0
    free_l2_count = 0.0
    free_endpoint_sum = 0.0
    free_endpoint_count = 0.0
    total_loss_ref_sum = 0.0
    batch_count = 0
    gate_vals: List[float] = []
    nonempty_count = 0
    free_semantic_state_valid_count = 0
    free_semantic_state_total_count = 0

    with torch.no_grad():
        for bi, raw_batch in enumerate(loader):
            if int(max_batches) > 0 and bi >= int(max_batches):
                break
            batch = _to_device(raw_batch, device=device, non_blocking=bool(pin_memory and device.type == "cuda"))

            tf_out = _teacher_forced_predict(
                stage1_model=stage1_model,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                future_semantic_state_head=future_semantic_state_head,
                semantic_state_feedback_adapter=semantic_state_feedback_adapter,
                semantic_state_feedback_enabled=semantic_state_feedback_enabled,
                semantic_state_feedback_alpha=semantic_state_feedback_alpha,
                semantic_state_feedback_stopgrad_state=semantic_state_feedback_stopgrad_state,
                semantic_state_feedback_mode=semantic_state_feedback_mode,
                structure_mode=str(structure_mode),
                trace_unit_tokenizer=trace_unit_tokenizer,
                trace_unit_factorized_state=trace_unit_factorized_state,
                trace_unit_handshake=trace_unit_handshake,
                trace_unit_broadcast=trace_unit_broadcast,
                trace_unit_disable_instance_path=bool(trace_unit_disable_instance_path),
                batch=batch,
                obs_len=int(obs_len),
                semantic_source_mainline=str(semantic_source_mainline),
            )
            fr_out = _free_rollout_predict(
                stage1_model=stage1_model,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                future_semantic_state_head=future_semantic_state_head,
                semantic_state_feedback_adapter=semantic_state_feedback_adapter,
                semantic_state_feedback_enabled=semantic_state_feedback_enabled,
                semantic_state_feedback_alpha=semantic_state_feedback_alpha,
                semantic_state_feedback_stopgrad_state=semantic_state_feedback_stopgrad_state,
                semantic_state_feedback_mode=semantic_state_feedback_mode,
                structure_mode=str(structure_mode),
                trace_unit_tokenizer=trace_unit_tokenizer,
                trace_unit_factorized_state=trace_unit_factorized_state,
                trace_unit_handshake=trace_unit_handshake,
                trace_unit_broadcast=trace_unit_broadcast,
                trace_unit_disable_instance_path=bool(trace_unit_disable_instance_path),
                batch=batch,
                obs_len=int(obs_len),
                fut_len=int(fut_len),
                semantic_source_mainline=str(semantic_source_mainline),
            )
            free_semantic_state_total_count += 1
            free_semantic_state_valid_count += int(bool(fr_out.get("future_semantic_state_output_valid", False)))

            tf_sq = ((tf_out["pred_coord"] - tf_out["target_coord"]) ** 2).sum(dim=-1)
            tf_mask = tf_out["valid_mask"].float()
            tf_sse += float((tf_sq * tf_mask).sum().item())
            tf_count += float(tf_mask.sum().item())

            free_l2 = torch.sqrt(((fr_out["pred_coord"] - fr_out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
            free_mask = fr_out["valid_mask"].float()
            free_l2_sum += float((free_l2 * free_mask).sum().item())
            free_l2_count += float(free_mask.sum().item())

            endpoint_l2 = torch.sqrt(
                ((fr_out["pred_coord"][:, -1] - fr_out["target_coord"][:, -1]) ** 2).sum(dim=-1).clamp_min(1e-12)
            )
            endpoint_mask = fr_out["valid_mask"][:, -1].float()
            free_endpoint_sum += float((endpoint_l2 * endpoint_mask).sum().item())
            free_endpoint_count += float(endpoint_mask.sum().item())

            total_loss_ref_sum += float(_masked_mse_coord(tf_out["pred_coord"], tf_out["target_coord"], tf_out["valid_mask"]).item())
            gate_vals.append(float(tf_out["gate_mean"]))
            nonempty_count += 1 if bool(tf_out["semantic_input_nonempty"]) else 0
            batch_count += 1

    teacher_forced_coord_loss = float(tf_sse / max(tf_count, 1.0))
    free_rollout_coord_mean_l2 = float(free_l2_sum / max(free_l2_count, 1.0))
    free_rollout_endpoint_l2 = float(free_endpoint_sum / max(free_endpoint_count, 1.0))
    total_loss_reference = float(total_loss_ref_sum / max(batch_count, 1))

    return {
        "teacher_forced_coord_loss": teacher_forced_coord_loss,
        "free_rollout_coord_mean_l2": free_rollout_coord_mean_l2,
        "free_rollout_endpoint_l2": free_rollout_endpoint_l2,
        "total_loss_reference": total_loss_reference,
        "tapvid_style_eval": {
            "compatible": False,
            "status": "not_supported_in_current_stage2_trainer",
        },
        "tapvid3d_limited_eval": {
            "compatible": False,
            "status": "not_supported_in_current_stage2_trainer",
        },
        "semantic_branch_metrics": {
            "eval_gate_mean": float(sum(gate_vals) / max(len(gate_vals), 1)),
            "semantic_input_nonempty_ratio": float(nonempty_count / max(batch_count, 1)),
            "eval_batches": int(batch_count),
            "free_rollout_semantic_state_output": bool(future_semantic_state_head is not None),
            "semantic_state_feedback_in_free_rollout": False,
            "future_semantic_state_output_valid_ratio": float(free_semantic_state_valid_count / max(free_semantic_state_total_count, 1)),
        },
    }


def _available_tertiary_metric(metrics: Dict[str, Any]) -> float:
    try:
        return float(metrics.get("teacher_forced_coord_loss", 1e9))
    except Exception:
        return 1e9


def _rank_key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("free_rollout_endpoint_l2", 1e9)),
        float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
        float(_available_tertiary_metric(metrics)),
    )


def _metric_triplet(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "teacher_forced_coord_loss": float(metrics.get("teacher_forced_coord_loss", 1e9)),
        "free_rollout_coord_mean_l2": float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
        "free_rollout_endpoint_l2": float(metrics.get("free_rollout_endpoint_l2", 1e9)),
        "total_loss_reference": float(metrics.get("total_loss_reference", 1e9)),
    }


def _semantic_hard_composite_score(metrics: Dict[str, Any]) -> float:
    endpoint = float(metrics.get("free_rollout_endpoint_l2", 1e9))
    coord = float(metrics.get("free_rollout_coord_mean_l2", 1e9))
    return float(0.7 * endpoint + 0.3 * coord)


class SemanticRescueAuxHeads(torch.nn.Module):
    def __init__(self, semantic_dim: int, target_dim: int = 10, readout_dim: int | None = None) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.feature_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(semantic_dim)),
            torch.nn.Linear(int(semantic_dim), self.target_dim),
        )
        self.endpoint_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(semantic_dim)),
            torch.nn.Linear(int(semantic_dim), 2),
        )
        self.readout_feature_head: torch.nn.Module | None = None
        if readout_dim is not None and int(readout_dim) > 0:
            self.readout_feature_head = torch.nn.Sequential(
                torch.nn.LayerNorm(int(readout_dim)),
                torch.nn.Linear(int(readout_dim), self.target_dim),
            )

    def forward(self, semantic_tokens: torch.Tensor, readout_tokens: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        out = {
            "feature_target": self.feature_head(semantic_tokens),
            "endpoint": self.endpoint_head(semantic_tokens),
        }
        if readout_tokens is not None and self.readout_feature_head is not None:
            out["readout_feature_target"] = self.readout_feature_head(readout_tokens)
        return out


def _bootstrap_targets_from_batch(
    *,
    batch: Dict[str, Any],
    cache: Dict[str, torch.Tensor],
    device: torch.device,
    target_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    fallback_raw = batch["semantic_features"].to(device=device, dtype=torch.float32)
    fallback = torch.zeros(
        (*fallback_raw.shape[:-1], int(target_dim)),
        device=device,
        dtype=torch.float32,
    )
    fallback_dim = min(int(fallback_raw.shape[-1]), int(target_dim))
    fallback[..., :fallback_dim] = fallback_raw[..., :fallback_dim]
    if not cache:
        valid = batch["semantic_mask"].to(device=device, dtype=torch.bool)
        return fallback, valid, 0.0

    targets = torch.zeros_like(fallback)
    valid = torch.zeros(batch["semantic_mask"].shape, dtype=torch.bool, device=device)
    hits = 0
    total = 0
    metas = batch.get("meta", [])
    for bi, meta in enumerate(metas if isinstance(metas, list) else []):
        if not isinstance(meta, dict):
            continue
        key = f"{str(meta.get('dataset', '')).upper()}::{str(meta.get('clip_id', ''))}"
        total += 1
        target = cache.get(key)
        if target is None:
            targets[bi] = fallback[bi]
            valid[bi] = batch["semantic_mask"][bi].to(device=device, dtype=torch.bool)
            continue
        dim = min(int(target.numel()), int(target_dim))
        targets[bi, :, :dim] = target[:dim].to(device=device, dtype=torch.float32)[None, :]
        valid[bi] = batch["semantic_mask"][bi].to(device=device, dtype=torch.bool)
        hits += 1
    coverage = float(hits / max(total, 1))
    return targets, valid, coverage


def _clamp01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def _sparse_topk_mask(scores: torch.Tensor, valid: torch.Tensor, keep_ratio: float, min_tokens: int) -> torch.Tensor:
    mask = torch.zeros_like(valid, dtype=torch.bool)
    if scores.ndim != 2 or valid.ndim != 2:
        return mask
    ratio = _clamp01(float(keep_ratio))
    min_keep = max(int(min_tokens), 1)
    for bi in range(int(scores.shape[0])):
        valid_idx = torch.nonzero(valid[bi], as_tuple=False).squeeze(-1)
        n = int(valid_idx.numel())
        if n <= 0:
            continue
        keep_count = max(min_keep, int(np.ceil(float(n) * ratio)))
        keep_count = min(max(keep_count, 1), n)
        vals = scores[bi, valid_idx]
        pick = torch.topk(vals, k=keep_count, largest=True, sorted=False).indices
        mask[bi, valid_idx[pick]] = True
    return mask


def _sparse_quantile_mask(scores: torch.Tensor, valid: torch.Tensor, quantile: float, min_tokens: int) -> torch.Tensor:
    keep_ratio = max(1.0 - _clamp01(float(quantile)), 0.0)
    return _sparse_topk_mask(scores=scores, valid=valid, keep_ratio=keep_ratio, min_tokens=min_tokens)


def _hard_topk_query_mask_v2(scores: torch.Tensor, valid: torch.Tensor, topk: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = torch.zeros_like(valid, dtype=torch.bool)
    if scores.shape != valid.shape or scores.ndim != 3:
        return mask, {
            "activated_query_count_mean": 0.0,
            "activated_query_ratio": 0.0,
            "per_batch_sparsity_mean": 0.0,
            "per_batch_sparsity_std": 0.0,
            "raw_quantile_ratio": 0.0,
            "capped_ratio": 0.0,
            "actual_gate_positive_ratio": 0.0,
        }

    counts: List[float] = []
    ratios: List[float] = []
    k = max(int(topk), 1)
    for bi in range(int(scores.shape[0])):
        flat_valid = valid[bi].reshape(-1)
        flat_scores = scores[bi].reshape(-1)
        valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        n = int(valid_idx.numel())
        if n <= 0:
            counts.append(0.0)
            ratios.append(0.0)
            continue
        keep = min(k, n)
        pick_local = torch.topk(flat_scores[valid_idx], k=keep, largest=True, sorted=False).indices
        chosen = valid_idx[pick_local]
        flat_mask = torch.zeros_like(flat_valid, dtype=torch.bool)
        flat_mask[chosen] = True
        mask[bi] = flat_mask.view_as(valid[bi])
        counts.append(float(keep))
        ratios.append(float(keep / max(n, 1)))

    mean_ratio = float(sum(ratios) / max(len(ratios), 1))
    return mask, {
        "activated_query_count_mean": float(sum(counts) / max(len(counts), 1)),
        "activated_query_ratio": mean_ratio,
        "per_batch_sparsity_mean": mean_ratio,
        "per_batch_sparsity_std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
        "raw_quantile_ratio": mean_ratio,
        "capped_ratio": mean_ratio,
        "actual_gate_positive_ratio": mean_ratio,
    }


def _capped_quantile_query_mask_v2(
    scores: torch.Tensor,
    valid: torch.Tensor,
    quantile: float,
    max_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = torch.zeros_like(valid, dtype=torch.bool)
    if scores.shape != valid.shape or scores.ndim != 3:
        return mask, {
            "activated_query_count_mean": 0.0,
            "activated_query_ratio": 0.0,
            "per_batch_sparsity_mean": 0.0,
            "per_batch_sparsity_std": 0.0,
            "raw_quantile_ratio": 0.0,
            "capped_ratio": 0.0,
            "actual_gate_positive_ratio": 0.0,
        }

    q = _clamp01(float(quantile))
    cap = _clamp01(float(max_ratio))
    counts: List[float] = []
    final_ratios: List[float] = []
    raw_ratios: List[float] = []
    cap_ratios: List[float] = []
    for bi in range(int(scores.shape[0])):
        flat_valid = valid[bi].reshape(-1)
        flat_scores = scores[bi].reshape(-1)
        valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        n = int(valid_idx.numel())
        if n <= 0:
            counts.append(0.0)
            final_ratios.append(0.0)
            raw_ratios.append(0.0)
            cap_ratios.append(0.0)
            continue

        vals = flat_scores[valid_idx]
        threshold = float(torch.quantile(vals.detach(), q).item()) if n >= 2 else float(vals[0].detach().item())
        raw_keep_idx = valid_idx[vals >= threshold]
        if int(raw_keep_idx.numel()) <= 0:
            raw_keep_idx = valid_idx[torch.topk(vals, k=1, largest=True, sorted=False).indices]

        max_keep = max(1, int(np.floor(float(n) * cap)))
        max_keep = min(max_keep, n)
        chosen = raw_keep_idx
        if int(chosen.numel()) > max_keep:
            raw_scores = flat_scores[chosen]
            keep_local = torch.topk(raw_scores, k=max_keep, largest=True, sorted=False).indices
            chosen = chosen[keep_local]

        flat_mask = torch.zeros_like(flat_valid, dtype=torch.bool)
        flat_mask[chosen] = True
        mask[bi] = flat_mask.view_as(valid[bi])

        raw_ratios.append(float(raw_keep_idx.numel() / max(n, 1)))
        cap_ratios.append(float(max_keep / max(n, 1)))
        final_ratios.append(float(chosen.numel() / max(n, 1)))
        counts.append(float(chosen.numel()))

    mean_ratio = float(sum(final_ratios) / max(len(final_ratios), 1))
    return mask, {
        "activated_query_count_mean": float(sum(counts) / max(len(counts), 1)),
        "activated_query_ratio": mean_ratio,
        "per_batch_sparsity_mean": mean_ratio,
        "per_batch_sparsity_std": float(np.std(final_ratios, ddof=1)) if len(final_ratios) > 1 else 0.0,
        "raw_quantile_ratio": float(sum(raw_ratios) / max(len(raw_ratios), 1)),
        "capped_ratio": float(sum(cap_ratios) / max(len(cap_ratios), 1)),
        "actual_gate_positive_ratio": mean_ratio,
    }


def _semantic_rescue_loss(
    *,
    mode: str,
    aux_heads: SemanticRescueAuxHeads | None,
    tf_out: Dict[str, Any],
    batch: Dict[str, Any],
    bootstrap_cache: Dict[str, torch.Tensor],
    device: torch.device,
    current_step: int,
    resume_global_step: int,
    semantic_alignment_loss_weight: float,
    query_persistence_consistency_loss_weight: float,
    readout_semantic_alignment_loss_weight: float,
    persistence_contrastive_or_ranking_loss_weight: float,
    semantic_aux_subset_weighting_strength: float,
    confidence_gated_alignment_loss_weight: float,
    sparse_persistence_contrastive_loss_weight: float,
    confidence_gating_margin_threshold: float,
    confidence_gating_temperature: float,
    semantic_hard_score_threshold: float,
    aux_loss_delay_steps: int,
    aux_loss_ramp_steps: int,
    v4_sparse_gating_family: str,
    v4_gating_quantile: float,
    v4_topk_token_ratio: float,
    v4_topk_min_tokens: int,
    v4_gate_min_strength: float,
    v4_persistence_value_quantile: float,
    v4_persistence_max_pairs: int,
    v4_persistence_margin: float,
    v5_gating_family: str,
    v5_topk_query_k: int,
    v5_capped_quantile: float,
    v5_max_affected_ratio: float,
    v5_gate_min_strength: float,
    v5_max_pairs_per_sample: int,
    v5_hard_negative_cap: int,
    v5_pair_sampling_temperature: float,
    v6_gating_family: str,
    v6_topk_query_k: int,
    v6_capped_quantile: float,
    v6_max_affected_ratio: float,
    v6_gate_min_strength: float,
    v6_strict_max_pairs_per_sample: int,
    v6_hard_negative_cap: int,
    v6_pair_sampling_temperature: float,
    v6_guaranteed_min_pairs_per_sample: int,
    v6_two_level_pair_mining_enabled: bool,
    v6_relaxed_motion_threshold: float,
    v6_relaxed_area_jump_threshold: float,
    v6_relaxed_small_query_threshold: float,
    v6_relaxed_appearance_shift_threshold: float,
    v6_relaxed_center_interaction_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if aux_heads is None or str(mode) == "none":
        zero = tf_out["pred_coord"].sum() * 0.0
        return zero, {
            "semantic_rescue_loss": 0.0,
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": 0.0,
            "persistence_contrastive_or_ranking_loss": 0.0,
            "confidence_gated_alignment_loss": 0.0,
            "sparse_persistence_contrastive_loss": 0.0,
            "confidence_gated_affected_sample_ratio": 0.0,
            "low_confidence_sample_ratio": 0.0,
            "confidence_metric_threshold": float(confidence_gating_margin_threshold),
            "confidence_metric_temperature": float(confidence_gating_temperature),
            "confidence_metric_definition": 0.0,
            "positive_pair_count": 0.0,
            "hard_negative_count": 0.0,
            "effective_pair_coverage_ratio": 0.0,
            "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
            "aux_loss_delay_steps": float(aux_loss_delay_steps),
            "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": 0.0,
            "actual_gate_positive_ratio": 0.0,
            "activated_query_count": 0.0,
            "activated_query_ratio": 0.0,
            "per_batch_sparsity_mean": 0.0,
            "per_batch_sparsity_std": 0.0,
            "raw_quantile_ratio": 0.0,
            "capped_ratio": 0.0,
            "valuable_pair_ratio": 0.0,
            "max_pairs_per_sample": float(v5_max_pairs_per_sample),
            "hard_negative_cap": float(v5_hard_negative_cap),
            "pair_sampling_temperature": float(v5_pair_sampling_temperature),
            "final_effective_aux_weight": 0.0,
            "fallback_trigger_rate": 0.0,
            "guaranteed_pair_count": 0.0,
            "strict_pair_ratio": 0.0,
            "fallback_pair_ratio": 0.0,
        }

    semantic_tokens = tf_out["semantic_tokens"]
    token_mask = batch["token_mask"].to(device=device, dtype=torch.bool)
    semantic_mask = batch["semantic_mask"].to(device=device, dtype=torch.bool) & token_mask
    readout_tokens = tf_out.get("future_fused_hidden")
    readout_tokens = readout_tokens[:, -1] if isinstance(readout_tokens, torch.Tensor) and readout_tokens.ndim == 4 else None
    aux = aux_heads(semantic_tokens, readout_tokens=readout_tokens)

    weighted_terms: List[torch.Tensor] = []
    weight_sum = 0.0
    cache_hit_ratio = 0.0
    mode_norm = str(mode).strip().lower()
    align_weight = float(semantic_alignment_loss_weight)
    query_weight = float(query_persistence_consistency_loss_weight)
    if mode_norm == "align" and align_weight <= 0.0:
        align_weight = 1.0
    if mode_norm == "querypersist" and query_weight <= 0.0:
        query_weight = 1.0
    if mode_norm.startswith("v2"):
        readout_align_weight = float(readout_semantic_alignment_loss_weight)
        contrastive_weight = float(persistence_contrastive_or_ranking_loss_weight)
        if mode_norm == "v2readoutalign" and readout_align_weight <= 0.0:
            readout_align_weight = 1.0
        if mode_norm == "v2readoutpersist":
            if readout_align_weight <= 0.0:
                readout_align_weight = 1.0
            if contrastive_weight <= 0.0:
                contrastive_weight = 0.25
        if mode_norm == "v2readouthard" and readout_align_weight <= 0.0:
            readout_align_weight = 1.0

        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache,
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        valid = semantic_mask & target_valid
        sample_weights = _semantic_hard_sample_weights(
            batch=batch,
            device=device,
            strength=float(semantic_aux_subset_weighting_strength),
        )[:, None]
        aux_weights = torch.where(valid, sample_weights.expand_as(valid).to(torch.float32), torch.zeros_like(valid, dtype=torch.float32))
        denom = aux_weights.sum().clamp_min(1.0)
        readout_align_loss = semantic_tokens.sum() * 0.0
        contrastive_loss = semantic_tokens.sum() * 0.0

        readout_pred = aux.get("readout_feature_target", aux["feature_target"])
        if readout_align_weight > 0.0:
            pred = torch.nn.functional.normalize(readout_pred, dim=-1)
            tgt = torch.nn.functional.normalize(target, dim=-1)
            cosine = 1.0 - (pred * tgt).sum(dim=-1)
            readout_align_loss = (cosine * aux_weights).sum() / denom
            weighted_terms.append(float(readout_align_weight) * readout_align_loss)
            weight_sum += float(readout_align_weight)

        if contrastive_weight > 0.0:
            flat_valid = valid.reshape(-1)
            flat_pred = torch.nn.functional.normalize(readout_pred.reshape(-1, readout_pred.shape[-1])[flat_valid], dim=-1)
            flat_tgt = torch.nn.functional.normalize(target.reshape(-1, target.shape[-1])[flat_valid], dim=-1)
            if flat_pred.shape[0] > 1:
                logits = flat_pred @ flat_tgt.T / 0.07
                labels = torch.arange(flat_pred.shape[0], device=device)
                contrastive_loss = 0.5 * (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                )
                weighted_terms.append(float(contrastive_weight) * contrastive_loss)
                weight_sum += float(contrastive_weight)

        if not weighted_terms:
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "confidence_gated_alignment_loss": 0.0,
                "sparse_persistence_contrastive_loss": 0.0,
                "confidence_gated_affected_sample_ratio": 0.0,
                "low_confidence_sample_ratio": 0.0,
                "confidence_metric_threshold": float(confidence_gating_margin_threshold),
                "confidence_metric_temperature": float(confidence_gating_temperature),
                "confidence_metric_definition": 0.0,
                "positive_pair_count": 0.0,
                "hard_negative_count": 0.0,
                "effective_pair_coverage_ratio": 0.0,
                "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
                "aux_loss_delay_steps": float(aux_loss_delay_steps),
                "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
            }
        loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
        return loss, {
            "semantic_rescue_loss": float(loss.detach().cpu().item()),
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": float(readout_align_loss.detach().cpu().item()),
            "persistence_contrastive_or_ranking_loss": float(contrastive_loss.detach().cpu().item()),
            "confidence_gated_alignment_loss": 0.0,
            "sparse_persistence_contrastive_loss": 0.0,
            "confidence_gated_affected_sample_ratio": 0.0,
            "low_confidence_sample_ratio": 0.0,
            "confidence_metric_threshold": float(confidence_gating_margin_threshold),
            "confidence_metric_temperature": float(confidence_gating_temperature),
            "confidence_metric_definition": 0.0,
            "positive_pair_count": 0.0,
            "hard_negative_count": 0.0,
            "effective_pair_coverage_ratio": 0.0,
            "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
            "aux_loss_delay_steps": float(aux_loss_delay_steps),
            "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
        }

    if mode_norm.startswith("v3"):
        readout_align_weight = float(confidence_gated_alignment_loss_weight)
        sparse_weight = float(sparse_persistence_contrastive_loss_weight)
        if mode_norm == "v3confalign" and readout_align_weight <= 0.0:
            readout_align_weight = 1.0
        if mode_norm in {"v3confpersist", "v3confpersistdelay", "v3confhardsidecar"}:
            if readout_align_weight <= 0.0:
                readout_align_weight = 1.0
            if sparse_weight <= 0.0:
                sparse_weight = 0.05

        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache,
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        valid = semantic_mask & target_valid
        hard_stats = _semantic_hard_sample_stats(batch=batch, device=device)
        hard_score = hard_stats["hard_score"][:, None].expand_as(valid).to(torch.float32)
        hard_relevance = (
            (hard_stats["area_range"] >= float(semantic_hard_score_threshold))
            | (hard_stats["center_interaction"] >= 0.30)
            | (hard_stats["appearance_shift"] >= 0.20)
            | (hard_stats["small_score"] >= 0.25)
        )[:, None].expand_as(valid)

        readout_pred = aux.get("readout_feature_target", aux["feature_target"])
        pred = torch.nn.functional.normalize(readout_pred, dim=-1)
        tgt = torch.nn.functional.normalize(target, dim=-1)
        pos_sim = (pred * tgt).sum(dim=-1)
        neg_sim = torch.full_like(pos_sim, -1.0)
        flat_valid = valid.reshape(-1)
        if int(flat_valid.sum().item()) >= 2:
            flat_pred = pred.reshape(-1, pred.shape[-1])[flat_valid]
            flat_tgt = tgt.reshape(-1, tgt.shape[-1])[flat_valid]
            flat_sim = flat_pred @ flat_tgt.T
            flat_sim = flat_sim.masked_fill(
                torch.eye(flat_sim.shape[0], device=device, dtype=torch.bool),
                -1e9,
            )
            flat_neg = flat_sim.max(dim=-1).values
            flat_neg = torch.where(torch.isfinite(flat_neg), flat_neg, torch.full_like(flat_neg, -1.0))
            neg_sim_flat = torch.full((flat_valid.shape[0],), -1.0, device=device, dtype=torch.float32)
            neg_sim_flat[flat_valid] = flat_neg
            neg_sim = neg_sim_flat.view_as(pos_sim)
        margin = pos_sim - neg_sim
        gate_raw = torch.sigmoid((float(confidence_gating_margin_threshold) - margin) / max(float(confidence_gating_temperature), 1e-4))
        gate = gate_raw * (1.0 + 0.75 * hard_score)
        gate = torch.where(valid, gate, torch.zeros_like(gate))
        gate = gate.clamp(0.0, 3.0)
        gate_denom = gate.sum().clamp_min(1.0)
        conf_align_loss = semantic_tokens.sum() * 0.0
        if readout_align_weight > 0.0:
            cosine = 1.0 - pos_sim
            conf_align_loss = (cosine * gate).sum() / gate_denom
            weighted_terms.append(float(readout_align_weight) * conf_align_loss)
            weight_sum += float(readout_align_weight)

        future_readout = tf_out.get("future_fused_hidden")
        sparse_loss = semantic_tokens.sum() * 0.0
        positive_pair_count = 0.0
        hard_negative_count = 0.0
        pair_coverage_ratio = 0.0
        if (
            sparse_weight > 0.0
            and isinstance(future_readout, torch.Tensor)
            and future_readout.ndim == 4
            and future_readout.shape[1] >= 2
            and aux_heads.readout_feature_head is not None
        ):
            future_proj = aux_heads.readout_feature_head(future_readout)
            first_proj = torch.nn.functional.normalize(future_proj[:, 0], dim=-1)
            last_proj = torch.nn.functional.normalize(future_proj[:, -1], dim=-1)
            selected = valid & hard_relevance
            total_candidates = float(selected.float().sum().item())
            flat_sel = selected.reshape(-1)
            if int(flat_sel.sum().item()) >= 2:
                flat_first = first_proj.reshape(-1, first_proj.shape[-1])[flat_sel]
                flat_last = last_proj.reshape(-1, last_proj.shape[-1])[flat_sel]
                logits = flat_first @ flat_last.T
                pos = torch.diagonal(logits)
                logits_wo_diag = logits.masked_fill(torch.eye(logits.shape[0], device=device, dtype=torch.bool), -1e9)
                max_neg = logits_wo_diag.max(dim=-1).values
                max_neg = torch.where(torch.isfinite(max_neg), max_neg, torch.full_like(max_neg, -1.0))
                sparse_loss = torch.relu(0.10 + max_neg - pos).mean()
                weighted_terms.append(float(sparse_weight) * sparse_loss)
                weight_sum += float(sparse_weight)
                positive_pair_count = float(flat_first.shape[0])
                hard_negative_count = float((logits_wo_diag > (pos[:, None] - 0.10)).float().sum().item())
                pair_coverage_ratio = float(positive_pair_count / max(total_candidates, 1.0))
            else:
                positive_pair_count = float(flat_sel.sum().item())
                pair_coverage_ratio = float(positive_pair_count / max(total_candidates, 1.0))

        if not weighted_terms:
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "confidence_gated_alignment_loss": 0.0,
                "sparse_persistence_contrastive_loss": 0.0,
                "confidence_gated_affected_sample_ratio": float(((gate > 0.25) & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
                "low_confidence_sample_ratio": float(((gate_raw > 0.5) & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
                "confidence_metric_threshold": float(confidence_gating_margin_threshold),
                "confidence_metric_temperature": float(confidence_gating_temperature),
                "confidence_metric_definition": 1.0,
                "positive_pair_count": float(positive_pair_count),
                "hard_negative_count": float(hard_negative_count),
                "effective_pair_coverage_ratio": float(pair_coverage_ratio),
                "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
                "aux_loss_delay_steps": float(aux_loss_delay_steps),
                "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
            }
        loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
        return loss, {
            "semantic_rescue_loss": float(loss.detach().cpu().item()),
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "persistence_contrastive_or_ranking_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "sparse_persistence_contrastive_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_affected_sample_ratio": float(((gate > 0.25) & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
            "low_confidence_sample_ratio": float(((gate_raw > 0.5) & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
            "confidence_metric_threshold": float(confidence_gating_margin_threshold),
            "confidence_metric_temperature": float(confidence_gating_temperature),
            "confidence_metric_definition": 1.0,
            "positive_pair_count": float(positive_pair_count),
            "hard_negative_count": float(hard_negative_count),
            "effective_pair_coverage_ratio": float(pair_coverage_ratio),
            "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
            "aux_loss_delay_steps": float(aux_loss_delay_steps),
            "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
        }

    if mode_norm.startswith("v6") or mode_norm.startswith("v7"):
        readout_align_weight = float(confidence_gated_alignment_loss_weight)
        sparse_weight = float(sparse_persistence_contrastive_loss_weight)
        if readout_align_weight <= 0.0:
            readout_align_weight = 1.0
        if mode_norm == "v7alignonly":
            sparse_weight = 0.0
        elif sparse_weight <= 0.0:
            sparse_weight = 0.05

        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache,
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        future_readout = tf_out.get("future_fused_hidden")
        if (
            not isinstance(future_readout, torch.Tensor)
            or future_readout.ndim != 4
            or aux_heads.readout_feature_head is None
        ):
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "confidence_gated_alignment_loss": 0.0,
                "sparse_persistence_contrastive_loss": 0.0,
                "confidence_gated_affected_sample_ratio": 0.0,
                "low_confidence_sample_ratio": 0.0,
                "confidence_metric_threshold": float(confidence_gating_margin_threshold),
                "confidence_metric_temperature": float(confidence_gating_temperature),
                "confidence_metric_definition": 4.0,
                "positive_pair_count": 0.0,
                "hard_negative_count": 0.0,
                "effective_pair_coverage_ratio": 0.0,
                "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
                "aux_loss_delay_steps": float(aux_loss_delay_steps),
                "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
                "actual_gate_positive_ratio": 0.0,
                "activated_query_count": 0.0,
                "activated_query_ratio": 0.0,
                "per_batch_sparsity_mean": 0.0,
                "per_batch_sparsity_std": 0.0,
                "raw_quantile_ratio": 0.0,
                "capped_ratio": 0.0,
                "valuable_pair_ratio": 0.0,
                "sparse_gate_selected_ratio": 0.0,
                "high_value_pair_ratio": 0.0,
                "high_value_pair_count": 0.0,
                "persistence_candidate_pair_count": 0.0,
                "max_pairs_per_sample": float(v6_strict_max_pairs_per_sample),
                "hard_negative_cap": float(v6_hard_negative_cap),
                "pair_sampling_temperature": float(v6_pair_sampling_temperature),
                "final_effective_aux_weight": 0.0,
                "fallback_trigger_rate": 0.0,
                "guaranteed_pair_count": 0.0,
                "strict_pair_ratio": 0.0,
                "fallback_pair_ratio": 0.0,
            }

        query_valid = tf_out["valid_mask"] & target_valid[:, None, :]
        hard_stats = _semantic_hard_sample_stats(batch=batch, device=device)
        hard_score = hard_stats["hard_score"][:, None, None].expand_as(query_valid).to(torch.float32)
        small_score = hard_stats["small_score"][:, None, None].expand_as(query_valid).to(torch.float32)
        center_interaction = hard_stats["center_interaction"][:, None, None].expand_as(query_valid).to(torch.float32)
        appearance_shift = hard_stats["appearance_shift"][:, None, None].expand_as(query_valid).to(torch.float32)

        future_proj = aux_heads.readout_feature_head(future_readout)
        pred = torch.nn.functional.normalize(future_proj, dim=-1)
        tgt = torch.nn.functional.normalize(target[:, None, :, :].expand(-1, pred.shape[1], -1, -1), dim=-1)
        pos_sim = (pred * tgt).sum(dim=-1)
        neg_sim = torch.full_like(pos_sim, -1.0)

        flat_valid = query_valid.reshape(-1)
        if int(flat_valid.sum().item()) >= 2:
            flat_pred = pred.reshape(-1, pred.shape[-1])[flat_valid]
            flat_tgt = tgt.reshape(-1, tgt.shape[-1])[flat_valid]
            sample_ids = torch.arange(pred.shape[0], device=device)[:, None, None].expand_as(query_valid).reshape(-1)[flat_valid]
            flat_sim = flat_pred @ flat_tgt.T
            same_sample = sample_ids[:, None] == sample_ids[None, :]
            flat_sim = flat_sim.masked_fill(same_sample, -1e9)
            flat_neg = flat_sim.max(dim=-1).values
            flat_neg = torch.where(torch.isfinite(flat_neg), flat_neg, torch.full_like(flat_neg, -1.0))
            neg_sim_flat = torch.full((flat_valid.shape[0],), -1.0, device=device, dtype=torch.float32)
            neg_sim_flat[flat_valid] = flat_neg
            neg_sim = neg_sim_flat.view_as(pos_sim)

        future_state = batch["fut_state"]
        area = (future_state[..., 6].abs() * future_state[..., 7].abs()).clamp(0.0, 1.0)
        small_query = (1.0 - area).clamp(0.0, 1.0)
        coord = future_state[..., 0:2]
        step_motion = torch.zeros_like(pos_sim)
        if coord.shape[1] >= 2:
            motion_delta = torch.sqrt(((coord[:, 1:] - coord[:, :-1]) ** 2).sum(dim=-1).clamp_min(1e-12))
            step_motion[:, 1:] = motion_delta
        motion_norm = step_motion / step_motion.amax(dim=1, keepdim=True).clamp_min(1e-6)
        area_jump = torch.zeros_like(pos_sim)
        if area.shape[1] >= 2:
            area_delta = (area[:, 1:] - area[:, :-1]).abs()
            area_jump[:, 1:] = area_delta
        area_jump = area_jump / area_jump.amax(dim=1, keepdim=True).clamp_min(1e-6)

        margin = pos_sim - neg_sim
        gate_raw = torch.sigmoid((float(confidence_gating_margin_threshold) - margin) / max(float(confidence_gating_temperature), 1e-4))
        difficulty = (1.0 - pos_sim).clamp(0.0, 2.0)
        query_score = (
            0.35 * gate_raw
            + 0.20 * difficulty
            + 0.15 * motion_norm
            + 0.10 * area_jump
            + 0.10 * hard_score
            + 0.10 * small_query
        ).clamp(0.0, 3.0)

        family = str(v6_gating_family).strip().lower()
        if family == "hard_topk_query_gating_v2":
            sparse_selected, sparse_stats = _hard_topk_query_mask_v2(
                scores=query_score,
                valid=query_valid,
                topk=max(int(v6_topk_query_k), 1),
            )
            family_code = 5.0
        else:
            sparse_selected, sparse_stats = _capped_quantile_query_mask_v2(
                scores=query_score,
                valid=query_valid,
                quantile=float(v6_capped_quantile),
                max_ratio=float(v6_max_affected_ratio),
            )
            family_code = 6.0

        gate = torch.where(
            sparse_selected,
            query_score.clamp_min(float(max(v6_gate_min_strength, 0.0))),
            torch.zeros_like(query_score),
        )
        gate_denom = gate.sum().clamp_min(1.0)

        conf_align_loss = semantic_tokens.sum() * 0.0
        if readout_align_weight > 0.0:
            cosine = 1.0 - pos_sim
            conf_align_loss = (cosine * gate).sum() / gate_denom
            weighted_terms.append(float(readout_align_weight) * conf_align_loss)
            weight_sum += float(readout_align_weight)

        positive_pair_count = 0.0
        hard_negative_count = 0.0
        effective_pair_coverage_ratio = 0.0
        valuable_pair_ratio = 0.0
        valuable_pair_count = 0.0
        candidate_pair_count = 0.0
        strict_pair_ratio = 0.0
        fallback_pair_ratio = 0.0
        fallback_trigger_rate = 0.0
        guaranteed_pair_count = 0.0
        strict_pair_count = 0.0
        fallback_pair_count = 0.0
        fallback_trigger_count = 0.0
        guaranteed_pair_total = 0.0
        eligible_sample_count = 0.0
        sparse_loss_terms: List[torch.Tensor] = []

        if sparse_weight > 0.0:
            strict_query = sparse_selected & query_valid & (
                (motion_norm >= 0.40)
                | (area_jump >= 0.35)
                | (small_query >= 0.65)
                | (appearance_shift >= 0.25)
                | (center_interaction >= 0.35)
                | (hard_score >= float(semantic_hard_score_threshold))
            )
            relaxed_query = sparse_selected & query_valid & (
                (motion_norm >= float(v6_relaxed_motion_threshold))
                | (area_jump >= float(v6_relaxed_area_jump_threshold))
                | (small_query >= float(v6_relaxed_small_query_threshold))
                | (appearance_shift >= float(v6_relaxed_appearance_shift_threshold))
                | (center_interaction >= float(v6_relaxed_center_interaction_threshold))
                | (hard_score >= float(max(0.05, semantic_hard_score_threshold * 0.75)))
            )

            temp = max(float(v6_pair_sampling_temperature), 1e-4)
            pair_budget = max(int(v6_strict_max_pairs_per_sample), 1)
            guaranteed_min = max(int(v6_guaranteed_min_pairs_per_sample), 0)
            hard_neg_cap = max(int(v6_hard_negative_cap), 1)

            flat_all_pred = pred.reshape(-1, pred.shape[-1])
            flat_all_valid = query_valid.reshape(-1)
            flat_all_pred_valid = flat_all_pred[flat_all_valid]
            flat_all_sid_valid = torch.arange(pred.shape[0], device=device)[:, None, None].expand_as(query_valid).reshape(-1)[flat_all_valid]

            for bi in range(int(pred.shape[0])):
                base_sel = sparse_selected[bi].reshape(-1)
                n_base = int(base_sel.sum().item())
                if n_base < 2:
                    continue
                eligible_sample_count += 1.0
                candidate_pair_count += float((n_base * (n_base - 1)) // 2)

                strict_idx = torch.nonzero(strict_query[bi].reshape(-1), as_tuple=False).squeeze(-1)
                relaxed_idx = torch.nonzero(relaxed_query[bi].reshape(-1), as_tuple=False).squeeze(-1)
                base_idx = torch.nonzero(base_sel, as_tuple=False).squeeze(-1)
                score_flat = query_score[bi].reshape(-1)
                pred_flat = pred[bi].reshape(-1, pred.shape[-1])

                def _rank_pairs(indices: torch.Tensor) -> List[Tuple[int, int]]:
                    if int(indices.numel()) < 2:
                        return []
                    pair_i, pair_j = torch.triu_indices(int(indices.numel()), int(indices.numel()), offset=1, device=indices.device)
                    src_idx = indices[pair_i]
                    dst_idx = indices[pair_j]
                    pair_values = ((score_flat[src_idx] + score_flat[dst_idx]) * 0.5) / temp
                    order = torch.argsort(pair_values, descending=True)
                    ranked: List[Tuple[int, int]] = []
                    for oi in order.tolist():
                        a = int(src_idx[int(oi)].item())
                        b = int(dst_idx[int(oi)].item())
                        if a == b:
                            continue
                        ranked.append((a, b))
                    return ranked

                strict_pairs = _rank_pairs(strict_idx)
                strict_pair_set = {(min(a, b), max(a, b)) for a, b in strict_pairs}
                pair_budget_eff = max(pair_budget, guaranteed_min)
                selected_pairs = strict_pairs[:pair_budget_eff]
                triggered = False

                if guaranteed_min > 0 and len(selected_pairs) < guaranteed_min:
                    fallback_pool = relaxed_idx if bool(v6_two_level_pair_mining_enabled) else base_idx
                    fallback_pairs = _rank_pairs(fallback_pool)
                    fallback_pairs = [p for p in fallback_pairs if (min(p[0], p[1]), max(p[0], p[1])) not in strict_pair_set]
                    if fallback_pairs:
                        need = guaranteed_min - len(selected_pairs)
                        selected_pairs.extend(fallback_pairs[:need])
                        triggered = True

                if guaranteed_min > 0 and len(selected_pairs) < guaranteed_min:
                    base_pairs = _rank_pairs(base_idx)
                    existing = {(min(a, b), max(a, b)) for a, b in selected_pairs}
                    base_pairs = [p for p in base_pairs if (min(p[0], p[1]), max(p[0], p[1])) not in existing]
                    if base_pairs:
                        need = guaranteed_min - len(selected_pairs)
                        selected_pairs.extend(base_pairs[:need])
                        triggered = True

                if triggered:
                    fallback_trigger_count += 1.0

                if len(selected_pairs) <= 0:
                    continue

                guaranteed_pair_total += float(len(selected_pairs))
                valuable_pair_count += float(len(selected_pairs))

                strict_selected_local = 0.0
                fallback_selected_local = 0.0
                for src_i, dst_j in selected_pairs:
                    pair_key = (min(src_i, dst_j), max(src_i, dst_j))
                    if pair_key in strict_pair_set:
                        strict_selected_local += 1.0
                    else:
                        fallback_selected_local += 1.0

                    src = pred_flat[int(src_i)]
                    pos = torch.dot(src, pred_flat[int(dst_j)])
                    neg_mask = flat_all_sid_valid != int(bi)
                    if int(neg_mask.sum().item()) <= 0:
                        continue
                    neg_logits = torch.matmul(flat_all_pred_valid[neg_mask], src)
                    if int(neg_logits.numel()) > hard_neg_cap:
                        neg_logits = torch.topk(neg_logits, k=hard_neg_cap, largest=True, sorted=False).values
                    max_neg = neg_logits.mean()
                    sparse_loss_terms.append(torch.relu(float(v4_persistence_margin) + max_neg - pos))
                    positive_pair_count += 1.0
                    hard_negative_count += float(neg_logits.numel())

                strict_pair_count += float(strict_selected_local)
                fallback_pair_count += float(fallback_selected_local)

            if sparse_loss_terms:
                sparse_loss = torch.stack(sparse_loss_terms).mean()
                weighted_terms.append(float(sparse_weight) * sparse_loss)
                weight_sum += float(sparse_weight)
            else:
                sparse_loss = semantic_tokens.sum() * 0.0

            total_selected_pairs = float(strict_pair_count + fallback_pair_count)
            valuable_pair_ratio = float(total_selected_pairs / max(candidate_pair_count, 1.0))
            strict_pair_ratio = float(strict_pair_count / max(total_selected_pairs, 1.0))
            fallback_pair_ratio = float(fallback_pair_count / max(total_selected_pairs, 1.0))
            effective_pair_coverage_ratio = float(positive_pair_count / max(candidate_pair_count, 1.0))
            fallback_trigger_rate = float(fallback_trigger_count / max(eligible_sample_count, 1.0))
            guaranteed_pair_count = float(guaranteed_pair_total / max(eligible_sample_count, 1.0))
        else:
            sparse_loss = semantic_tokens.sum() * 0.0

        aux_schedule_scale = float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps))
        final_effective_aux_weight = float(aux_schedule_scale)

        if not weighted_terms:
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "confidence_gated_alignment_loss": 0.0,
                "sparse_persistence_contrastive_loss": 0.0,
                "confidence_gated_affected_sample_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
                "low_confidence_sample_ratio": float(((gate_raw > 0.5) & query_valid).float().sum().item() / max(query_valid.float().sum().item(), 1.0)),
                "confidence_metric_threshold": float(confidence_gating_margin_threshold),
                "confidence_metric_temperature": float(confidence_gating_temperature),
                "confidence_metric_definition": 4.0,
                "positive_pair_count": float(positive_pair_count),
                "hard_negative_count": float(hard_negative_count),
                "effective_pair_coverage_ratio": float(effective_pair_coverage_ratio),
                "aux_schedule_scale": float(aux_schedule_scale),
                "aux_loss_delay_steps": float(aux_loss_delay_steps),
                "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
                "actual_gate_positive_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
                "activated_query_count": float(sparse_stats["activated_query_count_mean"]),
                "activated_query_ratio": float(sparse_stats["activated_query_ratio"]),
                "per_batch_sparsity_mean": float(sparse_stats["per_batch_sparsity_mean"]),
                "per_batch_sparsity_std": float(sparse_stats["per_batch_sparsity_std"]),
                "raw_quantile_ratio": float(sparse_stats["raw_quantile_ratio"]),
                "capped_ratio": float(sparse_stats["capped_ratio"]),
                "valuable_pair_ratio": float(valuable_pair_ratio),
                "sparse_gate_selected_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
                "high_value_pair_ratio": float(valuable_pair_ratio),
                "high_value_pair_count": float(valuable_pair_count),
                "persistence_candidate_pair_count": float(candidate_pair_count),
                "max_pairs_per_sample": float(v6_strict_max_pairs_per_sample),
                "hard_negative_cap": float(v6_hard_negative_cap),
                "pair_sampling_temperature": float(v6_pair_sampling_temperature),
                "final_effective_aux_weight": float(final_effective_aux_weight),
                "fallback_trigger_rate": float(fallback_trigger_rate),
                "guaranteed_pair_count": float(guaranteed_pair_count),
                "strict_pair_ratio": float(strict_pair_ratio),
                "fallback_pair_ratio": float(fallback_pair_ratio),
                "v4_sparse_gating_family_code": float(family_code),
            }

        loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
        return loss, {
            "semantic_rescue_loss": float(loss.detach().cpu().item()),
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "persistence_contrastive_or_ranking_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "sparse_persistence_contrastive_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_affected_sample_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
            "low_confidence_sample_ratio": float(((gate_raw > 0.5) & query_valid).float().sum().item() / max(query_valid.float().sum().item(), 1.0)),
            "confidence_metric_threshold": float(confidence_gating_margin_threshold),
            "confidence_metric_temperature": float(confidence_gating_temperature),
            "confidence_metric_definition": 4.0,
            "positive_pair_count": float(positive_pair_count),
            "hard_negative_count": float(hard_negative_count),
            "effective_pair_coverage_ratio": float(effective_pair_coverage_ratio),
            "aux_schedule_scale": float(aux_schedule_scale),
            "aux_loss_delay_steps": float(aux_loss_delay_steps),
            "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
            "actual_gate_positive_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
            "activated_query_count": float(sparse_stats["activated_query_count_mean"]),
            "activated_query_ratio": float(sparse_stats["activated_query_ratio"]),
            "per_batch_sparsity_mean": float(sparse_stats["per_batch_sparsity_mean"]),
            "per_batch_sparsity_std": float(sparse_stats["per_batch_sparsity_std"]),
            "raw_quantile_ratio": float(sparse_stats["raw_quantile_ratio"]),
            "capped_ratio": float(sparse_stats["capped_ratio"]),
            "valuable_pair_ratio": float(valuable_pair_ratio),
            "sparse_gate_selected_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
            "high_value_pair_ratio": float(valuable_pair_ratio),
            "high_value_pair_count": float(valuable_pair_count),
            "persistence_candidate_pair_count": float(candidate_pair_count),
            "max_pairs_per_sample": float(v6_strict_max_pairs_per_sample),
            "hard_negative_cap": float(v6_hard_negative_cap),
            "pair_sampling_temperature": float(v6_pair_sampling_temperature),
            "final_effective_aux_weight": float(final_effective_aux_weight),
            "fallback_trigger_rate": float(fallback_trigger_rate),
            "guaranteed_pair_count": float(guaranteed_pair_count),
            "strict_pair_ratio": float(strict_pair_ratio),
            "fallback_pair_ratio": float(fallback_pair_ratio),
            "v4_sparse_gating_family_code": float(family_code),
        }

    if mode_norm.startswith("v5"):
        readout_align_weight = float(confidence_gated_alignment_loss_weight)
        sparse_weight = float(sparse_persistence_contrastive_loss_weight)
        if readout_align_weight <= 0.0:
            readout_align_weight = 1.0
        if sparse_weight <= 0.0:
            sparse_weight = 0.05

        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache,
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        future_readout = tf_out.get("future_fused_hidden")
        if (
            not isinstance(future_readout, torch.Tensor)
            or future_readout.ndim != 4
            or aux_heads.readout_feature_head is None
        ):
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "confidence_gated_alignment_loss": 0.0,
                "sparse_persistence_contrastive_loss": 0.0,
                "confidence_gated_affected_sample_ratio": 0.0,
                "low_confidence_sample_ratio": 0.0,
                "confidence_metric_threshold": float(confidence_gating_margin_threshold),
                "confidence_metric_temperature": float(confidence_gating_temperature),
                "confidence_metric_definition": 3.0,
                "positive_pair_count": 0.0,
                "hard_negative_count": 0.0,
                "effective_pair_coverage_ratio": 0.0,
                "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
                "aux_loss_delay_steps": float(aux_loss_delay_steps),
                "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
                "actual_gate_positive_ratio": 0.0,
                "activated_query_count": 0.0,
                "activated_query_ratio": 0.0,
                "per_batch_sparsity_mean": 0.0,
                "per_batch_sparsity_std": 0.0,
                "raw_quantile_ratio": 0.0,
                "capped_ratio": 0.0,
                "valuable_pair_ratio": 0.0,
                "sparse_gate_selected_ratio": 0.0,
                "high_value_pair_ratio": 0.0,
                "high_value_pair_count": 0.0,
                "persistence_candidate_pair_count": 0.0,
                "max_pairs_per_sample": float(v5_max_pairs_per_sample),
                "hard_negative_cap": float(v5_hard_negative_cap),
                "pair_sampling_temperature": float(v5_pair_sampling_temperature),
                "final_effective_aux_weight": 0.0,
            }

        query_valid = tf_out["valid_mask"] & target_valid[:, None, :]
        hard_stats = _semantic_hard_sample_stats(batch=batch, device=device)
        hard_score = hard_stats["hard_score"][:, None, None].expand_as(query_valid).to(torch.float32)
        small_score = hard_stats["small_score"][:, None, None].expand_as(query_valid).to(torch.float32)
        center_interaction = hard_stats["center_interaction"][:, None, None].expand_as(query_valid).to(torch.float32)
        appearance_shift = hard_stats["appearance_shift"][:, None, None].expand_as(query_valid).to(torch.float32)

        future_proj = aux_heads.readout_feature_head(future_readout)
        pred = torch.nn.functional.normalize(future_proj, dim=-1)
        tgt = torch.nn.functional.normalize(target[:, None, :, :].expand(-1, pred.shape[1], -1, -1), dim=-1)
        pos_sim = (pred * tgt).sum(dim=-1)
        neg_sim = torch.full_like(pos_sim, -1.0)

        flat_valid = query_valid.reshape(-1)
        if int(flat_valid.sum().item()) >= 2:
            flat_pred = pred.reshape(-1, pred.shape[-1])[flat_valid]
            flat_tgt = tgt.reshape(-1, tgt.shape[-1])[flat_valid]
            sample_ids = torch.arange(pred.shape[0], device=device)[:, None, None].expand_as(query_valid).reshape(-1)[flat_valid]
            flat_sim = flat_pred @ flat_tgt.T
            same_sample = sample_ids[:, None] == sample_ids[None, :]
            flat_sim = flat_sim.masked_fill(same_sample, -1e9)
            flat_neg = flat_sim.max(dim=-1).values
            flat_neg = torch.where(torch.isfinite(flat_neg), flat_neg, torch.full_like(flat_neg, -1.0))
            neg_sim_flat = torch.full((flat_valid.shape[0],), -1.0, device=device, dtype=torch.float32)
            neg_sim_flat[flat_valid] = flat_neg
            neg_sim = neg_sim_flat.view_as(pos_sim)

        future_state = batch["fut_state"]
        area = (future_state[..., 6].abs() * future_state[..., 7].abs()).clamp(0.0, 1.0)
        small_query = (1.0 - area).clamp(0.0, 1.0)
        coord = future_state[..., 0:2]
        step_motion = torch.zeros_like(pos_sim)
        if coord.shape[1] >= 2:
            motion_delta = torch.sqrt(((coord[:, 1:] - coord[:, :-1]) ** 2).sum(dim=-1).clamp_min(1e-12))
            step_motion[:, 1:] = motion_delta
        motion_norm = step_motion / step_motion.amax(dim=1, keepdim=True).clamp_min(1e-6)
        area_jump = torch.zeros_like(pos_sim)
        if area.shape[1] >= 2:
            area_delta = (area[:, 1:] - area[:, :-1]).abs()
            area_jump[:, 1:] = area_delta
        area_jump = area_jump / area_jump.amax(dim=1, keepdim=True).clamp_min(1e-6)

        margin = pos_sim - neg_sim
        gate_raw = torch.sigmoid((float(confidence_gating_margin_threshold) - margin) / max(float(confidence_gating_temperature), 1e-4))
        difficulty = (1.0 - pos_sim).clamp(0.0, 2.0)
        query_score = (
            0.35 * gate_raw
            + 0.20 * difficulty
            + 0.15 * motion_norm
            + 0.10 * area_jump
            + 0.10 * hard_score
            + 0.10 * small_query
        ).clamp(0.0, 3.0)

        family = str(v5_gating_family).strip().lower()
        if family == "hard_topk_query_gating_v2":
            sparse_selected, sparse_stats = _hard_topk_query_mask_v2(
                scores=query_score,
                valid=query_valid,
                topk=max(int(v5_topk_query_k), 1),
            )
            family_code = 3.0
        else:
            sparse_selected, sparse_stats = _capped_quantile_query_mask_v2(
                scores=query_score,
                valid=query_valid,
                quantile=float(v5_capped_quantile),
                max_ratio=float(v5_max_affected_ratio),
            )
            family_code = 4.0

        gate = torch.where(
            sparse_selected,
            query_score.clamp_min(float(max(v5_gate_min_strength, 0.0))),
            torch.zeros_like(query_score),
        )
        gate_denom = gate.sum().clamp_min(1.0)

        conf_align_loss = semantic_tokens.sum() * 0.0
        if readout_align_weight > 0.0:
            cosine = 1.0 - pos_sim
            conf_align_loss = (cosine * gate).sum() / gate_denom
            weighted_terms.append(float(readout_align_weight) * conf_align_loss)
            weight_sum += float(readout_align_weight)

        positive_pair_count = 0.0
        hard_negative_count = 0.0
        effective_pair_coverage_ratio = 0.0
        valuable_pair_ratio = 0.0
        valuable_pair_count = 0.0
        candidate_pair_count = 0.0
        sparse_loss_terms: List[torch.Tensor] = []

        if sparse_weight > 0.0:
            valuable_query = sparse_selected & query_valid & (
                (motion_norm >= 0.40)
                | (area_jump >= 0.35)
                | (small_query >= 0.65)
                | (appearance_shift >= 0.25)
                | (center_interaction >= 0.35)
                | (hard_score >= float(semantic_hard_score_threshold))
            )
            valuable_query_count = float(valuable_query.float().sum().item())
            selected_query_count = float(sparse_selected.float().sum().item())

            temp = max(float(v5_pair_sampling_temperature), 1e-4)
            max_pairs = max(int(v5_max_pairs_per_sample), 1)
            hard_neg_cap = max(int(v5_hard_negative_cap), 1)
            flat_all_pred = pred.reshape(-1, pred.shape[-1])
            flat_all_valid = query_valid.reshape(-1)
            flat_all_pred_valid = flat_all_pred[flat_all_valid]
            flat_all_sid_valid = torch.arange(pred.shape[0], device=device)[:, None, None].expand_as(query_valid).reshape(-1)[flat_all_valid]

            for bi in range(int(pred.shape[0])):
                sel = valuable_query[bi].reshape(-1)
                base_sel = sparse_selected[bi].reshape(-1)
                n_base = int(base_sel.sum().item())
                if n_base >= 2:
                    candidate_pair_count += float((n_base * (n_base - 1)) // 2)
                idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
                if int(idx.numel()) < 2:
                    continue

                valuable_pair_count += float((int(idx.numel()) * (int(idx.numel()) - 1)) // 2)
                local_pred = pred[bi].reshape(-1, pred.shape[-1])[idx]
                local_scores = query_score[bi].reshape(-1)[idx]
                pair_i, pair_j = torch.triu_indices(int(idx.numel()), int(idx.numel()), offset=1, device=device)
                if int(pair_i.numel()) <= 0:
                    continue
                pair_value = ((local_scores[pair_i] + local_scores[pair_j]) * 0.5) / temp
                if int(pair_i.numel()) > max_pairs:
                    keep_local = torch.topk(pair_value, k=max_pairs, largest=True, sorted=False).indices
                    pair_i = pair_i[keep_local]
                    pair_j = pair_j[keep_local]
                    pair_value = pair_value[keep_local]

                for pi, pj in zip(pair_i.tolist(), pair_j.tolist()):
                    src = local_pred[int(pi)]
                    pos = torch.dot(src, local_pred[int(pj)])
                    neg_mask = flat_all_sid_valid != int(bi)
                    if int(neg_mask.sum().item()) <= 0:
                        continue
                    neg_logits = torch.matmul(flat_all_pred_valid[neg_mask], src)
                    if int(neg_logits.numel()) > hard_neg_cap:
                        neg_logits = torch.topk(neg_logits, k=hard_neg_cap, largest=True, sorted=False).values
                    max_neg = neg_logits.mean()
                    sparse_loss_terms.append(torch.relu(float(v4_persistence_margin) + max_neg - pos))
                    positive_pair_count += 1.0
                    hard_negative_count += float(neg_logits.numel())

            if sparse_loss_terms:
                sparse_loss = torch.stack(sparse_loss_terms).mean()
                weighted_terms.append(float(sparse_weight) * sparse_loss)
                weight_sum += float(sparse_weight)
            else:
                sparse_loss = semantic_tokens.sum() * 0.0
            valuable_pair_ratio = float(valuable_pair_count / max(candidate_pair_count, 1.0))
            effective_pair_coverage_ratio = float(positive_pair_count / max(candidate_pair_count, 1.0))
        else:
            sparse_loss = semantic_tokens.sum() * 0.0

        aux_schedule_scale = float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps))
        final_effective_aux_weight = float(aux_schedule_scale)

        if not weighted_terms:
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "confidence_gated_alignment_loss": 0.0,
                "sparse_persistence_contrastive_loss": 0.0,
                "confidence_gated_affected_sample_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
                "low_confidence_sample_ratio": float(((gate_raw > 0.5) & query_valid).float().sum().item() / max(query_valid.float().sum().item(), 1.0)),
                "confidence_metric_threshold": float(confidence_gating_margin_threshold),
                "confidence_metric_temperature": float(confidence_gating_temperature),
                "confidence_metric_definition": 3.0,
                "positive_pair_count": float(positive_pair_count),
                "hard_negative_count": float(hard_negative_count),
                "effective_pair_coverage_ratio": float(effective_pair_coverage_ratio),
                "aux_schedule_scale": float(aux_schedule_scale),
                "aux_loss_delay_steps": float(aux_loss_delay_steps),
                "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
                "actual_gate_positive_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
                "activated_query_count": float(sparse_stats["activated_query_count_mean"]),
                "activated_query_ratio": float(sparse_stats["activated_query_ratio"]),
                "per_batch_sparsity_mean": float(sparse_stats["per_batch_sparsity_mean"]),
                "per_batch_sparsity_std": float(sparse_stats["per_batch_sparsity_std"]),
                "raw_quantile_ratio": float(sparse_stats["raw_quantile_ratio"]),
                "capped_ratio": float(sparse_stats["capped_ratio"]),
                "valuable_pair_ratio": float(valuable_pair_ratio),
                "sparse_gate_selected_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
                "high_value_pair_ratio": float(valuable_pair_ratio),
                "high_value_pair_count": float(valuable_pair_count),
                "persistence_candidate_pair_count": float(candidate_pair_count),
                "max_pairs_per_sample": float(v5_max_pairs_per_sample),
                "hard_negative_cap": float(v5_hard_negative_cap),
                "pair_sampling_temperature": float(v5_pair_sampling_temperature),
                "final_effective_aux_weight": float(final_effective_aux_weight),
                "v4_sparse_gating_family_code": float(family_code),
            }

        loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
        return loss, {
            "semantic_rescue_loss": float(loss.detach().cpu().item()),
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "persistence_contrastive_or_ranking_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "sparse_persistence_contrastive_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_affected_sample_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
            "low_confidence_sample_ratio": float(((gate_raw > 0.5) & query_valid).float().sum().item() / max(query_valid.float().sum().item(), 1.0)),
            "confidence_metric_threshold": float(confidence_gating_margin_threshold),
            "confidence_metric_temperature": float(confidence_gating_temperature),
            "confidence_metric_definition": 3.0,
            "positive_pair_count": float(positive_pair_count),
            "hard_negative_count": float(hard_negative_count),
            "effective_pair_coverage_ratio": float(effective_pair_coverage_ratio),
            "aux_schedule_scale": float(aux_schedule_scale),
            "aux_loss_delay_steps": float(aux_loss_delay_steps),
            "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
            "actual_gate_positive_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
            "activated_query_count": float(sparse_stats["activated_query_count_mean"]),
            "activated_query_ratio": float(sparse_stats["activated_query_ratio"]),
            "per_batch_sparsity_mean": float(sparse_stats["per_batch_sparsity_mean"]),
            "per_batch_sparsity_std": float(sparse_stats["per_batch_sparsity_std"]),
            "raw_quantile_ratio": float(sparse_stats["raw_quantile_ratio"]),
            "capped_ratio": float(sparse_stats["capped_ratio"]),
            "valuable_pair_ratio": float(valuable_pair_ratio),
            "sparse_gate_selected_ratio": float(sparse_stats["actual_gate_positive_ratio"]),
            "high_value_pair_ratio": float(valuable_pair_ratio),
            "high_value_pair_count": float(valuable_pair_count),
            "persistence_candidate_pair_count": float(candidate_pair_count),
            "max_pairs_per_sample": float(v5_max_pairs_per_sample),
            "hard_negative_cap": float(v5_hard_negative_cap),
            "pair_sampling_temperature": float(v5_pair_sampling_temperature),
            "final_effective_aux_weight": float(final_effective_aux_weight),
            "v4_sparse_gating_family_code": float(family_code),
        }

    if mode_norm.startswith("v4"):
        readout_align_weight = float(confidence_gated_alignment_loss_weight)
        sparse_weight = float(sparse_persistence_contrastive_loss_weight)
        if readout_align_weight <= 0.0:
            readout_align_weight = 1.0
        if sparse_weight <= 0.0:
            sparse_weight = 0.05

        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache,
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        valid = semantic_mask & target_valid
        hard_stats = _semantic_hard_sample_stats(batch=batch, device=device)
        hard_score = hard_stats["hard_score"][:, None].expand_as(valid).to(torch.float32)
        hard_relevance = (
            (hard_stats["area_range"] >= float(semantic_hard_score_threshold))
            | (hard_stats["center_interaction"] >= 0.30)
            | (hard_stats["appearance_shift"] >= 0.20)
            | (hard_stats["small_score"] >= 0.25)
        )[:, None].expand_as(valid)

        readout_pred = aux.get("readout_feature_target", aux["feature_target"])
        pred = torch.nn.functional.normalize(readout_pred, dim=-1)
        tgt = torch.nn.functional.normalize(target, dim=-1)
        pos_sim = (pred * tgt).sum(dim=-1)
        neg_sim = torch.full_like(pos_sim, -1.0)
        flat_valid = valid.reshape(-1)
        if int(flat_valid.sum().item()) >= 2:
            flat_pred = pred.reshape(-1, pred.shape[-1])[flat_valid]
            flat_tgt = tgt.reshape(-1, tgt.shape[-1])[flat_valid]
            flat_sim = flat_pred @ flat_tgt.T
            flat_sim = flat_sim.masked_fill(torch.eye(flat_sim.shape[0], device=device, dtype=torch.bool), -1e9)
            flat_neg = flat_sim.max(dim=-1).values
            flat_neg = torch.where(torch.isfinite(flat_neg), flat_neg, torch.full_like(flat_neg, -1.0))
            neg_sim_flat = torch.full((flat_valid.shape[0],), -1.0, device=device, dtype=torch.float32)
            neg_sim_flat[flat_valid] = flat_neg
            neg_sim = neg_sim_flat.view_as(pos_sim)

        margin = pos_sim - neg_sim
        gate_raw = torch.sigmoid((float(confidence_gating_margin_threshold) - margin) / max(float(confidence_gating_temperature), 1e-4))
        difficulty = (1.0 - pos_sim).clamp(0.0, 2.0)
        confidence_score = (gate_raw * (1.0 + 0.75 * hard_score)).clamp(0.0, 3.0)
        quantile_score = (0.60 * confidence_score + 0.25 * difficulty + 0.15 * hard_score).clamp(0.0, 3.0)
        topk_score = (0.65 * difficulty + 0.25 * confidence_score + 0.10 * hard_score).clamp(0.0, 3.0)

        family = str(v4_sparse_gating_family).strip().lower()
        min_tokens = max(int(v4_topk_min_tokens), 1)
        if family == "topk_query_gating":
            sparse_selected = _sparse_topk_mask(
                scores=topk_score,
                valid=valid,
                keep_ratio=float(v4_topk_token_ratio),
                min_tokens=min_tokens,
            )
            sparse_score = topk_score
            family_code = 2.0
        else:
            sparse_selected = _sparse_quantile_mask(
                scores=quantile_score,
                valid=valid,
                quantile=float(v4_gating_quantile),
                min_tokens=min_tokens,
            )
            sparse_score = quantile_score
            family_code = 1.0

        min_gate_strength = max(float(v4_gate_min_strength), 0.0)
        gate = torch.where(sparse_selected, sparse_score.clamp_min(min_gate_strength), torch.zeros_like(sparse_score))
        gate_denom = gate.sum().clamp_min(1.0)

        conf_align_loss = semantic_tokens.sum() * 0.0
        if readout_align_weight > 0.0:
            cosine = 1.0 - pos_sim
            conf_align_loss = (cosine * gate).sum() / gate_denom
            weighted_terms.append(float(readout_align_weight) * conf_align_loss)
            weight_sum += float(readout_align_weight)

        future_readout = tf_out.get("future_fused_hidden")
        sparse_loss = semantic_tokens.sum() * 0.0
        positive_pair_count = 0.0
        hard_negative_count = 0.0
        pair_coverage_ratio = 0.0
        candidate_pair_count = 0.0
        high_value_pair_count = 0.0
        high_value_pair_ratio = 0.0
        persistence_value_threshold = 0.0
        if (
            sparse_weight > 0.0
            and isinstance(future_readout, torch.Tensor)
            and future_readout.ndim == 4
            and future_readout.shape[1] >= 2
            and aux_heads.readout_feature_head is not None
        ):
            future_proj = aux_heads.readout_feature_head(future_readout)
            first_proj = torch.nn.functional.normalize(future_proj[:, 0], dim=-1)
            last_proj = torch.nn.functional.normalize(future_proj[:, -1], dim=-1)
            selected = sparse_selected & hard_relevance & valid
            candidate_pair_count = float(selected.float().sum().item())

            pair_value = (sparse_score * (1.0 + hard_score) * (difficulty + 0.1)).clamp(0.0, 10.0)
            high_value_selected = selected
            flat_selected = selected.reshape(-1)
            if int(flat_selected.sum().item()) > 0:
                selected_values = pair_value.reshape(-1)[flat_selected]
                value_q = _clamp01(float(v4_persistence_value_quantile))
                if int(selected_values.numel()) >= 2:
                    persistence_value_threshold = float(torch.quantile(selected_values.detach(), value_q).item())
                    high_value_selected = selected & (pair_value >= persistence_value_threshold)

                flat_high = high_value_selected.reshape(-1)
                max_pairs = int(v4_persistence_max_pairs)
                if max_pairs > 0 and int(flat_high.sum().item()) > max_pairs:
                    hv_idx = torch.nonzero(flat_high, as_tuple=False).squeeze(-1)
                    hv_vals = pair_value.reshape(-1)[hv_idx]
                    keep_local = torch.topk(hv_vals, k=max_pairs, largest=True, sorted=False).indices
                    keep_idx = hv_idx[keep_local]
                    pruned = torch.zeros_like(flat_high)
                    pruned[keep_idx] = True
                    high_value_selected = pruned.view_as(selected)
                    flat_high = pruned

                if int(flat_high.sum().item()) < 2 and int(flat_selected.sum().item()) >= 2:
                    cand_idx = torch.nonzero(flat_selected, as_tuple=False).squeeze(-1)
                    cand_vals = pair_value.reshape(-1)[cand_idx]
                    keep_n = min(2, int(cand_idx.numel()))
                    keep_local = torch.topk(cand_vals, k=keep_n, largest=True, sorted=False).indices
                    keep_idx = cand_idx[keep_local]
                    fallback = torch.zeros_like(flat_selected)
                    fallback[keep_idx] = True
                    high_value_selected = fallback.view_as(selected)

            flat_sel = high_value_selected.reshape(-1)
            high_value_pair_count = float(flat_sel.float().sum().item())
            high_value_pair_ratio = float(high_value_pair_count / max(candidate_pair_count, 1.0))
            if int(flat_sel.sum().item()) >= 2:
                flat_first = first_proj.reshape(-1, first_proj.shape[-1])[flat_sel]
                flat_last = last_proj.reshape(-1, last_proj.shape[-1])[flat_sel]
                logits = flat_first @ flat_last.T
                pos = torch.diagonal(logits)
                logits_wo_diag = logits.masked_fill(torch.eye(logits.shape[0], device=device, dtype=torch.bool), -1e9)
                max_neg = logits_wo_diag.max(dim=-1).values
                max_neg = torch.where(torch.isfinite(max_neg), max_neg, torch.full_like(max_neg, -1.0))
                sparse_loss = torch.relu(float(v4_persistence_margin) + max_neg - pos).mean()
                weighted_terms.append(float(sparse_weight) * sparse_loss)
                weight_sum += float(sparse_weight)
                positive_pair_count = float(flat_first.shape[0])
                hard_negative_count = float((logits_wo_diag > (pos[:, None] - float(v4_persistence_margin))).float().sum().item())
                pair_coverage_ratio = float(positive_pair_count / max(candidate_pair_count, 1.0))
            else:
                positive_pair_count = float(flat_sel.sum().item())
                pair_coverage_ratio = float(positive_pair_count / max(candidate_pair_count, 1.0))

        sparse_gate_selected_ratio = float(sparse_selected.float().sum().item() / max(valid.float().sum().item(), 1.0))
        aux_schedule_base = float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps))
        aux_schedule_scaled = float(aux_schedule_base * aux_schedule_base)

        if not weighted_terms:
            zero = tf_out["pred_coord"].sum() * 0.0
            return zero, {
                "semantic_rescue_loss": 0.0,
                "semantic_alignment_loss": 0.0,
                "query_persistence_consistency_loss": 0.0,
                "readout_semantic_alignment_loss": 0.0,
                "persistence_contrastive_or_ranking_loss": 0.0,
                "confidence_gated_alignment_loss": 0.0,
                "sparse_persistence_contrastive_loss": 0.0,
                "confidence_gated_affected_sample_ratio": float(sparse_gate_selected_ratio),
                "low_confidence_sample_ratio": float(((gate_raw > 0.5) & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
                "confidence_metric_threshold": float(confidence_gating_margin_threshold),
                "confidence_metric_temperature": float(confidence_gating_temperature),
                "confidence_metric_definition": 2.0,
                "positive_pair_count": float(positive_pair_count),
                "hard_negative_count": float(hard_negative_count),
                "effective_pair_coverage_ratio": float(pair_coverage_ratio),
                "aux_schedule_scale": float(aux_schedule_scaled),
                "aux_loss_delay_steps": float(aux_loss_delay_steps),
                "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
                "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
                "whether_main_rollout_loss_reweighted": False,
                "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
                "sparse_gate_selected_ratio": float(sparse_gate_selected_ratio),
                "high_value_pair_ratio": float(high_value_pair_ratio),
                "high_value_pair_count": float(high_value_pair_count),
                "persistence_candidate_pair_count": float(candidate_pair_count),
                "persistence_value_threshold": float(persistence_value_threshold),
                "v4_sparse_gating_family_code": float(family_code),
            }

        loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
        return loss, {
            "semantic_rescue_loss": float(loss.detach().cpu().item()),
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "persistence_contrastive_or_ranking_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_alignment_loss": float(conf_align_loss.detach().cpu().item()),
            "sparse_persistence_contrastive_loss": float(sparse_loss.detach().cpu().item()),
            "confidence_gated_affected_sample_ratio": float(sparse_gate_selected_ratio),
            "low_confidence_sample_ratio": float(((gate_raw > 0.5) & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
            "confidence_metric_threshold": float(confidence_gating_margin_threshold),
            "confidence_metric_temperature": float(confidence_gating_temperature),
            "confidence_metric_definition": 2.0,
            "positive_pair_count": float(positive_pair_count),
            "hard_negative_count": float(hard_negative_count),
            "effective_pair_coverage_ratio": float(pair_coverage_ratio),
            "aux_schedule_scale": float(aux_schedule_scaled),
            "aux_loss_delay_steps": float(aux_loss_delay_steps),
            "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
            "sparse_gate_selected_ratio": float(sparse_gate_selected_ratio),
            "high_value_pair_ratio": float(high_value_pair_ratio),
            "high_value_pair_count": float(high_value_pair_count),
            "persistence_candidate_pair_count": float(candidate_pair_count),
            "persistence_value_threshold": float(persistence_value_threshold),
            "v4_sparse_gating_family_code": float(family_code),
        }

    align_loss = semantic_tokens.sum() * 0.0
    query_loss = semantic_tokens.sum() * 0.0
    if align_weight > 0.0:
        target, target_valid, cache_hit_ratio = _bootstrap_targets_from_batch(
            batch=batch,
            cache=bootstrap_cache if mode_norm == "bootstrapplabel" else {},
            device=device,
            target_dim=int(aux_heads.target_dim),
        )
        valid = semantic_mask & target_valid
        denom = valid.float().sum().clamp_min(1.0)
        pred = torch.nn.functional.normalize(aux["feature_target"], dim=-1)
        tgt = torch.nn.functional.normalize(target, dim=-1)
        cosine = 1.0 - (pred * tgt).sum(dim=-1)
        align_loss = (cosine * valid.float()).sum() / denom
        weighted_terms.append(float(align_weight) * align_loss)
        weight_sum += float(align_weight)

    if query_weight > 0.0:
        endpoint_target = batch["fut_state"][:, -1, :, 0:2].to(device=device, dtype=torch.float32)
        endpoint_valid = batch["fut_valid"][:, -1].to(device=device, dtype=torch.bool) & semantic_mask
        denom = endpoint_valid.float().sum().clamp_min(1.0)
        sq = ((aux["endpoint"] - endpoint_target) ** 2).sum(dim=-1)
        query_loss = (sq * endpoint_valid.float()).sum() / denom
        weighted_terms.append(float(query_weight) * query_loss)
        weight_sum += float(query_weight)

    if not weighted_terms:
        zero = tf_out["pred_coord"].sum() * 0.0
        return zero, {
            "semantic_rescue_loss": 0.0,
            "semantic_alignment_loss": 0.0,
            "query_persistence_consistency_loss": 0.0,
            "readout_semantic_alignment_loss": 0.0,
            "persistence_contrastive_or_ranking_loss": 0.0,
            "confidence_gated_alignment_loss": 0.0,
            "sparse_persistence_contrastive_loss": 0.0,
            "confidence_gated_affected_sample_ratio": 0.0,
            "low_confidence_sample_ratio": 0.0,
            "confidence_metric_threshold": float(confidence_gating_margin_threshold),
            "confidence_metric_temperature": float(confidence_gating_temperature),
            "confidence_metric_definition": 0.0,
            "positive_pair_count": 0.0,
            "hard_negative_count": 0.0,
            "effective_pair_coverage_ratio": 0.0,
            "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
            "aux_loss_delay_steps": float(aux_loss_delay_steps),
            "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
            "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
            "whether_main_rollout_loss_reweighted": False,
            "semantic_bootstrap_cache_hit_ratio": 0.0,
        }

    loss = sum(weighted_terms) / max(float(weight_sum), 1e-6)
    return loss, {
        "semantic_rescue_loss": float(loss.detach().cpu().item()),
        "semantic_alignment_loss": float(align_loss.detach().cpu().item()),
        "query_persistence_consistency_loss": float(query_loss.detach().cpu().item()),
        "readout_semantic_alignment_loss": 0.0,
        "persistence_contrastive_or_ranking_loss": 0.0,
        "confidence_gated_alignment_loss": 0.0,
        "sparse_persistence_contrastive_loss": 0.0,
        "confidence_gated_affected_sample_ratio": 0.0,
        "low_confidence_sample_ratio": 0.0,
        "confidence_metric_threshold": float(confidence_gating_margin_threshold),
        "confidence_metric_temperature": float(confidence_gating_temperature),
        "confidence_metric_definition": 0.0,
        "positive_pair_count": 0.0,
        "hard_negative_count": 0.0,
        "effective_pair_coverage_ratio": 0.0,
        "aux_schedule_scale": float(_aux_schedule_scale(current_step, resume_global_step, aux_loss_delay_steps, aux_loss_ramp_steps)),
        "aux_loss_delay_steps": float(aux_loss_delay_steps),
        "aux_loss_ramp_steps": float(aux_loss_ramp_steps),
        "semantic_aux_subset_weighting_strength": float(semantic_aux_subset_weighting_strength),
        "whether_main_rollout_loss_reweighted": False,
        "semantic_bootstrap_cache_hit_ratio": float(cache_hit_ratio),
    }

def _entity_temporal_appearance_drift(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb_temporal = batch.get("semantic_rgb_crop_temporal")
    mask_temporal = batch.get("semantic_mask_crop_temporal")
    temporal_valid = batch.get("semantic_temporal_valid")
    token_mask = batch.get("token_mask")
    if (
        not isinstance(rgb_temporal, torch.Tensor)
        or not isinstance(mask_temporal, torch.Tensor)
        or not isinstance(temporal_valid, torch.Tensor)
        or not isinstance(token_mask, torch.Tensor)
    ):
        bsz = int(batch["obs_state"].shape[0])
        entity_count = int(batch["obs_state"].shape[2])
        return (
            torch.zeros((bsz, entity_count), device=device, dtype=torch.float32),
            torch.zeros((bsz, entity_count), device=device, dtype=torch.bool),
        )
    rgb_temporal = rgb_temporal.to(device=device, dtype=torch.float32)
    mask_temporal = mask_temporal.to(device=device, dtype=torch.float32)
    temporal_valid = temporal_valid.to(device=device, dtype=torch.bool)
    token_mask = token_mask.to(device=device, dtype=torch.bool)
    bsz, entity_count, _, _, _, _ = rgb_temporal.shape
    scores = torch.zeros((bsz, entity_count), device=device, dtype=torch.float32)
    valid_entity = torch.zeros((bsz, entity_count), device=device, dtype=torch.bool)

    def _masked_rgb_signature(rgb_crop: torch.Tensor, mask_crop: torch.Tensor) -> torch.Tensor:
        mask = mask_crop.clamp(0.0, 1.0)
        denom = mask.sum().clamp_min(1e-6)
        vec = (rgb_crop * mask).reshape(3, -1).sum(dim=-1) / denom
        return torch.nn.functional.normalize(vec, dim=-1)

    for b_idx in range(int(bsz)):
        for ent_idx in range(int(entity_count)):
            if not bool(token_mask[b_idx, ent_idx].item()):
                continue
            valid_steps = torch.nonzero(temporal_valid[b_idx, ent_idx], as_tuple=False).flatten()
            if valid_steps.numel() < 2:
                continue
            first_idx = int(valid_steps[0].item())
            last_idx = int(valid_steps[-1].item())
            early_sig = _masked_rgb_signature(
                rgb_temporal[b_idx, ent_idx, first_idx],
                mask_temporal[b_idx, ent_idx, first_idx],
            )
            late_sig = _masked_rgb_signature(
                rgb_temporal[b_idx, ent_idx, last_idx],
                mask_temporal[b_idx, ent_idx, last_idx],
            )
            cosine = (early_sig * late_sig).sum().clamp(-1.0, 1.0)
            scores[b_idx, ent_idx] = (1.0 - cosine).clamp(0.0, 2.0)
            valid_entity[b_idx, ent_idx] = True
    return scores, valid_entity


def _entity_temporal_local_appearance_delta(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb_temporal = batch.get("semantic_rgb_crop_temporal")
    mask_temporal = batch.get("semantic_mask_crop_temporal")
    temporal_valid = batch.get("semantic_temporal_valid")
    token_mask = batch.get("token_mask")
    if (
        not isinstance(rgb_temporal, torch.Tensor)
        or not isinstance(mask_temporal, torch.Tensor)
        or not isinstance(temporal_valid, torch.Tensor)
        or not isinstance(token_mask, torch.Tensor)
    ):
        bsz = int(batch["obs_state"].shape[0])
        entity_count = int(batch["obs_state"].shape[2])
        return (
            torch.zeros((bsz, entity_count), device=device, dtype=torch.float32),
            torch.zeros((bsz, entity_count), device=device, dtype=torch.bool),
        )
    rgb_temporal = rgb_temporal.to(device=device, dtype=torch.float32)
    mask_temporal = mask_temporal.to(device=device, dtype=torch.float32)
    temporal_valid = temporal_valid.to(device=device, dtype=torch.bool)
    token_mask = token_mask.to(device=device, dtype=torch.bool)
    bsz, entity_count, _, _, _, _ = rgb_temporal.shape
    scores = torch.zeros((bsz, entity_count), device=device, dtype=torch.float32)
    valid_entity = torch.zeros((bsz, entity_count), device=device, dtype=torch.bool)

    def _masked_mean_std(rgb_crop: torch.Tensor, mask_crop: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask_crop.clamp(0.0, 1.0)
        denom = mask.sum().clamp_min(1e-6)
        flat = (rgb_crop * mask).reshape(3, -1)
        mean = flat.sum(dim=-1) / denom
        sq = ((rgb_crop - mean[:, None, None]) ** 2 * mask).reshape(3, -1).sum(dim=-1) / denom
        std = sq.clamp_min(0.0).sqrt()
        return mean, std

    for b_idx in range(int(bsz)):
        for ent_idx in range(int(entity_count)):
            if not bool(token_mask[b_idx, ent_idx].item()):
                continue
            valid_steps = torch.nonzero(temporal_valid[b_idx, ent_idx], as_tuple=False).flatten()
            if valid_steps.numel() < 2:
                continue
            first_idx = int(valid_steps[0].item())
            last_idx = int(valid_steps[-1].item())
            mean_early, std_early = _masked_mean_std(
                rgb_temporal[b_idx, ent_idx, first_idx],
                mask_temporal[b_idx, ent_idx, first_idx],
            )
            mean_late, std_late = _masked_mean_std(
                rgb_temporal[b_idx, ent_idx, last_idx],
                mask_temporal[b_idx, ent_idx, last_idx],
            )
            mean_delta = (mean_early - mean_late).abs().mean()
            std_delta = (std_early - std_late).abs().mean()
            scores[b_idx, ent_idx] = (mean_delta + 0.5 * std_delta).clamp(0.0, 2.0)
            valid_entity[b_idx, ent_idx] = True
    return scores, valid_entity


def _entity_temporal_appearance_signature(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb_temporal = batch.get("semantic_rgb_crop_temporal")
    mask_temporal = batch.get("semantic_mask_crop_temporal")
    temporal_valid = batch.get("semantic_temporal_valid")
    token_mask = batch.get("token_mask")
    if (
        not isinstance(rgb_temporal, torch.Tensor)
        or not isinstance(mask_temporal, torch.Tensor)
        or not isinstance(temporal_valid, torch.Tensor)
        or not isinstance(token_mask, torch.Tensor)
    ):
        bsz = int(batch["obs_state"].shape[0])
        entity_count = int(batch["obs_state"].shape[2])
        return (
            torch.zeros((bsz, entity_count, 6), device=device, dtype=torch.float32),
            torch.zeros((bsz, entity_count), device=device, dtype=torch.bool),
        )
    rgb_temporal = rgb_temporal.to(device=device, dtype=torch.float32)
    mask_temporal = mask_temporal.to(device=device, dtype=torch.float32)
    temporal_valid = temporal_valid.to(device=device, dtype=torch.bool)
    token_mask = token_mask.to(device=device, dtype=torch.bool)
    bsz, entity_count, _, _, _, _ = rgb_temporal.shape
    out = torch.zeros((bsz, entity_count, 6), device=device, dtype=torch.float32)
    valid_entity = torch.zeros((bsz, entity_count), device=device, dtype=torch.bool)

    def _masked_signature(rgb_crop: torch.Tensor, mask_crop: torch.Tensor) -> torch.Tensor:
        mask = mask_crop.clamp(0.0, 1.0)
        denom = mask.sum().clamp_min(1e-6)
        flat = (rgb_crop * mask).reshape(3, -1)
        mean = flat.sum(dim=-1) / denom
        sq = ((rgb_crop - mean[:, None, None]) ** 2 * mask).reshape(3, -1).sum(dim=-1) / denom
        std = sq.clamp_min(0.0).sqrt()
        return torch.cat([mean, std], dim=0)

    for b_idx in range(int(bsz)):
        for ent_idx in range(int(entity_count)):
            if not bool(token_mask[b_idx, ent_idx].item()):
                continue
            valid_steps = torch.nonzero(temporal_valid[b_idx, ent_idx], as_tuple=False).flatten()
            if valid_steps.numel() <= 0:
                continue
            sigs = []
            for step_idx in valid_steps.tolist():
                sigs.append(
                    _masked_signature(
                        rgb_temporal[b_idx, ent_idx, int(step_idx)],
                        mask_temporal[b_idx, ent_idx, int(step_idx)],
                    )
                )
            if not sigs:
                continue
            out[b_idx, ent_idx] = torch.stack(sigs, dim=0).mean(dim=0)
            valid_entity[b_idx, ent_idx] = True
    return out, valid_entity


def _appearance_high_mask(
    *,
    appearance_signal: torch.Tensor,
    valid_mask: torch.Tensor,
    min_threshold: float,
    high_quantile: float,
) -> Tuple[torch.Tensor, float]:
    valid_vals = appearance_signal[valid_mask]
    if valid_vals.numel() <= 0:
        return torch.zeros_like(valid_mask, dtype=torch.bool), float(min_threshold)
    quant = float(valid_vals.detach().quantile(float(min(max(high_quantile, 0.0), 1.0))).item()) if valid_vals.numel() >= 2 else float(valid_vals[0].detach().item())
    threshold = max(float(min_threshold), quant)
    return (valid_mask & (appearance_signal >= threshold)), float(threshold)


def _pair_iou_xywh(box_i: torch.Tensor, box_j: torch.Tensor) -> float:
    xi1 = float(box_i[0].item() - 0.5 * box_i[2].item())
    yi1 = float(box_i[1].item() - 0.5 * box_i[3].item())
    xi2 = float(box_i[0].item() + 0.5 * box_i[2].item())
    yi2 = float(box_i[1].item() + 0.5 * box_i[3].item())
    xj1 = float(box_j[0].item() - 0.5 * box_j[2].item())
    yj1 = float(box_j[1].item() - 0.5 * box_j[3].item())
    xj2 = float(box_j[0].item() + 0.5 * box_j[2].item())
    yj2 = float(box_j[1].item() + 0.5 * box_j[3].item())
    inter_w = max(min(xi2, xj2) - max(xi1, xj1), 0.0)
    inter_h = max(min(yi2, yj2) - max(yi1, yj1), 0.0)
    inter = inter_w * inter_h
    area_i = max(xi2 - xi1, 0.0) * max(yi2 - yi1, 0.0)
    area_j = max(xj2 - xj1, 0.0) * max(yj2 - yj1, 0.0)
    denom = max(area_i + area_j - inter, 1e-6)
    return float(inter / denom)


def _tusb_v3p1_context_scores(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    ambiguity_min_dist_weight: float = 0.40,
    ambiguity_iou_weight: float = 0.35,
    ambiguity_motion_cross_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    obs_state = batch["obs_state"].to(device=device, dtype=torch.float32)
    obs_valid = batch["obs_valid"].to(device=device, dtype=torch.bool)
    token_mask = batch["token_mask"].to(device=device, dtype=torch.bool)
    valid = obs_valid & token_mask[:, None, :]
    centers = obs_state[..., 0:2]
    wh = obs_state[..., 6:8].clamp(0.0, 1.0)
    appearance_teacher_drift_by_entity, appearance_drift_valid = _entity_temporal_appearance_drift(batch, device)
    local_appearance_delta_by_entity, local_appearance_valid = _entity_temporal_local_appearance_delta(batch, device)
    appearance_drift_by_entity = torch.maximum(
        appearance_teacher_drift_by_entity,
        (0.60 * appearance_teacher_drift_by_entity + 0.40 * (local_appearance_delta_by_entity / 0.25).clamp(0.0, 2.0)),
    )
    appearance_valid = appearance_drift_valid | local_appearance_valid
    bsz, obs_len, entity_count, _ = obs_state.shape
    ambiguity_score = torch.zeros((bsz,), device=device, dtype=torch.float32)
    occlusion_risk = torch.zeros((bsz,), device=device, dtype=torch.float32)
    long_gap_like = torch.zeros((bsz,), device=device, dtype=torch.float32)
    appearance_drift = torch.zeros((bsz,), device=device, dtype=torch.float32)
    weight_sum = max(
        float(ambiguity_min_dist_weight) + float(ambiguity_iou_weight) + float(ambiguity_motion_cross_weight),
        1e-6,
    )
    temporal_valid = batch.get("semantic_temporal_valid")
    if isinstance(temporal_valid, torch.Tensor):
        temporal_valid = temporal_valid.to(device=device, dtype=torch.bool)
    for b_idx in range(int(bsz)):
        pair_risks: List[float] = []
        for t_idx in range(int(obs_len)):
            active_idx = torch.nonzero(valid[b_idx, t_idx], as_tuple=False).flatten()
            if active_idx.numel() < 2:
                continue
            for i_pos in range(int(active_idx.numel())):
                for j_pos in range(i_pos + 1, int(active_idx.numel())):
                    i = int(active_idx[i_pos].item())
                    j = int(active_idx[j_pos].item())
                    dist = float(torch.linalg.norm(centers[b_idx, t_idx, i] - centers[b_idx, t_idx, j]).detach().cpu().item())
                    dist_risk = max(0.0, 1.0 - dist / 0.25)
                    box_i = torch.cat([centers[b_idx, t_idx, i], wh[b_idx, t_idx, i]], dim=0)
                    box_j = torch.cat([centers[b_idx, t_idx, j], wh[b_idx, t_idx, j]], dim=0)
                    iou = _pair_iou_xywh(box_i, box_j)
                    motion_cross = 0.0
                    if t_idx + 1 < int(obs_len) and bool(valid[b_idx, t_idx + 1, i].item()) and bool(valid[b_idx, t_idx + 1, j].item()):
                        move_i = centers[b_idx, t_idx + 1, i] - centers[b_idx, t_idx, i]
                        move_j = centers[b_idx, t_idx + 1, j] - centers[b_idx, t_idx, j]
                        norm_i = float(torch.linalg.norm(move_i).detach().cpu().item())
                        norm_j = float(torch.linalg.norm(move_j).detach().cpu().item())
                        if norm_i > 1e-6 and norm_j > 1e-6:
                            cosine = float(
                                torch.nn.functional.cosine_similarity(move_i[None, :], move_j[None, :], dim=-1)[0]
                                .detach()
                                .cpu()
                                .item()
                            )
                            motion_cross = max(0.0, (1.0 - cosine) * 0.5) * max(0.0, 1.0 - dist / 0.35)
                    risk = (
                        float(ambiguity_min_dist_weight) * dist_risk
                        + float(ambiguity_iou_weight) * iou
                        + float(ambiguity_motion_cross_weight) * motion_cross
                    ) / weight_sum
                    pair_risks.append(float(risk))
        if pair_risks:
            top_risks = sorted(pair_risks, reverse=True)[: max(min(len(pair_risks), 5), 1)]
            ambiguity_score[b_idx] = float(sum(top_risks) / max(len(top_risks), 1))
        if isinstance(temporal_valid, torch.Tensor):
            occlusion_vals: List[float] = []
            long_gap_vals: List[float] = []
            for ent_idx in range(int(entity_count)):
                valid_steps = torch.nonzero(temporal_valid[b_idx, ent_idx], as_tuple=False).flatten()
                if valid_steps.numel() < 2:
                    continue
                first_idx = int(valid_steps[0].item())
                last_idx = int(valid_steps[-1].item())
                span = max(last_idx - first_idx + 1, 1)
                coverage = float(valid_steps.numel()) / float(span)
                gap_ratio = float(1.0 - coverage)
                occlusion_vals.append(gap_ratio)
                if span >= 3:
                    long_gap_vals.append(gap_ratio)
            if occlusion_vals:
                occlusion_risk[b_idx] = float(sum(occlusion_vals) / max(len(occlusion_vals), 1))
            if long_gap_vals:
                long_gap_like[b_idx] = float(sum(long_gap_vals) / max(len(long_gap_vals), 1))
        valid_drift = appearance_drift_by_entity[b_idx][appearance_valid[b_idx]]
        if valid_drift.numel() > 0:
            appearance_drift[b_idx] = float(valid_drift.mean().detach().cpu().item())
    return {
        "ambiguity_risk": ambiguity_score.detach(),
        "appearance_drift": appearance_drift.detach(),
        "occlusion_risk": occlusion_risk.detach(),
        "long_gap_like": long_gap_like.detach(),
        "appearance_drift_by_entity": appearance_drift_by_entity.detach(),
        "appearance_drift_entity_valid": appearance_valid.detach(),
        "appearance_teacher_drift_by_entity": appearance_teacher_drift_by_entity.detach(),
        "local_appearance_delta_by_entity": local_appearance_delta_by_entity.detach(),
    }


def _semantic_hard_sample_stats(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.cat([batch["obs_state"], batch["fut_state"]], dim=1).to(device=device, dtype=torch.float32)
    valid = torch.cat([batch["obs_valid"], batch["fut_valid"]], dim=1).to(device=device, dtype=torch.bool)
    token_mask = batch["token_mask"].to(device=device, dtype=torch.bool)
    vt = valid & token_mask[:, None, :]
    coords = state[..., 0:2]
    area = (state[..., 6] * state[..., 7]).clamp(0.0, 1.0)
    step_motion = torch.sqrt(((coords[:, 1:] - coords[:, :-1]) ** 2).sum(dim=-1).clamp_min(1e-12))
    motion_valid = vt[:, 1:] & vt[:, :-1]
    motion = (step_motion * motion_valid.float()).sum(dim=(1, 2)) / motion_valid.float().sum(dim=(1, 2)).clamp_min(1.0)
    area_masked = torch.where(vt, area, torch.zeros_like(area))
    area_mean = area_masked.sum(dim=(1, 2)) / vt.float().sum(dim=(1, 2)).clamp_min(1.0)
    area_max = torch.where(vt, area, torch.full_like(area, -1.0)).amax(dim=(1, 2))
    area_min = torch.where(vt, area, torch.full_like(area, 2.0)).amin(dim=(1, 2))
    area_range = (area_max - area_min).clamp_min(0.0)
    small_score = (0.05 - area_mean).clamp_min(0.0) / 0.05
    center_dist = torch.sqrt(((coords[..., 0] - 0.5) ** 2 + (coords[..., 1] - 0.5) ** 2).clamp_min(1e-12))
    center_score = (1.0 - center_dist / 0.75).clamp(0.0, 1.0)
    center_interaction = (center_score * vt.float()).sum(dim=(1, 2)) / vt.float().sum(dim=(1, 2)).clamp_min(1.0)
    semantic_features = batch["semantic_features"].to(device=device, dtype=torch.float32)
    color_std_mean = semantic_features[..., 3:6].mean(dim=(1, 2)).clamp_min(0.0)
    tusb_ctx = _tusb_v3p1_context_scores(batch=batch, device=device)
    appearance_shift = torch.maximum(color_std_mean, tusb_ctx["appearance_drift"])
    hard_score = (
        0.35 * (motion / 0.05)
        + 0.20 * (area_range / 0.25)
        + 0.20 * small_score
        + 0.15 * center_interaction
        + 0.10 * (appearance_shift / 0.25)
    ).clamp(0.0, 2.5)
    return {
        "motion": motion.detach(),
        "area_mean": area_mean.detach(),
        "area_range": area_range.detach(),
        "small_score": small_score.detach(),
        "center_interaction": center_interaction.detach(),
        "appearance_shift": appearance_shift.detach(),
        "ambiguity_risk": tusb_ctx["ambiguity_risk"].detach(),
        "occlusion_risk": tusb_ctx["occlusion_risk"].detach(),
        "long_gap_like": tusb_ctx["long_gap_like"].detach(),
        "hard_score": hard_score.detach(),
    }


def _semantic_hard_sample_weights(batch: Dict[str, Any], device: torch.device, strength: float) -> torch.Tensor:
    bsz = int(batch["obs_state"].shape[0])
    if float(strength) <= 0.0:
        return torch.ones((bsz,), device=device, dtype=torch.float32)
    hard_score = _semantic_hard_sample_stats(batch=batch, device=device)["hard_score"]
    return 1.0 + float(strength) * hard_score


def _tusb_v3p1_curriculum_weights(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    curriculum_weight: float,
    ambiguity_weight: float,
    appearance_weight: float,
    occlusion_weight: float,
    longgap_weight: float,
    appearance_high_threshold: float = 0.18,
    appearance_high_quantile: float = 0.80,
) -> torch.Tensor:
    bsz = int(batch["obs_state"].shape[0])
    if float(curriculum_weight) <= 0.0:
        return torch.ones((bsz,), device=device, dtype=torch.float32)
    stats = _semantic_hard_sample_stats(batch=batch, device=device)
    ambiguity_high = (stats["ambiguity_risk"] >= 0.45).to(torch.float32)
    appearance_valid = stats["appearance_shift"] > 0.0
    appearance_high, _ = _appearance_high_mask(
        appearance_signal=stats["appearance_shift"],
        valid_mask=appearance_valid,
        min_threshold=float(appearance_high_threshold),
        high_quantile=float(appearance_high_quantile),
    )
    occlusion_high = (stats["occlusion_risk"] >= 0.20).to(torch.float32)
    longgap_high = (stats["long_gap_like"] >= 0.20).to(torch.float32)
    weighted = (
        float(ambiguity_weight) * ambiguity_high
        + float(appearance_weight) * appearance_high.to(torch.float32)
        + float(occlusion_weight) * occlusion_high
        + float(longgap_weight) * longgap_high
    )
    denom = max(
        float(ambiguity_weight) + float(appearance_weight) + float(occlusion_weight) + float(longgap_weight),
        1e-6,
    )
    return 1.0 + float(curriculum_weight) * (weighted / denom)


def _aux_schedule_scale(global_step: int, resume_global_step: int, delay_steps: int, ramp_steps: int) -> float:
    step_offset = max(int(global_step) - int(resume_global_step), 0)
    if step_offset < int(delay_steps):
        return 0.0
    if int(ramp_steps) <= 0:
        return 1.0
    return float(min(max(step_offset - int(delay_steps), 0) / max(int(ramp_steps), 1), 1.0))


def _semantic_hard_subset_indices(path_value: str) -> List[int]:
    target = str(path_value).strip()
    if not target:
        return []
    p = Path(target)
    if not p.exists():
        return []
    try:
        payload = _safe_json(p)
    except Exception:
        return []
    indices = set()
    subsets = payload.get("subsets", {}) if isinstance(payload.get("subsets", {}), dict) else {}
    for name in [
        "occlusion_reappearance",
        "crossing_or_interaction_ambiguity",
        "small_object_or_low_area",
        "appearance_change_or_semantic_shift",
    ]:
        meta = subsets.get(name, {}) if isinstance(subsets.get(name, {}), dict) else {}
        for item in meta.get("items", []) if isinstance(meta.get("items", []), list) else []:
            if isinstance(item, dict) and "dataset_index" in item:
                indices.add(int(item["dataset_index"]))
    return sorted(indices)


def _weighted_teacher_loss(
    pred_coord: torch.Tensor,
    target_coord: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    sq = ((pred_coord - target_coord) ** 2).sum(dim=-1)
    mask_f = valid_mask.float()
    weights = sample_weights[:, None, None].to(device=pred_coord.device, dtype=torch.float32)
    denom = (mask_f * weights).sum().clamp_min(1.0)
    return (sq * mask_f * weights).sum() / denom


def _split_counts_used(summary: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, meta in summary.items():
        if not isinstance(meta, dict):
            continue
        out[str(name)] = int(meta.get("sample_count", 0) or 0)
    return out


def _full_usage_flag(max_samples_value: Any) -> bool:
    try:
        return int(max_samples_value) < 0
    except Exception:
        return False


def _build_progress_payload(
    *,
    args: Any,
    status: str,
    global_step: int,
    target_steps: int,
    train_summary: Dict[str, Dict[str, Any]],
    val_summary: Dict[str, Dict[str, Any]],
    run_metadata: Dict[str, Any],
    runtime_meta: Dict[str, Any],
    checkpoint_dir: Path,
    best_ckpt: Path,
    latest_ckpt: Path,
    eval_history: List[Dict[str, Any]],
    best_metric_so_far: Dict[str, Any] | None,
) -> Dict[str, Any]:
    latest_event = eval_history[-1] if eval_history and isinstance(eval_history[-1], dict) else {}
    latest_metrics = latest_event.get("metrics", {}) if isinstance(latest_event.get("metrics", {}), dict) else {}
    return {
        "generated_at_utc": now_iso(),
        "run_name": str(args.run_name),
        "status": str(status),
        "current_mainline_semantic_source": str(args.semantic_source_mainline),
        "datasets_bound_for_train": [str(x) for x in args.dataset_names],
        "datasets_bound_for_eval": [str(x) for x in args.dataset_names],
        "runtime": runtime_meta,
        "run_metadata": run_metadata,
        "global_step": int(global_step),
        "train_steps_target": int(target_steps),
        "progress_ratio": float(float(global_step) / float(max(target_steps, 1))),
        "whether_full_train_used": bool(_full_usage_flag(args.max_samples_train)),
        "whether_full_val_used": bool(_full_usage_flag(args.max_samples_val)),
        "effective_train_sample_count_per_dataset": _split_counts_used(train_summary),
        "effective_val_sample_count_per_dataset": _split_counts_used(val_summary),
        "checkpoint_inventory": {
            "checkpoint_dir": str(checkpoint_dir),
            "best": str(best_ckpt),
            "latest": str(latest_ckpt),
            "best_exists": bool(best_ckpt.exists()),
            "latest_exists": bool(latest_ckpt.exists()),
        },
        "latest_eval_metrics": _metric_triplet(latest_metrics),
        "best_metric_so_far": best_metric_so_far if isinstance(best_metric_so_far, dict) else None,
    }


def _write_progress_snapshot(path_value: str, payload: Dict[str, Any]) -> None:
    target = str(path_value).strip()
    if not target:
        return
    _write_json(Path(target), payload)


def _checkpoint_payload(
    *,
    args: Any,
    global_step: int,
    best_metric_so_far: Dict[str, Any] | None,
    eval_history: List[Dict[str, Any]],
    semantic_encoder: SemanticEncoder,
    semantic_fusion: SemanticFusion,
    readout_head: torch.nn.Linear,
    future_semantic_state_head: SemanticTraceStateHead | None,
    semantic_state_feedback_adapter: SemanticStateFeedbackAdapter | None,
    semantic_rescue_heads: SemanticRescueAuxHeads | None,
    trace_unit_tokenizer: TraceUnitTokenizer | None,
    trace_unit_factorized_state: TraceUnitFactorizedState | None,
    trace_unit_handshake: TraceUnitHandshake | None,
    trace_unit_broadcast: TraceUnitBroadcast | None,
    optimizer: torch.optim.Optimizer,
    run_metadata: Dict[str, Any],
    stage1_model: TraceCausalTransformer | None = None,
) -> Dict[str, Any]:
    payload = {
        "run_name": str(args.run_name),
        "global_step": int(global_step),
        "best_metric_so_far": best_metric_so_far if isinstance(best_metric_so_far, dict) else None,
        "eval_history": eval_history,
        "semantic_encoder_state_dict": semantic_encoder.state_dict(),
        "semantic_fusion_state_dict": semantic_fusion.state_dict(),
        "readout_head_state_dict": readout_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "run_metadata": run_metadata,
    }
    if future_semantic_state_head is not None:
        payload["future_semantic_state_head_state_dict"] = future_semantic_state_head.state_dict()
    if semantic_state_feedback_adapter is not None:
        payload["semantic_state_feedback_adapter_state_dict"] = semantic_state_feedback_adapter.state_dict()
    if semantic_rescue_heads is not None:
        payload["semantic_rescue_heads_state_dict"] = semantic_rescue_heads.state_dict()
    if trace_unit_tokenizer is not None:
        payload["trace_unit_tokenizer_state_dict"] = trace_unit_tokenizer.state_dict()
    if trace_unit_factorized_state is not None:
        payload["trace_unit_factorized_state_state_dict"] = trace_unit_factorized_state.state_dict()
    if trace_unit_handshake is not None:
        payload["trace_unit_handshake_state_dict"] = trace_unit_handshake.state_dict()
    if trace_unit_broadcast is not None:
        payload["trace_unit_broadcast_state_dict"] = trace_unit_broadcast.state_dict()
    if stage1_model is not None and any(bool(p.requires_grad) for p in stage1_model.parameters()):
        payload["stage1_model_state_dict"] = stage1_model.state_dict()
        payload["stage1_partial_unfreeze_config"] = {
            "mode": str(getattr(args, "stage1_partial_unfreeze_mode", "none")),
            "layer_count": int(getattr(args, "stage1_partial_unfreeze_layer_count", 0) or 0),
            "lr_scale": float(getattr(args, "stage1_partial_unfreeze_lr_scale", 0.0) or 0.0),
        }
    return payload


def _count_trainable_params(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _count_total_params(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters()))


def _freeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


def _build_future_semantic_head_only_audit(
    *,
    enabled: bool,
    warmup_steps: int,
    freeze_non_head: bool,
    stage1_model: torch.nn.Module,
    semantic_encoder: torch.nn.Module,
    semantic_fusion: torch.nn.Module,
    readout_head: torch.nn.Module,
    future_semantic_state_head: torch.nn.Module | None,
    semantic_state_feedback_adapter: torch.nn.Module | None,
    semantic_rescue_heads: torch.nn.Module | None,
    trace_unit_tokenizer: torch.nn.Module | None,
    trace_unit_factorized_state: torch.nn.Module | None,
    trace_unit_handshake: torch.nn.Module | None,
    trace_unit_broadcast: torch.nn.Module | None,
) -> dict[str, Any]:
    head_trainable = _count_trainable_params(future_semantic_state_head)
    non_head_modules = [
        ("stage1_model", stage1_model),
        ("semantic_encoder", semantic_encoder),
        ("semantic_fusion", semantic_fusion),
        ("readout_head", readout_head),
        ("semantic_state_feedback_adapter", semantic_state_feedback_adapter),
        ("semantic_rescue_heads", semantic_rescue_heads),
        ("trace_unit_tokenizer", trace_unit_tokenizer),
        ("trace_unit_factorized_state", trace_unit_factorized_state),
        ("trace_unit_handshake", trace_unit_handshake),
        ("trace_unit_broadcast", trace_unit_broadcast),
    ]
    per_non_head = {name: _count_trainable_params(module) for name, module in non_head_modules}
    non_head_trainable = int(sum(per_non_head.values()))
    total_trainable = int(head_trainable + non_head_trainable)
    return {
        "future_semantic_head_only_warmup": bool(enabled),
        "future_semantic_head_only_warmup_steps": int(warmup_steps),
        "freeze_non_future_semantic_head_during_warmup": bool(freeze_non_head),
        "total_trainable_params": total_trainable,
        "future_semantic_state_head_trainable_params": head_trainable,
        "non_future_semantic_head_trainable_params": non_head_trainable,
        "non_future_semantic_head_trainable_params_by_module": per_non_head,
        "head_only_boundary_ok": bool((not enabled) or (head_trainable > 0 and non_head_trainable == 0)),
    }


def _unfreeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = True


def _build_future_semantic_controlled_joint_audit(
    *,
    enabled: bool,
    stage1_model: torch.nn.Module,
    semantic_encoder: torch.nn.Module,
    semantic_fusion: torch.nn.Module,
    readout_head: torch.nn.Module,
    future_semantic_state_head: torch.nn.Module | None,
    semantic_state_feedback_adapter: torch.nn.Module | None,
    semantic_rescue_heads: torch.nn.Module | None,
    trace_unit_tokenizer: torch.nn.Module | None,
    trace_unit_factorized_state: torch.nn.Module | None,
    trace_unit_handshake: torch.nn.Module | None,
    trace_unit_broadcast: torch.nn.Module | None,
) -> dict[str, Any]:
    modules = {
        "stage1_model": stage1_model,
        "semantic_encoder": semantic_encoder,
        "semantic_fusion": semantic_fusion,
        "semantic_fusion.semantic_proj": getattr(semantic_fusion, "semantic_proj", None),
        "semantic_fusion.gate": getattr(semantic_fusion, "gate", None),
        "semantic_fusion.norm": getattr(semantic_fusion, "norm", None),
        "readout_head": readout_head,
        "future_semantic_state_head": future_semantic_state_head,
        "semantic_state_feedback_adapter": semantic_state_feedback_adapter,
        "semantic_rescue_heads": semantic_rescue_heads,
        "trace_unit_tokenizer": trace_unit_tokenizer,
        "trace_unit_factorized_state": trace_unit_factorized_state,
        "trace_unit_handshake": trace_unit_handshake,
        "trace_unit_broadcast": trace_unit_broadcast,
    }
    trainable_by_module = {name: _count_trainable_params(module) for name, module in modules.items()}
    total_by_module = {name: _count_total_params(module) for name, module in modules.items()}
    frozen_by_module = {name: int(total_by_module[name] - trainable_by_module[name]) for name in modules}
    allowed_positive = {
        "future_semantic_state_head",
        "semantic_fusion.semantic_proj",
        "readout_head",
        "semantic_state_feedback_adapter",
    }
    disallowed_trainable = {
        name: count
        for name, count in trainable_by_module.items()
        if int(count) > 0 and name not in allowed_positive and not name.startswith("semantic_fusion.")
    }
    semantic_fusion_disallowed = {
        name: count
        for name, count in trainable_by_module.items()
        if name in {"semantic_fusion.gate", "semantic_fusion.norm"} and int(count) > 0
    }
    # The aggregate semantic_fusion module includes semantic_proj trainable params;
    # keep detailed submodule checks authoritative.
    disallowed_trainable.pop("semantic_fusion", None)
    disallowed_trainable.update(semantic_fusion_disallowed)
    stage1_trainable = int(trainable_by_module["stage1_model"])
    trace_main_trainable = any(
        int(trainable_by_module[name]) > 0
        for name in [
            "trace_unit_tokenizer",
            "trace_unit_factorized_state",
            "trace_unit_handshake",
            "trace_unit_broadcast",
        ]
    )
    head_trainable = int(trainable_by_module["future_semantic_state_head"]) > 0
    boundary_ok = bool((not enabled) or (stage1_trainable == 0 and not trace_main_trainable and head_trainable and not disallowed_trainable))
    return {
        "future_semantic_controlled_joint": bool(enabled),
        "joint_training_scope": "minimal_adapter_readout",
        "allowed_trainable_modules": sorted(allowed_positive),
        "forbidden_trainable_modules": [
            "stage1_model",
            "semantic_encoder",
            "semantic_fusion.gate",
            "semantic_fusion.norm",
            "semantic_rescue_heads",
            "trace_unit_tokenizer",
            "trace_unit_factorized_state",
            "trace_unit_handshake",
            "trace_unit_broadcast",
        ],
        "trainable_param_count_total": int(sum(trainable_by_module.values()) - trainable_by_module["semantic_fusion"]),
        "trainable_param_count_by_module": trainable_by_module,
        "frozen_param_count_by_module": frozen_by_module,
        "stage1_trainable_param_count": stage1_trainable,
        "trace_backbone_trainable": bool(trace_main_trainable),
        "future_semantic_state_head_trainable": bool(head_trainable),
        "semantic_state_feedback_adapter_trainable": bool(int(trainable_by_module["semantic_state_feedback_adapter"]) > 0),
        "disallowed_trainable_param_count_by_module": disallowed_trainable,
        "controlled_joint_boundary_ok": boundary_ok,
    }


def _configure_future_semantic_controlled_joint_trainability(
    *,
    stage1_model: torch.nn.Module,
    semantic_encoder: torch.nn.Module,
    semantic_fusion: torch.nn.Module,
    readout_head: torch.nn.Module,
    future_semantic_state_head: torch.nn.Module | None,
    semantic_state_feedback_adapter: torch.nn.Module | None,
    semantic_rescue_heads: torch.nn.Module | None,
    trace_unit_tokenizer: torch.nn.Module | None,
    trace_unit_factorized_state: torch.nn.Module | None,
    trace_unit_handshake: torch.nn.Module | None,
    trace_unit_broadcast: torch.nn.Module | None,
    train_semantic_fusion_proj: bool,
    train_readout_head: bool,
) -> None:
    for module in [
        stage1_model,
        semantic_encoder,
        semantic_fusion,
        readout_head,
        future_semantic_state_head,
        semantic_state_feedback_adapter,
        semantic_rescue_heads,
        trace_unit_tokenizer,
        trace_unit_factorized_state,
        trace_unit_handshake,
        trace_unit_broadcast,
    ]:
        _freeze_module(module)
    _unfreeze_module(future_semantic_state_head)
    _unfreeze_module(semantic_state_feedback_adapter)
    if bool(train_semantic_fusion_proj):
        _unfreeze_module(getattr(semantic_fusion, "semantic_proj", None))
    if bool(train_readout_head):
        _unfreeze_module(readout_head)


def _sample_has_reappearance_positive(sample: dict[str, Any], obs_len: int, fut_len: int, slot_count: int) -> bool:
    fut_valid = sample.get("fut_valid")
    obs_valid = sample.get("obs_valid")
    token_mask = sample.get("semantic_mask", sample.get("token_mask"))
    if not isinstance(fut_valid, torch.Tensor) or fut_valid.ndim != 2:
        return False
    if not isinstance(obs_valid, torch.Tensor) or obs_valid.ndim != 2:
        return False
    h = min(int(fut_len), int(fut_valid.shape[0]))
    k = min(int(slot_count), int(fut_valid.shape[1]), int(obs_valid.shape[1]))
    if h <= 0 or k <= 0:
        return False
    future_visibility = fut_valid[:h, :k].to(dtype=torch.bool)
    obs = obs_valid[: int(obs_len), :k].to(dtype=torch.bool)
    obs_seen_any = obs.any(dim=0)
    obs_endpoint_visible = obs[-1] if obs.shape[0] > 0 else torch.zeros_like(obs_seen_any)
    obs_occluded = obs_seen_any & (~obs.all(dim=0))
    gate = ((~obs_endpoint_visible) | obs_occluded) & obs_seen_any
    if isinstance(token_mask, torch.Tensor) and token_mask.ndim == 1:
        gate = gate & token_mask[:k].to(dtype=torch.bool)
    return bool((future_visibility & gate[None, :]).any().item())


def _build_reappearance_positive_sampling_plan(
    dataset: Stage2SemanticDataset,
    *,
    obs_len: int,
    fut_len: int,
    slot_count: int,
    target_min_batch_ratio: float,
) -> dict[str, Any]:
    positives: list[int] = []
    total = int(len(dataset))
    for idx in range(total):
        try:
            if _sample_has_reappearance_positive(dataset[idx], obs_len=obs_len, fut_len=fut_len, slot_count=slot_count):
                positives.append(idx)
        except Exception:
            continue
    sample_positive_rate = float(len(positives) / max(total, 1))
    if sample_positive_rate <= 0.0:
        oversample_factor = 1.0
    else:
        oversample_factor = max(1.0, float(target_min_batch_ratio) * (1.0 - sample_positive_rate) / (sample_positive_rate * max(1.0 - float(target_min_batch_ratio), 1e-6)))
    return {
        "total_train_samples": total,
        "samples_with_reappearance_positive": int(len(positives)),
        "positive_indices": positives,
        "sample_positive_rate": sample_positive_rate,
        "estimated_batches_with_positive_under_batch_size_1": sample_positive_rate,
        "target_min_batch_ratio": float(target_min_batch_ratio),
        "recommended_oversample_factor": float(oversample_factor),
    }


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    set_seed(int(args.seed))

    if int(args.train_steps) <= 0:
        raise ValueError("train_steps must be > 0")
    if int(args.save_every_n_steps) <= 0:
        raise ValueError("save_every_n_steps must be > 0")

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_json = Path(str(args.run_summary_json))
    run_summary_json.parent.mkdir(parents=True, exist_ok=True)
    progress_json = str(args.progress_json).strip()

    best_ckpt = output_dir / "best.pt"
    latest_ckpt = output_dir / "latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime_meta: Dict[str, Any] = {
        "source": "manual",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "single_gpu_only": True,
    }
    if bool(args.use_recommended_runtime):
        rt = _load_runtime_config(args.recommended_runtime_json)
        runtime_meta = {
            "source": "recommended_runtime_json",
            "path": str(args.recommended_runtime_json),
            "num_workers": int(rt.num_workers),
            "pin_memory": bool(rt.pin_memory),
            "persistent_workers": bool(rt.persistent_workers),
            "prefetch_factor": int(rt.prefetch_factor),
            "single_gpu_only": bool(rt.single_gpu_only),
            "selected_gpu_id_runtime_json": int(rt.selected_gpu_id),
            "required_mem_gb": float(rt.required_mem_gb),
            "safety_margin_gb": float(rt.safety_margin_gb),
        }

    stage2_contract = _safe_json(args.stage2_contract_path)
    binding = _extract_binding(stage2_contract)
    core_names = [str(x) for x in binding.get("core", [])]

    train_cfg = Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.train_split),
        contract_path=str(args.stage2_contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_train),
        semantic_patch_radius=int(args.semantic_patch_radius),
        semantic_crop_size=int(args.semantic_crop_size),
        semantic_source_mainline=str(args.semantic_source_mainline),
        semantic_temporal_window=int(args.local_temporal_window),
        predecode_cache_path=str(args.predecode_cache_path),
        teacher_semantic_cache_path=str(args.teacher_semantic_cache_path),
        max_entities_per_sample=int(args.max_entities_per_sample),
    )
    val_cfg = Stage2SemanticDatasetConfig(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.val_split),
        contract_path=str(args.stage2_contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_val),
        semantic_patch_radius=int(args.semantic_patch_radius),
        semantic_crop_size=int(args.semantic_crop_size),
        semantic_source_mainline=str(args.semantic_source_mainline),
        semantic_temporal_window=int(args.local_temporal_window),
        predecode_cache_path=str(args.predecode_cache_path),
        teacher_semantic_cache_path=str(args.teacher_semantic_cache_path),
        max_entities_per_sample=int(args.max_entities_per_sample),
    )

    train_ds = Stage2SemanticDataset(train_cfg)
    val_ds = Stage2SemanticDataset(val_cfg)
    train_summary = dict(train_ds.dataset_summary)
    val_summary = dict(val_ds.dataset_summary)
    core_ready, core_details = _core_dataset_ready(
        train_summary=train_summary,
        val_summary=val_summary,
        core_names=core_names,
        usage=binding.get("usage", {}),
    )

    num_workers = int(runtime_meta.get("num_workers", 0))
    pin_memory = bool(runtime_meta.get("pin_memory", False))
    persistent_workers = bool(runtime_meta.get("persistent_workers", False))
    prefetch_factor = int(runtime_meta.get("prefetch_factor", 2))
    reappearance_positive_sampling_plan = _build_reappearance_positive_sampling_plan(
        train_ds,
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        slot_count=int(args.max_tokens),
        target_min_batch_ratio=float(args.reappearance_positive_min_batch_ratio),
    )
    train_sampler = None
    train_shuffle = True
    if bool(args.reappearance_positive_oversample):
        positive_indices = set(int(x) for x in reappearance_positive_sampling_plan.get("positive_indices", []))
        if positive_indices:
            factor = float(reappearance_positive_sampling_plan.get("recommended_oversample_factor", 1.0))
            weights = [factor if idx in positive_indices else 1.0 for idx in range(len(train_ds))]
            train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_shuffle = False

    train_loader_kwargs: Dict[str, Any] = {
        "dataset": train_ds,
        "batch_size": int(args.batch_size),
        "shuffle": bool(train_shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": stage2_semantic_collate_fn,
    }
    if train_sampler is not None:
        train_loader_kwargs["sampler"] = train_sampler
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        train_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=stage2_semantic_collate_fn,
    )

    stage1_model, stage1_meta = _load_frozen_stage1_backbone(args=args, device=device)

    semantic_encoder = SemanticEncoder(
        SemanticEncoderConfig(
            input_dim=10,
            hidden_dim=int(args.semantic_hidden_dim),
            output_dim=int(args.semantic_embed_dim),
            dropout=0.1,
            mainline_source=str(args.semantic_source_mainline),
            legacy_source=str(args.legacy_semantic_source),
            local_temporal_window=int(args.local_temporal_window),
            local_temporal_fuse_weight=float(args.local_temporal_fuse_weight),
        )
    ).to(device)
    fusion_hidden_dim = int(stage1_model.config.d_model)
    semantic_fusion = SemanticFusion(
        SemanticFusionConfig(
            hidden_dim=fusion_hidden_dim,
            semantic_dim=int(args.semantic_embed_dim),
            dropout=0.1,
        )
    ).to(device)
    readout_head = torch.nn.Linear(fusion_hidden_dim, 2).to(device)
    future_semantic_state_head: SemanticTraceStateHead | None = None
    if bool(args.enable_future_semantic_state_head):
        future_semantic_state_head = SemanticTraceStateHead(
            SemanticTraceStateHeadConfig(
                hidden_dim=int(fusion_hidden_dim),
                semantic_embedding_dim=int(args.future_semantic_embedding_dim),
                identity_embedding_dim=int(args.future_semantic_embedding_dim),
                hypothesis_count=int(args.future_hypothesis_count),
                enable_extent_head=bool(args.enable_future_extent_head),
                enable_multi_hypothesis_head=bool(args.enable_future_multihypothesis_head)
                or int(args.future_hypothesis_count) > 1,
            )
        ).to(device)
    semantic_state_feedback_adapter: SemanticStateFeedbackAdapter | None = None
    if bool(args.enable_semantic_state_feedback):
        semantic_state_feedback_adapter = SemanticStateFeedbackAdapter(
            SemanticStateFeedbackConfig(
                hidden_dim=int(fusion_hidden_dim),
                semantic_embedding_dim=int(args.future_semantic_embedding_dim),
                identity_embedding_dim=int(args.future_semantic_embedding_dim),
                alpha=float(args.semantic_state_feedback_alpha),
            )
        ).to(device)
    trace_unit_tokenizer: TraceUnitTokenizer | None = None
    trace_unit_factorized_state: TraceUnitFactorizedState | None = None
    trace_unit_handshake: TraceUnitHandshake | None = None
    trace_unit_broadcast: TraceUnitBroadcast | None = None
    structure_mode = str(args.stage2_structure_mode).strip().lower()
    if structure_mode == "trace_unit_semantic_binding":
        trace_unit_tokenizer = TraceUnitTokenizer(
            TraceUnitTokenizerConfig(
                hidden_dim=int(fusion_hidden_dim),
                semantic_dim=int(args.semantic_embed_dim),
                state_dim=STATE_DIM,
                teacher_prior_dim=int(args.trace_unit_teacher_prior_dim),
                unit_dim=int(args.trace_unit_dim),
                unit_count=int(args.trace_unit_count),
                slot_iters=int(args.trace_unit_slot_iters),
                assignment_topk=int(args.trace_unit_assignment_topk),
                assignment_temperature=float(args.trace_unit_assignment_temperature),
                use_instance_prior_bias=bool(args.trace_unit_use_instance_prior_bias),
            )
        ).to(device)
        trace_unit_factorized_state = TraceUnitFactorizedState(
            TraceUnitFactorizedStateConfig(
                unit_dim=int(args.trace_unit_dim),
                dyn_update=str(args.trace_unit_dyn_update),
                sem_update=str(args.trace_unit_sem_update),
                sem_alpha_min=float(args.trace_unit_sem_alpha_min),
                sem_alpha_max=float(args.trace_unit_sem_alpha_max),
            )
        ).to(device)
        trace_unit_handshake = TraceUnitHandshake(
            TraceUnitHandshakeConfig(
                unit_dim=int(args.trace_unit_dim),
                handshake_dim=int(args.trace_unit_handshake_dim),
                layers=int(args.trace_unit_handshake_layers),
                writeback=str(args.trace_unit_handshake_writeback),
            )
        ).to(device)
        trace_unit_broadcast = TraceUnitBroadcast(
            TraceUnitBroadcastConfig(
                hidden_dim=int(fusion_hidden_dim),
                unit_dim=int(args.trace_unit_dim),
                residual_weight=float(args.trace_unit_broadcast_residual_weight),
                stopgrad_semantic=bool(args.trace_unit_broadcast_stopgrad_semantic),
            )
        ).to(device)
    semantic_rescue_heads: SemanticRescueAuxHeads | None = None
    rescue_mode = str(args.semantic_rescue_mode).strip().lower()
    rescue_weight = float(args.semantic_rescue_weight)
    bootstrap_cache = _load_bootstrap_cache(str(args.semantic_bootstrap_cache_path))
    if rescue_mode != "none" and rescue_weight > 0.0:
        semantic_rescue_heads = SemanticRescueAuxHeads(
            semantic_dim=int(args.semantic_embed_dim),
            target_dim=int(args.semantic_bootstrap_target_dim),
            readout_dim=int(fusion_hidden_dim),
        ).to(device)

    if bool(args.future_semantic_head_only_warmup) and bool(args.future_semantic_controlled_joint):
        raise ValueError("--future-semantic-head-only-warmup and --future-semantic-controlled-joint are mutually exclusive")

    if bool(args.future_semantic_head_only_warmup):
        if future_semantic_state_head is None:
            raise ValueError("--future-semantic-head-only-warmup requires --enable-future-semantic-state-head")
        if not bool(args.freeze_non_future_semantic_head_during_warmup):
            raise ValueError(
                "--future-semantic-head-only-warmup requires --freeze-non-future-semantic-head-during-warmup "
                "so the rollout trunk cannot train accidentally"
            )
        for module in [
            stage1_model,
            semantic_encoder,
            semantic_fusion,
            readout_head,
            semantic_rescue_heads,
            trace_unit_tokenizer,
            trace_unit_factorized_state,
            trace_unit_handshake,
            trace_unit_broadcast,
            semantic_state_feedback_adapter,
        ]:
            _freeze_module(module)
        for param in future_semantic_state_head.parameters():
            param.requires_grad = True
    elif bool(args.future_semantic_controlled_joint):
        if future_semantic_state_head is None:
            raise ValueError("--future-semantic-controlled-joint requires --enable-future-semantic-state-head")
        _configure_future_semantic_controlled_joint_trainability(
            stage1_model=stage1_model,
            semantic_encoder=semantic_encoder,
            semantic_fusion=semantic_fusion,
            readout_head=readout_head,
            future_semantic_state_head=future_semantic_state_head,
            semantic_state_feedback_adapter=semantic_state_feedback_adapter,
            semantic_rescue_heads=semantic_rescue_heads,
            trace_unit_tokenizer=trace_unit_tokenizer,
            trace_unit_factorized_state=trace_unit_factorized_state,
            trace_unit_handshake=trace_unit_handshake,
            trace_unit_broadcast=trace_unit_broadcast,
            train_semantic_fusion_proj=bool(args.future_semantic_joint_train_semantic_fusion_proj),
            train_readout_head=bool(args.future_semantic_joint_train_readout_head),
        )
        if semantic_state_feedback_adapter is not None:
            for param in semantic_state_feedback_adapter.parameters():
                param.requires_grad = True

    future_semantic_head_only_warmup_audit = _build_future_semantic_head_only_audit(
        enabled=bool(args.future_semantic_head_only_warmup),
        warmup_steps=int(args.future_semantic_head_only_warmup_steps),
        freeze_non_head=bool(args.freeze_non_future_semantic_head_during_warmup),
        stage1_model=stage1_model,
        semantic_encoder=semantic_encoder,
        semantic_fusion=semantic_fusion,
        readout_head=readout_head,
        future_semantic_state_head=future_semantic_state_head,
        semantic_state_feedback_adapter=semantic_state_feedback_adapter,
        semantic_rescue_heads=semantic_rescue_heads,
        trace_unit_tokenizer=trace_unit_tokenizer,
        trace_unit_factorized_state=trace_unit_factorized_state,
        trace_unit_handshake=trace_unit_handshake,
        trace_unit_broadcast=trace_unit_broadcast,
    )
    if bool(args.future_semantic_head_only_warmup) and not bool(future_semantic_head_only_warmup_audit["head_only_boundary_ok"]):
        raise RuntimeError(f"head-only warmup boundary failed: {future_semantic_head_only_warmup_audit}")
    future_semantic_controlled_joint_audit = _build_future_semantic_controlled_joint_audit(
        enabled=bool(args.future_semantic_controlled_joint),
        stage1_model=stage1_model,
        semantic_encoder=semantic_encoder,
        semantic_fusion=semantic_fusion,
        readout_head=readout_head,
        future_semantic_state_head=future_semantic_state_head,
        semantic_state_feedback_adapter=semantic_state_feedback_adapter,
        semantic_rescue_heads=semantic_rescue_heads,
        trace_unit_tokenizer=trace_unit_tokenizer,
        trace_unit_factorized_state=trace_unit_factorized_state,
        trace_unit_handshake=trace_unit_handshake,
        trace_unit_broadcast=trace_unit_broadcast,
    )
    if bool(args.future_semantic_controlled_joint) and not bool(future_semantic_controlled_joint_audit["controlled_joint_boundary_ok"]):
        raise RuntimeError(f"controlled joint boundary failed: {future_semantic_controlled_joint_audit}")

    trainable_params: List[torch.nn.Parameter] = []
    stage1_trainable_params: List[torch.nn.Parameter] = [p for p in stage1_model.parameters() if p.requires_grad]
    modules_for_training: List[torch.nn.Module] = [semantic_encoder, semantic_fusion, readout_head]
    if future_semantic_state_head is not None:
        modules_for_training.append(future_semantic_state_head)
    if semantic_state_feedback_adapter is not None:
        modules_for_training.append(semantic_state_feedback_adapter)
    if semantic_rescue_heads is not None:
        modules_for_training.append(semantic_rescue_heads)
    for optional_module in [
        trace_unit_tokenizer,
        trace_unit_factorized_state,
        trace_unit_handshake,
        trace_unit_broadcast,
    ]:
        if optional_module is not None:
            modules_for_training.append(optional_module)
    for module in modules_for_training:
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])

    pre_frozen_parameter_count = int(stage1_meta.get("parameter_count", 0))
    pre_stage1_trainable_parameter_count = int(stage1_meta.get("trainable_parameter_count", 0))
    pre_trainable_parameter_count = int(sum(p.numel() for p in trainable_params))
    total_trainable_parameter_count = int(pre_trainable_parameter_count + sum(p.numel() for p in stage1_trainable_params))

    print(f"[stage2-smalltrain] pre_frozen_parameter_count={pre_frozen_parameter_count}")
    print(f"[stage2-smalltrain] pre_stage1_trainable_parameter_count={pre_stage1_trainable_parameter_count}")
    print(f"[stage2-smalltrain] pre_stage2_trainable_parameter_count={pre_trainable_parameter_count}")

    optimizer_param_groups: List[Dict[str, Any]] = []
    if trainable_params:
        optimizer_param_groups.append({"params": trainable_params, "lr": float(args.lr)})
    if stage1_trainable_params:
        optimizer_param_groups.append({
            "params": stage1_trainable_params,
            "lr": float(args.lr) * float(args.stage1_partial_unfreeze_lr_scale),
        })
    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))

    gpu_selection = {}
    try:
        gpu_selection = json.loads(str(os.environ.get("TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON", "")) or "{}")
        if not isinstance(gpu_selection, dict):
            gpu_selection = {}
    except Exception:
        gpu_selection = {}

    run_metadata: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "started_at_utc": now_iso(),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "gpu_selection": gpu_selection,
        "stage2_structure_mode": str(structure_mode),
        "current_mainline_semantic_source": str(args.semantic_source_mainline),
        "legacy_semantic_source": str(args.legacy_semantic_source),
        "instance_aware_data_path_enabled": True,
        "semantic_rescue_mode": str(rescue_mode),
        "semantic_rescue_weight": float(rescue_weight),
        "semantic_bootstrap_cache_path": str(args.semantic_bootstrap_cache_path),
        "semantic_bootstrap_target_dim": int(args.semantic_bootstrap_target_dim),
        "semantic_alignment_loss_weight": float(args.semantic_alignment_loss_weight),
        "query_persistence_consistency_loss_weight": float(args.query_persistence_consistency_loss_weight),
        "semantic_hard_curriculum_weight": float(args.semantic_hard_curriculum_weight),
        "readout_semantic_alignment_loss_weight": float(args.readout_semantic_alignment_loss_weight),
        "persistence_contrastive_ranking_loss_weight": float(args.persistence_contrastive_ranking_loss_weight),
        "semantic_aux_subset_weighting_strength": float(args.semantic_aux_subset_weighting_strength),
        "confidence_gated_alignment_loss_weight": float(args.confidence_gated_alignment_loss_weight),
        "sparse_persistence_contrastive_loss_weight": float(args.sparse_persistence_contrastive_loss_weight),
        "enable_future_semantic_state_head": bool(args.enable_future_semantic_state_head),
        "future_semantic_embedding_dim": int(args.future_semantic_embedding_dim),
        "future_semantic_loss_weight": float(args.future_semantic_loss_weight),
        "future_visibility_loss_weight": float(args.future_visibility_loss_weight),
        "future_reappearance_loss_weight": float(args.future_reappearance_loss_weight),
        "future_reappearance_event_loss_weight": float(args.future_reappearance_event_loss_weight),
        "future_reappearance_pos_weight": str(args.future_reappearance_pos_weight),
        "future_reappearance_pos_weight_max": float(args.future_reappearance_pos_weight_max),
        "future_reappearance_mask_policy": str(args.future_reappearance_mask_policy),
        "reappearance_positive_oversample": bool(args.reappearance_positive_oversample),
        "reappearance_positive_min_batch_ratio": float(args.reappearance_positive_min_batch_ratio),
        "reappearance_positive_sampling_plan": reappearance_positive_sampling_plan,
        "future_identity_belief_loss_weight": float(args.future_identity_belief_loss_weight),
        "future_uncertainty_loss_weight": float(args.future_uncertainty_loss_weight),
        "future_hypothesis_count": int(args.future_hypothesis_count),
        "future_hypothesis_loss_weight": float(args.future_hypothesis_loss_weight),
        "enable_future_extent_head": bool(args.enable_future_extent_head),
        "enable_future_multihypothesis_head": bool(args.enable_future_multihypothesis_head),
        "future_semantic_head_only_warmup": bool(args.future_semantic_head_only_warmup),
        "future_semantic_head_only_warmup_steps": int(args.future_semantic_head_only_warmup_steps),
        "freeze_non_future_semantic_head_during_warmup": bool(args.freeze_non_future_semantic_head_during_warmup),
        "future_semantic_head_only_warmup_audit": future_semantic_head_only_warmup_audit,
        "future_semantic_controlled_joint": bool(args.future_semantic_controlled_joint),
        "future_semantic_joint_train_semantic_fusion_proj": bool(args.future_semantic_joint_train_semantic_fusion_proj),
        "future_semantic_joint_train_readout_head": bool(args.future_semantic_joint_train_readout_head),
        "future_semantic_controlled_joint_audit": future_semantic_controlled_joint_audit,
        "semantic_state_feedback_enabled": bool(args.enable_semantic_state_feedback),
        "semantic_state_feedback_mode": str(args.semantic_state_feedback_mode),
        "semantic_state_feedback_alpha": float(args.semantic_state_feedback_alpha),
        "semantic_state_feedback_stopgrad_state": bool(args.semantic_state_feedback_stopgrad_state),
        "semantic_state_feedback_trainable_params": _count_trainable_params(semantic_state_feedback_adapter),
        "semantic_state_feedback_total_params": _count_total_params(semantic_state_feedback_adapter),
        "future_semantic_state_head_default_enabled": False,
        "confidence_metric_definition": "margin between positive CLIP-target cosine and hardest in-batch negative cosine on readout projection",
        "confidence_gating_margin_threshold": float(args.confidence_gating_margin_threshold),
        "confidence_gating_temperature": float(args.confidence_gating_temperature),
        "semantic_hard_score_threshold": float(args.semantic_hard_score_threshold),
        "aux_loss_delay_steps": int(args.aux_loss_delay_steps),
        "aux_loss_ramp_steps": int(args.aux_loss_ramp_steps),
        "v4_sparse_gating_family": str(args.v4_sparse_gating_family),
        "v4_gating_quantile": float(args.v4_gating_quantile),
        "v4_topk_token_ratio": float(args.v4_topk_token_ratio),
        "v4_topk_min_tokens": int(args.v4_topk_min_tokens),
        "v4_gate_min_strength": float(args.v4_gate_min_strength),
        "v4_persistence_value_quantile": float(args.v4_persistence_value_quantile),
        "v4_persistence_max_pairs": int(args.v4_persistence_max_pairs),
        "v4_persistence_margin": float(args.v4_persistence_margin),
        "v5_gating_family": str(args.v5_gating_family),
        "v5_topk_query_k": int(args.v5_topk_query_k),
        "v5_capped_quantile": float(args.v5_capped_quantile),
        "v5_max_affected_ratio": float(args.v5_max_affected_ratio),
        "v5_gate_min_strength": float(args.v5_gate_min_strength),
        "v5_max_pairs_per_sample": int(args.v5_max_pairs_per_sample),
        "v5_hard_negative_cap": int(args.v5_hard_negative_cap),
        "v5_pair_sampling_temperature": float(args.v5_pair_sampling_temperature),
        "v6_gating_family": str(args.v6_gating_family),
        "v6_topk_query_k": int(args.v6_topk_query_k),
        "v6_capped_quantile": float(args.v6_capped_quantile),
        "v6_max_affected_ratio": float(args.v6_max_affected_ratio),
        "v6_gate_min_strength": float(args.v6_gate_min_strength),
        "v6_strict_max_pairs_per_sample": int(args.v6_strict_max_pairs_per_sample),
        "v6_hard_negative_cap": int(args.v6_hard_negative_cap),
        "v6_pair_sampling_temperature": float(args.v6_pair_sampling_temperature),
        "v6_guaranteed_min_pairs_per_sample": int(args.v6_guaranteed_min_pairs_per_sample),
        "v6_two_level_pair_mining_enabled": bool(args.v6_two_level_pair_mining_enabled),
        "v6_relaxed_motion_threshold": float(args.v6_relaxed_motion_threshold),
        "v6_relaxed_area_jump_threshold": float(args.v6_relaxed_area_jump_threshold),
        "v6_relaxed_small_query_threshold": float(args.v6_relaxed_small_query_threshold),
        "v6_relaxed_appearance_shift_threshold": float(args.v6_relaxed_appearance_shift_threshold),
        "v6_relaxed_center_interaction_threshold": float(args.v6_relaxed_center_interaction_threshold),
        "semantic_hard_sidecar_enabled": bool(args.semantic_hard_sidecar_enabled),
        "whether_main_rollout_loss_reweighted": bool((not str(rescue_mode).startswith("v2")) and float(args.semantic_hard_curriculum_weight) > 0.0),
        "semantic_bootstrap_cache_item_count": int(len(bootstrap_cache)),
        "skip_resume_optimizer": bool(args.skip_resume_optimizer),
        "resume_optimizer_loaded": False,
        "resume_optimizer_skip_reason": "",
        "stage1_partial_unfreeze_mode": str(args.stage1_partial_unfreeze_mode),
        "stage1_partial_unfreeze_layer_count": int(args.stage1_partial_unfreeze_layer_count),
        "stage1_partial_unfreeze_lr_scale": float(args.stage1_partial_unfreeze_lr_scale),
        "stage1_partial_unfreeze_active": bool(pre_stage1_trainable_parameter_count > 0),
        "trace_unit_count": int(args.trace_unit_count),
        "trace_unit_dim": int(args.trace_unit_dim),
        "trace_unit_slot_iters": int(args.trace_unit_slot_iters),
        "trace_unit_assignment_topk": int(args.trace_unit_assignment_topk),
        "trace_unit_assignment_temperature": float(args.trace_unit_assignment_temperature),
        "trace_unit_use_instance_prior_bias": bool(args.trace_unit_use_instance_prior_bias),
        "trace_unit_disable_instance_path": bool(args.trace_unit_disable_instance_path),
        "trace_unit_teacher_prior_dim": int(args.trace_unit_teacher_prior_dim),
        "trace_unit_sem_alpha_min": float(args.trace_unit_sem_alpha_min),
        "trace_unit_sem_alpha_max": float(args.trace_unit_sem_alpha_max),
        "trace_unit_handshake_dim": int(args.trace_unit_handshake_dim),
        "trace_unit_handshake_layers": int(args.trace_unit_handshake_layers),
        "trace_unit_broadcast_residual_weight": float(args.trace_unit_broadcast_residual_weight),
        "trace_unit_broadcast_stopgrad_semantic": bool(args.trace_unit_broadcast_stopgrad_semantic),
        "trace_unit_assignment_sparsity_weight": float(args.trace_unit_assignment_sparsity_weight),
            "trace_unit_assignment_temporal_consistency_weight": float(args.trace_unit_assignment_temporal_consistency_weight),
            "trace_unit_semantic_inertia_weight": float(args.trace_unit_semantic_inertia_weight),
            "trace_unit_instance_consistency_weight": float(args.trace_unit_instance_consistency_weight),
            "trace_unit_instance_binding_weight": float(args.trace_unit_instance_binding_weight),
            "trace_unit_interinstance_repulse_weight": float(args.trace_unit_interinstance_repulse_weight),
            "trace_unit_unit_purity_weight": float(args.trace_unit_unit_purity_weight),
            "trace_unit_instance_id_source": str(args.trace_unit_instance_id_source),
            "trace_unit_instance_conf_threshold": float(args.trace_unit_instance_conf_threshold),
            "trace_unit_ambiguity_repulse_weight": float(args.trace_unit_ambiguity_repulse_weight),
            "trace_unit_ambiguity_risk_threshold": float(args.trace_unit_ambiguity_risk_threshold),
            "trace_unit_ambiguity_min_dist_weight": float(args.trace_unit_ambiguity_min_dist_weight),
            "trace_unit_ambiguity_iou_weight": float(args.trace_unit_ambiguity_iou_weight),
            "trace_unit_ambiguity_motion_cross_weight": float(args.trace_unit_ambiguity_motion_cross_weight),
            "trace_unit_appearance_refine_weight": float(args.trace_unit_appearance_refine_weight),
            "trace_unit_dynsem_decorrelation_weight": float(args.trace_unit_dynsem_decorrelation_weight),
            "trace_unit_utilization_weight": float(args.trace_unit_utilization_weight),
        "trace_unit_min_active_target": float(args.trace_unit_min_active_target),
        "trace_unit_diversity_weight": float(args.trace_unit_diversity_weight),
        "trace_unit_top2_floor_weight": float(args.trace_unit_top2_floor_weight),
        "trace_unit_top2_mass_floor": float(args.trace_unit_top2_mass_floor),
        "trace_unit_hardsubset_curriculum_weight": float(args.trace_unit_hardsubset_curriculum_weight),
        "trace_unit_hardsubset_ambiguity_weight": float(args.trace_unit_hardsubset_ambiguity_weight),
        "trace_unit_hardsubset_appearance_weight": float(args.trace_unit_hardsubset_appearance_weight),
        "trace_unit_hardsubset_occlusion_weight": float(args.trace_unit_hardsubset_occlusion_weight),
        "trace_unit_hardsubset_longgap_weight": float(args.trace_unit_hardsubset_longgap_weight),
        "teacher_semantic_cache_path": str(args.teacher_semantic_cache_path),
        "max_entities_per_sample": int(args.max_entities_per_sample),
    }

    resolved_resume = _resolve_resume_path(
        resume_from=str(args.resume_from),
        auto_resume_latest=bool(args.auto_resume_latest),
        latest_path=latest_ckpt,
    )

    global_step = 0
    resume_global_step_loaded = 0
    new_best_written_this_run = False
    best_metric_so_far: Dict[str, Any] | None = None
    eval_history: List[Dict[str, Any]] = []
    if resolved_resume:
        payload = _safe_load_checkpoint(resolved_resume, device=device)
        semantic_encoder_missing, semantic_encoder_unexpected = semantic_encoder.load_state_dict(
            payload.get("semantic_encoder_state_dict", {}),
            strict=False,
        )
        if semantic_encoder_missing:
            run_metadata["semantic_encoder_resume_missing_keys"] = list(semantic_encoder_missing)
        if semantic_encoder_unexpected:
            run_metadata["semantic_encoder_resume_unexpected_keys"] = list(semantic_encoder_unexpected)
        semantic_fusion.load_state_dict(payload.get("semantic_fusion_state_dict", {}))
        readout_head.load_state_dict(payload.get("readout_head_state_dict", {}))
        if future_semantic_state_head is not None and isinstance(payload.get("future_semantic_state_head_state_dict", None), dict):
            future_semantic_state_head.load_state_dict(payload["future_semantic_state_head_state_dict"], strict=False)
        if semantic_state_feedback_adapter is not None and isinstance(payload.get("semantic_state_feedback_adapter_state_dict", None), dict):
            semantic_state_feedback_adapter.load_state_dict(payload["semantic_state_feedback_adapter_state_dict"], strict=False)
        if trace_unit_tokenizer is not None and isinstance(payload.get("trace_unit_tokenizer_state_dict", None), dict):
            trace_unit_tokenizer.load_state_dict(payload["trace_unit_tokenizer_state_dict"], strict=False)
        if trace_unit_factorized_state is not None and isinstance(payload.get("trace_unit_factorized_state_state_dict", None), dict):
            trace_unit_factorized_state.load_state_dict(payload["trace_unit_factorized_state_state_dict"], strict=False)
        if trace_unit_handshake is not None and isinstance(payload.get("trace_unit_handshake_state_dict", None), dict):
            trace_unit_handshake.load_state_dict(payload["trace_unit_handshake_state_dict"], strict=False)
        if trace_unit_broadcast is not None and isinstance(payload.get("trace_unit_broadcast_state_dict", None), dict):
            trace_unit_broadcast.load_state_dict(payload["trace_unit_broadcast_state_dict"], strict=False)
        if isinstance(payload.get("stage1_model_state_dict", None), dict):
            stage1_model.load_state_dict(payload["stage1_model_state_dict"], strict=False)
        if semantic_rescue_heads is not None and isinstance(payload.get("semantic_rescue_heads_state_dict", None), dict):
            semantic_rescue_heads.load_state_dict(payload["semantic_rescue_heads_state_dict"])
        if isinstance(payload.get("optimizer_state_dict", None), dict):
            if bool(args.skip_resume_optimizer):
                run_metadata["resume_optimizer_skip_reason"] = "explicit_skip_resume_optimizer"
            else:
                try:
                    optimizer.load_state_dict(payload["optimizer_state_dict"])
                    run_metadata["resume_optimizer_loaded"] = True
                except ValueError as exc:
                    run_metadata["resume_optimizer_skip_reason"] = f"incompatible_optimizer_state: {exc}"
        global_step = int(payload.get("global_step", 0) or 0)
        resume_global_step_loaded = int(global_step)
        if isinstance(payload.get("best_metric_so_far", None), dict):
            best_metric_so_far = payload.get("best_metric_so_far")
        if isinstance(payload.get("eval_history", None), list):
            eval_history = [x for x in payload.get("eval_history", []) if isinstance(x, dict)]
        run_metadata["resumed_from"] = str(resolved_resume)
    else:
        run_metadata["resumed_from"] = ""

    semantic_encoder.train()
    semantic_fusion.train()
    readout_head.train()
    if future_semantic_state_head is not None:
        future_semantic_state_head.train()
    if semantic_state_feedback_adapter is not None:
        semantic_state_feedback_adapter.train()
    if semantic_rescue_heads is not None:
        semantic_rescue_heads.train()
    if trace_unit_tokenizer is not None:
        trace_unit_tokenizer.train()
    if trace_unit_factorized_state is not None:
        trace_unit_factorized_state.train()
    if trace_unit_handshake is not None:
        trace_unit_handshake.train()
    if trace_unit_broadcast is not None:
        trace_unit_broadcast.train()
    if bool(args.future_semantic_head_only_warmup):
        semantic_encoder.eval()
        semantic_fusion.eval()
        readout_head.eval()
        if semantic_rescue_heads is not None:
            semantic_rescue_heads.eval()
        if trace_unit_tokenizer is not None:
            trace_unit_tokenizer.eval()
        if trace_unit_factorized_state is not None:
            trace_unit_factorized_state.eval()
        if trace_unit_handshake is not None:
            trace_unit_handshake.eval()
        if trace_unit_broadcast is not None:
            trace_unit_broadcast.eval()
        if semantic_state_feedback_adapter is not None:
            semantic_state_feedback_adapter.eval()
        if future_semantic_state_head is not None:
            future_semantic_state_head.train()
    elif bool(args.future_semantic_controlled_joint):
        semantic_encoder.eval()
        semantic_fusion.train()
        readout_head.train() if bool(args.future_semantic_joint_train_readout_head) else readout_head.eval()
        if semantic_rescue_heads is not None:
            semantic_rescue_heads.eval()
        # Controlled joint training keeps these trunks parameter-frozen, but the
        # future-state losses still backpropagate through their hidden states to
        # the tiny adapter/readout slice. cuDNN RNN backward requires train mode
        # even when the recurrent parameters themselves are frozen.
        if trace_unit_tokenizer is not None:
            trace_unit_tokenizer.train()
        if trace_unit_factorized_state is not None:
            trace_unit_factorized_state.train()
        if trace_unit_handshake is not None:
            trace_unit_handshake.train()
        if trace_unit_broadcast is not None:
            trace_unit_broadcast.train()
        if future_semantic_state_head is not None:
            future_semantic_state_head.train()
        if semantic_state_feedback_adapter is not None:
            semantic_state_feedback_adapter.train()

    train_iter = iter(train_loader)
    step_checkpoints: List[str] = sorted(str(p) for p in output_dir.glob("step_*.pt"))
    best_semantic_hard_ckpt = output_dir / "best_semantic_hard.pt"
    stage1_grad_detected_any = False
    semantic_grad_norm_latest = 0.0
    teacher_loss_history: List[float] = []
    rescue_loss_history: List[float] = []
    semantic_alignment_loss_history: List[float] = []
    query_persistence_loss_history: List[float] = []
    readout_alignment_loss_history: List[float] = []
    persistence_contrastive_loss_history: List[float] = []
    confidence_gated_alignment_loss_history: List[float] = []
    sparse_persistence_contrastive_loss_history: List[float] = []
    confidence_gated_affected_ratio_history: List[float] = []
    low_confidence_ratio_history: List[float] = []
    pair_coverage_ratio_history: List[float] = []
    positive_pair_count_history: List[float] = []
    hard_negative_count_history: List[float] = []
    actual_gate_positive_ratio_history: List[float] = []
    activated_query_count_history: List[float] = []
    activated_query_ratio_history: List[float] = []
    per_batch_sparsity_mean_history: List[float] = []
    per_batch_sparsity_std_history: List[float] = []
    raw_quantile_ratio_history: List[float] = []
    capped_ratio_history: List[float] = []
    valuable_pair_ratio_history: List[float] = []
    fallback_trigger_rate_history: List[float] = []
    guaranteed_pair_count_history: List[float] = []
    strict_pair_ratio_history: List[float] = []
    fallback_pair_ratio_history: List[float] = []
    final_effective_aux_weight_history: List[float] = []
    aux_schedule_scale_history: List[float] = []
    sparse_gate_selected_ratio_history: List[float] = []
    high_value_pair_ratio_history: List[float] = []
    high_value_pair_count_history: List[float] = []
    persistence_candidate_pair_count_history: List[float] = []
    rescue_cache_hit_history: List[float] = []
    semantic_hard_weight_mean_history: List[float] = []
    gate_history: List[float] = []
    future_semantic_state_loss_history: List[float] = []
    future_visibility_loss_history: List[float] = []
    future_reappearance_loss_history: List[float] = []
    future_reappearance_event_loss_history: List[float] = []
    future_semantic_embedding_loss_history: List[float] = []
    future_identity_belief_loss_history: List[float] = []
    future_uncertainty_loss_history: List[float] = []
    future_hypothesis_loss_history: List[float] = []
    future_semantic_state_valid_history: List[float] = []
    future_visibility_supervised_ratio_history: List[float] = []
    future_reappearance_supervised_ratio_history: List[float] = []
    future_visibility_positive_rate_history: List[float] = []
    future_reappearance_positive_rate_history: List[float] = []
    future_reappearance_event_positive_rate_history: List[float] = []
    future_reappearance_risk_entry_ratio_history: List[float] = []
    future_reappearance_risk_slot_ratio_history: List[float] = []
    future_reappearance_pos_weight_history: List[float] = []
    future_reappearance_event_pos_weight_history: List[float] = []
    future_reappearance_head_available_history: List[float] = []
    future_reappearance_event_head_available_history: List[float] = []
    future_reappearance_loss_uses_independent_logit_history: List[float] = []
    future_reappearance_event_loss_uses_independent_logit_history: List[float] = []
    future_visibility_target_source_history: List[str] = []
    future_visibility_target_quality_history: List[str] = []
    trace_unit_loss_history: List[float] = []
    trace_unit_assignment_entropy_history: List[float] = []
    trace_unit_top2_ratio_history: List[float] = []
    trace_unit_active_unit_count_history: List[float] = []
    trace_unit_z_dyn_drift_history: List[float] = []
    trace_unit_z_sem_drift_history: List[float] = []
    trace_unit_z_sem_to_dyn_ratio_history: List[float] = []
    trace_unit_same_instance_consistency_history: List[float] = []
    trace_unit_same_instance_match_rate_history: List[float] = []
    trace_unit_same_instance_assignment_cosine_history: List[float] = []
    trace_unit_diff_instance_separation_history: List[float] = []
    trace_unit_diff_instance_collision_history: List[float] = []
    trace_unit_ambiguity_collision_history: List[float] = []
    trace_unit_confuser_collision_history: List[float] = []
    trace_unit_appearance_highrisk_match_history: List[float] = []
    trace_unit_appearance_high_ratio_history: List[float] = []
    trace_unit_batch_appearance_high_ratio_history: List[float] = []
    trace_unit_step_appearance_high_count_history: List[float] = []
    trace_unit_appearance_signal_valid_count_history: List[float] = []
    trace_unit_appearance_refine_nonzero_history: List[float] = []
    trace_unit_purity_history: List[float] = []
    trace_unit_track_stability_history: List[float] = []
    trace_unit_target_entity_consistency_history: List[float] = []
    trace_unit_semantic_stability_history: List[float] = []
    trace_unit_broadcast_norm_history: List[float] = []
    true_instance_entity_count_history: List[float] = []
    pseudo_entity_count_history: List[float] = []
    fallback_entity_count_history: List[float] = []
    true_instance_ratio_per_batch_history: List[float] = []
    semantic_nonempty_count = 0
    optimizer_steps_this_run = 0
    best_semantic_hard_metric: Dict[str, Any] | None = None

    semantic_hard_loader: DataLoader | None = None
    if bool(args.semantic_hard_sidecar_enabled):
        hard_indices = _semantic_hard_subset_indices(str(args.semantic_hard_manifest_path))
        if hard_indices:
            hard_cfg = Stage2SemanticDatasetConfig(
                dataset_names=[str(x) for x in args.dataset_names],
                split=str(args.val_split),
                contract_path=str(args.stage2_contract_path),
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
                max_tokens=int(args.max_tokens),
                max_samples_per_dataset=-1,
                semantic_patch_radius=int(args.semantic_patch_radius),
                semantic_crop_size=int(args.semantic_crop_size),
                semantic_source_mainline=str(args.semantic_source_mainline),
                semantic_temporal_window=int(args.local_temporal_window),
                predecode_cache_path=str(args.predecode_cache_path),
                teacher_semantic_cache_path=str(args.teacher_semantic_cache_path),
                max_entities_per_sample=int(args.max_entities_per_sample),
            )
            hard_ds_full = Stage2SemanticDataset(hard_cfg)
            hard_subset = torch.utils.data.Subset(hard_ds_full, hard_indices)
            semantic_hard_loader = DataLoader(
                hard_subset,
                batch_size=max(1, int(args.batch_size)),
                shuffle=False,
                num_workers=0,
                pin_memory=bool(pin_memory),
                collate_fn=stage2_semantic_collate_fn,
            )

    target_steps = int(args.train_steps)
    eval_interval = int(args.eval_interval)
    save_every = int(args.save_every_n_steps)
    progress_heartbeat_every = int(min(max(eval_interval, 1), 50))

    _write_progress_snapshot(
        progress_json,
        _build_progress_payload(
            args=args,
            status="initialized",
            global_step=int(global_step),
            target_steps=int(target_steps),
            train_summary=train_summary,
            val_summary=val_summary,
            run_metadata=run_metadata,
            runtime_meta=runtime_meta,
            checkpoint_dir=output_dir,
            best_ckpt=best_ckpt,
            latest_ckpt=latest_ckpt,
            eval_history=eval_history,
            best_metric_so_far=best_metric_so_far,
        ),
    )

    def _save_latest_and_optional_step(step_now: int, save_step: bool) -> None:
        payload = _checkpoint_payload(
            args=args,
            global_step=int(step_now),
            best_metric_so_far=best_metric_so_far,
            eval_history=eval_history,
            semantic_encoder=semantic_encoder,
            semantic_fusion=semantic_fusion,
            readout_head=readout_head,
            future_semantic_state_head=future_semantic_state_head,
            semantic_state_feedback_adapter=semantic_state_feedback_adapter,
            semantic_rescue_heads=semantic_rescue_heads,
            trace_unit_tokenizer=trace_unit_tokenizer,
            trace_unit_factorized_state=trace_unit_factorized_state,
            trace_unit_handshake=trace_unit_handshake,
            trace_unit_broadcast=trace_unit_broadcast,
            optimizer=optimizer,
            run_metadata=run_metadata,
            stage1_model=stage1_model,
        )
        _atomic_torch_save(payload, latest_ckpt)
        if save_step:
            step_path = output_dir / f"step_{int(step_now):07d}.pt"
            _atomic_torch_save(payload, step_path)
            sp = str(step_path)
            if sp not in step_checkpoints:
                step_checkpoints.append(sp)

    while global_step < target_steps:
        try:
            raw_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_batch = next(train_iter)

        meta_rows = raw_batch.get("meta", []) if isinstance(raw_batch, dict) else []
        batch_true_count = 0.0
        batch_pseudo_count = 0.0
        batch_fallback_count = 0.0
        batch_entity_total = 0.0
        for meta_row in meta_rows if isinstance(meta_rows, list) else []:
            if not isinstance(meta_row, dict):
                continue
            batch_true_count += float(meta_row.get("true_instance_entity_count", 0.0) or 0.0)
            batch_pseudo_count += float(meta_row.get("pseudo_entity_count", 0.0) or 0.0)
            batch_fallback_count += float(meta_row.get("fallback_entity_count", 0.0) or 0.0)
            batch_entity_total += float(meta_row.get("entity_count", 0.0) or 0.0)
        batch_true_ratio = float(batch_true_count / max(batch_entity_total, 1.0))
        true_instance_entity_count_history.append(float(batch_true_count))
        pseudo_entity_count_history.append(float(batch_pseudo_count))
        fallback_entity_count_history.append(float(batch_fallback_count))
        true_instance_ratio_per_batch_history.append(float(batch_true_ratio))
        run_metadata["true_instance_entity_count_mean_so_far"] = float(
            sum(true_instance_entity_count_history) / max(len(true_instance_entity_count_history), 1)
        )
        run_metadata["pseudo_entity_count_mean_so_far"] = float(
            sum(pseudo_entity_count_history) / max(len(pseudo_entity_count_history), 1)
        )
        run_metadata["fallback_entity_count_mean_so_far"] = float(
            sum(fallback_entity_count_history) / max(len(fallback_entity_count_history), 1)
        )
        run_metadata["true_instance_ratio_per_batch_mean_so_far"] = float(
            sum(true_instance_ratio_per_batch_history) / max(len(true_instance_ratio_per_batch_history), 1)
        )

        batch = _to_device(raw_batch, device=device, non_blocking=bool(pin_memory and device.type == "cuda"))
        tf_out = _teacher_forced_predict(
            stage1_model=stage1_model,
            semantic_encoder=semantic_encoder,
            semantic_fusion=semantic_fusion,
            readout_head=readout_head,
            structure_mode=str(structure_mode),
            trace_unit_tokenizer=trace_unit_tokenizer,
            trace_unit_factorized_state=trace_unit_factorized_state,
            trace_unit_handshake=trace_unit_handshake,
            trace_unit_broadcast=trace_unit_broadcast,
            trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
            batch=batch,
            obs_len=int(args.obs_len),
            semantic_source_mainline=str(args.semantic_source_mainline),
            allow_stage1_grad=bool(pre_stage1_trainable_parameter_count > 0),
            future_semantic_state_head=future_semantic_state_head,
            semantic_state_feedback_adapter=semantic_state_feedback_adapter,
            semantic_state_feedback_enabled=bool(args.enable_semantic_state_feedback),
            semantic_state_feedback_alpha=float(args.semantic_state_feedback_alpha),
            semantic_state_feedback_stopgrad_state=bool(args.semantic_state_feedback_stopgrad_state),
            semantic_state_feedback_mode=str(args.semantic_state_feedback_mode),
        )

        main_rollout_reweight_strength = 0.0 if str(rescue_mode).startswith("v2") else float(args.semantic_hard_curriculum_weight)
        semantic_hard_weights = _semantic_hard_sample_weights(
            batch=batch,
            device=device,
            strength=float(main_rollout_reweight_strength),
        )
        tusb_v3p1_curriculum_weights = _tusb_v3p1_curriculum_weights(
            batch=batch,
            device=device,
            curriculum_weight=float(args.trace_unit_hardsubset_curriculum_weight),
            ambiguity_weight=float(args.trace_unit_hardsubset_ambiguity_weight),
            appearance_weight=float(args.trace_unit_hardsubset_appearance_weight),
            occlusion_weight=float(args.trace_unit_hardsubset_occlusion_weight),
            longgap_weight=float(args.trace_unit_hardsubset_longgap_weight),
            appearance_high_threshold=float(args.trace_unit_appearance_high_threshold),
            appearance_high_quantile=float(args.trace_unit_appearance_high_quantile),
        )
        teacher_loss = _weighted_teacher_loss(
            tf_out["pred_coord"],
            tf_out["target_coord"],
            tf_out["valid_mask"],
            semantic_hard_weights * tusb_v3p1_curriculum_weights,
        )
        rescue_loss, rescue_info = _semantic_rescue_loss(
            mode=str(rescue_mode),
            aux_heads=semantic_rescue_heads,
            tf_out=tf_out,
            batch=batch,
            bootstrap_cache=bootstrap_cache,
            device=device,
            current_step=int(global_step),
            resume_global_step=int(resume_global_step_loaded),
            semantic_alignment_loss_weight=float(args.semantic_alignment_loss_weight),
            query_persistence_consistency_loss_weight=float(args.query_persistence_consistency_loss_weight),
            readout_semantic_alignment_loss_weight=float(args.readout_semantic_alignment_loss_weight),
            persistence_contrastive_or_ranking_loss_weight=float(args.persistence_contrastive_ranking_loss_weight),
            semantic_aux_subset_weighting_strength=float(args.semantic_aux_subset_weighting_strength),
            confidence_gated_alignment_loss_weight=float(args.confidence_gated_alignment_loss_weight),
            sparse_persistence_contrastive_loss_weight=float(args.sparse_persistence_contrastive_loss_weight),
            confidence_gating_margin_threshold=float(args.confidence_gating_margin_threshold),
            confidence_gating_temperature=float(args.confidence_gating_temperature),
            semantic_hard_score_threshold=float(args.semantic_hard_score_threshold),
            aux_loss_delay_steps=int(args.aux_loss_delay_steps),
            aux_loss_ramp_steps=int(args.aux_loss_ramp_steps),
            v4_sparse_gating_family=str(args.v4_sparse_gating_family),
            v4_gating_quantile=float(args.v4_gating_quantile),
            v4_topk_token_ratio=float(args.v4_topk_token_ratio),
            v4_topk_min_tokens=int(args.v4_topk_min_tokens),
            v4_gate_min_strength=float(args.v4_gate_min_strength),
            v4_persistence_value_quantile=float(args.v4_persistence_value_quantile),
            v4_persistence_max_pairs=int(args.v4_persistence_max_pairs),
            v4_persistence_margin=float(args.v4_persistence_margin),
            v5_gating_family=str(args.v5_gating_family),
            v5_topk_query_k=int(args.v5_topk_query_k),
            v5_capped_quantile=float(args.v5_capped_quantile),
            v5_max_affected_ratio=float(args.v5_max_affected_ratio),
            v5_gate_min_strength=float(args.v5_gate_min_strength),
            v5_max_pairs_per_sample=int(args.v5_max_pairs_per_sample),
            v5_hard_negative_cap=int(args.v5_hard_negative_cap),
            v5_pair_sampling_temperature=float(args.v5_pair_sampling_temperature),
            v6_gating_family=str(args.v6_gating_family),
            v6_topk_query_k=int(args.v6_topk_query_k),
            v6_capped_quantile=float(args.v6_capped_quantile),
            v6_max_affected_ratio=float(args.v6_max_affected_ratio),
            v6_gate_min_strength=float(args.v6_gate_min_strength),
            v6_strict_max_pairs_per_sample=int(args.v6_strict_max_pairs_per_sample),
            v6_hard_negative_cap=int(args.v6_hard_negative_cap),
            v6_pair_sampling_temperature=float(args.v6_pair_sampling_temperature),
            v6_guaranteed_min_pairs_per_sample=int(args.v6_guaranteed_min_pairs_per_sample),
            v6_two_level_pair_mining_enabled=bool(args.v6_two_level_pair_mining_enabled),
            v6_relaxed_motion_threshold=float(args.v6_relaxed_motion_threshold),
            v6_relaxed_area_jump_threshold=float(args.v6_relaxed_area_jump_threshold),
            v6_relaxed_small_query_threshold=float(args.v6_relaxed_small_query_threshold),
            v6_relaxed_appearance_shift_threshold=float(args.v6_relaxed_appearance_shift_threshold),
            v6_relaxed_center_interaction_threshold=float(args.v6_relaxed_center_interaction_threshold),
        )
        trace_unit_loss, trace_unit_info = _trace_unit_regularization_loss(
            structure_mode=str(structure_mode),
            trace_unit_aux=tf_out.get("trace_unit_aux", {}),
            batch=batch,
            device=device,
            assignment_sparsity_weight=float(args.trace_unit_assignment_sparsity_weight),
            assignment_temporal_consistency_weight=float(args.trace_unit_assignment_temporal_consistency_weight),
            semantic_inertia_weight=float(args.trace_unit_semantic_inertia_weight),
            instance_consistency_weight=float(args.trace_unit_instance_consistency_weight),
            instance_binding_weight=float(args.trace_unit_instance_binding_weight),
            interinstance_repulse_weight=float(args.trace_unit_interinstance_repulse_weight),
            unit_purity_weight=float(args.trace_unit_unit_purity_weight),
            instance_conf_threshold=float(args.trace_unit_instance_conf_threshold),
            ambiguity_repulse_weight=float(args.trace_unit_ambiguity_repulse_weight),
            ambiguity_risk_threshold=float(args.trace_unit_ambiguity_risk_threshold),
            ambiguity_min_dist_weight=float(args.trace_unit_ambiguity_min_dist_weight),
            ambiguity_iou_weight=float(args.trace_unit_ambiguity_iou_weight),
            ambiguity_motion_cross_weight=float(args.trace_unit_ambiguity_motion_cross_weight),
            confuser_separation_weight=float(args.trace_unit_confuser_separation_weight),
            confuser_risk_threshold=float(args.trace_unit_confuser_risk_threshold),
            confuser_appearance_weight=float(args.trace_unit_confuser_appearance_weight),
            confuser_motion_weight=float(args.trace_unit_confuser_motion_weight),
            confuser_overlap_weight=float(args.trace_unit_confuser_overlap_weight),
            appearance_refine_weight=float(args.trace_unit_appearance_refine_weight),
            appearance_high_threshold=float(args.trace_unit_appearance_high_threshold),
            appearance_high_quantile=float(args.trace_unit_appearance_high_quantile),
            dynsem_decorrelation_weight=float(args.trace_unit_dynsem_decorrelation_weight),
            utilization_weight=float(args.trace_unit_utilization_weight),
            min_active_target=float(args.trace_unit_min_active_target),
            diversity_weight=float(args.trace_unit_diversity_weight),
            top2_floor_weight=float(args.trace_unit_top2_floor_weight),
            top2_mass_floor=float(args.trace_unit_top2_mass_floor),
        )
        future_semantic_state_loss = tf_out["pred_coord"].sum() * 0.0
        future_semantic_state_info: Dict[str, Any] = {
            "future_trace_coord_loss": 0.0,
            "future_visibility_loss": 0.0,
            "future_reappearance_loss": 0.0,
            "future_reappearance_event_loss": 0.0,
            "future_semantic_embedding_loss": 0.0,
            "future_identity_belief_loss": 0.0,
            "future_uncertainty_loss": 0.0,
            "future_hypothesis_loss": 0.0,
            "future_semantic_state_loss": 0.0,
            "future_semantic_state_output_valid": bool(future_semantic_state_head is None),
        }
        if future_semantic_state_head is not None and tf_out.get("future_semantic_trace_state") is not None:
            coord_error = torch.sqrt(((tf_out["pred_coord"] - tf_out["target_coord"]) ** 2).sum(dim=-1).clamp_min(1e-12))
            visibility_targets = build_future_visibility_reappearance_targets(
                batch=batch,
                out=tf_out,
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
                slot_count=int(tf_out["pred_coord"].shape[2]),
                reappearance_mask_policy=str(args.future_reappearance_mask_policy),
            )
            future_semantic_state_loss, future_semantic_state_info = compute_future_semantic_state_losses(
                state=tf_out["future_semantic_trace_state"],
                semantic_target=tf_out["semantic_tokens"],
                identity_target=tf_out["semantic_tokens"],
                visibility_target=visibility_targets.future_visibility_target,
                valid_mask=tf_out["valid_mask"],
                coord_error=coord_error.detach(),
                instance_confidence=batch.get("semantic_entity_true_instance_confidence"),
                cfg=FutureSemanticStateLossConfig(
                    semantic_loss_weight=float(args.future_semantic_loss_weight),
                    visibility_loss_weight=float(args.future_visibility_loss_weight),
                    reappearance_loss_weight=float(args.future_reappearance_loss_weight),
                    reappearance_event_loss_weight=float(args.future_reappearance_event_loss_weight),
                    reappearance_pos_weight=args.future_reappearance_pos_weight,
                    reappearance_pos_weight_max=float(args.future_reappearance_pos_weight_max),
                    identity_belief_loss_weight=float(args.future_identity_belief_loss_weight),
                    uncertainty_loss_weight=float(args.future_uncertainty_loss_weight),
                    hypothesis_loss_weight=float(args.future_hypothesis_loss_weight),
                    instance_conf_threshold=float(args.trace_unit_instance_conf_threshold),
                ),
                visibility_mask=visibility_targets.future_visibility_mask,
                reappearance_target=visibility_targets.future_reappearance_target,
                reappearance_mask=visibility_targets.future_reappearance_mask,
                reappearance_event_target=visibility_targets.future_reappearance_event_target,
                reappearance_event_mask=visibility_targets.future_reappearance_event_mask,
                visibility_target_info=visibility_targets.to_loss_info(),
            )
        aux_schedule_scale = float(rescue_info.get("aux_schedule_scale", 1.0))
        total_train_loss = (
            teacher_loss
            + float(rescue_weight) * float(aux_schedule_scale) * rescue_loss
            + trace_unit_loss
            + future_semantic_state_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_train_loss.backward()

        if float(args.clip_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.clip_grad_norm))

        stage1_grad_detected = False
        for p in stage1_model.parameters():
            if p.grad is not None and float(p.grad.detach().abs().sum().item()) > 0.0:
                stage1_grad_detected = True
                break
        stage1_grad_detected_any = bool(stage1_grad_detected_any or stage1_grad_detected)

        semantic_grad_sq = 0.0
        for p in trainable_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            semantic_grad_sq += float((g * g).sum().item())
        semantic_grad_norm_latest = float(np.sqrt(max(semantic_grad_sq, 0.0)))

        optimizer.step()

        global_step += 1
        optimizer_steps_this_run += 1

        teacher_loss_history.append(float(teacher_loss.detach().cpu().item()))
        rescue_loss_history.append(float(rescue_info.get("semantic_rescue_loss", 0.0)))
        semantic_alignment_loss_history.append(float(rescue_info.get("semantic_alignment_loss", 0.0)))
        query_persistence_loss_history.append(float(rescue_info.get("query_persistence_consistency_loss", 0.0)))
        readout_alignment_loss_history.append(float(rescue_info.get("readout_semantic_alignment_loss", 0.0)))
        persistence_contrastive_loss_history.append(float(rescue_info.get("persistence_contrastive_or_ranking_loss", 0.0)))
        confidence_gated_alignment_loss_history.append(float(rescue_info.get("confidence_gated_alignment_loss", 0.0)))
        sparse_persistence_contrastive_loss_history.append(float(rescue_info.get("sparse_persistence_contrastive_loss", 0.0)))
        confidence_gated_affected_ratio_history.append(float(rescue_info.get("confidence_gated_affected_sample_ratio", 0.0)))
        low_confidence_ratio_history.append(float(rescue_info.get("low_confidence_sample_ratio", 0.0)))
        pair_coverage_ratio_history.append(float(rescue_info.get("effective_pair_coverage_ratio", 0.0)))
        positive_pair_count_history.append(float(rescue_info.get("positive_pair_count", 0.0)))
        hard_negative_count_history.append(float(rescue_info.get("hard_negative_count", 0.0)))
        actual_gate_positive_ratio_history.append(float(rescue_info.get("actual_gate_positive_ratio", 0.0)))
        activated_query_count_history.append(float(rescue_info.get("activated_query_count", 0.0)))
        activated_query_ratio_history.append(float(rescue_info.get("activated_query_ratio", 0.0)))
        per_batch_sparsity_mean_history.append(float(rescue_info.get("per_batch_sparsity_mean", 0.0)))
        per_batch_sparsity_std_history.append(float(rescue_info.get("per_batch_sparsity_std", 0.0)))
        raw_quantile_ratio_history.append(float(rescue_info.get("raw_quantile_ratio", 0.0)))
        capped_ratio_history.append(float(rescue_info.get("capped_ratio", 0.0)))
        valuable_pair_ratio_history.append(float(rescue_info.get("valuable_pair_ratio", 0.0)))
        fallback_trigger_rate_history.append(float(rescue_info.get("fallback_trigger_rate", 0.0)))
        guaranteed_pair_count_history.append(float(rescue_info.get("guaranteed_pair_count", 0.0)))
        strict_pair_ratio_history.append(float(rescue_info.get("strict_pair_ratio", 0.0)))
        fallback_pair_ratio_history.append(float(rescue_info.get("fallback_pair_ratio", 0.0)))
        final_effective_aux_weight_history.append(float(rescue_info.get("final_effective_aux_weight", 0.0)))
        aux_schedule_scale_history.append(float(aux_schedule_scale))
        sparse_gate_selected_ratio_history.append(float(rescue_info.get("sparse_gate_selected_ratio", 0.0)))
        high_value_pair_ratio_history.append(float(rescue_info.get("high_value_pair_ratio", 0.0)))
        high_value_pair_count_history.append(float(rescue_info.get("high_value_pair_count", 0.0)))
        persistence_candidate_pair_count_history.append(float(rescue_info.get("persistence_candidate_pair_count", 0.0)))
        rescue_cache_hit_history.append(float(rescue_info.get("semantic_bootstrap_cache_hit_ratio", 0.0)))
        semantic_hard_weight_mean_history.append(float((semantic_hard_weights * tusb_v3p1_curriculum_weights).detach().mean().cpu().item()))
        gate_history.append(float(tf_out["gate_mean"]))
        future_semantic_state_loss_history.append(float(future_semantic_state_info.get("future_semantic_state_loss", 0.0)))
        future_visibility_loss_history.append(float(future_semantic_state_info.get("future_visibility_loss", 0.0)))
        future_reappearance_loss_history.append(float(future_semantic_state_info.get("future_reappearance_loss", 0.0)))
        future_reappearance_event_loss_history.append(float(future_semantic_state_info.get("future_reappearance_event_loss", 0.0)))
        future_semantic_embedding_loss_history.append(float(future_semantic_state_info.get("future_semantic_embedding_loss", 0.0)))
        future_identity_belief_loss_history.append(float(future_semantic_state_info.get("future_identity_belief_loss", 0.0)))
        future_uncertainty_loss_history.append(float(future_semantic_state_info.get("future_uncertainty_loss", 0.0)))
        future_hypothesis_loss_history.append(float(future_semantic_state_info.get("future_hypothesis_loss", 0.0)))
        future_semantic_state_valid_history.append(1.0 if bool(future_semantic_state_info.get("future_semantic_state_output_valid", False)) else 0.0)
        future_visibility_supervised_ratio_history.append(float(future_semantic_state_info.get("future_visibility_supervised_ratio", 0.0)))
        future_reappearance_supervised_ratio_history.append(float(future_semantic_state_info.get("future_reappearance_supervised_ratio", 0.0)))
        future_visibility_positive_rate_history.append(float(future_semantic_state_info.get("future_visibility_positive_rate", 0.0)))
        future_reappearance_positive_rate_history.append(float(future_semantic_state_info.get("future_reappearance_positive_rate", 0.0)))
        future_reappearance_event_positive_rate_history.append(float(future_semantic_state_info.get("future_reappearance_event_positive_rate", 0.0)))
        future_reappearance_risk_entry_ratio_history.append(float(future_semantic_state_info.get("future_reappearance_risk_entry_ratio", 0.0)))
        future_reappearance_risk_slot_ratio_history.append(float(future_semantic_state_info.get("future_reappearance_risk_slot_ratio", 0.0)))
        if future_semantic_state_info.get("future_reappearance_pos_weight") is not None:
            future_reappearance_pos_weight_history.append(float(future_semantic_state_info.get("future_reappearance_pos_weight", 0.0)))
        if future_semantic_state_info.get("future_reappearance_event_pos_weight") is not None:
            future_reappearance_event_pos_weight_history.append(float(future_semantic_state_info.get("future_reappearance_event_pos_weight", 0.0)))
        future_reappearance_head_available_history.append(
            1.0 if bool(future_semantic_state_info.get("future_reappearance_head_available", False)) else 0.0
        )
        future_reappearance_event_head_available_history.append(
            1.0 if bool(future_semantic_state_info.get("future_reappearance_event_head_available", False)) else 0.0
        )
        future_reappearance_loss_uses_independent_logit_history.append(
            1.0 if bool(future_semantic_state_info.get("future_reappearance_loss_uses_independent_logit", False)) else 0.0
        )
        future_reappearance_event_loss_uses_independent_logit_history.append(
            1.0 if bool(future_semantic_state_info.get("future_reappearance_event_loss_uses_independent_logit", False)) else 0.0
        )
        future_visibility_target_source_history.append(str(future_semantic_state_info.get("future_visibility_target_source", "")))
        future_visibility_target_quality_history.append(str(future_semantic_state_info.get("future_visibility_target_quality", "")))
        trace_unit_loss_history.append(float(trace_unit_info.get("trace_unit_loss", 0.0)))
        trace_unit_assignment_entropy_history.append(float(trace_unit_info.get("assignment_entropy_mean", 0.0)))
        trace_unit_top2_ratio_history.append(float(trace_unit_info.get("actual_top2_assignment_ratio", 0.0)))
        trace_unit_active_unit_count_history.append(float(trace_unit_info.get("active_unit_count_mean", 0.0)))
        trace_unit_z_dyn_drift_history.append(float(trace_unit_info.get("z_dyn_drift_mean", 0.0)))
        trace_unit_z_sem_drift_history.append(float(trace_unit_info.get("z_sem_drift_mean", 0.0)))
        trace_unit_z_sem_to_dyn_ratio_history.append(float(trace_unit_info.get("z_sem_to_z_dyn_drift_ratio", 0.0)))
        trace_unit_same_instance_consistency_history.append(float(trace_unit_info.get("same_instance_within_unit_consistency", 0.0)))
        trace_unit_same_instance_match_rate_history.append(float(trace_unit_info.get("same_instance_dominant_unit_match_rate", 0.0)))
        trace_unit_same_instance_assignment_cosine_history.append(float(trace_unit_info.get("same_instance_assignment_cosine", 0.0)))
        trace_unit_diff_instance_separation_history.append(float(trace_unit_info.get("different_instance_between_unit_separation", 0.0)))
        trace_unit_diff_instance_collision_history.append(float(trace_unit_info.get("different_instance_dominant_unit_collision_rate", 0.0)))
        trace_unit_ambiguity_collision_history.append(float(trace_unit_info.get("ambiguity_highrisk_pair_collision_rate", 0.0)))
        trace_unit_confuser_collision_history.append(float(trace_unit_info.get("confuser_highrisk_pair_collision_rate", 0.0)))
        trace_unit_appearance_highrisk_match_history.append(float(trace_unit_info.get("appearance_drift_highrisk_same_instance_match_rate", 0.0)))
        trace_unit_appearance_high_ratio_history.append(float(trace_unit_info.get("appearance_drift_high_ratio", 0.0)))
        trace_unit_batch_appearance_high_ratio_history.append(float(trace_unit_info.get("batch_appearance_drift_high_ratio", 0.0)))
        trace_unit_step_appearance_high_count_history.append(float(trace_unit_info.get("step_appearance_drift_high_count", 0.0)))
        trace_unit_appearance_signal_valid_count_history.append(float(trace_unit_info.get("appearance_signal_valid_count", 0.0)))
        trace_unit_appearance_refine_nonzero_history.append(float(trace_unit_info.get("appearance_refine_loss_nonzero", 0.0)))
        trace_unit_purity_history.append(float(trace_unit_info.get("unit_purity_by_instance_id", 0.0)))
        trace_unit_track_stability_history.append(float(trace_unit_info.get("unit_track_stability_over_time", 0.0)))
        trace_unit_target_entity_consistency_history.append(float(trace_unit_info.get("target_entity_to_dominant_unit_consistency", 0.0)))
        trace_unit_semantic_stability_history.append(float(trace_unit_info.get("unit_semantic_stability_over_time", 0.0)))
        trace_unit_broadcast_norm_history.append(float(trace_unit_info.get("broadcast_residual_norm_mean", 0.0)))
        semantic_nonempty_count += 1 if bool(tf_out["semantic_input_nonempty"]) else 0

        if global_step % progress_heartbeat_every == 0:
            _write_progress_snapshot(
                progress_json,
                _build_progress_payload(
                    args=args,
                    status="running",
                    global_step=int(global_step),
                    target_steps=int(target_steps),
                    train_summary=train_summary,
                    val_summary=val_summary,
                    run_metadata=run_metadata,
                    runtime_meta=runtime_meta,
                    checkpoint_dir=output_dir,
                    best_ckpt=best_ckpt,
                    latest_ckpt=latest_ckpt,
                    eval_history=eval_history,
                    best_metric_so_far=best_metric_so_far,
                ),
            )

        should_eval = bool(eval_interval > 0 and global_step % eval_interval == 0)
        if global_step == target_steps:
            should_eval = True

        if should_eval:
            metrics = _evaluate(
                stage1_model=stage1_model,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                future_semantic_state_head=future_semantic_state_head,
                structure_mode=str(structure_mode),
                trace_unit_tokenizer=trace_unit_tokenizer,
                trace_unit_factorized_state=trace_unit_factorized_state,
                trace_unit_handshake=trace_unit_handshake,
                trace_unit_broadcast=trace_unit_broadcast,
                trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
                loader=val_loader,
                device=device,
                pin_memory=bool(pin_memory),
                obs_len=int(args.obs_len),
                fut_len=int(args.fut_len),
                max_batches=int(args.eval_max_batches),
                semantic_source_mainline=str(args.semantic_source_mainline),
                semantic_state_feedback_adapter=semantic_state_feedback_adapter,
                semantic_state_feedback_enabled=bool(args.enable_semantic_state_feedback),
                semantic_state_feedback_alpha=float(args.semantic_state_feedback_alpha),
                semantic_state_feedback_stopgrad_state=bool(args.semantic_state_feedback_stopgrad_state),
                semantic_state_feedback_mode=str(args.semantic_state_feedback_mode),
            )
            semantic_hard_payload: Dict[str, Any] | None = None
            if semantic_hard_loader is not None:
                hard_metrics = _evaluate(
                    stage1_model=stage1_model,
                    semantic_encoder=semantic_encoder,
                    semantic_fusion=semantic_fusion,
                    readout_head=readout_head,
                    future_semantic_state_head=future_semantic_state_head,
                    structure_mode=str(structure_mode),
                    trace_unit_tokenizer=trace_unit_tokenizer,
                    trace_unit_factorized_state=trace_unit_factorized_state,
                    trace_unit_handshake=trace_unit_handshake,
                    trace_unit_broadcast=trace_unit_broadcast,
                    trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
                    loader=semantic_hard_loader,
                    device=device,
                    pin_memory=bool(pin_memory),
                    obs_len=int(args.obs_len),
                    fut_len=int(args.fut_len),
                    max_batches=-1,
                    semantic_source_mainline=str(args.semantic_source_mainline),
                    semantic_state_feedback_adapter=semantic_state_feedback_adapter,
                    semantic_state_feedback_enabled=bool(args.enable_semantic_state_feedback),
                    semantic_state_feedback_alpha=float(args.semantic_state_feedback_alpha),
                    semantic_state_feedback_stopgrad_state=bool(args.semantic_state_feedback_stopgrad_state),
                    semantic_state_feedback_mode=str(args.semantic_state_feedback_mode),
                )
                semantic_hard_payload = {
                    "metrics": hard_metrics,
                    "semantic_hard_sidecar_score": float(_semantic_hard_composite_score(hard_metrics)),
                }
            rk = _rank_key(metrics)
            event = {
                "global_step": int(global_step),
                "metrics": metrics,
                "rank_key": [float(rk[0]), float(rk[1]), float(rk[2])],
            }
            if semantic_hard_payload is not None:
                event["semantic_hard_sidecar"] = semantic_hard_payload
            eval_history.append(event)

            if best_metric_so_far is None or tuple(event["rank_key"]) < tuple(best_metric_so_far.get("rank_key", [1e9, 1e9, 1e9])):
                best_metric_so_far = {
                    "global_step": int(global_step),
                    "metrics": metrics,
                    "rank_key": [float(rk[0]), float(rk[1]), float(rk[2])],
                }
                best_payload = _checkpoint_payload(
                    args=args,
                    global_step=int(global_step),
                    best_metric_so_far=best_metric_so_far,
                    eval_history=eval_history,
                    semantic_encoder=semantic_encoder,
                    semantic_fusion=semantic_fusion,
                    readout_head=readout_head,
                    future_semantic_state_head=future_semantic_state_head,
                    semantic_state_feedback_adapter=semantic_state_feedback_adapter,
                    semantic_rescue_heads=semantic_rescue_heads,
                    trace_unit_tokenizer=trace_unit_tokenizer,
                    trace_unit_factorized_state=trace_unit_factorized_state,
                    trace_unit_handshake=trace_unit_handshake,
                    trace_unit_broadcast=trace_unit_broadcast,
                    optimizer=optimizer,
                    run_metadata=run_metadata,
                )
                _atomic_torch_save(best_payload, best_ckpt)
                new_best_written_this_run = True

            if semantic_hard_payload is not None:
                sidecar_score = float(semantic_hard_payload["semantic_hard_sidecar_score"])
                prev_score = float((best_semantic_hard_metric or {}).get("semantic_hard_sidecar_score", 1e9))
                if best_semantic_hard_metric is None or sidecar_score < prev_score:
                    best_semantic_hard_metric = {
                        "global_step": int(global_step),
                        "semantic_hard_sidecar_score": float(sidecar_score),
                        "metrics": semantic_hard_payload["metrics"],
                    }
                    sidecar_payload = _checkpoint_payload(
                        args=args,
                        global_step=int(global_step),
                        best_metric_so_far=best_metric_so_far,
                        eval_history=eval_history,
                        semantic_encoder=semantic_encoder,
                        semantic_fusion=semantic_fusion,
                        readout_head=readout_head,
                        future_semantic_state_head=future_semantic_state_head,
                        semantic_state_feedback_adapter=semantic_state_feedback_adapter,
                        semantic_rescue_heads=semantic_rescue_heads,
                        trace_unit_tokenizer=trace_unit_tokenizer,
                        trace_unit_factorized_state=trace_unit_factorized_state,
                        trace_unit_handshake=trace_unit_handshake,
                        trace_unit_broadcast=trace_unit_broadcast,
                        optimizer=optimizer,
                        run_metadata=run_metadata,
                    )
                    _atomic_torch_save(sidecar_payload, best_semantic_hard_ckpt)

            _write_progress_snapshot(
                progress_json,
                _build_progress_payload(
                    args=args,
                    status="running",
                    global_step=int(global_step),
                    target_steps=int(target_steps),
                    train_summary=train_summary,
                    val_summary=val_summary,
                    run_metadata=run_metadata,
                    runtime_meta=runtime_meta,
                    checkpoint_dir=output_dir,
                    best_ckpt=best_ckpt,
                    latest_ckpt=latest_ckpt,
                    eval_history=eval_history,
                    best_metric_so_far=best_metric_so_far,
                ),
            )

        should_save_step = bool(global_step % save_every == 0)
        should_save_latest = bool(should_save_step or global_step == target_steps)
        if should_save_latest:
            _save_latest_and_optional_step(step_now=int(global_step), save_step=bool(should_save_step))

    if best_metric_so_far is None:
        metrics = _evaluate(
            stage1_model=stage1_model,
            semantic_encoder=semantic_encoder,
            semantic_fusion=semantic_fusion,
            readout_head=readout_head,
            future_semantic_state_head=future_semantic_state_head,
            structure_mode=str(structure_mode),
            trace_unit_tokenizer=trace_unit_tokenizer,
            trace_unit_factorized_state=trace_unit_factorized_state,
            trace_unit_handshake=trace_unit_handshake,
            trace_unit_broadcast=trace_unit_broadcast,
            trace_unit_disable_instance_path=bool(args.trace_unit_disable_instance_path),
            loader=val_loader,
            device=device,
            pin_memory=bool(pin_memory),
            obs_len=int(args.obs_len),
            fut_len=int(args.fut_len),
            max_batches=int(args.eval_max_batches),
            semantic_source_mainline=str(args.semantic_source_mainline),
        )
        rk = _rank_key(metrics)
        best_metric_so_far = {
            "global_step": int(global_step),
            "metrics": metrics,
            "rank_key": [float(rk[0]), float(rk[1]), float(rk[2])],
        }

    if not best_ckpt.exists():
        inherited_best_step = int((best_metric_so_far or {}).get("global_step", -1))
        if (
            resolved_resume
            and (not new_best_written_this_run)
            and inherited_best_step <= int(resume_global_step_loaded)
            and Path(str(resolved_resume)).resolve() != best_ckpt.resolve()
        ):
            shutil.copy2(str(resolved_resume), str(best_ckpt))
        else:
            best_payload = _checkpoint_payload(
                args=args,
                global_step=int(global_step),
                best_metric_so_far=best_metric_so_far,
                eval_history=eval_history,
                semantic_encoder=semantic_encoder,
                semantic_fusion=semantic_fusion,
                readout_head=readout_head,
                future_semantic_state_head=future_semantic_state_head,
                semantic_state_feedback_adapter=semantic_state_feedback_adapter,
                semantic_rescue_heads=semantic_rescue_heads,
                trace_unit_tokenizer=trace_unit_tokenizer,
                trace_unit_factorized_state=trace_unit_factorized_state,
                trace_unit_handshake=trace_unit_handshake,
                trace_unit_broadcast=trace_unit_broadcast,
                optimizer=optimizer,
                run_metadata=run_metadata,
            )
            _atomic_torch_save(best_payload, best_ckpt)
    if not latest_ckpt.exists():
        _save_latest_and_optional_step(step_now=int(global_step), save_step=False)

    final_metrics = dict(best_metric_so_far.get("metrics", {})) if isinstance(best_metric_so_far, dict) else {}
    final_metric_triplet = _metric_triplet(final_metrics)
    frozen_count = int(stage1_meta.get("parameter_count", 0))
    trainable_count = int(total_trainable_parameter_count)
    stage1_trainable_count = int(stage1_meta.get("trainable_parameter_count", 0))

    best_checkpoint_metric = _metric_triplet(
        (best_metric_so_far.get("metrics", {}) if isinstance(best_metric_so_far, dict) and isinstance(best_metric_so_far.get("metrics", {}), dict) else final_metrics)
    )
    latest_event = eval_history[-1] if eval_history and isinstance(eval_history[-1], dict) else {}
    latest_metrics = latest_event.get("metrics", {}) if isinstance(latest_event.get("metrics", {}), dict) else final_metrics
    latest_checkpoint_metric = _metric_triplet(latest_metrics)
    best_semantic_hard_triplet = _metric_triplet(
        (best_semantic_hard_metric.get("metrics", {}) if isinstance(best_semantic_hard_metric, dict) and isinstance(best_semantic_hard_metric.get("metrics", {}), dict) else {})
    )

    train_split_counts_used = _split_counts_used(train_summary)
    val_split_counts_used = _split_counts_used(val_summary)

    boundary_ok = bool((stage1_trainable_count == 0 and (not stage1_grad_detected_any)) or (stage1_trainable_count > 0 and stage1_grad_detected_any))
    run_stable = bool(
        np.isfinite(float(final_metrics.get("teacher_forced_coord_loss", np.inf)))
        and np.isfinite(float(final_metrics.get("free_rollout_coord_mean_l2", np.inf)))
        and np.isfinite(float(final_metrics.get("free_rollout_endpoint_l2", np.inf)))
    )

    train_total_count = int(sum(train_split_counts_used.values()))
    effective_batch = int(args.batch_size)
    epochs_completed = 0.0
    if train_total_count > 0:
        epochs_completed = float(optimizer_steps_this_run * effective_batch) / float(train_total_count)
    finite_loss_values = [
        *teacher_loss_history,
        *future_semantic_state_loss_history,
        *future_visibility_loss_history,
        *future_reappearance_loss_history,
        *future_reappearance_event_loss_history,
        *future_semantic_embedding_loss_history,
        *future_identity_belief_loss_history,
        *future_uncertainty_loss_history,
    ]
    loss_finite_ratio = (
        float(sum(1 for value in finite_loss_values if np.isfinite(float(value))) / max(len(finite_loss_values), 1))
        if finite_loss_values
        else 1.0
    )
    future_semantic_state_output_valid_ratio = float(sum(future_semantic_state_valid_history) / max(len(future_semantic_state_valid_history), 1))
    future_visibility_loss_mean = float(sum(future_visibility_loss_history) / max(len(future_visibility_loss_history), 1))
    future_reappearance_loss_mean = float(sum(future_reappearance_loss_history) / max(len(future_reappearance_loss_history), 1))
    future_reappearance_event_loss_mean = float(sum(future_reappearance_event_loss_history) / max(len(future_reappearance_event_loss_history), 1))
    future_reappearance_positive_rate_mean = float(sum(future_reappearance_positive_rate_history) / max(len(future_reappearance_positive_rate_history), 1))
    future_reappearance_event_positive_rate_mean = float(sum(future_reappearance_event_positive_rate_history) / max(len(future_reappearance_event_positive_rate_history), 1))
    future_reappearance_risk_entry_ratio_mean = float(sum(future_reappearance_risk_entry_ratio_history) / max(len(future_reappearance_risk_entry_ratio_history), 1))
    future_reappearance_pos_weight_mean = float(sum(future_reappearance_pos_weight_history) / max(len(future_reappearance_pos_weight_history), 1))
    future_reappearance_event_pos_weight_mean = float(sum(future_reappearance_event_pos_weight_history) / max(len(future_reappearance_event_pos_weight_history), 1))

    payload = {
        "generated_at_utc": now_iso(),
        "run_name": str(args.run_name),
        "train_steps": int(optimizer_steps_this_run),
        "loss_finite_ratio": float(loss_finite_ratio),
        "output_valid_ratio": float(future_semantic_state_output_valid_ratio),
        "future_visibility_loss_mean": future_visibility_loss_mean,
        "future_reappearance_loss_mean": future_reappearance_loss_mean,
        "future_reappearance_event_loss_mean": future_reappearance_event_loss_mean,
        "future_reappearance_positive_rate_mean": future_reappearance_positive_rate_mean,
        "future_reappearance_event_positive_rate_mean": future_reappearance_event_positive_rate_mean,
        "future_reappearance_risk_entry_ratio_mean": future_reappearance_risk_entry_ratio_mean,
        "future_reappearance_pos_weight_mean": future_reappearance_pos_weight_mean,
        "future_reappearance_event_pos_weight_mean": future_reappearance_event_pos_weight_mean,
        "objective": "Stage2 training run on frozen Stage1 220m backbone",
        "stage2_structure_mode": str(structure_mode),
        "current_mainline_semantic_source": str(args.semantic_source_mainline),
        "legacy_semantic_source": str(args.legacy_semantic_source),
        "stage2_contract_path": str(args.stage2_contract_path),
        "stage2_data_binding": {
            "core": core_names,
            "optional_extension": binding.get("optional_extension", []),
            "excluded": binding.get("excluded", []),
            "run_datasets": [str(x) for x in args.dataset_names],
        },
        "datasets_bound_for_train": [str(x) for x in args.dataset_names],
        "datasets_bound_for_eval": [str(x) for x in args.dataset_names],
        "runtime": runtime_meta,
        "run_metadata": run_metadata,
        "training_budget": {
            "train_steps_target": int(target_steps),
            "train_steps_completed": int(global_step),
            "optimizer_steps_this_invocation": int(optimizer_steps_this_run),
            "batch_size": int(args.batch_size),
            "eval_interval": int(eval_interval),
            "eval_max_batches": int(args.eval_max_batches),
            "save_every_n_steps": int(save_every),
        },
        "dataset_summary": {
            "train": train_summary,
            "val": val_summary,
        },
        "whether_full_train_used": bool(_full_usage_flag(args.max_samples_train)),
        "whether_full_val_used": bool(_full_usage_flag(args.max_samples_val)),
        "effective_train_sample_count_per_dataset": train_split_counts_used,
        "effective_val_sample_count_per_dataset": val_split_counts_used,
        "core_dataset_inputs": {
            "ready": bool(core_ready),
            "details": core_details,
        },
        "stage1_backbone": {
            "load_success": True,
            **stage1_meta,
        },
        "parameter_count_frozen": int(frozen_count),
        "parameter_count_trainable": int(trainable_count),
        "freeze_trainable_boundary": {
            "stage1_trainable_parameter_count": int(stage1_trainable_count),
            "semantic_trainable_parameter_count": int(trainable_count),
            "stage1_grad_detected_after_backward": bool(stage1_grad_detected_any),
            "semantic_grad_norm_latest": float(semantic_grad_norm_latest),
            "boundary_ok": bool(boundary_ok),
        },
        "future_semantic_head_only_warmup_audit": future_semantic_head_only_warmup_audit,
        "future_semantic_controlled_joint_audit": future_semantic_controlled_joint_audit,
        "semantic_branch_metrics": {
            "train_gate_mean": float(sum(gate_history) / max(len(gate_history), 1)),
            "train_semantic_input_nonempty_ratio": float(semantic_nonempty_count / max(len(gate_history), 1)),
            "semantic_crop_size": int(args.semantic_crop_size),
            "current_mainline_semantic_source": str(args.semantic_source_mainline),
            "legacy_semantic_source": str(args.legacy_semantic_source),
            "legacy_semantic_feature_dim": 10,
            "semantic_rescue_mode": str(rescue_mode),
            "semantic_rescue_weight": float(rescue_weight),
            "semantic_bootstrap_target_dim": int(args.semantic_bootstrap_target_dim),
            "semantic_alignment_loss_weight": float(args.semantic_alignment_loss_weight),
            "query_persistence_consistency_loss_weight": float(args.query_persistence_consistency_loss_weight),
            "semantic_hard_curriculum_weight": float(args.semantic_hard_curriculum_weight),
            "readout_semantic_alignment_loss_weight": float(args.readout_semantic_alignment_loss_weight),
            "persistence_contrastive_ranking_loss_weight": float(args.persistence_contrastive_ranking_loss_weight),
            "semantic_aux_subset_weighting_strength": float(args.semantic_aux_subset_weighting_strength),
            "confidence_gated_alignment_loss_weight": float(args.confidence_gated_alignment_loss_weight),
            "sparse_persistence_contrastive_loss_weight": float(args.sparse_persistence_contrastive_loss_weight),
            "confidence_metric_definition": "margin between positive CLIP-target cosine and hardest in-batch negative cosine on readout projection",
            "confidence_gating_margin_threshold": float(args.confidence_gating_margin_threshold),
            "confidence_gating_temperature": float(args.confidence_gating_temperature),
            "semantic_hard_score_threshold": float(args.semantic_hard_score_threshold),
            "aux_loss_delay_steps": int(args.aux_loss_delay_steps),
            "aux_loss_ramp_steps": int(args.aux_loss_ramp_steps),
            "v4_sparse_gating_family": str(args.v4_sparse_gating_family),
            "v4_gating_quantile": float(args.v4_gating_quantile),
            "v4_topk_token_ratio": float(args.v4_topk_token_ratio),
            "v4_topk_min_tokens": int(args.v4_topk_min_tokens),
            "v4_gate_min_strength": float(args.v4_gate_min_strength),
            "v4_persistence_value_quantile": float(args.v4_persistence_value_quantile),
            "v4_persistence_max_pairs": int(args.v4_persistence_max_pairs),
            "v4_persistence_margin": float(args.v4_persistence_margin),
            "v5_gating_family": str(args.v5_gating_family),
            "v5_topk_query_k": int(args.v5_topk_query_k),
            "v5_capped_quantile": float(args.v5_capped_quantile),
            "v5_max_affected_ratio": float(args.v5_max_affected_ratio),
            "v5_gate_min_strength": float(args.v5_gate_min_strength),
            "v5_max_pairs_per_sample": int(args.v5_max_pairs_per_sample),
            "v5_hard_negative_cap": int(args.v5_hard_negative_cap),
            "v5_pair_sampling_temperature": float(args.v5_pair_sampling_temperature),
            "v6_gating_family": str(args.v6_gating_family),
            "v6_topk_query_k": int(args.v6_topk_query_k),
            "v6_capped_quantile": float(args.v6_capped_quantile),
            "v6_max_affected_ratio": float(args.v6_max_affected_ratio),
            "v6_gate_min_strength": float(args.v6_gate_min_strength),
            "v6_strict_max_pairs_per_sample": int(args.v6_strict_max_pairs_per_sample),
            "v6_hard_negative_cap": int(args.v6_hard_negative_cap),
            "v6_pair_sampling_temperature": float(args.v6_pair_sampling_temperature),
            "v6_guaranteed_min_pairs_per_sample": int(args.v6_guaranteed_min_pairs_per_sample),
            "v6_two_level_pair_mining_enabled": bool(args.v6_two_level_pair_mining_enabled),
            "v6_relaxed_motion_threshold": float(args.v6_relaxed_motion_threshold),
            "v6_relaxed_area_jump_threshold": float(args.v6_relaxed_area_jump_threshold),
            "v6_relaxed_small_query_threshold": float(args.v6_relaxed_small_query_threshold),
            "v6_relaxed_appearance_shift_threshold": float(args.v6_relaxed_appearance_shift_threshold),
            "v6_relaxed_center_interaction_threshold": float(args.v6_relaxed_center_interaction_threshold),
            "whether_main_rollout_loss_reweighted": bool((not str(rescue_mode).startswith("v2")) and float(args.semantic_hard_curriculum_weight) > 0.0),
            "semantic_rescue_loss_mean": float(sum(rescue_loss_history) / max(len(rescue_loss_history), 1)),
            "semantic_alignment_loss_mean": float(sum(semantic_alignment_loss_history) / max(len(semantic_alignment_loss_history), 1)),
            "query_persistence_consistency_loss_mean": float(sum(query_persistence_loss_history) / max(len(query_persistence_loss_history), 1)),
            "readout_semantic_alignment_loss_mean": float(sum(readout_alignment_loss_history) / max(len(readout_alignment_loss_history), 1)),
            "persistence_contrastive_or_ranking_loss_mean": float(sum(persistence_contrastive_loss_history) / max(len(persistence_contrastive_loss_history), 1)),
            "confidence_gated_alignment_loss_mean": float(sum(confidence_gated_alignment_loss_history) / max(len(confidence_gated_alignment_loss_history), 1)),
            "sparse_persistence_contrastive_loss_mean": float(sum(sparse_persistence_contrastive_loss_history) / max(len(sparse_persistence_contrastive_loss_history), 1)),
            "confidence_gated_affected_sample_ratio_mean": float(sum(confidence_gated_affected_ratio_history) / max(len(confidence_gated_affected_ratio_history), 1)),
            "low_confidence_sample_ratio_mean": float(sum(low_confidence_ratio_history) / max(len(low_confidence_ratio_history), 1)),
            "effective_pair_coverage_ratio_mean": float(sum(pair_coverage_ratio_history) / max(len(pair_coverage_ratio_history), 1)),
            "positive_pair_count_mean": float(sum(positive_pair_count_history) / max(len(positive_pair_count_history), 1)),
            "hard_negative_count_mean": float(sum(hard_negative_count_history) / max(len(hard_negative_count_history), 1)),
            "actual_gate_positive_ratio_mean": float(sum(actual_gate_positive_ratio_history) / max(len(actual_gate_positive_ratio_history), 1)),
            "activated_query_count_mean": float(sum(activated_query_count_history) / max(len(activated_query_count_history), 1)),
            "activated_query_ratio_mean": float(sum(activated_query_ratio_history) / max(len(activated_query_ratio_history), 1)),
            "per_batch_sparsity_mean": float(sum(per_batch_sparsity_mean_history) / max(len(per_batch_sparsity_mean_history), 1)),
            "per_batch_sparsity_std_mean": float(sum(per_batch_sparsity_std_history) / max(len(per_batch_sparsity_std_history), 1)),
            "raw_quantile_ratio_mean": float(sum(raw_quantile_ratio_history) / max(len(raw_quantile_ratio_history), 1)),
            "capped_ratio_mean": float(sum(capped_ratio_history) / max(len(capped_ratio_history), 1)),
            "valuable_pair_ratio_mean": float(sum(valuable_pair_ratio_history) / max(len(valuable_pair_ratio_history), 1)),
            "fallback_trigger_rate_mean": float(sum(fallback_trigger_rate_history) / max(len(fallback_trigger_rate_history), 1)),
            "guaranteed_pair_count_mean": float(sum(guaranteed_pair_count_history) / max(len(guaranteed_pair_count_history), 1)),
            "strict_pair_ratio_mean": float(sum(strict_pair_ratio_history) / max(len(strict_pair_ratio_history), 1)),
            "fallback_pair_ratio_mean": float(sum(fallback_pair_ratio_history) / max(len(fallback_pair_ratio_history), 1)),
            "sparse_gate_selected_ratio_mean": float(sum(sparse_gate_selected_ratio_history) / max(len(sparse_gate_selected_ratio_history), 1)),
            "high_value_pair_ratio_mean": float(sum(high_value_pair_ratio_history) / max(len(high_value_pair_ratio_history), 1)),
            "high_value_pair_count_mean": float(sum(high_value_pair_count_history) / max(len(high_value_pair_count_history), 1)),
            "persistence_candidate_pair_count_mean": float(sum(persistence_candidate_pair_count_history) / max(len(persistence_candidate_pair_count_history), 1)),
            "aux_schedule_scale_mean": float(sum(aux_schedule_scale_history) / max(len(aux_schedule_scale_history), 1)),
            "final_effective_aux_weight_mean": float(float(rescue_weight) * (sum(final_effective_aux_weight_history) / max(len(final_effective_aux_weight_history), 1))),
            "final_effective_aux_weight": float(float(rescue_weight) * (final_effective_aux_weight_history[-1] if final_effective_aux_weight_history else 0.0)),
            "semantic_bootstrap_cache_hit_ratio_mean": float(sum(rescue_cache_hit_history) / max(len(rescue_cache_hit_history), 1)),
            "semantic_hard_sample_weight_mean": float(sum(semantic_hard_weight_mean_history) / max(len(semantic_hard_weight_mean_history), 1)),
            "semantic_bootstrap_cache_item_count": int(len(bootstrap_cache)),
        },
        "future_semantic_trace_state_metrics": {
            "enabled": bool(args.enable_future_semantic_state_head),
            "future_semantic_embedding_dim": int(args.future_semantic_embedding_dim),
            "future_hypothesis_count": int(args.future_hypothesis_count),
            "enable_future_extent_head": bool(args.enable_future_extent_head),
            "enable_future_multihypothesis_head": bool(args.enable_future_multihypothesis_head),
            "loss_weights_default_zero_preserve_official": bool(
                float(args.future_semantic_loss_weight) == 0.0
                and float(args.future_visibility_loss_weight) == 0.0
                and float(args.future_reappearance_loss_weight) == 0.0
                and float(args.future_identity_belief_loss_weight) == 0.0
                and float(args.future_uncertainty_loss_weight) == 0.0
                and float(args.future_hypothesis_loss_weight) == 0.0
            ),
            "future_trace_coord_loss_mean": 0.0,
            "future_visibility_loss_mean": future_visibility_loss_mean,
            "future_reappearance_loss_mean": future_reappearance_loss_mean,
            "future_reappearance_event_loss_mean": future_reappearance_event_loss_mean,
            "future_semantic_embedding_loss_mean": float(sum(future_semantic_embedding_loss_history) / max(len(future_semantic_embedding_loss_history), 1)),
            "future_identity_belief_loss_mean": float(sum(future_identity_belief_loss_history) / max(len(future_identity_belief_loss_history), 1)),
            "future_uncertainty_loss_mean": float(sum(future_uncertainty_loss_history) / max(len(future_uncertainty_loss_history), 1)),
            "future_hypothesis_loss_mean": float(sum(future_hypothesis_loss_history) / max(len(future_hypothesis_loss_history), 1)),
            "future_semantic_state_loss_mean": float(sum(future_semantic_state_loss_history) / max(len(future_semantic_state_loss_history), 1)),
            "future_semantic_state_output_valid_ratio": future_semantic_state_output_valid_ratio,
            "future_visibility_target_source": next((x for x in reversed(future_visibility_target_source_history) if x), "unavailable"),
            "future_visibility_target_quality": next((x for x in reversed(future_visibility_target_quality_history) if x), "weak_unavailable"),
            "future_visibility_supervised_ratio_mean": float(sum(future_visibility_supervised_ratio_history) / max(len(future_visibility_supervised_ratio_history), 1)),
            "future_reappearance_supervised_ratio_mean": float(sum(future_reappearance_supervised_ratio_history) / max(len(future_reappearance_supervised_ratio_history), 1)),
            "future_visibility_positive_rate_mean": float(sum(future_visibility_positive_rate_history) / max(len(future_visibility_positive_rate_history), 1)),
            "future_reappearance_positive_rate_mean": future_reappearance_positive_rate_mean,
            "future_reappearance_event_positive_rate_mean": future_reappearance_event_positive_rate_mean,
            "future_reappearance_risk_entry_ratio_mean": future_reappearance_risk_entry_ratio_mean,
            "future_reappearance_risk_slot_ratio_mean": float(sum(future_reappearance_risk_slot_ratio_history) / max(len(future_reappearance_risk_slot_ratio_history), 1)),
            "future_reappearance_mask_policy": str(args.future_reappearance_mask_policy),
            "future_reappearance_head_available": bool(
                sum(future_reappearance_head_available_history) / max(len(future_reappearance_head_available_history), 1) > 0.5
            ),
            "future_reappearance_event_head_available": bool(
                sum(future_reappearance_event_head_available_history) / max(len(future_reappearance_event_head_available_history), 1) > 0.5
            ),
            "future_reappearance_loss_uses_independent_logit": bool(
                sum(future_reappearance_loss_uses_independent_logit_history)
                / max(len(future_reappearance_loss_uses_independent_logit_history), 1)
                > 0.5
            ),
            "future_reappearance_event_loss_uses_independent_logit": bool(
                sum(future_reappearance_event_loss_uses_independent_logit_history)
                / max(len(future_reappearance_event_loss_uses_independent_logit_history), 1)
                > 0.5
            ),
            "future_reappearance_pos_weight_mean": future_reappearance_pos_weight_mean,
            "future_reappearance_event_pos_weight_mean": future_reappearance_event_pos_weight_mean,
            "future_reappearance_pos_weight_setting": str(args.future_reappearance_pos_weight),
            "future_reappearance_pos_weight_max": float(args.future_reappearance_pos_weight_max),
            "reappearance_positive_sampling_plan": reappearance_positive_sampling_plan,
        },
        "trace_unit_metrics": {
            "enabled": bool(structure_mode == "trace_unit_semantic_binding"),
            "trace_unit_count": int(args.trace_unit_count),
            "trace_unit_dim": int(args.trace_unit_dim),
            "trace_unit_slot_iters": int(args.trace_unit_slot_iters),
            "trace_unit_assignment_topk": int(args.trace_unit_assignment_topk),
            "trace_unit_handshake_layers": int(args.trace_unit_handshake_layers),
            "trace_unit_loss_mean": float(sum(trace_unit_loss_history) / max(len(trace_unit_loss_history), 1)),
            "assignment_entropy_mean": float(sum(trace_unit_assignment_entropy_history) / max(len(trace_unit_assignment_entropy_history), 1)),
            "actual_top2_assignment_ratio_mean": float(sum(trace_unit_top2_ratio_history) / max(len(trace_unit_top2_ratio_history), 1)),
            "active_unit_count_mean": float(sum(trace_unit_active_unit_count_history) / max(len(trace_unit_active_unit_count_history), 1)),
            "z_dyn_drift_mean": float(sum(trace_unit_z_dyn_drift_history) / max(len(trace_unit_z_dyn_drift_history), 1)),
            "z_sem_drift_mean": float(sum(trace_unit_z_sem_drift_history) / max(len(trace_unit_z_sem_drift_history), 1)),
            "z_sem_to_z_dyn_drift_ratio_mean": float(sum(trace_unit_z_sem_to_dyn_ratio_history) / max(len(trace_unit_z_sem_to_dyn_ratio_history), 1)),
            "same_instance_within_unit_consistency_mean": float(sum(trace_unit_same_instance_consistency_history) / max(len(trace_unit_same_instance_consistency_history), 1)),
            "same_instance_dominant_unit_match_rate_mean": float(sum(trace_unit_same_instance_match_rate_history) / max(len(trace_unit_same_instance_match_rate_history), 1)),
            "same_instance_assignment_cosine_mean": float(sum(trace_unit_same_instance_assignment_cosine_history) / max(len(trace_unit_same_instance_assignment_cosine_history), 1)),
            "different_instance_between_unit_separation_mean": float(sum(trace_unit_diff_instance_separation_history) / max(len(trace_unit_diff_instance_separation_history), 1)),
            "different_instance_dominant_unit_collision_rate_mean": float(sum(trace_unit_diff_instance_collision_history) / max(len(trace_unit_diff_instance_collision_history), 1)),
            "ambiguity_highrisk_pair_collision_rate_mean": float(sum(trace_unit_ambiguity_collision_history) / max(len(trace_unit_ambiguity_collision_history), 1)),
            "confuser_highrisk_pair_collision_rate_mean": float(sum(trace_unit_confuser_collision_history) / max(len(trace_unit_confuser_collision_history), 1)),
            "appearance_drift_highrisk_same_instance_match_rate_mean": float(sum(trace_unit_appearance_highrisk_match_history) / max(len(trace_unit_appearance_highrisk_match_history), 1)),
            "appearance_drift_high_ratio_mean": float(sum(trace_unit_appearance_high_ratio_history) / max(len(trace_unit_appearance_high_ratio_history), 1)),
            "batch_appearance_drift_high_ratio_mean": float(sum(trace_unit_batch_appearance_high_ratio_history) / max(len(trace_unit_batch_appearance_high_ratio_history), 1)),
            "step_appearance_drift_high_count_mean": float(sum(trace_unit_step_appearance_high_count_history) / max(len(trace_unit_step_appearance_high_count_history), 1)),
            "appearance_signal_valid_count_mean": float(sum(trace_unit_appearance_signal_valid_count_history) / max(len(trace_unit_appearance_signal_valid_count_history), 1)),
            "appearance_refine_loss_nonzero_ratio": float(sum(trace_unit_appearance_refine_nonzero_history) / max(len(trace_unit_appearance_refine_nonzero_history), 1)),
            "unit_purity_by_instance_id_mean": float(sum(trace_unit_purity_history) / max(len(trace_unit_purity_history), 1)),
            "unit_track_stability_over_time_mean": float(sum(trace_unit_track_stability_history) / max(len(trace_unit_track_stability_history), 1)),
            "target_entity_to_dominant_unit_consistency_mean": float(sum(trace_unit_target_entity_consistency_history) / max(len(trace_unit_target_entity_consistency_history), 1)),
            "unit_semantic_stability_over_time_mean": float(sum(trace_unit_semantic_stability_history) / max(len(trace_unit_semantic_stability_history), 1)),
            "broadcast_residual_norm_mean": float(sum(trace_unit_broadcast_norm_history) / max(len(trace_unit_broadcast_norm_history), 1)),
        },
        "instance_aware_density": {
            "true_instance_entity_count_mean": float(sum(true_instance_entity_count_history) / max(len(true_instance_entity_count_history), 1)),
            "pseudo_entity_count_mean": float(sum(pseudo_entity_count_history) / max(len(pseudo_entity_count_history), 1)),
            "fallback_entity_count_mean": float(sum(fallback_entity_count_history) / max(len(fallback_entity_count_history), 1)),
            "true_instance_ratio_per_batch_mean": float(sum(true_instance_ratio_per_batch_history) / max(len(true_instance_ratio_per_batch_history), 1)),
        },
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "teacher_forced_coord_loss",
            "total_loss_usage": "reference_only",
        },
        "comparison_sorting": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "teacher_forced_coord_loss",
            "total_loss_usage": "reference_only",
        },
        "teacher_forced_coord_loss": float(final_metric_triplet["teacher_forced_coord_loss"]),
        "free_rollout_coord_mean_l2": float(final_metric_triplet["free_rollout_coord_mean_l2"]),
        "free_rollout_endpoint_l2": float(final_metric_triplet["free_rollout_endpoint_l2"]),
        "best_checkpoint_metric": {
            "global_step": int((best_metric_so_far or {}).get("global_step", -1)),
            "metrics": best_checkpoint_metric,
            "rank_key": _rank_key(best_checkpoint_metric),
        },
        "latest_checkpoint_metric": {
            "global_step": int(latest_event.get("global_step", -1) or -1),
            "metrics": latest_checkpoint_metric,
            "rank_key": _rank_key(latest_checkpoint_metric),
        },
        "semantic_hard_sidecar_metric": {
            "enabled": bool(args.semantic_hard_sidecar_enabled),
            "checkpoint_path": str(best_semantic_hard_ckpt),
            "exists": bool(best_semantic_hard_ckpt.exists()),
            "global_step": int((best_semantic_hard_metric or {}).get("global_step", -1)),
            "semantic_hard_sidecar_score": float((best_semantic_hard_metric or {}).get("semantic_hard_sidecar_score", 1e9)),
            "metrics": best_semantic_hard_triplet,
            "score_formula": "mean endpoint L2 over semantic-hard subsets: occlusion_reappearance, crossing_or_interaction_ambiguity, small_object_or_low_area, appearance_change_or_semantic_shift",
        },
        "sidecar_checkpoint_selection": {
            "overall_best_checkpoint": str(best_ckpt),
            "semantic_hard_best_checkpoint": str(best_semantic_hard_ckpt),
            "overall_best_global_step": int((best_metric_so_far or {}).get("global_step", -1)),
            "semantic_hard_best_global_step": int((best_semantic_hard_metric or {}).get("global_step", -1)),
            "same_checkpoint_selected": bool(int((best_metric_so_far or {}).get("global_step", -1)) == int((best_semantic_hard_metric or {}).get("global_step", -2))),
            "sidecar_truly_diverged": bool(
                bool(args.semantic_hard_sidecar_enabled)
                and int((best_semantic_hard_metric or {}).get("global_step", -1)) >= 0
                and int((best_metric_so_far or {}).get("global_step", -1)) != int((best_semantic_hard_metric or {}).get("global_step", -2))
            ),
        },
        "overall_vs_semantic_hard_best_delta": {
            "same_step": bool(int((best_metric_so_far or {}).get("global_step", -1)) == int((best_semantic_hard_metric or {}).get("global_step", -2))),
            "endpoint_l2_delta": float(best_semantic_hard_triplet["free_rollout_endpoint_l2"] - best_checkpoint_metric["free_rollout_endpoint_l2"]),
            "coord_mean_l2_delta": float(best_semantic_hard_triplet["free_rollout_coord_mean_l2"] - best_checkpoint_metric["free_rollout_coord_mean_l2"]),
            "semantic_hard_sidecar_score_delta": float((best_semantic_hard_metric or {}).get("semantic_hard_sidecar_score", 1e9) - _semantic_hard_composite_score(best_checkpoint_metric)),
        },
        "train_split_counts_used": train_split_counts_used,
        "val_split_counts_used": val_split_counts_used,
        "train_split_total_count_used": int(train_total_count),
        "val_split_total_count_used": int(sum(val_split_counts_used.values())),
        "frozen_parameter_count": int(frozen_count),
        "trainable_parameter_count": int(trainable_count),
        "boundary_ok": bool(boundary_ok),
        "training_progress": {
            "optimizer_steps": int(optimizer_steps_this_run),
            "effective_batch": int(effective_batch),
            "epochs_completed": float(epochs_completed),
            "eval_interval": int(eval_interval),
            "save_every_n_steps": int(save_every),
        },
        "final_metrics": final_metrics,
        "best_metric_so_far": best_metric_so_far,
        "eval_history": eval_history,
        "checkpoint_inventory": {
            "checkpoint_dir": str(output_dir),
            "best": str(best_ckpt),
            "latest": str(latest_ckpt),
            "best_semantic_hard": str(best_semantic_hard_ckpt),
            "step_checkpoints": sorted(step_checkpoints),
            "resume_from": str(resolved_resume),
            "auto_resume_latest": bool(args.auto_resume_latest),
        },
        "run_stable": bool(run_stable),
    }

    _write_json(run_summary_json, payload)
    _write_progress_snapshot(
        progress_json,
        _build_progress_payload(
            args=args,
            status="completed",
            global_step=int(global_step),
            target_steps=int(target_steps),
            train_summary=train_summary,
            val_summary=val_summary,
            run_metadata=run_metadata,
            runtime_meta=runtime_meta,
            checkpoint_dir=output_dir,
            best_ckpt=best_ckpt,
            latest_ckpt=latest_ckpt,
            eval_history=eval_history,
            best_metric_so_far=best_metric_so_far,
        ),
    )

    print(f"[stage2-smalltrain] run_name={args.run_name}")
    print(f"[stage2-smalltrain] run_summary_json={run_summary_json}")
    print(f"[stage2-smalltrain] checkpoint_dir={output_dir}")
    print(f"[stage2-smalltrain] best_checkpoint={best_ckpt}")
    print(f"[stage2-smalltrain] latest_checkpoint={latest_ckpt}")
    print(f"[stage2-smalltrain] free_rollout_endpoint_l2={float(final_metrics.get('free_rollout_endpoint_l2', 1e9)):.6f}")
    print(f"[stage2-smalltrain] boundary_ok={bool(boundary_ok)}")


if __name__ == "__main__":
    main()
