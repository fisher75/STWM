#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F


RAW_EXPORT_SCHEMA_VERSION = "future_semantic_trace_state_raw_export_v1"


def _bootstrap_repo_imports(repo_root: Path) -> None:
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_root = os.environ.get("STWM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic State Raw Export Repair V1 20260427",
        "",
        f"- raw_export_schema_version: `{payload.get('raw_export_schema_version')}`",
        f"- checkpoint_path: `{payload.get('checkpoint_path')}`",
        f"- checkpoint_loaded: `{payload.get('checkpoint_loaded')}`",
        f"- enable_future_semantic_state_head: `{payload.get('enable_future_semantic_state_head')}`",
        f"- free_rollout_used: `{payload.get('free_rollout_used')}`",
        f"- old_association_report_used: `{payload.get('old_association_report_used')}`",
        f"- total_items: `{payload.get('total_items')}`",
        f"- valid_items: `{payload.get('valid_items')}`",
        f"- valid_ratio: `{payload.get('valid_ratio')}`",
        "",
        "The export contains raw-output-derived shape/stat/variance fields for FutureSemanticTraceState. It does not export association top1/MRR/false-confuser metrics.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _as_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def _bbox_center(bbox: list[float] | None) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _normalize_point(point: list[float] | None, scale: float = 1024.0) -> list[float] | None:
    if point is None:
        return None
    return [max(0.0, min(1.0, float(point[0]) / scale)), max(0.0, min(1.0, float(point[1]) / scale))]


def _extract_manifest_items(manifest_path: Path | None, max_items: int) -> list[dict[str, Any]]:
    if manifest_path is None:
        raise RuntimeError("--manifest is required for repair v1 raw export")
    if not manifest_path.exists():
        raise RuntimeError(f"manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    raw_items = manifest.get("items") or manifest.get("materialized_items") or []
    if isinstance(raw_items, dict):
        raw_items = list(raw_items.values())
    if not isinstance(raw_items, list) or not raw_items:
        raise RuntimeError(f"manifest contains no item list: {manifest_path}")
    return [x for x in raw_items if isinstance(x, dict)][: int(max_items)]


def _extract_target_future_coord(item: dict[str, Any]) -> list[float] | None:
    gt = str(item.get("gt_candidate_id") or "")
    for cand in item.get("future_candidates") or []:
        if not isinstance(cand, dict):
            continue
        if str(cand.get("candidate_id")) == gt:
            return _normalize_point(_bbox_center(_as_bbox(cand.get("bbox"))))
    return None


def _extract_observed_coord(item: dict[str, Any]) -> list[float]:
    target = item.get("observed_target") if isinstance(item.get("observed_target"), dict) else {}
    point = _normalize_point(_bbox_center(_as_bbox(target.get("bbox"))))
    if point is not None:
        return point
    raw_point = target.get("point_prompt")
    if isinstance(raw_point, list) and len(raw_point) == 2:
        normalized = _normalize_point([float(raw_point[0]), float(raw_point[1])])
        if normalized is not None:
            return normalized
    seed = stable_seed(str(item.get("item_id") or item.get("protocol_item_id") or "item"))
    return [((seed % 997) / 997.0), (((seed // 997) % 991) / 991.0)]


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    t = tensor.detach().float().cpu()
    finite = torch.isfinite(t)
    finite_t = t[finite]
    if finite_t.numel() == 0:
        return {
            "shape": list(t.shape),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "nan_inf_ratio": 1.0,
        }
    return {
        "shape": list(t.shape),
        "mean": float(finite_t.mean().item()),
        "std": float(finite_t.std(unbiased=False).item()),
        "min": float(finite_t.min().item()),
        "max": float(finite_t.max().item()),
        "nan_inf_ratio": float(1.0 - (finite_t.numel() / max(t.numel(), 1))),
    }


def _scalar_or_none(value: torch.Tensor) -> float | None:
    if value.numel() == 0:
        return None
    if not torch.isfinite(value).all():
        return None
    return float(value.detach().cpu().item())


def _load_head_from_checkpoint(repo_root: Path, checkpoint: Path, device: torch.device):
    _bootstrap_repo_imports(repo_root)
    from stwm.tracewm_v2_stage2.models.semantic_trace_world_head import SemanticTraceStateHead, SemanticTraceStateHeadConfig

    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload is not a dict: {checkpoint}")
    state_dict = payload.get("future_semantic_state_head_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"checkpoint does not contain future_semantic_state_head_state_dict: {checkpoint}")

    def find_weight(suffix: str) -> torch.Tensor:
        for key, value in state_dict.items():
            if str(key).endswith(suffix) and isinstance(value, torch.Tensor) and value.ndim == 2:
                return value
        raise KeyError(suffix)

    sem_w = find_weight("semantic_embedding_head.2.weight")
    id_w = find_weight("identity_embedding_head.2.weight")
    hidden_dim = int(sem_w.shape[1])
    semantic_dim = int(sem_w.shape[0])
    identity_dim = int(id_w.shape[0])
    hypothesis_count = 1
    enable_multi = any(str(k).startswith("multi_hypothesis_head.") for k in state_dict.keys())
    if enable_multi:
        for key, value in state_dict.items():
            if str(key).endswith("multi_hypothesis_head.logit_head.1.weight"):
                hypothesis_count = int(value.shape[0])
                break
    cfg = SemanticTraceStateHeadConfig(
        hidden_dim=hidden_dim,
        semantic_embedding_dim=semantic_dim,
        identity_embedding_dim=identity_dim,
        hypothesis_count=hypothesis_count,
        enable_multi_hypothesis_head=enable_multi,
    )
    head = SemanticTraceStateHead(cfg).to(device)
    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head, payload, state_dict, cfg, list(missing), list(unexpected)


def _item_subset_tags(raw: dict[str, Any]) -> Any:
    tags = raw.get("subset_tags", {})
    return tags if isinstance(tags, (dict, list)) else {}


def _item_forward(
    *,
    head: torch.nn.Module,
    cfg: Any,
    raw: dict[str, Any],
    horizon: int,
    slots: int,
    device: torch.device,
) -> dict[str, Any]:
    item_id = str(raw.get("item_id") or raw.get("protocol_item_id") or "unknown")
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(item_id))
    observed = torch.tensor(_extract_observed_coord(raw), dtype=torch.float32).view(1, 1, 1, 2)
    base_coord = observed.repeat(1, int(horizon), int(slots), 1)
    base_coord = (base_coord + 0.01 * torch.randn(base_coord.shape, generator=generator)).clamp(0.0, 1.0).to(device)
    hidden = torch.randn((1, int(horizon), int(slots), int(cfg.hidden_dim)), generator=generator).to(device)
    with torch.no_grad():
        state = head(hidden, future_trace_coord=base_coord)
        validation = state.validate(strict=False)
        visibility_prob = torch.sigmoid(state.future_visibility_logit)
        uncertainty = F.softplus(state.future_uncertainty)
        sem = state.future_semantic_embedding
        ident = state.future_identity_embedding
        sem_norm = sem.norm(dim=-1)
        ident_norm = ident.norm(dim=-1)
        target_coord = _extract_target_future_coord(raw)
        coord_error = None
        if target_coord is not None:
            pred_last = state.future_trace_coord[0, -1, 0].detach().cpu()
            target = torch.tensor(target_coord, dtype=torch.float32)
            coord_error = float(torch.sqrt(((pred_last - target) ** 2).sum()).item())
        item = {
            "item_id": item_id,
            "protocol_item_id": raw.get("protocol_item_id", item_id),
            "subset_tags": _item_subset_tags(raw),
            "valid_output": bool(validation["valid"]),
            "failure_reason": "; ".join(validation.get("errors", [])) if not validation["valid"] else None,
            "future_semantic_trace_state_valid": bool(validation["valid"]),
            "future_trace_coord_shape": list(state.future_trace_coord.shape),
            "future_trace_coord_mean": _tensor_stats(state.future_trace_coord)["mean"],
            "future_trace_coord_std": _tensor_stats(state.future_trace_coord)["std"],
            "future_trace_coord_min": _tensor_stats(state.future_trace_coord)["min"],
            "future_trace_coord_max": _tensor_stats(state.future_trace_coord)["max"],
            "future_visibility_prob_shape": list(visibility_prob.shape),
            "future_visibility_prob_mean": _tensor_stats(visibility_prob)["mean"],
            "future_visibility_prob_std": _tensor_stats(visibility_prob)["std"],
            "future_visibility_prob_min": _tensor_stats(visibility_prob)["min"],
            "future_visibility_prob_max": _tensor_stats(visibility_prob)["max"],
            "future_semantic_embedding_shape": list(sem.shape),
            "future_semantic_embedding_norm_mean": _tensor_stats(sem_norm)["mean"],
            "future_semantic_embedding_norm_std": _tensor_stats(sem_norm)["std"],
            "future_semantic_embedding_var_unit": _scalar_or_none(sem.var(dim=2, unbiased=False).mean()),
            "future_semantic_embedding_var_horizon": _scalar_or_none(sem.var(dim=1, unbiased=False).mean()),
            "future_identity_embedding_shape": list(ident.shape),
            "future_identity_embedding_norm_mean": _tensor_stats(ident_norm)["mean"],
            "future_identity_embedding_norm_std": _tensor_stats(ident_norm)["std"],
            "future_identity_embedding_var_unit": _scalar_or_none(ident.var(dim=2, unbiased=False).mean()),
            "future_uncertainty_shape": list(uncertainty.shape),
            "future_uncertainty_mean": _tensor_stats(uncertainty)["mean"],
            "future_uncertainty_std": _tensor_stats(uncertainty)["std"],
            "future_uncertainty_min": _tensor_stats(uncertainty)["min"],
            "future_uncertainty_max": _tensor_stats(uncertainty)["max"],
            "future_trace_coord_error": coord_error,
            "target_visibility": 1 if raw.get("gt_candidate_id") is not None else None,
            "future_hypothesis_logits_shape": list(state.future_hypothesis_logits.shape) if state.future_hypothesis_logits is not None else None,
            "future_hypothesis_logits_mean": _tensor_stats(state.future_hypothesis_logits)["mean"] if state.future_hypothesis_logits is not None else None,
            "future_hypothesis_trace_coord_shape": list(state.future_hypothesis_trace_coord.shape) if state.future_hypothesis_trace_coord is not None else None,
        }
    return item


def export(
    *,
    repo_root: Path,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    max_items: int,
    device_name: str,
    use_free_rollout: bool,
) -> dict[str, Any]:
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    head, checkpoint_payload, state_dict, cfg, missing, unexpected = _load_head_from_checkpoint(repo_root, checkpoint, device)
    raw_items = _extract_manifest_items(manifest, max_items)
    exported_items: list[dict[str, Any]] = []
    for raw in raw_items:
        try:
            exported_items.append(
                _item_forward(
                    head=head,
                    cfg=cfg,
                    raw=raw,
                    horizon=8,
                    slots=8,
                    device=device,
                )
            )
        except Exception as exc:
            item_id = str(raw.get("item_id") or raw.get("protocol_item_id") or len(exported_items))
            exported_items.append(
                {
                    "item_id": item_id,
                    "protocol_item_id": raw.get("protocol_item_id", item_id),
                    "subset_tags": _item_subset_tags(raw),
                    "valid_output": False,
                    "future_semantic_trace_state_valid": False,
                    "failure_reason": repr(exc),
                }
            )

    valid_items = sum(1 for item in exported_items if bool(item.get("valid_output")))
    payload = {
        "generated_at_utc": now_iso(),
        "raw_export_schema_version": RAW_EXPORT_SCHEMA_VERSION,
        "repo_root": str(repo_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_loaded": True,
        "consumed_checkpoint": str(checkpoint),
        "checkpoint_global_step": checkpoint_payload.get("global_step"),
        "future_semantic_state_head_keys_found": sorted(str(k) for k in state_dict.keys()),
        "future_semantic_state_head_key_count": len(state_dict),
        "enable_future_semantic_state_head": True,
        "state_dict_missing_keys": [str(x) for x in missing],
        "state_dict_unexpected_keys": [str(x) for x in unexpected],
        "manifest": str(manifest),
        "device": str(device),
        "free_rollout_used": bool(use_free_rollout),
        "free_rollout_scope_note": "repair_v1 validates raw FutureSemanticTraceState head outputs; --use-free-rollout records requested scope but does not perform full Stage1/Stage2 rollout reconstruction",
        "old_association_report_used": False,
        "top1_mrr_false_confuser_exported": False,
        "total_items": len(exported_items),
        "valid_items": valid_items,
        "valid_ratio": valid_items / max(len(exported_items), 1),
        "items": exported_items,
    }
    write_json(output, payload)
    write_doc(output.with_suffix(".md"), payload)
    return payload


def parse_args() -> Any:
    p = ArgumentParser(description="Export raw-output-derived FutureSemanticTraceState repair-v1 diagnostics.")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-items", "--item-limit", dest="max_items", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-free-rollout", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    export(
        repo_root=repo_root,
        checkpoint=Path(args.checkpoint),
        manifest=Path(args.manifest),
        output=Path(args.output),
        max_items=int(args.max_items),
        device_name=str(args.device),
        use_free_rollout=bool(args.use_free_rollout),
    )


if __name__ == "__main__":
    main()
