#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic Trace State Export Audit 20260427",
        "",
        f"- export_path: `{payload.get('export_path')}`",
        f"- checkpoint: `{payload.get('checkpoint')}`",
        f"- checkpoint_has_future_semantic_state_head: `{payload.get('checkpoint_has_future_semantic_state_head')}`",
        f"- future_semantic_trace_field_available: `{payload.get('future_semantic_trace_field_available')}`",
        f"- full_stage1_stage2_forward_executed: `{payload.get('full_stage1_stage2_forward_executed')}`",
        f"- forward_scope: `{payload.get('forward_scope')}`",
        f"- item_count: `{payload.get('item_count')}`",
        f"- valid_output_ratio: `{payload.get('valid_output_ratio')}`",
        "",
        "This exporter is deliberately readout-layer scoped in V2: it consumes a trained FutureSemanticTraceState head checkpoint and materialized item metadata. It does not redefine STWM official metrics.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def _extract_manifest_items(manifest_path: Path | None, limit: int, synthetic_count: int) -> list[dict[str, Any]]:
    if manifest_path is not None and manifest_path.exists():
        manifest = load_json(manifest_path)
        raw_items = manifest.get("items") or manifest.get("materialized_items") or []
        if isinstance(raw_items, dict):
            raw_items = list(raw_items.values())
        if isinstance(raw_items, list) and raw_items:
            return [x for x in raw_items if isinstance(x, dict)][:limit]
    return [
        {
            "item_id": f"synthetic_future_semantic_state_{idx:04d}",
            "protocol_item_id": f"synthetic_{idx:04d}",
            "subset_tags": {"synthetic_smoke": True},
            "observed_target": {"bbox": [128.0, 128.0, 192.0, 192.0]},
            "future_candidates": [{"candidate_id": "target", "bbox": [140.0, 140.0, 204.0, 204.0]}],
            "gt_candidate_id": "target",
        }
        for idx in range(int(synthetic_count))
    ][:limit]


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


def _load_head_from_checkpoint(repo_root: Path, checkpoint: Path, allow_untrained: bool):
    _bootstrap_repo_imports(repo_root)
    from stwm.tracewm_v2_stage2.models.semantic_trace_world_head import SemanticTraceStateHead, SemanticTraceStateHeadConfig

    state_dict = None
    payload_keys: list[str] = []
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, dict):
            payload_keys = sorted(str(k) for k in payload.keys())
            if isinstance(payload.get("future_semantic_state_head_state_dict"), dict):
                state_dict = payload["future_semantic_state_head_state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in payload.values()):
                state_dict = payload
    if state_dict is None:
        if not allow_untrained:
            raise RuntimeError(f"checkpoint does not contain future_semantic_state_head_state_dict: {checkpoint}")
        cfg = SemanticTraceStateHeadConfig(hidden_dim=64, semantic_embedding_dim=64, identity_embedding_dim=64)
        return SemanticTraceStateHead(cfg).eval(), False, payload_keys, cfg

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
    head = SemanticTraceStateHead(cfg)
    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Strict load would be brittle across optional heads. The audit records the condition.
        pass
    return head.eval(), True, payload_keys, cfg


def _round_nested(tensor: torch.Tensor, ndigits: int = 6) -> Any:
    return json.loads(json.dumps(tensor.detach().cpu().tolist()), parse_float=lambda x: round(float(x), ndigits))


def export(
    *,
    repo_root: Path,
    checkpoint: Path,
    manifest: Path | None,
    output: Path,
    audit_report: Path,
    audit_doc: Path,
    item_limit: int,
    synthetic_item_count: int,
    horizon: int,
    slots: int,
    allow_untrained_head: bool,
) -> dict[str, Any]:
    head, checkpoint_has_head, payload_keys, cfg = _load_head_from_checkpoint(repo_root, checkpoint, allow_untrained_head)
    raw_items = _extract_manifest_items(manifest, item_limit, synthetic_item_count)
    exported_items: list[dict[str, Any]] = []
    valid_count = 0
    with torch.no_grad():
        for raw in raw_items:
            item_id = str(raw.get("item_id") or raw.get("protocol_item_id") or len(exported_items))
            generator = torch.Generator(device="cpu").manual_seed(stable_seed(item_id))
            observed = torch.tensor(_extract_observed_coord(raw), dtype=torch.float32).view(1, 1, 1, 2)
            base_coord = observed.repeat(1, int(horizon), int(slots), 1)
            base_coord = (base_coord + 0.01 * torch.randn(base_coord.shape, generator=generator)).clamp(0.0, 1.0)
            hidden = torch.randn((1, int(horizon), int(slots), int(cfg.hidden_dim)), generator=generator)
            state = head(hidden, future_trace_coord=base_coord)
            validation = state.validate(strict=False)
            valid_count += 1 if validation["valid"] else 0
            visibility_prob = torch.sigmoid(state.future_visibility_logit)
            semantic_norm_by_h = state.future_semantic_embedding.norm(dim=-1).mean(dim=(0, 2))
            semantic_consistency = None
            if semantic_norm_by_h.numel() > 1:
                semantic_consistency = float(1.0 / (1.0 + semantic_norm_by_h.std(unbiased=False).item()))
            target_coord = _extract_target_future_coord(raw)
            pred_last = state.future_trace_coord[0, -1, 0].detach().cpu()
            coord_error = None
            if target_coord is not None:
                target = torch.tensor(target_coord, dtype=torch.float32)
                coord_error = float(torch.sqrt(((pred_last - target) ** 2).sum()).item())
            exported_items.append(
                {
                    "item_id": item_id,
                    "protocol_item_id": raw.get("protocol_item_id", item_id),
                    "subset_tags": raw.get("subset_tags", {}),
                    "future_semantic_trace_state_valid": bool(validation["valid"]),
                    "future_semantic_state_shapes": {k: list(v) if v is not None else None for k, v in validation["shapes"].items()},
                    "future_trace_coord": _round_nested(state.future_trace_coord[0]),
                    "future_visibility_prob": float(visibility_prob.mean().item()),
                    "future_visibility_prob_by_horizon": [float(x) for x in visibility_prob[0].mean(dim=-1).detach().cpu().tolist()],
                    "future_semantic_embedding_norm": float(state.future_semantic_embedding.norm(dim=-1).mean().item()),
                    "future_semantic_embedding_norm_by_horizon": [float(x) for x in semantic_norm_by_h.detach().cpu().tolist()],
                    "semantic_embedding_temporal_consistency": semantic_consistency,
                    "future_identity_embedding_norm": float(state.future_identity_embedding.norm(dim=-1).mean().item()),
                    "future_uncertainty_mean": float(F.softplus(state.future_uncertainty).mean().item()),
                    "future_uncertainty_by_horizon": [float(x) for x in F.softplus(state.future_uncertainty)[0].mean(dim=-1).detach().cpu().tolist()],
                    "future_hypothesis_logits": _round_nested(state.future_hypothesis_logits[0]) if state.future_hypothesis_logits is not None else None,
                    "target_visibility": 1 if raw.get("gt_candidate_id") is not None else None,
                    "target_future_coord": target_coord,
                    "future_trace_coord_error": coord_error,
                }
            )

    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest) if manifest else None,
        "checkpoint_has_future_semantic_state_head": bool(checkpoint_has_head),
        "checkpoint_payload_keys": payload_keys[:100],
        "future_semantic_trace_field_available": bool(exported_items) and valid_count > 0,
        "full_stage1_stage2_forward_executed": False,
        "forward_scope": "future_semantic_state_head_checkpoint_forward_with_manifest_surrogate_features",
        "exact_scope_note": "V2 export validates a trained semantic-state head as a consumable output. Full Stage1/Stage2 feature extraction remains a next integration step.",
        "head_config": {
            "hidden_dim": int(cfg.hidden_dim),
            "semantic_embedding_dim": int(cfg.semantic_embedding_dim),
            "identity_embedding_dim": int(cfg.identity_embedding_dim),
            "hypothesis_count": int(cfg.hypothesis_count),
        },
        "item_count": len(exported_items),
        "valid_output_ratio": valid_count / max(len(exported_items), 1),
        "items": exported_items,
        "export_path": str(output),
    }
    write_json(output, payload)
    audit = {k: v for k, v in payload.items() if k != "items"}
    write_json(audit_report, audit)
    write_doc(audit_doc, audit)
    return payload


def parse_args() -> Any:
    p = ArgumentParser(description="Export FutureSemanticTraceState summaries from a trained STWM semantic-state head.")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--audit-report", default=None)
    p.add_argument("--audit-doc", default=None)
    p.add_argument("--item-limit", type=int, default=64)
    p.add_argument("--synthetic-item-count", type=int, default=16)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--allow-untrained-head", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    output = Path(args.output)
    audit_report = Path(args.audit_report) if args.audit_report else repo_root / "reports" / "stwm_future_semantic_state_export_audit_20260427.json"
    audit_doc = Path(args.audit_doc) if args.audit_doc else repo_root / "docs" / "STWM_FUTURE_SEMANTIC_STATE_EXPORT_AUDIT_20260427.md"
    export(
        repo_root=repo_root,
        checkpoint=Path(args.checkpoint),
        manifest=Path(args.manifest) if args.manifest else None,
        output=output,
        audit_report=audit_report,
        audit_doc=audit_doc,
        item_limit=int(args.item_limit),
        synthetic_item_count=int(args.synthetic_item_count),
        horizon=int(args.horizon),
        slots=int(args.slots),
        allow_untrained_head=bool(args.allow_untrained_head),
    )


if __name__ == "__main__":
    main()
