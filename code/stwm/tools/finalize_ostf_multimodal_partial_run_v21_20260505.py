#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json
from stwm.tools.ostf_v20_common_20260502 import load_combo_rows, load_context_cache
from stwm.tools.train_ostf_multimodal_v21_20260502 import _build_model, _combo_for_model, _evaluate_model


TRAIN_RE = re.compile(r"\[V21\]\[train\]\s+(.*)")
VAL_RE = re.compile(r"\[V21\]\[val\]\s+(.*)")


def _parse_kv_blob(blob: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for token in blob.strip().split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except Exception:
            out[k] = v
    return out


def _parse_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    max_step = 0
    if not path.exists():
        return train_rows, val_rows, max_step
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            row = _parse_kv_blob(m.group(1))
            if "step" in row:
                max_step = max(max_step, int(row["step"]))
            train_rows.append(row)
            continue
        m = VAL_RE.search(line)
        if m:
            row = _parse_kv_blob(m.group(1))
            if "step" in row:
                max_step = max(max_step, int(row["step"]))
            val_rows.append(row)
    return train_rows, val_rows, max_step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-kind", required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--context-cache-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    combo = _combo_for_model(args.model_kind, args.horizon)
    rows, proto_centers = load_combo_rows(combo, seed=args.seed)
    ctx_map = load_context_cache(ROOT / args.context_cache_path)
    model = _build_model(args.model_kind, args.horizon).to(device)
    ckpt = torch.load(ROOT / args.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    train_all, train_ds, train_sub, train_rows_mm, train_div = _evaluate_model(model, rows["train"], ctx_map, args.batch_size, device)
    val_all, val_ds, val_sub, val_rows_mm, val_div = _evaluate_model(model, rows["val"], ctx_map, args.batch_size, device)
    test_all, test_ds, test_sub, test_rows_mm, test_div = _evaluate_model(model, rows["test"], ctx_map, args.batch_size, device)
    loss_history, val_history, max_step = _parse_log(ROOT / args.log_path)

    report = {
        "audit_name": "stwm_ostf_v21_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "finalized_from_best_checkpoint": True,
        "experiment_name": args.experiment_name,
        "model_kind": args.model_kind,
        "horizon": args.horizon,
        "seed": args.seed,
        "source_combo": combo,
        "steps": max_step,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_path": args.checkpoint_path,
        "best_checkpoint_path": args.checkpoint_path,
        "context_cache_path": args.context_cache_path,
        "log_path": args.log_path,
        "best_val_score": max((float(r.get("score", float("-inf"))) for r in val_history), default=None),
        "loss_history": loss_history,
        "val_history": val_history,
        "train_metrics": train_all,
        "val_metrics": val_all,
        "test_metrics": test_all,
        "train_metrics_by_dataset": train_ds,
        "val_metrics_by_dataset": val_ds,
        "test_metrics_by_dataset": test_ds,
        "train_subset_metrics": train_sub,
        "val_subset_metrics": val_sub,
        "test_subset_metrics": test_sub,
        "train_diversity_valid": train_div,
        "val_diversity_valid": val_div,
        "test_diversity_valid": test_div,
        "item_scores": test_rows_mm,
        "metric_note": "V21 run finalized from current best checkpoint to preserve ablation evidence while main M512 full-length training continues.",
    }
    out_path = ROOT / f"reports/stwm_ostf_v21_runs/{args.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, report)
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
