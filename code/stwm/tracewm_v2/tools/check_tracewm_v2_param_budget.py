#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json

from stwm.tracewm_v2.models.causal_trace_transformer import (
    TraceCausalTransformer,
    build_tracewm_v2_config,
    estimate_parameter_count,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    parser = ArgumentParser(description="Check TraceWM v2 transformer parameter budget")
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/tracewm_stage1_v2_param_budget_20260408.json")
    parser.add_argument("--instantiate", action="store_true")
    return parser.parse_args()


def _entry(preset: str, instantiate: bool) -> Dict[str, Any]:
    cfg = build_tracewm_v2_config(preset)
    est = int(estimate_parameter_count(cfg))
    out: Dict[str, Any] = {
        "preset": preset,
        "config": cfg.__dict__,
        "estimated_parameter_count": est,
        "in_220m_range": bool(200_000_000 <= est <= 240_000_000),
    }

    if instantiate:
        model = TraceCausalTransformer(cfg)
        out["instantiated_parameter_count"] = int(model.parameter_count())
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": now_iso(),
        "presets": {
            "prototype_220m": _entry("prototype_220m", instantiate=bool(args.instantiate)),
            "debug_small": _entry("debug_small", instantiate=bool(args.instantiate)),
        },
    }

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[param-budget] wrote {out}")


if __name__ == "__main__":
    main()
