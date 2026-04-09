#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict
from stwm.tracewm_v2_stage2.tools.stage2_eval_protocol import (
    build_eval_fix_comparison,
    build_eval_fix_run_summary,
    write_results_markdown,
)
import json


def parse_args() -> Any:
    p = ArgumentParser(description="Summarize Stage2 semantic hardening core vs core+burst runs")
    p.add_argument("--core-run-json", required=True)
    p.add_argument("--core-plus-burst-run-json", required=True)
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--results-md", required=True)
    return p.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    core_raw = _read_json(args.core_run_json)
    burst_raw = _read_json(args.core_plus_burst_run_json)

    core = build_eval_fix_run_summary(core_raw, run_label="stage2_smalltrain_cropenc_core")
    burst = build_eval_fix_run_summary(burst_raw, run_label="stage2_smalltrain_cropenc_core_plus_burst")

    comparison = build_eval_fix_comparison(
        core,
        burst,
        round_name="stage2_semantic_hardening_20260408",
    )

    comparison_json = Path(args.comparison_json)
    results_md = Path(args.results_md)
    _write_json(comparison_json, comparison)
    write_results_markdown(results_md, comparison, title="Stage2 Semantic Hardening Results")

    print(f"[stage2-semhard-summary] comparison_json={comparison_json}")
    print(f"[stage2-semhard-summary] results_md={results_md}")
    print(f"[stage2-semhard-summary] crop_based_semantic_encoder_ran_through={comparison['crop_based_semantic_encoder_ran_through']}")
    print(f"[stage2-semhard-summary] final_recommended_mainline={comparison['final_recommended_mainline']}")
    print(f"[stage2-semhard-summary] can_continue_stage2_training={comparison['can_continue_stage2_training']}")
    print(f"[stage2-semhard-summary] next_step_choice={comparison['next_step_choice']}")


if __name__ == "__main__":
    main()