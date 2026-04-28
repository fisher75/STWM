#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import sys


def _apply_process_title_normalization() -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode == "off":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", "python")).strip() or "python"
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


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
        "# STWM Reappearance Random Head Baseline V1 20260427",
        "",
        f"- checkpoint: `{payload.get('checkpoint')}`",
        f"- mode: `{payload.get('mode')}`",
        f"- max_items: `{payload.get('max_items')}`",
        f"- seed_list: `{payload.get('seed_list')}`",
        f"- event_AUROC_mean: `{payload.get('event_AUROC_mean')}`",
        f"- event_AP_mean: `{payload.get('event_AP_mean')}`",
        f"- per_horizon_AUROC_mean: `{payload.get('per_horizon_AUROC_mean')}`",
        f"- per_horizon_AP_mean: `{payload.get('per_horizon_AP_mean')}`",
        f"- random_head_distribution_ready: `{payload.get('random_head_distribution_ready')}`",
        f"- exact_blocking_reason: `{payload.get('exact_blocking_reason')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    m = mean(values)
    assert m is not None
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def metric_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "std": std(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> None:
    _apply_process_title_normalization()
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from stwm.tools.export_future_semantic_trace_state_20260427 import export
    from stwm.tools.eval_future_semantic_trace_queries_20260427 import run_raw_export_mode

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in str(args.seeds).replace(",", " ").split() if x.strip()]
    per_seed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for seed in seeds:
        export_path = out_dir / f"reappearance_random_head_seed_{seed:03d}_export.json"
        eval_path = out_dir / f"reappearance_random_head_seed_{seed:03d}_eval.json"
        eval_doc = out_dir / f"reappearance_random_head_seed_{seed:03d}_eval.md"
        try:
            export_payload = export(
                repo_root=repo_root,
                checkpoint=checkpoint,
                manifest=manifest,
                output=export_path,
                max_items=int(args.max_items),
                device_name=str(args.device),
                mode=str(args.mode),
                reappearance_mask_policy="at_risk_only",
                reappearance_random_seed=seed,
                force_random_reappearance_head=True,
            )
            eval_payload = run_raw_export_mode(export_path, eval_path, eval_doc, repo_root)
            overall = eval_payload.get("overall", {})
            per_seed.append(
                {
                    "seed": seed,
                    "export_path": str(export_path),
                    "eval_path": str(eval_path),
                    "valid_output_ratio": overall.get("valid_output_ratio"),
                    "event_AUROC": overall.get("future_reappearance_event_AUROC"),
                    "event_AP": overall.get("future_reappearance_event_AP"),
                    "event_positive_rate": overall.get("future_reappearance_event_positive_rate"),
                    "per_horizon_AUROC": overall.get("future_reappearance_AUROC"),
                    "per_horizon_AP": overall.get("future_reappearance_AP"),
                    "per_horizon_positive_rate": overall.get("future_reappearance_positive_rate"),
                    "output_degenerate": overall.get("output_degenerate"),
                    "old_association_report_used": eval_payload.get("old_association_report_used"),
                    "full_free_rollout_executed": export_payload.get("full_free_rollout_executed"),
                    "reset_reappearance_heads": export_payload.get("reset_reappearance_heads"),
                }
            )
        except Exception as exc:
            failures.append({"seed": seed, "exact_error": repr(exc)})

    event_aurocs = [float(x["event_AUROC"]) for x in per_seed if isinstance(x.get("event_AUROC"), (int, float))]
    event_aps = [float(x["event_AP"]) for x in per_seed if isinstance(x.get("event_AP"), (int, float))]
    horizon_aurocs = [float(x["per_horizon_AUROC"]) for x in per_seed if isinstance(x.get("per_horizon_AUROC"), (int, float))]
    horizon_aps = [float(x["per_horizon_AP"]) for x in per_seed if isinstance(x.get("per_horizon_AP"), (int, float))]
    event_positive_rates = [float(x["event_positive_rate"]) for x in per_seed if isinstance(x.get("event_positive_rate"), (int, float))]
    ready = bool(len(per_seed) == len(seeds) and event_aps and event_aurocs)
    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "mode": str(args.mode),
        "device": str(args.device),
        "max_items": int(args.max_items),
        "seed_list": seeds,
        "successful_seed_count": len(per_seed),
        "failed_seed_count": len(failures),
        "per_seed": per_seed,
        "failures": failures,
        "event_AUROC_mean": metric_stats(event_aurocs)["mean"],
        "event_AUROC_std": metric_stats(event_aurocs)["std"],
        "event_AUROC_min": metric_stats(event_aurocs)["min"],
        "event_AUROC_max": metric_stats(event_aurocs)["max"],
        "event_AP_mean": metric_stats(event_aps)["mean"],
        "event_AP_std": metric_stats(event_aps)["std"],
        "event_AP_min": metric_stats(event_aps)["min"],
        "event_AP_max": metric_stats(event_aps)["max"],
        "event_positive_rate_mean": mean(event_positive_rates),
        "per_horizon_AUROC_mean": metric_stats(horizon_aurocs)["mean"],
        "per_horizon_AUROC_std": metric_stats(horizon_aurocs)["std"],
        "per_horizon_AUROC_min": metric_stats(horizon_aurocs)["min"],
        "per_horizon_AUROC_max": metric_stats(horizon_aurocs)["max"],
        "per_horizon_AP_mean": metric_stats(horizon_aps)["mean"],
        "per_horizon_AP_std": metric_stats(horizon_aps)["std"],
        "per_horizon_AP_min": metric_stats(horizon_aps)["min"],
        "per_horizon_AP_max": metric_stats(horizon_aps)["max"],
        "best_random_seed_by_event_AP": max(per_seed, key=lambda x: float(x.get("event_AP") or -1.0)).get("seed") if per_seed else None,
        "best_random_seed_by_event_AUROC": max(per_seed, key=lambda x: float(x.get("event_AUROC") or -1.0)).get("seed") if per_seed else None,
        "random_head_distribution_ready": ready,
        "exact_blocking_reason": None if ready else "not all requested seeds produced event AUROC/AP",
    }
    write_json(Path(args.out_report), payload)
    write_doc(Path(args.out_doc), payload)


def parse_args() -> Any:
    parser = ArgumentParser(description="Estimate random reappearance-head baseline distribution using full-model free rollout exports.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", default="full_model_free_rollout", choices=["full_model_teacher_forced", "full_model_free_rollout"])
    parser.add_argument("--max-items", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="0 1 2 3 4 5 6 7 8 9")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-doc", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
