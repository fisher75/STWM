from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import time

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.modules.semantic_adapter import SemanticAdapter


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Check and optionally repair STWM V4.2 semantic cache for a train-manifest subset")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument(
        "--manifest",
        default="/home/chen034/workspace/stwm/manifests/realsplits/stwm_v4_2_vspw_vipseg_train_v1.json",
    )
    parser.add_argument("--cache-dir", default="/home/chen034/workspace/stwm/data/cache/semantic_summaries")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--repair", action="store_true", help="When set, quarantine bad cache and rebuild from source")
    parser.add_argument("--max-problem-examples", type=int, default=200)
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_semantic_cache_healthcheck.json",
    )
    return parser


def _exc_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = build_parser().parse_args()

    sample_limit = int(args.sample_limit) if int(args.sample_limit) > 0 else None
    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=sample_limit)
    adapter = SemanticAdapter(cache_dir=args.cache_dir, use_cache=True)

    inspected_samples = len(dataset.samples)
    total_checked = 0
    ok_cache_count = 0
    missing_cache_count = 0
    bad_cache_count = 0
    rebuilt_count = 0
    unhandled_count = 0
    problem_examples: list[dict[str, object]] = []

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for sample in dataset.samples:
        num_steps = len(sample.frame_paths)
        if num_steps <= 0:
            continue

        cache_path = adapter.cache_path_for_sample(sample.text_labels, num_steps, clip_id=sample.clip_id)
        if not cache_path.exists():
            missing_cache_count += 1
            continue

        total_checked += 1

        try:
            adapter._load_from_cache(cache_path)
            ok_cache_count += 1
            continue
        except Exception as exc:
            recoverable = adapter.is_cache_error_recoverable(exc)
            issue: dict[str, object] = {
                "clip_id": str(sample.clip_id),
                "cache_path": str(cache_path),
                "error": _exc_text(exc),
                "recoverable": bool(recoverable),
                "repair_attempted": bool(args.repair),
            }

            if recoverable:
                bad_cache_count += 1
                if bool(args.repair):
                    try:
                        summary = adapter.encode(
                            sample.text_labels,
                            num_steps,
                            metadata=sample.metadata,
                            clip_id=sample.clip_id,
                        )
                        rebuilt = bool(summary.metadata.get("cache_rebuilt", False))
                        issue["repaired"] = rebuilt
                        issue["cache_quarantine_path"] = str(summary.metadata.get("cache_quarantine_path", ""))
                        if rebuilt:
                            rebuilt_count += 1
                        else:
                            unhandled_count += 1
                            issue["repair_error"] = "repair path did not mark cache_rebuilt"
                    except Exception as repair_exc:
                        unhandled_count += 1
                        issue["repaired"] = False
                        issue["repair_error"] = _exc_text(repair_exc)
                else:
                    issue["repaired"] = False
                    unhandled_count += 1
            else:
                unhandled_count += 1
                issue["repaired"] = False

            if len(problem_examples) < max(1, int(args.max_problem_examples)):
                problem_examples.append(issue)

    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")

    report = {
        "manifest": str(Path(args.manifest)),
        "data_root": str(Path(args.data_root)),
        "cache_dir": str(Path(args.cache_dir)),
        "sample_limit": int(args.sample_limit),
        "repair_enabled": bool(args.repair),
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": {
            "inspected_samples": int(inspected_samples),
            "total_checked": int(total_checked),
            "ok_cache_count": int(ok_cache_count),
            "missing_cache_count": int(missing_cache_count),
            "bad_cache_count": int(bad_cache_count),
            "rebuilt_count": int(rebuilt_count),
            "unhandled_count": int(unhandled_count),
        },
        "problem_examples": problem_examples,
    }

    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"[semantic-cache-health] report={output_path}")


if __name__ == "__main__":
    main()
