from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
import time


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Wait for enough warmup/nowarm audit rows, then write comparison markdown")
    parser.add_argument("--nowarm-report", required=True)
    parser.add_argument("--warmup-report", required=True)
    parser.add_argument("--output-doc", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--min-rows", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    return parser


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    out: list[dict] = []
    for row in rows:
        if isinstance(row, dict) and "step" in row:
            out.append(row)
    out.sort(key=lambda x: int(x.get("step", 0)))
    return out


def _summary(rows: list[dict], key: str) -> dict[str, float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return {"first": 0.0, "median": 0.0, "last": 0.0, "min": 0.0, "max": 0.0}
    return {
        "first": float(values[0]),
        "median": float(statistics.median(values)),
        "last": float(values[-1]),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _fmt(summary: dict[str, float]) -> str:
    return (
        f"first={summary['first']:.9g}, median={summary['median']:.9g}, "
        f"last={summary['last']:.9g}, min={summary['min']:.9g}, max={summary['max']:.9g}"
    )


def _write_doc(output_doc: Path, nowarm_rows: list[dict], warmup_rows: list[dict], nowarm_path: Path, warmup_path: Path) -> None:
    n_traj = _summary(nowarm_rows, "g_traj_norm")
    w_traj = _summary(warmup_rows, "g_traj_norm")

    n_sem = _summary(nowarm_rows, "g_sem_norm")
    w_sem = _summary(warmup_rows, "g_sem_norm")

    n_cos = _summary(nowarm_rows, "cos_sem_traj")
    w_cos = _summary(warmup_rows, "cos_sem_traj")

    n_qpath = _summary(nowarm_rows, "qpath_g_query_norm")
    w_qpath = _summary(warmup_rows, "qpath_g_query_norm")

    lines = [
        "# STWM Warmup Gradient Comparison V1",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Status: Generated after multi-point frontend-default audits",
        "",
        "## Inputs",
        "",
        f"- nowarm: `{nowarm_path}`",
        f"- warmup: `{warmup_path}`",
        f"- nowarm rows: {len(nowarm_rows)}",
        f"- warmup rows: {len(warmup_rows)}",
        "",
        "## Core Metrics (first / median / last / min / max)",
        "",
        "### `||g_traj||`",
        f"- nowarm: {_fmt(n_traj)}",
        f"- warmup: {_fmt(w_traj)}",
        "",
        "### `||g_sem||`",
        f"- nowarm: {_fmt(n_sem)}",
        f"- warmup: {_fmt(w_sem)}",
        "",
        "### `cos(g_sem, g_traj)`",
        f"- nowarm: {_fmt(n_cos)}",
        f"- warmup: {_fmt(w_cos)}",
        "",
        "### `qpath ||g_query||` (query-path-aware anchor)",
        f"- nowarm: {_fmt(n_qpath)}",
        f"- warmup: {_fmt(w_qpath)}",
        "",
        "## Readout",
        "",
        "1. This comparison is computed on frontend-default run artifacts only.",
        "2. Positive shift in `cos(g_sem, g_traj)` indicates reduced direct conflict tendency.",
        "3. Query-path-aware `qpath ||g_query||` confirms query supervision path remains measurable.",
    ]
    output_doc.parent.mkdir(parents=True, exist_ok=True)
    output_doc.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    nowarm_path = Path(args.nowarm_report)
    warmup_path = Path(args.warmup_report)
    output_doc = Path(args.output_doc)

    started = time.time()
    poll_seconds = max(5, int(args.poll_seconds))
    min_rows = max(1, int(args.min_rows))
    timeout_seconds = max(0, int(args.timeout_seconds))

    while True:
        nowarm_rows = _load_rows(nowarm_path)
        warmup_rows = _load_rows(warmup_path)

        if len(nowarm_rows) >= min_rows and len(warmup_rows) >= min_rows:
            _write_doc(output_doc, nowarm_rows, warmup_rows, nowarm_path, warmup_path)
            print(str(output_doc))
            return

        if timeout_seconds > 0 and (time.time() - started) >= timeout_seconds:
            raise TimeoutError(
                f"timeout waiting for rows>={min_rows}: nowarm={len(nowarm_rows)}, warmup={len(warmup_rows)}"
            )

        print(
            f"waiting rows nowarm={len(nowarm_rows)} warmup={len(warmup_rows)} target={min_rows}",
            flush=True,
        )
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
