#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

from torch.utils.data import DataLoader, Dataset

from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset, stage1_v2_collate_fn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TimedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base
        self.getitem_times: List[float] = []

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Any:
        start = time.perf_counter()
        item = self.base[index]
        self.getitem_times.append(float(time.perf_counter() - start))
        return item


def parse_args() -> Any:
    parser = ArgumentParser(description="Profile Stage1-v2 dataloader settings")
    parser.add_argument("--contract-path", default="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json")
    parser.add_argument("--dataset-names", nargs="*", default=["pointodyssey", "kubric"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--fut-len", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-samples-per-dataset", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=30)
    parser.add_argument("--report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_dataloader_profile_20260408.json")
    parser.add_argument("--report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_DATALOADER_PROFILE_20260408.md")
    return parser.parse_args()


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _fmt_optional(raw: Any, digits: int = 6) -> str:
    if raw is None:
        return "unavailable"
    try:
        val = float(raw)
    except Exception:
        return "unavailable"
    return f"{val:.{digits}f}"


def _configs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for workers in [0, 4, 8, 16]:
        for pin in [False, True]:
            if workers == 0:
                out.append(
                    {
                        "num_workers": 0,
                        "pin_memory": bool(pin),
                        "persistent_workers": False,
                        "prefetch_factor": None,
                    }
                )
                continue
            for persistent in [False, True]:
                for prefetch in [2, 4]:
                    out.append(
                        {
                            "num_workers": int(workers),
                            "pin_memory": bool(pin),
                            "persistent_workers": bool(persistent),
                            "prefetch_factor": int(prefetch),
                        }
                    )
    return out


def _run_one(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    start_init = time.perf_counter()
    base = Stage1V2UnifiedDataset(
        dataset_names=[str(x) for x in args.dataset_names],
        split=str(args.split),
        contract_path=str(args.contract_path),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
    )
    timed = TimedDataset(base)
    collate_times: List[float] = []

    def timed_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        st = time.perf_counter()
        out = stage1_v2_collate_fn(batch)
        collate_times.append(float(time.perf_counter() - st))
        return out

    kwargs: Dict[str, Any] = {
        "dataset": timed,
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": bool(cfg["pin_memory"]),
        "collate_fn": timed_collate,
    }
    if int(cfg["num_workers"]) > 0:
        kwargs["persistent_workers"] = bool(cfg["persistent_workers"])
        kwargs["prefetch_factor"] = int(cfg["prefetch_factor"])

    loader = DataLoader(**kwargs)
    init_time = float(time.perf_counter() - start_init)

    warmup = max(int(args.warmup_batches), 0)
    measure = max(int(args.measure_batches), 1)

    measured = 0
    measure_start = 0.0
    for step, _batch in enumerate(loader):
        if step == warmup:
            measure_start = time.perf_counter()
        if step >= warmup:
            measured += 1
        if measured >= measure:
            break

    elapsed = float(time.perf_counter() - measure_start) if measure_start > 0 else 0.0
    bps = float(measured / elapsed) if elapsed > 0 else 0.0

    worker_timing_reliable = int(cfg["num_workers"]) == 0
    if worker_timing_reliable:
        mean_getitem_time_sec: Optional[float] = _mean(timed.getitem_times)
        mean_collate_time_sec: Optional[float] = _mean(collate_times)
        worker_timing_status = "available_main_process_single_worker"
    else:
        mean_getitem_time_sec = None
        mean_collate_time_sec = None
        worker_timing_status = "unavailable_multiprocess_workers"

    return {
        "config": cfg,
        "status": "pass",
        "worker_side_timing_reliable": bool(worker_timing_reliable),
        "worker_side_timing_status": worker_timing_status,
        "dataset_init_time_sec": init_time,
        "mean_getitem_time_sec": mean_getitem_time_sec,
        "mean_collate_time_sec": mean_collate_time_sec,
        "batches_per_sec": float(bps),
        "measured_batches": int(measured),
    }


def main() -> None:
    args = parse_args()
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for cfg in _configs():
        try:
            rows.append(_run_one(cfg, args))
        except Exception as exc:
            rows.append(
                {
                    "config": cfg,
                    "status": "fail",
                    "error": str(exc),
                    "worker_side_timing_reliable": False,
                    "worker_side_timing_status": "unavailable_due_error",
                    "dataset_init_time_sec": 0.0,
                    "mean_getitem_time_sec": None,
                    "mean_collate_time_sec": None,
                    "batches_per_sec": 0.0,
                    "measured_batches": 0,
                }
            )

    passed = [r for r in rows if r.get("status") == "pass"]
    best = max(passed, key=lambda x: float(x.get("batches_per_sec", 0.0))) if passed else None

    payload = {
        "generated_at_utc": now_iso(),
        "contract_path": str(args.contract_path),
        "metric_reliability": {
            "batches_per_sec": "reliable_for_all_num_workers",
            "dataset_init_time_sec": "reliable_for_all_num_workers",
            "mean_getitem_time_sec": "reliable_only_when_num_workers_eq_0",
            "mean_collate_time_sec": "reliable_only_when_num_workers_eq_0",
        },
        "reliability_notes": [
            "end_to_end loader throughput is measured in main process and available for all num_workers",
            "worker-side getitem/collate timing is only reliable when num_workers=0",
            "for num_workers>0, worker-side timing is marked unavailable instead of 0.0",
        ],
        "rows": rows,
        "best_config": best.get("config", {}) if best else {},
        "best_batches_per_sec": float(best.get("batches_per_sec", 0.0)) if best else 0.0,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage1-v2 Dataloader Profile",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- best_batches_per_sec: {payload['best_batches_per_sec']:.4f}",
        f"- best_config: {json.dumps(payload['best_config'], ensure_ascii=True)}",
        "- reliability: batches_per_sec and dataset_init_time_sec are reliable for all num_workers",
        "- reliability: mean_getitem_time_sec and mean_collate_time_sec are only reliable when num_workers=0",
        "- reliability: worker-side timing for num_workers>0 is explicitly marked unavailable",
        "",
        "| workers | pin_memory | persistent_workers | prefetch_factor | status | worker_timing_status | dataset_init_sec | mean_getitem_sec | mean_collate_sec | batches_per_sec |",
        "|---:|---|---|---:|---|---|---:|---|---|---:|",
    ]

    def _key(rec: Dict[str, Any]) -> tuple:
        cfg = rec.get("config", {})
        return (
            int(cfg.get("num_workers", 0)),
            int(bool(cfg.get("pin_memory", False))),
            int(bool(cfg.get("persistent_workers", False))),
            int(cfg.get("prefetch_factor", 0) or 0),
        )

    for row in sorted(rows, key=_key):
        cfg = row.get("config", {})
        lines.append(
            "| {w} | {pin} | {pw} | {pf} | {st} | {wt} | {init:.4f} | {gi} | {co} | {bps:.4f} |".format(
                w=int(cfg.get("num_workers", 0)),
                pin=bool(cfg.get("pin_memory", False)),
                pw=bool(cfg.get("persistent_workers", False)),
                pf=int(cfg.get("prefetch_factor", 0) or 0),
                st=str(row.get("status", "unknown")),
                wt=str(row.get("worker_side_timing_status", "unavailable")),
                init=float(row.get("dataset_init_time_sec", 0.0)),
                gi=_fmt_optional(row.get("mean_getitem_time_sec", None), digits=6),
                co=_fmt_optional(row.get("mean_collate_time_sec", None), digits=6),
                bps=float(row.get("batches_per_sec", 0.0)),
            )
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-dataloader-profile] report_json={report_json}")
    print(f"[stage1-v2-dataloader-profile] report_md={report_md}")


if __name__ == "__main__":
    main()
