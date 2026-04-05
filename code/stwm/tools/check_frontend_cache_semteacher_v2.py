from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import time
from typing import Any

import torch


REQUIRED_KEYS = [
    "semantic_features_teacher",
    "target_semantic_probs_teacher",
    "cache_version",
    "manifest_hash",
    "frontend_hash",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Healthcheck semteacher frontend cache required fields")
    parser.add_argument(
        "--cache-dir",
        default="/home/chen034/workspace/stwm/data/cache/frontend_cache_protocol_v2_full_v1",
    )
    parser.add_argument("--index", default="")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/stwm_frontend_cache_semteacher_health_v1.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = build_parser().parse_args()
    random.seed(int(args.seed))

    cache_dir = Path(args.cache_dir)
    index_path = Path(args.index) if str(args.index).strip() else (cache_dir / "index.json")
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")

    index_payload = json.loads(index_path.read_text())
    entries = index_payload.get("entries", []) if isinstance(index_payload, dict) else []
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("cache index entries empty")

    sample_n = max(1, min(int(args.sample_count), len(entries)))
    sampled = random.sample(entries, sample_n)

    missing_records: list[dict[str, Any]] = []
    bad_shape_records: list[dict[str, Any]] = []

    for e in sampled:
        shard_rel = str(e.get("shard", "")).strip()
        offset = int(e.get("offset", 0))
        clip_id = str(e.get("clip_id", "")).strip()
        shard_path = cache_dir / shard_rel
        if not shard_path.exists():
            missing_records.append({"clip_id": clip_id, "reason": f"missing_shard:{shard_rel}"})
            continue

        payload = _torch_load(shard_path)
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list) or offset < 0 or offset >= len(records):
            missing_records.append({"clip_id": clip_id, "reason": "invalid_offset_or_records"})
            continue

        rec = records[offset]
        if not isinstance(rec, dict):
            missing_records.append({"clip_id": clip_id, "reason": "record_not_dict"})
            continue

        missing = [k for k in REQUIRED_KEYS if k not in rec]
        if missing:
            missing_records.append({"clip_id": clip_id, "reason": "missing_keys", "missing": missing})
            continue

        try:
            sf = torch.as_tensor(rec["semantic_features_teacher"])  # [T,D]
            sp = torch.as_tensor(rec["target_semantic_probs_teacher"])  # [T,C]
            if sf.ndim != 2 or sp.ndim != 2 or sf.shape[0] != sp.shape[0]:
                bad_shape_records.append(
                    {
                        "clip_id": clip_id,
                        "semantic_features_teacher_shape": list(sf.shape),
                        "target_semantic_probs_teacher_shape": list(sp.shape),
                    }
                )
        except Exception as exc:
            bad_shape_records.append({"clip_id": clip_id, "error": f"{type(exc).__name__}: {exc}"})

    ok = len(missing_records) == 0 and len(bad_shape_records) == 0
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_dir": str(cache_dir),
        "index": str(index_path),
        "sample_count": int(sample_n),
        "required_keys": REQUIRED_KEYS,
        "ok": bool(ok),
        "missing_records": missing_records,
        "bad_shape_records": bad_shape_records,
    }

    out = Path(args.output_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"ok": bool(ok), "report": str(out), "sample_count": int(sample_n)}, indent=2))


if __name__ == "__main__":
    main()
