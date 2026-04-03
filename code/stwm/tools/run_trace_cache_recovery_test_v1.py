from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.modules.trace_adapter import TraceAdapter


def _worker(
  frame_paths: list[str],
  metadata: dict,
  clip_id: str,
  cache_dir: str,
  idx: int,
  queue: mp.Queue,
) -> None:
  try:
    adapter = TraceAdapter(cache_dir=cache_dir, use_cache=True)
    out = adapter.encode(frame_paths, metadata=metadata, clip_id=clip_id)
    queue.put(
      {
        "idx": idx,
        "ok": True,
        "cache_path": str(out.metadata.get("cache_path", "")),
        "cache_rebuilt": bool(out.metadata.get("cache_rebuilt", False)),
      }
    )
  except Exception as exc:
    queue.put({"idx": idx, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
  repo_root = Path(__file__).resolve().parents[3]
  report_path = repo_root / "reports" / "stwm_trace_cache_recovery_test_v1.json"

  cache_root = repo_root / "data" / "cache" / "trace_summaries_phase1_test"
  if cache_root.exists():
    shutil.rmtree(cache_root)
  cache_root.mkdir(parents=True, exist_ok=True)

  manifest = repo_root / "manifests" / "protocol_v2" / "train_v2.json"
  dataset = STWMDataset(repo_root / "data" / "external", manifest=manifest, limit=16)

  sample = None
  for item in dataset.samples:
    if len(item.frame_paths) >= 8:
      sample = item
      break
  if sample is None:
    raise RuntimeError("no sample for trace cache recovery test")

  adapter = TraceAdapter(cache_dir=cache_root, use_cache=True)
  first = adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
  cache_path = Path(first.metadata.get("cache_path", ""))
  if not cache_path.exists():
    raise RuntimeError("cache file was not created")

  # Corrupt cache in-place to verify quarantine + auto rebuild.
  with cache_path.open("wb") as fp:
    fp.write(b"not-a-valid-npz-payload")

  second = adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
  quarantine_dir = cache_root / "quarantine"
  quarantined = sorted(quarantine_dir.glob("*.npz")) if quarantine_dir.exists() else []

  # Cache version mismatch should also trigger safe rebuild.
  adapter_v3 = TraceAdapter(cache_dir=cache_root, use_cache=True, cache_version="trace_mask_center_v3")
  third = adapter_v3.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
  third_cache_path = Path(third.metadata.get("cache_path", ""))
  version_isolation_ok = bool(third_cache_path.exists()) and (third_cache_path != cache_path)

  concurrency_cache = cache_root / "concurrency_case"
  queue: mp.Queue = mp.Queue()
  processes = [
    mp.Process(
      target=_worker,
      args=(sample.frame_paths, sample.metadata, sample.clip_id, str(concurrency_cache), idx, queue),
    )
    for idx in range(2)
  ]

  for proc in processes:
    proc.start()
  for proc in processes:
    proc.join(timeout=120)

  rows = []
  for _ in range(2):
    if not queue.empty():
      rows.append(queue.get())
  rows = sorted(rows, key=lambda x: int(x.get("idx", 0)))

  concurrency_files = sorted(concurrency_cache.glob("*.npz")) if concurrency_cache.exists() else []
  concurrent_ok = (len(rows) == 2) and all(bool(r.get("ok")) for r in rows) and (len(concurrency_files) == 1)

  report = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "phase": "phase1_trace_cache_hardening",
    "sample_clip_id": sample.clip_id,
    "manifest_path": str(manifest),
    "manifest_hash": str(sample.metadata.get("manifest_hash", "")),
    "tests": {
      "corruption_rebuild": {
        "ok": bool(second.metadata.get("cache_rebuilt", False)),
        "cache_rebuild_reason": str(second.metadata.get("cache_rebuild_reason", "")),
        "cache_quarantine_path": str(second.metadata.get("cache_quarantine_path", "")),
        "cache_path_exists_after_rebuild": bool(cache_path.exists()),
        "quarantine_file_count": int(len(quarantined)),
      },
      "cache_version_key_isolation": {
        "ok": bool(version_isolation_ok),
        "cache_version": str(third.metadata.get("cache_version", "")),
        "cache_path": str(third_cache_path),
        "matches_v2_cache_path": bool(third_cache_path == cache_path),
      },
      "concurrent_encode_same_key": {
        "ok": bool(concurrent_ok),
        "process_results": rows,
        "npz_file_count": int(len(concurrency_files)),
      },
    },
  }

  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(json.dumps(report, indent=2))
  print(str(report_path))


if __name__ == "__main__":
  main()
