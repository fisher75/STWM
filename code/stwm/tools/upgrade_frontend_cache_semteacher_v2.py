from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import time
from typing import Any

import torch


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Upgrade frontend cache records to semteacher v2 schema fields")
    parser.add_argument(
        "--cache-dir",
        default="/home/chen034/workspace/stwm/data/cache/frontend_cache_protocol_v2_full_v1",
    )
    parser.add_argument("--index", default="")
    parser.add_argument(
        "--manifest",
        default="/home/chen034/workspace/stwm/manifests/protocol_v2/train_v2.json",
    )
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--teacher-backend-tag", default="cache_proxy_lift_v1")
    parser.add_argument(
        "--cache-version",
        default="frontend_cache_semteacher_v2",
    )
    return parser


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _frontend_hash(record: dict[str, Any]) -> str:
    md = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    payload = {
        "clip_id": str(record.get("clip_id", "")),
        "seq_len": int(record.get("seq_len", 0)),
        "frame_paths_head": [str(x) for x in md.get("frame_paths_head", [])] if isinstance(md.get("frame_paths_head"), list) else [],
        "frame_paths_tail": [str(x) for x in md.get("frame_paths_tail", [])] if isinstance(md.get("frame_paths_tail"), list) else [],
        "mask_paths_head": [str(x) for x in md.get("mask_paths_head", [])] if isinstance(md.get("mask_paths_head"), list) else [],
        "mask_paths_tail": [str(x) for x in md.get("mask_paths_tail", [])] if isinstance(md.get("mask_paths_tail"), list) else [],
    }
    return _sha1_text(json.dumps(payload, sort_keys=True))


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = build_parser().parse_args()

    cache_dir = Path(args.cache_dir)
    index_path = Path(args.index) if str(args.index).strip() else (cache_dir / "index.json")
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")
    if not index_path.exists():
        raise FileNotFoundError(f"cache index not found: {index_path}")

    out_dir = cache_dir
    if not bool(args.inplace):
        out_raw = str(args.output_dir).strip()
        if not out_raw:
            out_raw = str(cache_dir.parent / f"{cache_dir.name}_semteacher_v2")
        out_dir = Path(out_raw)
        out_dir.mkdir(parents=True, exist_ok=True)

    manifest_hash = ""
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest_hash = hashlib.sha1(manifest_path.read_bytes()).hexdigest()

    index_payload = json.loads(index_path.read_text())
    entries = index_payload.get("entries", []) if isinstance(index_payload, dict) else []
    if not isinstance(entries, list):
        raise RuntimeError("invalid cache index payload: entries is not list")

    shard_names = sorted({str(e.get("shard", "")).strip() for e in entries if str(e.get("shard", "")).strip()})
    if not shard_names:
        raise RuntimeError("no shard names discovered in index")

    upgraded_records = 0
    upgraded_shards = 0

    for shard_rel in shard_names:
        src_shard = cache_dir / shard_rel
        if not src_shard.exists():
            continue
        payload = _torch_load(src_shard)
        if not isinstance(payload, dict):
            continue
        records = payload.get("records")
        if not isinstance(records, list):
            continue

        changed = False
        for rec in records:
            if not isinstance(rec, dict):
                continue

            if "semantic_features_teacher" not in rec and "semantic_features" in rec:
                rec["semantic_features_teacher"] = rec["semantic_features"]
                changed = True
            if "target_semantic_probs_teacher" not in rec and "target_semantic_probs" in rec:
                rec["target_semantic_probs_teacher"] = rec["target_semantic_probs"]
                changed = True

            if "cache_version" not in rec or str(rec.get("cache_version", "")) != str(args.cache_version):
                rec["cache_version"] = str(args.cache_version)
                changed = True
            if "manifest_hash" not in rec or (manifest_hash and str(rec.get("manifest_hash", "")) != manifest_hash):
                rec["manifest_hash"] = manifest_hash or str(rec.get("manifest_hash", ""))
                changed = True
            if "frontend_hash" not in rec or not str(rec.get("frontend_hash", "")).strip():
                rec["frontend_hash"] = _frontend_hash(rec)
                changed = True

            md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            if not isinstance(md, dict):
                md = {}
            if str(md.get("semantic_teacher_backend", "")) != str(args.teacher_backend_tag):
                md["semantic_teacher_backend"] = str(args.teacher_backend_tag)
                changed = True
            if str(md.get("cache_version", "")) != str(args.cache_version):
                md["cache_version"] = str(args.cache_version)
                changed = True
            if manifest_hash and str(md.get("manifest_hash", "")) != manifest_hash:
                md["manifest_hash"] = manifest_hash
                changed = True
            if not str(md.get("frontend_hash", "")).strip():
                md["frontend_hash"] = str(rec.get("frontend_hash", ""))
                changed = True
            rec["metadata"] = md

            upgraded_records += 1

        if not changed and not bool(args.inplace):
            changed = True

        if changed:
            dst_shard = out_dir / shard_rel
            dst_shard.parent.mkdir(parents=True, exist_ok=True)
            payload["schema_version"] = str(args.cache_version)
            torch.save(payload, dst_shard)
            upgraded_shards += 1

    index_payload["schema_version"] = str(args.cache_version)
    index_payload["upgraded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    index_payload["manifest_hash"] = manifest_hash
    index_payload["teacher_backend_tag"] = str(args.teacher_backend_tag)
    index_payload["cache_version"] = str(args.cache_version)

    out_index = out_dir / "index.json"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_index.write_text(json.dumps(index_payload, indent=2))

    summary = {
        "cache_dir": str(cache_dir),
        "output_dir": str(out_dir),
        "index": str(out_index),
        "manifest": str(manifest_path),
        "manifest_hash": manifest_hash,
        "cache_version": str(args.cache_version),
        "teacher_backend_tag": str(args.teacher_backend_tag),
        "upgraded_shards": int(upgraded_shards),
        "visited_records": int(upgraded_records),
        "inplace": bool(args.inplace),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
