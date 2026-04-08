from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

from .constants import TRACE_CONTRACT_PATH


@dataclass(frozen=True)
class TraceCacheDatasetContract:
    dataset_name: str
    cache_root: str
    index_path: str
    source_root: str
    track_source: str
    enabled: bool


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_contract_payload(
    generated_at_utc: str,
    schema_version: str,
    feature_layout: Dict[str, int],
    dataset_entries: Iterable[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    datasets: List[Dict[str, Any]] = []
    for item in dataset_entries:
        if not isinstance(item, dict):
            continue
        datasets.append(dict(item))

    return {
        "generated_at_utc": generated_at_utc,
        "schema_version": schema_version,
        "feature_layout": dict(feature_layout),
        "datasets": datasets,
        "summary": dict(summary),
    }


def save_contract(payload: Dict[str, Any], path: Path | None = None) -> Path:
    out = path if path is not None else TRACE_CONTRACT_PATH
    save_json(payload, out)
    return out


def load_contract(path: Path | None = None) -> Dict[str, Any]:
    resolved = path if path is not None else TRACE_CONTRACT_PATH
    return load_json(resolved)


def dataset_index(contract_payload: Dict[str, Any]) -> Dict[str, TraceCacheDatasetContract]:
    out: Dict[str, TraceCacheDatasetContract] = {}
    for raw in contract_payload.get("datasets", []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("dataset_name", "")).strip()
        if not name:
            continue
        out[name] = TraceCacheDatasetContract(
            dataset_name=name,
            cache_root=str(raw.get("cache_root", "")),
            index_path=str(raw.get("index_path", "")),
            source_root=str(raw.get("source_root", "")),
            track_source=str(raw.get("track_source", "")),
            enabled=bool(raw.get("enabled", False)),
        )
    return out
