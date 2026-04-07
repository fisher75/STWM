from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import os

DEFAULT_CONTRACT_PATH = Path("/home/chen034/workspace/data/_manifests/stage1_data_contract_20260408.json")


@dataclass(frozen=True)
class DatasetContractEntry:
    dataset_name: str
    role_in_stage1: str
    status: str
    root_path: str
    required_subsets: list[str]
    required_modalities: list[str]
    allowed_missing_items: list[str]
    usable_for_train: bool
    usable_for_eval: bool
    notes: str


def resolve_contract_path(contract_path: str | Path | None = None) -> Path:
    if contract_path is not None:
        return Path(contract_path)
    env_path = os.environ.get("STAGE1_DATA_CONTRACT_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_CONTRACT_PATH


def load_stage1_contract(contract_path: str | Path | None = None) -> Dict[str, Any]:
    path = resolve_contract_path(contract_path)
    if not path.exists():
        raise FileNotFoundError(f"stage1 data contract not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "datasets" not in payload or not isinstance(payload["datasets"], list):
        raise ValueError(f"invalid stage1 data contract payload: {path}")
    payload["_contract_path"] = str(path)
    return payload


def _to_entry(raw: Dict[str, Any]) -> DatasetContractEntry:
    return DatasetContractEntry(
        dataset_name=str(raw.get("dataset_name", "")),
        role_in_stage1=str(raw.get("role_in_stage1", "not_in_first_wave")),
        status=str(raw.get("status", "unknown")),
        root_path=str(raw.get("root_path", "")),
        required_subsets=[str(x) for x in raw.get("required_subsets", [])],
        required_modalities=[str(x) for x in raw.get("required_modalities", [])],
        allowed_missing_items=[str(x) for x in raw.get("allowed_missing_items", [])],
        usable_for_train=bool(raw.get("usable_for_train", False)),
        usable_for_eval=bool(raw.get("usable_for_eval", False)),
        notes=str(raw.get("notes", "")),
    )


def contract_index(contract_payload: Dict[str, Any]) -> Dict[str, DatasetContractEntry]:
    out: Dict[str, DatasetContractEntry] = {}
    for item in contract_payload.get("datasets", []):
        if not isinstance(item, dict):
            continue
        entry = _to_entry(item)
        if entry.dataset_name:
            out[entry.dataset_name] = entry
    return out


def get_dataset_entry(contract_payload: Dict[str, Any], dataset_name: str) -> DatasetContractEntry | None:
    return contract_index(contract_payload).get(dataset_name)


def require_dataset_entry(contract_payload: Dict[str, Any], dataset_name: str) -> DatasetContractEntry:
    entry = get_dataset_entry(contract_payload, dataset_name)
    if entry is None:
        raise KeyError(f"dataset not found in contract: {dataset_name}")
    return entry


def stage1_ready_flags(contract_payload: Dict[str, Any]) -> Dict[str, bool]:
    summary = contract_payload.get("summary", {})
    return {
        "main_training_ready": bool(summary.get("stage1_main_training_ready", False)),
        "main_eval_ready": bool(summary.get("stage1_main_eval_ready", False)),
    }
