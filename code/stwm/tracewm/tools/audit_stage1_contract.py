#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


DATA_ROOT = Path("/home/chen034/workspace/data")
STWM_ROOT = Path("/home/chen034/workspace/stwm")

IN_STAGE1_AUDIT_JSON = DATA_ROOT / "_manifests" / "stage1_dataset_audit_20260407.json"
IN_STAGE1_AUDIT_MD = DATA_ROOT / "_manifests" / "stage1_dataset_audit_20260407.md"
IN_POINT_HARD_AFTER = DATA_ROOT / "_manifests" / "pointodyssey_hard_complete_after_20260407.json"
IN_KUBRIC_HARD_AFTER = DATA_ROOT / "_manifests" / "kubric_hard_complete_after_20260407.json"
IN_HARD_SUMMARY = DATA_ROOT / "_manifests" / "hard_complete_summary_20260407.json"

OUT_CONTRACT = DATA_ROOT / "_manifests" / "stage1_data_contract_20260408.json"
OUT_CONTRACT_SMOKE = STWM_ROOT / "reports" / "stage1_contract_smoke_20260408.json"


@dataclass
class ContractDataset:
    dataset_name: str
    role_in_stage1: str
    status: str
    root_path: str
    required_subsets: List[str]
    required_modalities: List[str]
    allowed_missing_items: List[str]
    usable_for_train: bool
    usable_for_eval: bool
    notes: str
    non_blocking_for_stage1: bool
    limited_eval_ready: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, f"missing:{path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"parse_error:{path}:{exc}"


def old_status_map(stage1_audit_payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not stage1_audit_payload:
        return out
    for item in stage1_audit_payload.get("datasets", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("dataset_name", "")).strip()
        status = str(item.get("status", "")).strip()
        if name and status:
            out[name] = status
    return out


def build_contract_entries(
    stage1_audit: Optional[Dict[str, Any]],
    point_after: Optional[Dict[str, Any]],
    kubric_after: Optional[Dict[str, Any]],
    hard_summary: Optional[Dict[str, Any]],
) -> List[ContractDataset]:
    old = old_status_map(stage1_audit)

    point_hard = bool(point_after and point_after.get("hard_complete_passed", False))
    point_counts = {
        "train": point_after.get("train_sequence_count") if point_after else None,
        "val": point_after.get("val_sequence_count") if point_after else None,
        "test": point_after.get("test_sequence_count") if point_after else None,
    }

    use_panning_required = int(kubric_after.get("use_panning_required", 1)) if kubric_after else 1
    kubric_movi_pass = bool(kubric_after and kubric_after.get("movi_e_hard_complete_passed", False))
    kubric_final_pass = bool(kubric_after and kubric_after.get("kubric_final_passed", False))
    kubric_complete = bool(use_panning_required == 0 and kubric_movi_pass and kubric_final_pass)
    panning_status = (
        kubric_after.get("panning_smoke_test", {}).get("status", "unknown") if kubric_after else "unknown"
    )

    tapvid_status = "complete"
    if old.get("tapvid") and old["tapvid"] != "complete":
        tapvid_status = old["tapvid"]

    tapvid3d_status = old.get("tapvid3d", "partial")
    limited_eval_ready = tapvid3d_status in {"partial", "complete"}

    dynamic_replica_status = old.get("dynamic_replica_data", "env_only")
    dynamic_stereo_status = old.get("dynamic_stereo", "env_only")

    out = [
        ContractDataset(
            dataset_name="pointodyssey",
            role_in_stage1="main_training",
            status="complete" if point_hard else old.get("pointodyssey", "partial"),
            root_path=str(DATA_ROOT / "pointodyssey"),
            required_subsets=["train", "val", "test"],
            required_modalities=["rgbs", "depths", "masks", "normals"],
            allowed_missing_items=[
                "top_level_preview_mp4_count_not_used_for_hard_complete",
                "trajectory_like_signal_not_required_for_entry_gate",
            ],
            usable_for_train=bool(point_hard),
            usable_for_eval=False,
            notes=(
                "Upgraded by hard-complete manifest (sequence criterion 131/15/13). "
                f"counts={point_counts}"
            ),
            non_blocking_for_stage1=False,
            limited_eval_ready=False,
        ),
        ContractDataset(
            dataset_name="kubric",
            role_in_stage1="main_training",
            status="complete" if kubric_complete else old.get("kubric", "partial"),
            root_path=str(DATA_ROOT / "kubric"),
            required_subsets=["tfds/movi_e"],
            required_modalities=["movi_e tfrecord shards", "dataset_info.json", "tfds smoke pass"],
            allowed_missing_items=[
                "panning_movi_e failed_smoke_test (non-blocking under movi_e-only standard)",
            ],
            usable_for_train=bool(kubric_complete),
            usable_for_eval=False,
            notes=(
                "movi_e-only standard applied (USE_PANNING_REQUIRED=0). "
                f"movi_pass={kubric_movi_pass}, kubric_final_pass={kubric_final_pass}, panning_status={panning_status}"
            ),
            non_blocking_for_stage1=False,
            limited_eval_ready=False,
        ),
        ContractDataset(
            dataset_name="tapvid",
            role_in_stage1="main_eval",
            status=tapvid_status,
            root_path=str(DATA_ROOT / "tapvid"),
            required_subsets=[
                "davis/tapvid_davis/tapvid_davis.pkl",
                "rgb_stacking/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl",
                "kinetics_labels/tapvid_kinetics/tapvid_kinetics.csv",
                "tapnet_repo/tapnet/tapvid/evaluation_datasets.py",
            ],
            required_modalities=["2d_points", "occluded"],
            allowed_missing_items=["kinetics_raw_videos_missing_non_blocking_for_eval"],
            usable_for_train=False,
            usable_for_eval=(tapvid_status == "complete"),
            notes="Main Stage 1 evaluation dataset.",
            non_blocking_for_stage1=False,
            limited_eval_ready=False,
        ),
        ContractDataset(
            dataset_name="tapvid3d",
            role_in_stage1="limited_eval",
            status=tapvid3d_status,
            root_path=str(DATA_ROOT / "tapvid3d"),
            required_subsets=["minival_dataset"],
            required_modalities=["tracks_XYZ", "visibility", "queries_xyt", "fx_fy_cx_cy"],
            allowed_missing_items=[
                "full_benchmark_scale_not_required_for_stage1_start",
                "missing_extrinsics_w2c_for_some_sources_allowed_in_limited_eval",
            ],
            usable_for_train=False,
            usable_for_eval=limited_eval_ready,
            notes="Limited eval only; full benchmark remains out of current Stage 1 gate.",
            non_blocking_for_stage1=True,
            limited_eval_ready=limited_eval_ready,
        ),
        ContractDataset(
            dataset_name="dynamic_replica_data",
            role_in_stage1="optional_future",
            status=dynamic_replica_status,
            root_path=str(DATA_ROOT / "dynamic_replica_data"),
            required_subsets=[],
            required_modalities=[],
            allowed_missing_items=["full_dynamic_replica_payload_not_required_for_first_wave"],
            usable_for_train=False,
            usable_for_eval=False,
            notes="Non-blocking optional enhancement path.",
            non_blocking_for_stage1=True,
            limited_eval_ready=False,
        ),
        ContractDataset(
            dataset_name="dynamic_stereo",
            role_in_stage1="not_in_first_wave",
            status=dynamic_stereo_status,
            root_path=str(DATA_ROOT / "dynamic_stereo"),
            required_subsets=[],
            required_modalities=[],
            allowed_missing_items=["dynamic_replica_runtime_data_not_required_for_first_wave"],
            usable_for_train=False,
            usable_for_eval=False,
            notes="Repository/env lane only for now.",
            non_blocking_for_stage1=True,
            limited_eval_ready=False,
        ),
    ]

    return out


def subset_exists(root: Path, subset: str) -> bool:
    p = root / subset
    return p.exists()


def run_contract_smoke(contract_payload: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    for item in contract_payload.get("datasets", []):
        dataset_name = item.get("dataset_name")
        root_path = Path(str(item.get("root_path", "")))
        required_subsets = item.get("required_subsets", [])
        required_modalities = item.get("required_modalities", [])
        non_blocking = bool(item.get("non_blocking_for_stage1", True))

        root_exists = root_path.exists() and root_path.is_dir()
        subset_result = {subset: subset_exists(root_path, subset) for subset in required_subsets}

        modality_ok = True
        modality_notes: List[str] = []

        if dataset_name == "pointodyssey":
            split_train = root_path / "train"
            seq_ok = False
            if split_train.exists() and split_train.is_dir():
                for seq in sorted(split_train.iterdir()):
                    if not seq.is_dir():
                        continue
                    if all((seq / m).is_dir() for m in ["rgbs", "depths", "masks", "normals"]):
                        seq_ok = True
                        break
            modality_ok = seq_ok
            if not seq_ok:
                modality_notes.append("no_sequence_with_all_key_modalities")

        elif dataset_name == "kubric":
            movi_dir = root_path / "tfds" / "movi_e"
            tfrecord_count = len(list(movi_dir.rglob("*.tfrecord*"))) if movi_dir.exists() else 0
            info_count = len(list(movi_dir.rglob("dataset_info.json"))) if movi_dir.exists() else 0
            modality_ok = tfrecord_count > 0 and info_count > 0
            if not modality_ok:
                modality_notes.append(f"tfrecord_count={tfrecord_count},dataset_info_count={info_count}")

        elif dataset_name == "tapvid":
            davis = root_path / "davis" / "tapvid_davis" / "tapvid_davis.pkl"
            rgb = root_path / "rgb_stacking" / "tapvid_rgb_stacking" / "tapvid_rgb_stacking.pkl"
            csv = root_path / "kinetics_labels" / "tapvid_kinetics" / "tapvid_kinetics.csv"
            reader = root_path / "tapnet_repo" / "tapnet" / "tapvid" / "evaluation_datasets.py"
            modality_ok = davis.exists() and rgb.exists() and csv.exists() and reader.exists()
            if not modality_ok:
                modality_notes.append("tapvid_required_eval_assets_missing")

        elif dataset_name == "tapvid3d":
            mini = root_path / "minival_dataset"
            npz_count = len(list(mini.rglob("*.npz"))) if mini.exists() else 0
            modality_ok = npz_count > 0
            if not modality_ok:
                modality_notes.append("tapvid3d_minival_npz_missing")

        else:
            modality_ok = True

        dataset_pass = root_exists and all(subset_result.values()) and modality_ok
        checks.append(
            {
                "dataset_name": dataset_name,
                "root_exists": root_exists,
                "subset_checks": subset_result,
                "required_modalities": required_modalities,
                "modality_check_passed": modality_ok,
                "modality_notes": modality_notes,
                "non_blocking_for_stage1": non_blocking,
                "passed": dataset_pass,
            }
        )

    blocking_failures = [c for c in checks if (not c["non_blocking_for_stage1"]) and (not c["passed"])]
    all_passed = len(blocking_failures) == 0

    return {
        "generated_at_utc": now_iso(),
        "all_passed": all_passed,
        "blocking_failures": blocking_failures,
        "checks": checks,
    }


def main() -> int:
    print(f"[contract] start: {now_iso()}")

    stage1_audit, err_audit = load_json(IN_STAGE1_AUDIT_JSON)
    point_after, err_point = load_json(IN_POINT_HARD_AFTER)
    kubric_after, err_kubric = load_json(IN_KUBRIC_HARD_AFTER)
    hard_summary, err_summary = load_json(IN_HARD_SUMMARY)

    load_warnings = [e for e in [err_audit, err_point, err_kubric, err_summary] if e]

    datasets = build_contract_entries(stage1_audit, point_after, kubric_after, hard_summary)

    main_training_ready = all(
        d.status == "complete" and d.usable_for_train for d in datasets if d.role_in_stage1 == "main_training"
    )
    main_eval_ready = all(
        d.status == "complete" and d.usable_for_eval for d in datasets if d.role_in_stage1 == "main_eval"
    )

    has_limited_eval_partial = any(
        d.role_in_stage1 == "limited_eval" and d.status in {"partial", "unknown"} for d in datasets
    )

    if main_training_ready and main_eval_ready and has_limited_eval_partial:
        final_recommendation = "GO_WITH_LIMITATIONS"
    elif main_training_ready and main_eval_ready:
        final_recommendation = "GO"
    elif main_training_ready or main_eval_ready:
        final_recommendation = "GO_WITH_LIMITATIONS"
    else:
        final_recommendation = "NO_GO"

    contract_payload: Dict[str, Any] = {
        "generated_at_utc": now_iso(),
        "data_root": str(DATA_ROOT),
        "decision": {
            "mode": "GO_WITH_LIMITATIONS",
            "policy_notes": [
                "PointOdyssey + Kubric + TAP-Vid are launch-critical",
                "TAPVid-3D is limited eval only",
                "DynamicReplica path is non-blocking optional",
            ],
        },
        "inputs": {
            "stage1_dataset_audit_json": str(IN_STAGE1_AUDIT_JSON),
            "stage1_dataset_audit_md": str(IN_STAGE1_AUDIT_MD),
            "pointodyssey_hard_complete_after": str(IN_POINT_HARD_AFTER),
            "kubric_hard_complete_after": str(IN_KUBRIC_HARD_AFTER),
            "hard_complete_summary": str(IN_HARD_SUMMARY),
            "load_warnings": load_warnings,
        },
        "datasets": [
            {
                "dataset_name": d.dataset_name,
                "role_in_stage1": d.role_in_stage1,
                "status": d.status,
                "root_path": d.root_path,
                "required_subsets": d.required_subsets,
                "required_modalities": d.required_modalities,
                "allowed_missing_items": d.allowed_missing_items,
                "usable_for_train": d.usable_for_train,
                "usable_for_eval": d.usable_for_eval,
                "notes": d.notes,
                "non_blocking_for_stage1": d.non_blocking_for_stage1,
                "limited_eval_ready": d.limited_eval_ready,
            }
            for d in datasets
        ],
        "summary": {
            "stage1_main_training_ready": main_training_ready,
            "stage1_main_eval_ready": main_eval_ready,
            "final_recommendation": final_recommendation,
        },
    }

    smoke = run_contract_smoke(contract_payload)

    OUT_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    OUT_CONTRACT_SMOKE.parent.mkdir(parents=True, exist_ok=True)
    OUT_CONTRACT.write_text(json.dumps(contract_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_CONTRACT_SMOKE.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[contract] wrote: {OUT_CONTRACT}")
    print(f"[contract] wrote: {OUT_CONTRACT_SMOKE}")
    print(f"[contract] stage1_main_training_ready={main_training_ready}")
    print(f"[contract] stage1_main_eval_ready={main_eval_ready}")
    print(f"[contract] final_recommendation={final_recommendation}")
    print(f"[contract] smoke_all_passed={smoke['all_passed']}")
    print(f"[contract] end: {now_iso()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
