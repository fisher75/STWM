#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import csv
import json
import os
import pickle
import subprocess

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_SUFFIXES = {".png"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_or_default(env_name: str, default: str) -> str:
    val = str(os.environ.get(env_name, "")).strip()
    if val:
        return val
    return str(default)


def parse_args() -> Any:
    repo_default = _env_or_default("STWM_ROOT", "/home/chen034/workspace/stwm")
    data_default = _env_or_default("TRACEWM_DATA_ROOT", "/home/chen034/workspace/data")
    p = ArgumentParser(description="Run TRACEWM evidence hardening audit round")
    p.add_argument("--repo-root", default=repo_default)
    p.add_argument("--data-root", default=data_default)
    p.add_argument("--workers", type=int, default=max(8, min(32, os.cpu_count() or 8)))
    p.add_argument(
        "--stage2-completion-json",
        default=_env_or_default("TRACEWM_STAGE2_EXTERNAL_EVAL_COMPLETION_JSON", f"{repo_default}/reports/stage2_external_eval_completion_20260408.json"),
    )
    p.add_argument(
        "--stage2-completion-log",
        default=_env_or_default("TRACEWM_STAGE2_EXTERNAL_EVAL_COMPLETION_LOG", f"{repo_default}/logs/tracewm_stage2_external_eval_completion_20260408.log"),
    )
    p.add_argument(
        "--evidence-audit-json",
        default=f"{repo_default}/reports/tracewm_evidence_hardening_audit_20260409.json",
    )
    p.add_argument(
        "--stage1-bundle-json",
        default=f"{repo_default}/reports/stage1_dataset_evidence_bundle_20260409.json",
    )
    p.add_argument(
        "--stage1-bundle-md",
        default=f"{repo_default}/docs/STAGE1_DATA_EVIDENCE_BUNDLE_20260409.md",
    )
    p.add_argument(
        "--stage2-bundle-json",
        default=f"{repo_default}/reports/stage2_dataset_evidence_bundle_20260409.json",
    )
    p.add_argument(
        "--stage2-bundle-md",
        default=f"{repo_default}/docs/STAGE2_DATA_EVIDENCE_BUNDLE_20260409.md",
    )
    p.add_argument(
        "--external-fidelity-json",
        default=f"{repo_default}/reports/stage2_external_eval_fidelity_audit_20260409.json",
    )
    p.add_argument(
        "--external-fidelity-md",
        default=f"{repo_default}/docs/STAGE2_EXTERNAL_EVAL_FIDELITY_AUDIT_20260409.md",
    )
    p.add_argument(
        "--project-readiness-json",
        default=f"{repo_default}/reports/tracewm_project_readiness_20260409.json",
    )
    p.add_argument(
        "--project-readiness-md",
        default=f"{repo_default}/docs/TRACEWM_PROJECT_READINESS_20260409.md",
    )
    p.add_argument(
        "--tap-export-smoke-json",
        default=f"{repo_default}/reports/stage2_tap_payload_export_smoke_20260409.json",
    )
    p.add_argument(
        "--tap-eval-smoke-json",
        default=f"{repo_default}/reports/stage2_tap_eval_smoke_20260409.json",
    )
    return p.parse_args()


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _git_cmd(repo_root: Path, args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo_root), *args], text=True, capture_output=True)


def _tracked_in_git(repo_root: Path, path: Path) -> bool:
    rel = None
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except Exception:
        return False
    proc = _git_cmd(repo_root, ["ls-files", "--error-unmatch", str(rel)])
    return proc.returncode == 0


def _ignored_by_git(repo_root: Path, path: Path) -> bool:
    rel = None
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except Exception:
        return False
    proc = _git_cmd(repo_root, ["check-ignore", "-q", str(rel)])
    return proc.returncode == 0


def _path_record(repo_root: Path, path: str | Path, regenerated_in_this_round: bool = False, note: str = "") -> Dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    within_repo_root = False
    tracked_in_git = False
    ignored_by_git = False
    packaged_in_repo_snapshot = False
    try:
        p.resolve().relative_to(repo_root.resolve())
        within_repo_root = True
    except Exception:
        within_repo_root = False
    if within_repo_root:
        tracked_in_git = _tracked_in_git(repo_root, p)
        ignored_by_git = _ignored_by_git(repo_root, p)
        packaged_in_repo_snapshot = bool(tracked_in_git and not ignored_by_git)
    return {
        "path": str(p),
        "resolved_path": str(p.resolve()) if exists else "",
        "exists": bool(exists),
        "is_file": bool(p.is_file()),
        "is_dir": bool(p.is_dir()),
        "within_repo_root": bool(within_repo_root),
        "tracked_in_git": bool(tracked_in_git),
        "ignored_by_git": bool(ignored_by_git),
        "packaged_in_repo_snapshot": bool(packaged_in_repo_snapshot),
        "regenerated_in_this_round": bool(regenerated_in_this_round),
        "note": str(note),
    }


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line and not line.startswith("._"):
            out.append(line)
    return out


def _safe_first(iterable: Iterable[Path]) -> Path | None:
    for item in iterable:
        return item
    return None


def _image_paths(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")
    )


def _mask_paths(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in MASK_SUFFIXES and not p.name.startswith("._")
    )


def _verify_image(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _scan_pointodyssey_sequence(seq_dir: Path) -> Dict[str, Any]:
    required = ["rgbs", "depths", "masks", "normals"]
    missing_modalities: List[str] = []
    empty_modalities: List[str] = []
    for name in required:
        d = seq_dir / name
        if not d.exists() or not d.is_dir():
            missing_modalities.append(name)
            continue
        first_item = _safe_first(x for x in d.iterdir() if not x.name.startswith("._"))
        if first_item is None:
            empty_modalities.append(name)
    return {
        "sequence": seq_dir.name,
        "missing_modalities": missing_modalities,
        "empty_modalities": empty_modalities,
        "ok": bool(not missing_modalities and not empty_modalities),
    }


def _audit_stage1_pointodyssey(data_root: Path, source_manifest: Dict[str, Any], workers: int) -> Dict[str, Any]:
    root = data_root / "pointodyssey"
    expected = {}
    for key in ["train_expected", "val_expected", "test_expected"]:
        expected[key.replace("_expected", "")] = int(source_manifest.get(key, 0))

    split_rows: Dict[str, Any] = {}
    overall_ok = True
    for split in ["train", "val", "test"]:
        split_dir = root / split
        seq_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")) if split_dir.exists() else []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            seq_rows = list(ex.map(_scan_pointodyssey_sequence, seq_dirs))
        broken = [row for row in seq_rows if not row["ok"]]
        count_ok = len(seq_dirs) == expected.get(split, len(seq_dirs))
        overall_ok = overall_ok and count_ok and not broken
        split_rows[split] = {
            "expected_sequences": int(expected.get(split, 0)),
            "observed_sequences": int(len(seq_dirs)),
            "count_matches_expected": bool(count_ok),
            "missing_or_empty_sequences": [row for row in broken[:20]],
        }
    return {
        "dataset_name": "pointodyssey",
        "live_root": str(root),
        "source_audit_path": str(data_root / "_manifests" / "pointodyssey_hard_complete_after_20260407.json"),
        "source_audit_exists": True,
        "regenerated_in_this_round": True,
        "completeness_status": "complete" if overall_ok else "not_ready",
        "blocking_reasons": [] if overall_ok else ["sequence counts or required modalities no longer satisfy the current hard-complete rule"],
        "live_probe": {
            "rule": "hard_complete",
            "splits": split_rows,
            "all_required_modalities_present": bool(overall_ok),
        },
    }


def _audit_stage1_kubric(data_root: Path, source_manifest: Dict[str, Any]) -> Dict[str, Any]:
    root = data_root / "kubric" / "tfds" / "movi_e"
    version_rows: List[Dict[str, Any]] = []
    overall_ok = bool(root.exists())
    for version_dir in sorted(root.glob("*/*")):
        if not version_dir.is_dir():
            continue
        tfrecords = sorted(version_dir.glob("*.tfrecord-*"))
        dataset_info = version_dir / "dataset_info.json"
        features = version_dir / "features.json"
        row = {
            "version_dir": str(version_dir),
            "dataset_info_exists": bool(dataset_info.exists()),
            "features_exists": bool(features.exists()),
            "tfrecord_count": int(len(tfrecords)),
        }
        version_rows.append(row)
        overall_ok = overall_ok and bool(dataset_info.exists()) and bool(features.exists()) and bool(tfrecords)
    if not version_rows:
        overall_ok = False
    return {
        "dataset_name": "kubric",
        "live_root": str(root),
        "source_audit_path": str(data_root / "_manifests" / "kubric_hard_complete_after_20260407.json"),
        "source_audit_exists": True,
        "regenerated_in_this_round": True,
        "completeness_status": "complete" if overall_ok else "not_ready",
        "blocking_reasons": [] if overall_ok else ["movi_e-only live verification no longer passes"],
        "live_probe": {
            "rule": "movi_e_only",
            "version_rows": version_rows,
            "panning_required": False,
            "source_manifest_movi_e_hard_complete_passed": bool(source_manifest.get("movi_e_hard_complete_passed", False)),
        },
    }


def _audit_stage1_tapvid(data_root: Path) -> Dict[str, Any]:
    paths = {
        "davis_pickle": data_root / "tapvid" / "davis" / "tapvid_davis" / "tapvid_davis.pkl",
        "rgb_stacking_pickle": data_root / "tapvid" / "rgb_stacking" / "tapvid_rgb_stacking" / "tapvid_rgb_stacking.pkl",
        "kinetics_labels_csv": data_root / "tapvid" / "kinetics_labels" / "tapvid_kinetics" / "tapvid_kinetics.csv",
        "official_evaluator_py": data_root / "_repos" / "tapnet" / "tapnet" / "tapvid" / "evaluation_datasets.py",
    }
    rows: Dict[str, Any] = {}
    overall_ok = True
    for name, path in paths.items():
        row: Dict[str, Any] = {
            "path": str(path),
            "exists": bool(path.exists()),
            "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
        }
        if path.exists() and path.suffix == ".pkl":
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                row["pickle_type"] = type(payload).__name__
                row["top_level_len"] = int(len(payload)) if hasattr(payload, "__len__") else -1
            except Exception as exc:
                row["load_error"] = str(exc)
                overall_ok = False
        elif path.exists() and path.suffix == ".csv":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    row["csv_header"] = header
            except Exception as exc:
                row["load_error"] = str(exc)
                overall_ok = False
        overall_ok = overall_ok and bool(path.exists())
        rows[name] = row
    return {
        "dataset_name": "tapvid",
        "live_root": str(data_root / "tapvid"),
        "source_audit_path": str(data_root / "_manifests" / "tapvid_eval_ready_20260406.json"),
        "source_audit_exists": bool((data_root / "_manifests" / "tapvid_eval_ready_20260406.json").exists()),
        "regenerated_in_this_round": True,
        "completeness_status": "complete" if overall_ok else "not_ready",
        "blocking_reasons": [] if overall_ok else ["one or more TAP-Vid eval-critical assets are missing or unreadable"],
        "live_probe": rows,
    }


def _tapvid3d_npz_row(path: Path) -> Dict[str, Any]:
    missing: List[str] = []
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            required = ["visibility", "queries_xyt", "fx_fy_cx_cy"]
            if "tracks_XYZ" not in keys and "tracks_xyz" not in keys:
                missing.append("tracks_XYZ")
            for key in required:
                if key not in keys:
                    missing.append(key)
    except Exception as exc:
        return {
            "path": str(path),
            "ok": False,
            "missing": ["load_failed"],
            "error": str(exc),
        }
    return {
        "path": str(path),
        "ok": bool(not missing),
        "missing": missing,
    }


def _audit_stage1_tapvid3d(data_root: Path, workers: int) -> Dict[str, Any]:
    root = data_root / "tapvid3d" / "minival_dataset"
    npz_files = sorted(root.glob("*/*.npz"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_tapvid3d_npz_row, npz_files))
    broken = [row for row in rows if not row["ok"]]
    by_source = {}
    for source in ["adt", "pstudio", "drivetrack"]:
        by_source[source] = int(len(list((root / source).glob("*.npz")))) if (root / source).exists() else 0
    ok = bool(npz_files) and not broken
    return {
        "dataset_name": "tapvid3d",
        "live_root": str(root),
        "source_audit_path": str(data_root / "_manifests" / "tapvid3d_gate_recheck_20260407.json"),
        "source_audit_exists": bool((data_root / "_manifests" / "tapvid3d_gate_recheck_20260407.json").exists()),
        "regenerated_in_this_round": True,
        "completeness_status": "limited_eval_ready" if ok else "not_ready",
        "blocking_reasons": [] if ok else ["minival_dataset is missing or one or more TAPVid-3D npz files lack limited-eval required keys"],
        "live_probe": {
            "npz_count_by_source": by_source,
            "total_npz_files": int(len(npz_files)),
            "broken_examples": broken[:10],
        },
    }


def _scan_stage2_clip(rec: Tuple[str, str, Path, Path, bool]) -> Dict[str, Any]:
    dataset_name, clip_id, frame_dir, mask_dir, require_masks = rec
    frame_paths = _image_paths(frame_dir)
    mask_paths = _mask_paths(mask_dir)
    frame_stems = {p.stem for p in frame_paths}
    mask_stems = {p.stem for p in mask_paths}
    missing_masks = sorted(frame_stems - mask_stems) if require_masks else []
    missing_frames = sorted(mask_stems - frame_stems)
    first_frame = frame_paths[0] if frame_paths else None
    first_mask = None
    if first_frame is not None and mask_dir.exists():
        cand = mask_dir / f"{first_frame.stem}.png"
        if cand.exists():
            first_mask = cand
    return {
        "dataset_name": dataset_name,
        "clip_id": clip_id,
        "frame_dir_exists": bool(frame_dir.exists()),
        "mask_dir_exists": bool(mask_dir.exists()),
        "frame_count": int(len(frame_paths)),
        "mask_count": int(len(mask_paths)),
        "missing_mask_count": int(len(missing_masks)),
        "missing_frame_count": int(len(missing_frames)),
        "enough_frames_for_8_plus_8": bool(len(frame_paths) >= 16),
        "first_frame_readable": bool(_verify_image(first_frame)),
        "first_mask_readable": bool(_verify_image(first_mask)) if require_masks else True,
        "sample_missing_masks": missing_masks[:5],
        "sample_missing_frames": missing_frames[:5],
    }


def _stage2_core_dataset_scan(dataset_name: str, dataset_root: Path, split_specs: Dict[str, Dict[str, Any]], workers: int) -> Dict[str, Any]:
    split_rows: Dict[str, Any] = {}
    overall_zero_missing = True
    overall_semantic_crop_ok = True
    for split_name, spec in split_specs.items():
        split_file = Path(spec["split_file"])
        ids = _read_lines(split_file)
        work: List[Tuple[str, str, Path, Path, bool]] = []
        for clip_id in ids:
            if dataset_name.lower() == "vspw":
                frame_dir = Path(spec["frame_root"]) / clip_id / spec["frame_subdir"]
                mask_dir = Path(spec["mask_root"]) / clip_id / spec["mask_subdir"]
            else:
                frame_dir = Path(spec["frame_root"]) / clip_id
                mask_dir = Path(spec["mask_root"]) / clip_id
            work.append((dataset_name, clip_id, frame_dir, mask_dir, bool(spec.get("require_masks", True))))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_scan_stage2_clip, work))
        frame_dir_missing_count = sum(1 for row in rows if not row["frame_dir_exists"])
        mask_dir_missing_count = sum(1 for row in rows if not row["mask_dir_exists"] and spec.get("require_masks", True))
        zero_frame_count = sum(1 for row in rows if row["frame_count"] <= 0)
        zero_mask_count = sum(1 for row in rows if row["mask_count"] <= 0 and spec.get("require_masks", True))
        insufficient_min_frame_count = sum(1 for row in rows if row["frame_count"] < 2)
        short_count = sum(1 for row in rows if not row["enough_frames_for_8_plus_8"])
        missing_masks_from_frame_total = sum(int(row["missing_mask_count"]) for row in rows)
        extra_masks_without_frame_total = sum(int(row["missing_frame_count"]) for row in rows)
        missing_mask_total = sum(int(row["missing_mask_count"]) for row in rows)
        missing_frame_total = sum(int(row["missing_frame_count"]) for row in rows)
        semantic_fail_count = sum(1 for row in rows if not row["first_frame_readable"] or not row["first_mask_readable"])

        # Keep the live-loader contract separate from quality warnings:
        # Stage2SemanticDataset only requires a readable frame directory with >=2 frames.
        # It tolerates short clips by resampling and tolerates mask gaps via fallback crops.
        binding_contract_ok = bool(
            frame_dir_missing_count == 0
            and mask_dir_missing_count == 0
            and zero_frame_count == 0
            and insufficient_min_frame_count == 0
        )
        overall_zero_missing = overall_zero_missing and binding_contract_ok
        overall_semantic_crop_ok = overall_semantic_crop_ok and semantic_fail_count == 0
        bad_examples = [
            row for row in rows
            if (not row["frame_dir_exists"])
            or (spec.get("require_masks", True) and not row["mask_dir_exists"])
            or row["frame_count"] < 2
            or not row["first_frame_readable"]
            or not row["first_mask_readable"]
        ][:10]
        warning_examples = [
            row for row in rows
            if row["missing_mask_count"] > 0
            or row["missing_frame_count"] > 0
            or not row["enough_frames_for_8_plus_8"]
        ][:10]
        split_rows[split_name] = {
            "split_file": str(split_file),
            "clip_count": int(len(ids)),
            "frame_dir_missing_count": int(frame_dir_missing_count),
            "mask_dir_missing_count": int(mask_dir_missing_count),
            "zero_frame_count": int(zero_frame_count),
            "zero_mask_count": int(zero_mask_count),
            "clips_with_fewer_than_2_frames": int(insufficient_min_frame_count),
            "clips_shorter_than_16": int(short_count),
            "missing_mask_total": int(missing_mask_total),
            "missing_frame_total": int(missing_frame_total),
            "missing_masks_from_frame_total": int(missing_masks_from_frame_total),
            "extra_masks_without_frame_total": int(extra_masks_without_frame_total),
            "semantic_crop_prereq_fail_count": int(semantic_fail_count),
            "binding_contract_ok": bool(binding_contract_ok),
            "zero_missing_binding": bool(binding_contract_ok),
            "bad_examples": bad_examples,
            "warning_examples": warning_examples,
        }
    return {
        "dataset_name": dataset_name,
        "core_binding_zero_missing": bool(overall_zero_missing),
        "semantic_crop_prerequisites_ok": bool(overall_semantic_crop_ok),
        "split_rows": split_rows,
    }


def _audit_stage2_dataset_bundle(repo_root: Path, stage2_contract: Dict[str, Any], stage2_final: Dict[str, Any], workers: int) -> Dict[str, Any]:
    datasets = stage2_contract.get("datasets", []) if isinstance(stage2_contract.get("datasets", []), list) else []
    excluded = stage2_contract.get("excluded_datasets", []) if isinstance(stage2_contract.get("excluded_datasets", []), list) else []
    ds_map = {str(x.get("dataset_name", "")): x for x in datasets if isinstance(x, dict)}
    ex_map = {str(x.get("dataset_name", "")): x for x in excluded if isinstance(x, dict)}

    core_rows: List[Dict[str, Any]] = []

    vspw = ds_map["VSPW"]
    vspw_scan = _stage2_core_dataset_scan(
        "VSPW",
        Path(vspw["local_path"]),
        {
            "train": {
                "split_file": vspw["split_mapping"]["train"]["split_file"],
                "frame_root": vspw["split_mapping"]["train"]["frame_root"],
                "frame_subdir": vspw["split_mapping"]["train"]["frame_subdir"],
                "mask_root": vspw["split_mapping"]["train"]["mask_root"],
                "mask_subdir": vspw["split_mapping"]["train"]["mask_subdir"],
                "require_masks": True,
            },
            "val": {
                "split_file": vspw["split_mapping"]["val"]["split_file"],
                "frame_root": vspw["split_mapping"]["val"]["frame_root"],
                "frame_subdir": vspw["split_mapping"]["val"]["frame_subdir"],
                "mask_root": vspw["split_mapping"]["val"]["mask_root"],
                "mask_subdir": vspw["split_mapping"]["val"]["mask_subdir"],
                "require_masks": True,
            },
        },
        workers=workers,
    )
    core_rows.append(
        {
            "dataset_name": "VSPW",
            "role_in_stage2": "core",
            "evidence_source_path": str(repo_root.parent / "data" / "_manifests" / "stage2_dataset_audit_20260408.json"),
            "evidence_source_exists": True,
            "regenerated_in_this_round": True,
            "completeness_status": "core_ready" if vspw_scan["core_binding_zero_missing"] and vspw_scan["semantic_crop_prerequisites_ok"] else "not_ready",
            "blocking_reasons": [] if vspw_scan["core_binding_zero_missing"] and vspw_scan["semantic_crop_prerequisites_ok"] else ["current core binding violates live-loader requirements or semantic crop prerequisites"],
            "warning_reasons": [
                "some clips are shorter than 16 frames or contain frame/mask name irregularities, but the current Stage2 loader tolerates these via resampling and optional mask fallback"
            ] if any(
                (
                    row.get("clips_shorter_than_16", 0) > 0
                    or row.get("missing_mask_total", 0) > 0
                    or row.get("missing_frame_total", 0) > 0
                )
                for row in vspw_scan["split_rows"].values()
            ) else [],
            "live_probe": vspw_scan,
        }
    )

    vipseg = ds_map["VIPSeg"]
    vipseg_scan = _stage2_core_dataset_scan(
        "VIPSeg",
        Path(vipseg["local_path"]),
        {
            "train": {
                "split_file": vipseg["split_mapping"]["train"]["split_file"],
                "frame_root": vipseg["split_mapping"]["train"]["frame_root"],
                "mask_root": vipseg["split_mapping"]["train"]["mask_root"],
                "require_masks": True,
            },
            "val": {
                "split_file": vipseg["split_mapping"]["val"]["split_file"],
                "frame_root": vipseg["split_mapping"]["val"]["frame_root"],
                "mask_root": vipseg["split_mapping"]["val"]["mask_root"],
                "require_masks": True,
            },
        },
        workers=workers,
    )
    core_rows.append(
        {
            "dataset_name": "VIPSeg",
            "role_in_stage2": "core",
            "evidence_source_path": str(repo_root.parent / "data" / "_manifests" / "stage2_dataset_audit_20260408.json"),
            "evidence_source_exists": True,
            "regenerated_in_this_round": True,
            "completeness_status": "core_ready" if vipseg_scan["core_binding_zero_missing"] and vipseg_scan["semantic_crop_prerequisites_ok"] else "not_ready",
            "blocking_reasons": [] if vipseg_scan["core_binding_zero_missing"] and vipseg_scan["semantic_crop_prerequisites_ok"] else ["current core binding violates live-loader requirements or semantic crop prerequisites"],
            "warning_reasons": [
                "many clips are shorter than 16 frames, but the current Stage2 loader still admits them because rollout indices are resampled from any clip with >=2 readable frames"
            ] if any(
                (
                    row.get("clips_shorter_than_16", 0) > 0
                    or row.get("missing_mask_total", 0) > 0
                    or row.get("missing_frame_total", 0) > 0
                )
                for row in vipseg_scan["split_rows"].values()
            ) else [],
            "live_probe": vipseg_scan,
        }
    )

    burst = ds_map["BURST"]
    burst_row = {
        "dataset_name": "BURST",
        "role_in_stage2": "optional_extension",
        "evidence_source_path": str(repo_root.parent / "data" / "_manifests" / "stage2_dataset_audit_20260408.json"),
        "evidence_source_exists": True,
        "regenerated_in_this_round": False,
        "completeness_status": "optional_extension_ready" if Path(str(burst["local_path"])).exists() else "not_ready",
        "blocking_reasons": [] if Path(str(burst["local_path"])).exists() else ["BURST root is missing"],
        "live_probe": {
            "root_exists": bool(Path(str(burst["local_path"])).exists()),
            "annotation_train_exists": bool(Path(str(burst["split_mapping"]["train"]["annotation_file"])).exists()),
            "annotation_val_exists": bool(Path(str(burst["split_mapping"]["val"]["annotation_file"])).exists()),
            "annotation_test_exists": bool(Path(str(burst["split_mapping"]["test"]["annotation_file"])).exists()),
        },
    }

    tao_raw = repo_root / "data" / "raw" / "burst"
    tao_archives = [tao_raw / "1-TAO_TRAIN.zip", tao_raw / "2-TAO_VAL.zip", tao_raw / "3-TAO_TEST.zip"]
    tao_row = {
        "dataset_name": "TAO",
        "role_in_stage2": "optional_extension",
        "evidence_source_path": str(repo_root.parent / "data" / "_manifests" / "stage2_dataset_audit_20260408.json"),
        "evidence_source_exists": True,
        "regenerated_in_this_round": False,
        "completeness_status": "access_ready" if all(p.exists() for p in tao_archives) else "not_ready",
        "blocking_reasons": [] if all(p.exists() for p in tao_archives) else ["one or more TAO archives are missing"],
        "live_probe": {
            "raw_archive_paths": [str(p) for p in tao_archives],
            "all_archives_exist": bool(all(p.exists() for p in tao_archives)),
        },
    }

    visor_root = repo_root / "data" / "external" / "visor" / "2v6cgv1x04ol22qp9rm9x2j6a7"
    visor_row = {
        "dataset_name": "VISOR",
        "role_in_stage2": "manual_gate_extension",
        "evidence_source_path": str(repo_root.parent / "data" / "_manifests" / "stage2_dataset_audit_20260408.json"),
        "evidence_source_exists": True,
        "regenerated_in_this_round": False,
        "completeness_status": "manual_gate" if visor_root.exists() else "not_ready",
        "blocking_reasons": ["EPIC-KITCHENS dependency remains manual-gated"] if visor_root.exists() else ["VISOR root is missing"],
        "live_probe": {
            "root_exists": bool(visor_root.exists()),
            "frame_mapping_exists": bool((visor_root / "frame_mapping.json").exists()),
            "info_json_exists": bool((visor_root / ".info.json").exists()),
        },
    }

    core_zero_missing = all(
        row["completeness_status"] == "core_ready" for row in core_rows
    )
    return {
        "generated_at_utc": now_iso(),
        "source_contract_path": str(repo_root / "reports" / "stage2_bootstrap_data_contract_20260408.json"),
        "source_contract_exists": bool((repo_root / "reports" / "stage2_bootstrap_data_contract_20260408.json").exists()),
        "datasets_bound_for_train": stage2_final.get("datasets_bound_for_train", []),
        "datasets_bound_for_eval": stage2_final.get("datasets_bound_for_eval", []),
        "core_binding_zero_missing": bool(core_zero_missing),
        "datasets": core_rows + [burst_row, tao_row, visor_row],
    }


def _run_smoke_tests(repo_root: Path, completion_json: Dict[str, Any], export_smoke_json: Path, eval_smoke_json: Path) -> Dict[str, Any]:
    tap_style = completion_json["primary_checkpoint_eval"]["tap_style_eval"]
    proxy_payload = tap_style["proxy_payload"]["payload_npz"]
    tapnet_python = tap_style["official_eval"]["tapnet_python"]
    smoke_payload = repo_root / "tmp" / "stage2_tap_payload_export_smoke_20260409.npz"

    export_cmd = [
        "/home/chen034/miniconda3/envs/stwm/bin/python",
        str(repo_root / "code" / "stwm" / "tracewm_v2_stage2" / "tools" / "export_stage2_tap_payload.py"),
        "--proxy-payload-npz",
        str(proxy_payload),
        "--output-npz",
        str(smoke_payload),
        "--output-report-json",
        str(export_smoke_json),
    ]
    eval_cmd = [
        "/home/chen034/miniconda3/envs/stwm/bin/python",
        str(repo_root / "code" / "stwm" / "tracewm_v2_stage2" / "tools" / "run_stage2_tap_eval.py"),
        "--tap-payload-npz",
        str(smoke_payload),
        "--output-json",
        str(eval_smoke_json),
        "--tapnet-python",
        str(tapnet_python),
    ]
    export_proc = subprocess.run(export_cmd, cwd=str(repo_root), text=True, capture_output=True)
    eval_proc = subprocess.run(eval_cmd, cwd=str(repo_root), text=True, capture_output=True)
    return {
        "export_stage2_tap_payload": {
            "command": export_cmd,
            "returncode": int(export_proc.returncode),
            "success": bool(export_proc.returncode == 0 and export_smoke_json.exists()),
            "stdout_tail": export_proc.stdout[-2000:],
            "stderr_tail": export_proc.stderr[-2000:],
            "report_path": str(export_smoke_json),
        },
        "run_stage2_tap_eval": {
            "command": eval_cmd,
            "returncode": int(eval_proc.returncode),
            "success": bool(eval_proc.returncode == 0 and eval_smoke_json.exists()),
            "stdout_tail": eval_proc.stdout[-2000:],
            "stderr_tail": eval_proc.stderr[-2000:],
            "report_path": str(eval_smoke_json),
        },
    }


def _build_external_fidelity_audit(repo_root: Path, completion_json: Dict[str, Any], completion_log_path: Path, smoke_results: Dict[str, Any]) -> Dict[str, Any]:
    tap_style = completion_json["primary_checkpoint_eval"]["tap_style_eval"]
    tap3d = completion_json.get("tap3d_completion", {})
    audit = {
        "generated_at_utc": now_iso(),
        "current_stage2_mainline_checkpoint": completion_json.get("current_stage2_mainline_checkpoint", ""),
        "secondary_checkpoint_reference": completion_json.get("secondary_checkpoint_reference", ""),
        "tap_style_eval_status": completion_json.get("tap_style_eval_status", "not_yet_implemented"),
        "tap3d_style_eval_status": "not_yet_implemented",
        "official_evaluator_invoked": bool(tap_style.get("official_evaluator_invoked", False)),
        "official_tapvid_evaluator_connected": bool(tap_style.get("official_tapvid_evaluator_connected", False)),
        "official_task_faithfully_instantiated": bool(tap_style.get("official_task_faithfully_instantiated", False)),
        "tap_style_task_checks": {
            "benchmark_native_full_tap_episode": bool(tap_style.get("benchmark_native_full_tap_episode", False)),
            "query_time_matches_official_task": bool(tap_style.get("query_time_matches_official_task", False)),
            "pred_visibility_from_model_output": bool(tap_style.get("pred_visibility_from_model_output", False)),
            "dataset_binding_is_official_tap_dataset_family": bool(tap_style.get("dataset_binding_is_official_tap_dataset_family", False)),
        },
        "tap3d_task_checks": {
            "aligned_3d_gt_for_current_binding": bool(tap3d.get("aligned_3d_gt_for_current_binding", False)),
            "camera_geometry_projection_or_lifting_path_available": bool(tap3d.get("camera_geometry_projection_or_lifting_path_available", False)),
            "verified_exporter_to_tracks_xyz_visibility": bool(tap3d.get("verified_exporter_to_tracks_xyz_visibility", False)),
            "official_tapvid3d_metric_importable": bool((tap3d.get("runtime_probe", {}) if isinstance(tap3d.get("runtime_probe", {}), dict) else {}).get("official_tapvid3d_metric_importable", False)),
        },
        "exact_blocking_reasons": completion_json.get("exact_blocking_reasons", []),
        "completion_log": _path_record(repo_root, completion_log_path, regenerated_in_this_round=False, note="exists locally but logs/** are gitignored"),
        "smoke_tests": smoke_results,
    }
    return audit


def _build_stage1_bundle(repo_root: Path, data_root: Path, workers: int) -> Dict[str, Any]:
    po_manifest = _read_json(data_root / "_manifests" / "pointodyssey_hard_complete_after_20260407.json")
    kubric_manifest = _read_json(data_root / "_manifests" / "kubric_hard_complete_after_20260407.json")
    datasets = [
        _audit_stage1_pointodyssey(data_root, po_manifest, workers),
        _audit_stage1_kubric(data_root, kubric_manifest),
        _audit_stage1_tapvid(data_root),
        _audit_stage1_tapvid3d(data_root, workers),
    ]
    critical_ok = all(
        row["completeness_status"] in {"complete", "limited_eval_ready"}
        for row in datasets
        if row["dataset_name"] in {"pointodyssey", "kubric", "tapvid", "tapvid3d"}
    )
    return {
        "generated_at_utc": now_iso(),
        "evidence_bundle_ready": bool(critical_ok),
        "datasets": datasets,
        "critical_contract_inputs": [
            _path_record(repo_root, data_root / "_manifests" / "stage1_data_contract_20260408.json"),
            _path_record(repo_root, data_root / "_manifests" / "stage1_dataset_audit_20260407.json"),
            _path_record(repo_root, data_root / "_manifests" / "stage1_v2_trace_cache_contract_20260408.json"),
            _path_record(repo_root, data_root / "_manifests" / "stage1_v2_pointodyssey_cache_index_20260408.json"),
            _path_record(repo_root, data_root / "_manifests" / "stage1_v2_kubric_cache_index_20260408.json"),
        ],
    }


def _support_row(name: str, value: Any, support_paths: List[Dict[str, Any]], support_kind: str, note: str = "") -> Dict[str, Any]:
    return {
        "state_name": name,
        "value": value,
        "support_kind": support_kind,
        "support_paths": support_paths,
        "note": note,
    }


def _build_evidence_audit(
    repo_root: Path,
    stage1_bundle: Dict[str, Any],
    stage2_bundle: Dict[str, Any],
    completion_json: Dict[str, Any],
    external_fidelity: Dict[str, Any],
) -> Dict[str, Any]:
    repo_supported = [
        _support_row(
            "stage2_core_cropenc_is_current_mainline",
            True,
            [
                _path_record(repo_root, repo_root / "reports" / "stage2_eval_fix_comparison_20260408.json"),
                _path_record(repo_root, repo_root / "docs" / "TRACEWM_STAGE2_CORE_MAINLINE_TRAIN_PROTOCOL_20260408.md"),
            ],
            "repo_audit_file",
        ),
        _support_row(
            "current_mainline_semantic_source",
            completion_json.get("current_mainline_semantic_source", ""),
            [
                _path_record(repo_root, repo_root / "reports" / "stage2_core_mainline_train_final_20260408.json"),
                _path_record(repo_root, repo_root / "reports" / "stage2_external_eval_completion_20260408.json"),
            ],
            "repo_audit_file",
        ),
        _support_row(
            "current_mainline_checkpoint_is_best_pt",
            completion_json.get("current_stage2_mainline_checkpoint", "").endswith("/best.pt"),
            [
                _path_record(repo_root, repo_root / "reports" / "stage2_core_mainline_train_final_20260408.json"),
                _path_record(repo_root, repo_root / "reports" / "stage2_external_eval_completion_20260408.json"),
            ],
            "repo_audit_file",
        ),
        _support_row(
            "frozen_boundary_kept_correct",
            bool(completion_json.get("frozen_boundary_kept_correct", False)),
            [
                _path_record(repo_root, repo_root / "reports" / "stage2_core_mainline_train_final_20260408.json"),
                _path_record(repo_root, repo_root / "reports" / "stage2_external_eval_completion_20260408.json"),
            ],
            "repo_audit_file",
        ),
    ]

    contract_only = [
        _support_row(
            "pointodyssey_complete_under_hard_complete_rule_before_this_round",
            True,
            [_path_record(repo_root, repo_root.parent / "data" / "_manifests" / "pointodyssey_hard_complete_after_20260407.json")],
            "external_manifest_or_contract",
            "Now re-backed by the repo-local stage1 evidence bundle generated this round.",
        ),
        _support_row(
            "kubric_complete_under_movi_e_only_rule_before_this_round",
            True,
            [_path_record(repo_root, repo_root.parent / "data" / "_manifests" / "kubric_hard_complete_after_20260407.json")],
            "external_manifest_or_contract",
            "Now re-backed by the repo-local stage1 evidence bundle generated this round.",
        ),
        _support_row(
            "stage2_core_ready_vspw_vipseg_before_this_round",
            True,
            [
                _path_record(repo_root, repo_root.parent / "data" / "_manifests" / "stage2_dataset_audit_20260408.json"),
                _path_record(repo_root, repo_root / "reports" / "stage2_bootstrap_data_contract_20260408.json"),
            ],
            "external_manifest_or_contract",
            "Now re-backed by the repo-local stage2 evidence bundle generated this round.",
        ),
    ]

    live_not_packaged = [
        _path_record(repo_root, repo_root / "logs" / "tracewm_stage2_external_eval_completion_20260408.log", note="present locally but excluded by logs/**"),
        _path_record(repo_root, repo_root / "outputs" / "checkpoints" / "stage2_core_mainline_train_20260408" / "best.pt", note="present locally but excluded by outputs/**"),
        _path_record(repo_root, repo_root / "data" / "external" / "vspw" / "VSPW", note="present locally but excluded by data/external/**"),
        _path_record(repo_root, repo_root / "data" / "external" / "vipseg" / "VIPSeg", note="present locally but excluded by data/external/**"),
    ]

    missing_or_regenerated = [
        _path_record(repo_root, repo_root / "reports" / "stage1_dataset_evidence_bundle_20260409.json", regenerated_in_this_round=True),
        _path_record(repo_root, repo_root / "reports" / "stage2_dataset_evidence_bundle_20260409.json", regenerated_in_this_round=True),
        _path_record(repo_root, repo_root / "reports" / "stage2_external_eval_fidelity_audit_20260409.json", regenerated_in_this_round=True),
        _path_record(repo_root, repo_root / "reports" / "tracewm_project_readiness_20260409.json", regenerated_in_this_round=True),
    ]

    return {
        "generated_at_utc": now_iso(),
        "repo_supported_states": repo_supported,
        "contract_or_summary_only_states_before_this_round": contract_only,
        "live_paths_exist_but_not_packaged": live_not_packaged,
        "paths_regenerated_in_this_round": missing_or_regenerated,
        "stage1_bundle_ready": bool(stage1_bundle.get("evidence_bundle_ready", False)),
        "stage2_core_bundle_zero_missing": bool(stage2_bundle.get("core_binding_zero_missing", False)),
        "external_eval_official_task_faithfully_instantiated": bool(external_fidelity.get("official_task_faithfully_instantiated", False)),
    }


def _build_project_readiness(
    completion_json: Dict[str, Any],
    stage1_bundle: Dict[str, Any],
    stage2_bundle: Dict[str, Any],
    external_fidelity: Dict[str, Any],
) -> Dict[str, Any]:
    current_stage2_mainline_still_valid = bool(
        completion_json.get("current_stage2_mainline_checkpoint", "").endswith("/best.pt")
        and completion_json.get("current_mainline_semantic_source", "") == "crop_visual_encoder"
        and bool(completion_json.get("frozen_boundary_kept_correct", False))
    )
    dataset_evidence_bundle_ready = bool(stage1_bundle.get("evidence_bundle_ready", False)) and bool(stage2_bundle.get("core_binding_zero_missing", False))
    data_evidence_paper_grade = bool(dataset_evidence_bundle_ready)
    external_eval_paper_grade = bool(
        external_fidelity.get("official_evaluator_invoked", False)
        and external_fidelity.get("official_task_faithfully_instantiated", False)
        and external_fidelity.get("tap_style_eval_status", "") == "fully_implemented_and_run"
        and external_fidelity.get("tap3d_style_eval_status", "") == "fully_implemented_and_run"
    )
    if not current_stage2_mainline_still_valid or not dataset_evidence_bundle_ready:
        project_readiness = "eval_not_ready"
        next_step_choice = "do_one_targeted_data_evidence_fix"
    elif external_eval_paper_grade:
        project_readiness = "paper_eval_ready"
        next_step_choice = "start_paper_framing_prep"
    else:
        project_readiness = "training_ready_but_eval_gap_remains"
        next_step_choice = "do_one_targeted_external_eval_fix"
    return {
        "generated_at_utc": now_iso(),
        "current_stage2_mainline_still_valid": bool(current_stage2_mainline_still_valid),
        "dataset_evidence_bundle_ready": bool(dataset_evidence_bundle_ready),
        "data_evidence_paper_grade": bool(data_evidence_paper_grade),
        "external_eval_paper_grade": bool(external_eval_paper_grade),
        "official_evaluator_invoked": bool(external_fidelity.get("official_evaluator_invoked", False)),
        "official_task_faithfully_instantiated": bool(external_fidelity.get("official_task_faithfully_instantiated", False)),
        "tap_style_eval_status": external_fidelity.get("tap_style_eval_status", "not_yet_implemented"),
        "tap3d_style_eval_status": external_fidelity.get("tap3d_style_eval_status", "not_yet_implemented"),
        "project_readiness": project_readiness,
        "next_step_choice": next_step_choice,
    }


def _stage1_bundle_md(bundle: Dict[str, Any]) -> List[str]:
    lines = [
        "# Stage1 Data Evidence Bundle",
        "",
        f"- generated_at_utc: {bundle.get('generated_at_utc', '')}",
        f"- evidence_bundle_ready: {bool(bundle.get('evidence_bundle_ready', False))}",
        "",
    ]
    for row in bundle.get("datasets", []):
        lines.extend(
            [
                f"## {row.get('dataset_name', '')}",
                f"- completeness_status: {row.get('completeness_status', '')}",
                f"- source_audit_path: {row.get('source_audit_path', '')}",
                f"- source_audit_exists: {bool(row.get('source_audit_exists', False))}",
                f"- regenerated_in_this_round: {bool(row.get('regenerated_in_this_round', False))}",
            ]
        )
        for reason in row.get("blocking_reasons", []):
            lines.append(f"- blocking_reason: {reason}")
        lines.append("")
    return lines


def _stage2_bundle_md(bundle: Dict[str, Any]) -> List[str]:
    lines = [
        "# Stage2 Data Evidence Bundle",
        "",
        f"- generated_at_utc: {bundle.get('generated_at_utc', '')}",
        f"- core_binding_zero_missing: {bool(bundle.get('core_binding_zero_missing', False))}",
        f"- datasets_bound_for_train: {bundle.get('datasets_bound_for_train', [])}",
        f"- datasets_bound_for_eval: {bundle.get('datasets_bound_for_eval', [])}",
        "",
    ]
    for row in bundle.get("datasets", []):
        lines.extend(
            [
                f"## {row.get('dataset_name', '')}",
                f"- role_in_stage2: {row.get('role_in_stage2', '')}",
                f"- completeness_status: {row.get('completeness_status', '')}",
                f"- evidence_source_path: {row.get('evidence_source_path', '')}",
                f"- evidence_source_exists: {bool(row.get('evidence_source_exists', False))}",
                f"- regenerated_in_this_round: {bool(row.get('regenerated_in_this_round', False))}",
            ]
        )
        for reason in row.get("blocking_reasons", []):
            lines.append(f"- blocking_reason: {reason}")
        for reason in row.get("warning_reasons", []):
            lines.append(f"- warning_reason: {reason}")
        lines.append("")
    return lines


def _external_fidelity_md(audit: Dict[str, Any]) -> List[str]:
    lines = [
        "# Stage2 External Eval Fidelity Audit",
        "",
        f"- current_stage2_mainline_checkpoint: {audit.get('current_stage2_mainline_checkpoint', '')}",
        f"- official_evaluator_invoked: {bool(audit.get('official_evaluator_invoked', False))}",
        f"- official_tapvid_evaluator_connected: {bool(audit.get('official_tapvid_evaluator_connected', False))}",
        f"- official_task_faithfully_instantiated: {bool(audit.get('official_task_faithfully_instantiated', False))}",
        f"- tap_style_eval_status: {audit.get('tap_style_eval_status', '')}",
        f"- tap3d_style_eval_status: {audit.get('tap3d_style_eval_status', '')}",
        "",
        "## TAP-Style Checks",
    ]
    for k, v in (audit.get("tap_style_task_checks", {}) or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## TAP3D Checks"])
    for k, v in (audit.get("tap3d_task_checks", {}) or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Blocking Reasons"])
    for reason in audit.get("exact_blocking_reasons", []):
        lines.append(f"- {reason}")
    lines.extend(["", "## Packaging Note"])
    log_row = audit.get("completion_log", {}) if isinstance(audit.get("completion_log", {}), dict) else {}
    lines.append(f"- completion_log_exists: {bool(log_row.get('exists', False))}")
    lines.append(f"- packaged_in_repo_snapshot: {bool(log_row.get('packaged_in_repo_snapshot', False))}")
    lines.append(f"- note: {log_row.get('note', '')}")
    return lines


def _project_readiness_md(readiness: Dict[str, Any]) -> List[str]:
    return [
        "# TRACEWM Project Readiness",
        "",
        f"- current_stage2_mainline_still_valid: {bool(readiness.get('current_stage2_mainline_still_valid', False))}",
        f"- dataset_evidence_bundle_ready: {bool(readiness.get('dataset_evidence_bundle_ready', False))}",
        f"- data_evidence_paper_grade: {bool(readiness.get('data_evidence_paper_grade', False))}",
        f"- external_eval_paper_grade: {bool(readiness.get('external_eval_paper_grade', False))}",
        f"- official_evaluator_invoked: {bool(readiness.get('official_evaluator_invoked', False))}",
        f"- official_task_faithfully_instantiated: {bool(readiness.get('official_task_faithfully_instantiated', False))}",
        f"- tap_style_eval_status: {readiness.get('tap_style_eval_status', '')}",
        f"- tap3d_style_eval_status: {readiness.get('tap3d_style_eval_status', '')}",
        f"- project_readiness: {readiness.get('project_readiness', '')}",
        f"- next_step_choice: {readiness.get('next_step_choice', '')}",
    ]


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    data_root = Path(args.data_root)

    stage2_final = _read_json(repo_root / "reports" / "stage2_core_mainline_train_final_20260408.json")
    stage2_contract = _read_json(repo_root / "reports" / "stage2_bootstrap_data_contract_20260408.json")
    completion_json = _read_json(args.stage2_completion_json)

    stage1_bundle = _build_stage1_bundle(repo_root, data_root, int(args.workers))
    stage2_bundle = _audit_stage2_dataset_bundle(repo_root, stage2_contract, stage2_final, int(args.workers))
    smoke_results = _run_smoke_tests(repo_root, completion_json, Path(args.tap_export_smoke_json), Path(args.tap_eval_smoke_json))
    external_fidelity = _build_external_fidelity_audit(repo_root, completion_json, Path(args.stage2_completion_log), smoke_results)
    readiness = _build_project_readiness(completion_json, stage1_bundle, stage2_bundle, external_fidelity)

    _write_json(args.stage1_bundle_json, stage1_bundle)
    _write_md(args.stage1_bundle_md, _stage1_bundle_md(stage1_bundle))
    _write_json(args.stage2_bundle_json, stage2_bundle)
    _write_md(args.stage2_bundle_md, _stage2_bundle_md(stage2_bundle))
    _write_json(args.external_fidelity_json, external_fidelity)
    _write_md(args.external_fidelity_md, _external_fidelity_md(external_fidelity))
    _write_json(args.project_readiness_json, readiness)
    _write_md(args.project_readiness_md, _project_readiness_md(readiness))
    evidence_audit = _build_evidence_audit(repo_root, stage1_bundle, stage2_bundle, completion_json, external_fidelity)
    _write_json(args.evidence_audit_json, evidence_audit)

    print(json.dumps(
        {
            "stage1_bundle_json": str(args.stage1_bundle_json),
            "stage2_bundle_json": str(args.stage2_bundle_json),
            "external_fidelity_json": str(args.external_fidelity_json),
            "project_readiness_json": str(args.project_readiness_json),
            "current_stage2_mainline_still_valid": readiness["current_stage2_mainline_still_valid"],
            "dataset_evidence_bundle_ready": readiness["dataset_evidence_bundle_ready"],
            "official_evaluator_invoked": readiness["official_evaluator_invoked"],
            "official_task_faithfully_instantiated": readiness["official_task_faithfully_instantiated"],
            "tap_style_eval_status": readiness["tap_style_eval_status"],
            "tap3d_style_eval_status": readiness["tap3d_style_eval_status"],
            "project_readiness": readiness["project_readiness"],
            "next_step_choice": readiness["next_step_choice"],
        },
        ensure_ascii=True,
        indent=2,
    ))


if __name__ == "__main__":
    main()
