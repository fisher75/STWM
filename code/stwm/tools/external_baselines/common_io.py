from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path("/home/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
BASELINES = ROOT / "baselines"
REPOS = BASELINES / "repos"
CHECKPOINTS = BASELINES / "checkpoints"
LOGS = BASELINES / "logs"
OUTPUTS = BASELINES / "outputs"
ENV_AUDIT = BASELINES / "env_audit"
ADAPTER_DIR = ROOT / "code" / "stwm" / "tools" / "external_baselines"
STWM_PYTHON = Path("/home/chen034/miniconda3/envs/stwm/bin/python")

SOURCE_REPORTS = [
    "reports/stwm_reacquisition_v2_task_build_20260425.json",
    "reports/stwm_reacquisition_v2_eval_20260425.json",
    "reports/stwm_reacquisition_v2_paper_assets_20260425.json",
    "reports/stwm_belief_final_eval_20260424.json",
    "reports/stwm_belief_true_ood_eval_20260424.json",
    "reports/stwm_false_confuser_analysis_20260425.json",
]

BASELINE_CONFIG = {
    "cutie": {
        "repo_dir": REPOS / "Cutie",
        "remote": "https://github.com/hkchengrex/Cutie.git",
        "license_names": ["LICENSE"],
        "readme_names": ["README.md"],
        "checkpoint_dir": CHECKPOINTS / "cutie",
        "download_command": [
            str(STWM_PYTHON),
            str(REPOS / "Cutie" / "cutie" / "utils" / "download_models.py"),
        ],
        "download_cwd": str(CHECKPOINTS / "cutie"),
        "import_modules": ["cutie", "cutie.inference.inference_core"],
        "required_item_fields": [
            "frame_paths",
            "observed_frame_indices",
            "future_frame_index",
            "observed_target_mask_or_box",
            "future_candidate_masks_or_boxes",
            "gt_candidate_id",
        ],
    },
    "sam2": {
        "repo_dir": REPOS / "sam2",
        "remote": "https://github.com/facebookresearch/sam2.git",
        "license_names": ["LICENSE"],
        "readme_names": ["README.md"],
        "checkpoint_dir": CHECKPOINTS / "sam2",
        "download_command": ["bash", str(REPOS / "sam2" / "checkpoints" / "download_ckpts.sh")],
        "download_cwd": str(CHECKPOINTS / "sam2"),
        "import_modules": ["sam2", "sam2.build_sam"],
        "required_item_fields": [
            "frame_paths",
            "observed_frame_indices",
            "future_frame_index",
            "observed_target_mask_or_box_or_point",
            "future_candidate_masks_or_boxes",
            "gt_candidate_id",
        ],
    },
    "cotracker": {
        "repo_dir": REPOS / "co-tracker",
        "remote": "https://github.com/facebookresearch/co-tracker.git",
        "license_names": ["LICENSE.md", "LICENSE"],
        "readme_names": ["README.md"],
        "checkpoint_dir": CHECKPOINTS / "cotracker",
        "download_command": None,
        "download_cwd": None,
        "import_modules": ["cotracker", "cotracker.predictor"],
        "required_item_fields": [
            "frame_paths",
            "observed_frame_indices",
            "future_frame_index",
            "observed_target_mask_or_box",
            "future_candidate_masks_or_boxes",
            "gt_candidate_id",
        ],
    },
}


@dataclass
class ExternalEvalItem:
    item_id: str
    dataset: str | None = None
    clip_id: str | None = None
    panel_name: str | None = None
    subset_tags: list[str] = field(default_factory=list)
    candidate_count: int | None = None
    gt_candidate_id: str | None = None
    frame_paths: list[str] = field(default_factory=list)
    observed_frame_indices: list[int] = field(default_factory=list)
    future_frame_index: int | None = None
    observed_target_mask_path: str | None = None
    observed_target_box: list[float] | None = None
    observed_target_point: list[float] | None = None
    future_candidate_masks: dict[str, str] = field(default_factory=dict)
    future_candidate_boxes: dict[str, list[float]] = field(default_factory=dict)
    source_rows: int = 0

    def has_observed_mask_or_box(self) -> bool:
        return bool(self.observed_target_mask_path or self.observed_target_box)

    def has_observed_prompt(self) -> bool:
        return bool(self.observed_target_mask_path or self.observed_target_box or self.observed_target_point)

    def has_future_candidates(self) -> bool:
        return bool(self.future_candidate_masks or self.future_candidate_boxes)


def ensure_dirs() -> None:
    for p in [REPORTS, DOCS, REPOS, LOGS, ENV_AUDIT, ADAPTER_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    for name in ["cutie", "sam2", "cotracker"]:
        (CHECKPOINTS / name).mkdir(parents=True, exist_ok=True)
        (OUTPUTS / name).mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_markdown(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    body.extend(lines)
    path.write_text("\n".join(body).rstrip() + "\n")


def sha256_json(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | str | None = None,
    timeout: int = 60,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "stdout_excerpt": proc.stdout[-4000:],
            "stderr_excerpt": proc.stderr[-4000:],
            "timeout": False,
            "wall_time_seconds": round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "stdout_excerpt": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_excerpt": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timeout": True,
            "wall_time_seconds": round(time.time() - started, 3),
            "exact_error": f"timeout_after_{timeout}s",
        }


def git_info(repo: Path) -> dict[str, Any]:
    if not repo.exists():
        return {"exists": False, "commit_hash": None, "git_remote": None}
    commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo, timeout=10)
    remote = run_cmd(["git", "remote", "get-url", "origin"], cwd=repo, timeout=10)
    return {
        "exists": True,
        "commit_hash": commit.get("stdout_excerpt", "").strip() if commit.get("returncode") == 0 else None,
        "git_remote": remote.get("stdout_excerpt", "").strip() if remote.get("returncode") == 0 else None,
        "git_errors": {
            "commit": commit if commit.get("returncode") != 0 else None,
            "remote": remote if remote.get("returncode") != 0 else None,
        },
    }


def build_clone_audit() -> dict[str, Any]:
    ensure_dirs()
    entries = {}
    for name, cfg in BASELINE_CONFIG.items():
        repo = Path(cfg["repo_dir"])
        info = git_info(repo)
        license_exists = any((repo / x).exists() for x in cfg["license_names"]) if repo.exists() else False
        readme_exists = any((repo / x).exists() for x in cfg["readme_names"]) if repo.exists() else False
        entries[name] = {
            "repo_name": name,
            "clone_success": bool(repo.exists() and (repo / ".git").exists()),
            "git_remote": info.get("git_remote"),
            "expected_remote": cfg["remote"],
            "commit_hash": info.get("commit_hash"),
            "license_file_exists": license_exists,
            "readme_exists": readme_exists,
            "repo_path": str(repo),
            "exact_error_if_failed": None if repo.exists() and (repo / ".git").exists() else "repo_directory_or_git_metadata_missing",
        }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_root": str(ROOT),
        "entries": entries,
        "all_clone_success": all(v["clone_success"] for v in entries.values()),
    }


def _import_check(module_names: list[str], repo_paths: list[Path], python: Path = STWM_PYTHON) -> dict[str, Any]:
    code = """
import importlib, json, sys, traceback
repo_paths = json.loads(sys.argv[1])
mods = json.loads(sys.argv[2])
for p in reversed(repo_paths):
    sys.path.insert(0, p)
out = {}
for mod in mods:
    try:
        m = importlib.import_module(mod)
        out[mod] = {"ok": True, "file": getattr(m, "__file__", None)}
    except Exception as e:
        out[mod] = {"ok": False, "error_type": type(e).__name__, "error": str(e).split("\\n")[0], "traceback_tail": traceback.format_exc()[-2000:]}
print(json.dumps(out, sort_keys=True))
"""
    if not python.exists():
        return {"ok": False, "exact_error": f"python_missing:{python}"}
    res = run_cmd(
        [str(python), "-c", code, json.dumps([str(p) for p in repo_paths]), json.dumps(module_names)],
        timeout=90,
    )
    try:
        parsed = json.loads(res.get("stdout_excerpt", "{}").strip().splitlines()[-1])
    except Exception:
        parsed = {}
    return {
        "ok": bool(parsed) and all(v.get("ok") for v in parsed.values()),
        "modules": parsed,
        "command_result": res if not parsed or any(not v.get("ok") for v in parsed.values()) else None,
    }


def _checkpoint_files(path: Path) -> list[dict[str, Any]]:
    exts = {".pt", ".pth", ".ckpt", ".safetensors", ".pkl"}
    files = []
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append({"path": str(p), "size_bytes": p.stat().st_size})
    return sorted(files, key=lambda x: x["path"])


def build_env_audit(*, attempt_downloads: bool = True, download_timeout: int = 45) -> dict[str, Any]:
    ensure_dirs()
    repo_paths = [Path(cfg["repo_dir"]) for cfg in BASELINE_CONFIG.values()]
    stwm_env_check = run_cmd(
        [
            str(STWM_PYTHON),
            "-c",
            "import sys, torch, torchvision, cv2, hydra, omegaconf; print(sys.executable); print(torch.__version__)",
        ],
        timeout=30,
    )
    entries = {}
    for name, cfg in BASELINE_CONFIG.items():
        ckpt_dir = Path(cfg["checkpoint_dir"])
        before = _checkpoint_files(ckpt_dir)
        download_attempt = None
        if attempt_downloads and not before and cfg.get("download_command"):
            log_path = LOGS / f"{name}_checkpoint_download_20260426.log"
            download_attempt = run_cmd(
                [str(x) for x in cfg["download_command"]],
                cwd=cfg.get("download_cwd"),
                timeout=download_timeout,
                env={"PYTHONPATH": os.pathsep.join(str(p) for p in repo_paths)},
            )
            log_path.write_text(json.dumps(download_attempt, indent=2, sort_keys=True) + "\n")
        after = _checkpoint_files(ckpt_dir)
        imports = _import_check([str(x) for x in cfg["import_modules"]], repo_paths)
        entries[name] = {
            "baseline_name": name,
            "import_ok": bool(imports["ok"]),
            "import_details": imports,
            "checkpoint_ready": bool(after),
            "checkpoint_files": after,
            "checkpoint_dir": str(ckpt_dir),
            "download_attempted": bool(download_attempt),
            "download_attempt": download_attempt,
            "exact_blocking_reason": None
            if imports["ok"] and after
            else "; ".join(
                x
                for x in [
                    None if imports["ok"] else "import_failed",
                    None if after else "checkpoint_missing_or_download_unfinished",
                ]
                if x
            ),
        }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_policy": {
            "stwm_environment_used_read_only": True,
            "stwm_python": str(STWM_PYTHON),
            "new_conda_env_created": False,
            "new_conda_env_reason": "not_needed_for_import_smoke; existing stwm env imports all cloned repos without installing packages",
            "main_base_environment_modified": False,
        },
        "stwm_env_check": stwm_env_check,
        "cutie_import_ok": entries["cutie"]["import_ok"],
        "sam2_import_ok": entries["sam2"]["import_ok"],
        "cotracker_import_ok": entries["cotracker"]["import_ok"],
        "cutie_checkpoint_ready": entries["cutie"]["checkpoint_ready"],
        "sam2_checkpoint_ready": entries["sam2"]["checkpoint_ready"],
        "cotracker_checkpoint_ready": entries["cotracker"]["checkpoint_ready"],
        "baseline_runnable_status": {
            name: bool(e["import_ok"] and e["checkpoint_ready"]) for name, e in entries.items()
        },
        "entries": entries,
    }


def _candidate_id_from_protocol_item_id(item_id: str | None) -> str | None:
    if not item_id:
        return None
    parts = str(item_id).split("::")
    return parts[-1] if len(parts) >= 2 else None


def _merge_row_into_item(items: dict[str, ExternalEvalItem], row: dict[str, Any]) -> None:
    item_id = str(row.get("protocol_item_id") or row.get("item_id") or "")
    if not item_id:
        return
    item = items.get(item_id)
    if item is None:
        item = ExternalEvalItem(item_id=item_id)
        items[item_id] = item
    item.dataset = item.dataset or row.get("dataset")
    item.clip_id = item.clip_id or row.get("clip_id")
    item.panel_name = item.panel_name or row.get("panel_name")
    tags = row.get("subset_tags")
    if isinstance(tags, list):
        item.subset_tags = sorted(set(item.subset_tags).union(str(t) for t in tags))
    if item.candidate_count is None and row.get("protocol_eval_context_entity_count") is not None:
        try:
            item.candidate_count = int(row["protocol_eval_context_entity_count"])
        except Exception:
            pass
    item.gt_candidate_id = item.gt_candidate_id or _candidate_id_from_protocol_item_id(item_id)
    item.source_rows += 1
    for key in ["frame_paths", "observed_frame_indices"]:
        if isinstance(row.get(key), list) and not getattr(item, key):
            setattr(item, key, [str(x) if key == "frame_paths" else int(x) for x in row[key]])
    if row.get("future_frame_index") is not None and item.future_frame_index is None:
        try:
            item.future_frame_index = int(row["future_frame_index"])
        except Exception:
            pass
    if row.get("observed_target_mask_path") and not item.observed_target_mask_path:
        item.observed_target_mask_path = str(row["observed_target_mask_path"])
    if isinstance(row.get("observed_target_box"), list) and not item.observed_target_box:
        item.observed_target_box = [float(x) for x in row["observed_target_box"][:4]]
    if isinstance(row.get("observed_target_point"), list) and not item.observed_target_point:
        item.observed_target_point = [float(x) for x in row["observed_target_point"][:2]]
    if isinstance(row.get("future_candidate_masks"), dict):
        item.future_candidate_masks.update({str(k): str(v) for k, v in row["future_candidate_masks"].items()})
    if isinstance(row.get("future_candidate_boxes"), dict):
        for k, v in row["future_candidate_boxes"].items():
            if isinstance(v, list) and len(v) >= 4:
                item.future_candidate_boxes[str(k)] = [float(x) for x in v[:4]]


def load_source_items() -> tuple[list[ExternalEvalItem], dict[str, Any]]:
    manifest_path = REPORTS / "stwm_external_baseline_item_manifest_20260426.json"
    if manifest_path.exists():
        status = {
            str(manifest_path.relative_to(ROOT)): {
                "path": str(manifest_path),
                "exists": True,
                "valid_json": False,
                "rows_seen": 0,
                "source_type": "materialized_external_baseline_manifest",
            }
        }
        try:
            manifest = load_json(manifest_path)
            status[str(manifest_path.relative_to(ROOT))]["valid_json"] = True
        except Exception as exc:
            status[str(manifest_path.relative_to(ROOT))]["exact_error"] = str(exc)
            return [], status
        out: list[ExternalEvalItem] = []
        for row in manifest.get("items", []):
            observed = row.get("observed_target") or {}
            future_boxes = {
                str(c.get("candidate_id")): c.get("bbox")
                for c in row.get("future_candidates", [])
                if c.get("candidate_id") is not None and isinstance(c.get("bbox"), list)
            }
            future_masks = {
                str(c.get("candidate_id")): str(c.get("mask_path"))
                for c in row.get("future_candidates", [])
                if c.get("candidate_id") is not None and c.get("mask_path")
            }
            out.append(
                ExternalEvalItem(
                    item_id=str(row.get("protocol_item_id") or row.get("item_id")),
                    dataset=row.get("dataset"),
                    clip_id=row.get("clip_id"),
                    panel_name=",".join(row.get("source_protocol", [])) if isinstance(row.get("source_protocol"), list) else row.get("source_protocol"),
                    subset_tags=sorted(k for k, v in (row.get("subset_tags") or {}).items() if v),
                    candidate_count=len(row.get("future_candidates", [])),
                    gt_candidate_id=str(row.get("gt_candidate_id")) if row.get("gt_candidate_id") is not None else None,
                    frame_paths=[str(x) for x in row.get("frame_paths", [])],
                    observed_frame_indices=[int(x) for x in row.get("observed_frame_indices", [])],
                    future_frame_index=int(row["future_frame_index"]) if row.get("future_frame_index") is not None else None,
                    observed_target_mask_path=observed.get("mask_path"),
                    observed_target_box=observed.get("bbox"),
                    observed_target_point=observed.get("point_prompt"),
                    future_candidate_masks=future_masks,
                    future_candidate_boxes=future_boxes,
                    source_rows=1,
                )
            )
        status[str(manifest_path.relative_to(ROOT))]["rows_seen"] = len(out)
        return sorted(out, key=lambda x: x.item_id), status

    items: dict[str, ExternalEvalItem] = {}
    source_status = {}
    for rel in SOURCE_REPORTS:
        p = ROOT / rel
        status = {"path": str(p), "exists": p.exists(), "valid_json": False, "rows_seen": 0}
        if not p.exists():
            source_status[rel] = status
            continue
        try:
            data = load_json(p)
            status["valid_json"] = True
        except Exception as exc:
            status["exact_error"] = str(exc)
            source_status[rel] = status
            continue
        rows: list[dict[str, Any]] = []
        if rel.endswith("stwm_belief_final_eval_20260424.json"):
            for panel in data.get("panels", {}).values():
                rows.extend([r for r in panel.get("per_item_results", []) if isinstance(r, dict)])
        elif rel.endswith("stwm_belief_true_ood_eval_20260424.json"):
            for split in data.get("splits", {}).values():
                rows.extend([r for r in split.get("per_item_results", []) if isinstance(r, dict)])
        elif rel.endswith("stwm_false_confuser_analysis_20260425.json"):
            for group in data.get("groups", {}).values():
                for item_id in group.get("representative_item_ids", []) if isinstance(group, dict) else []:
                    rows.append({"protocol_item_id": item_id, "subset_tags": [str(group.get("group_name", "false_confuser"))]})
        status["rows_seen"] = len(rows)
        for row in rows:
            _merge_row_into_item(items, row)
        source_status[rel] = status
    return sorted(items.values(), key=lambda x: x.item_id), source_status


def missing_fields_for_baseline(item: ExternalEvalItem, baseline: str) -> list[str]:
    missing = []
    if not item.frame_paths:
        missing.append("missing_frame_paths")
    if not item.observed_frame_indices:
        missing.append("missing_observed_frame_indices")
    if item.future_frame_index is None:
        missing.append("missing_future_frame_index")
    if baseline == "sam2":
        if not item.has_observed_prompt():
            missing.append("missing_observed_target_mask_or_box_or_point")
    else:
        if not item.has_observed_mask_or_box():
            missing.append("missing_observed_target_mask_or_box")
    if not item.has_future_candidates():
        missing.append("missing_future_candidate_masks_or_boxes")
    if item.gt_candidate_id is None:
        missing.append("missing_gt_candidate_id")
    return missing


def build_item_audit() -> dict[str, Any]:
    items, source_status = load_source_items()
    per_baseline = {}
    union_runnable = set()
    skipped_reason_counts = Counter()
    examples = []
    for baseline in BASELINE_CONFIG:
        runnable = []
        skipped = []
        reason_counts = Counter()
        for item in items:
            missing = missing_fields_for_baseline(item, baseline)
            if missing:
                reason_key = "+".join(missing)
                reason_counts[reason_key] += 1
                if len(skipped) < 20:
                    skipped.append({"item_id": item.item_id, "missing_fields": missing, "clip_id": item.clip_id})
            else:
                runnable.append(item.item_id)
                union_runnable.add(item.item_id)
        skipped_reason_counts.update(reason_counts)
        per_baseline[baseline] = {
            "runnable_items": len(runnable),
            "runnable_item_ids_sample": runnable[:20],
            "skipped_items": len(items) - len(runnable),
            "skipped_reason_counts": dict(reason_counts),
            "skipped_examples": skipped,
        }
    for item in items[:10]:
        examples.append(asdict(item))
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_reports": source_status,
        "total_items_found": len(items),
        "runnable_items": len(union_runnable),
        "skipped_items": len(items) - len(union_runnable),
        "skipped_reason_counts": dict(skipped_reason_counts),
        "per_baseline_runnable_items": {k: v["runnable_items"] for k, v in per_baseline.items()},
        "per_baseline": per_baseline,
        "example_items": examples,
        "exact_blocking_reason": None
        if union_runnable
        else "existing STWM reports expose per-item rankings/subset tags but not raw frame paths, observed prompt masks/boxes, future frame indices, or future candidate masks/boxes required by VOS/tracking baselines",
    }


def read_report_or_none(path: Path) -> Any | None:
    try:
        return load_json(path) if path.exists() else None
    except Exception:
        return None


def write_clone_docs(audit: dict[str, Any]) -> None:
    lines = [
        "| baseline | clone_success | commit | license | readme | error |",
        "|---|---:|---|---:|---:|---|",
    ]
    for name, e in audit["entries"].items():
        lines.append(
            f"| {name} | `{e['clone_success']}` | `{e.get('commit_hash')}` | `{e['license_file_exists']}` | `{e['readme_exists']}` | {e.get('exact_error_if_failed') or ''} |"
        )
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_CLONE_AUDIT_20260426.md", "STWM External Baseline Clone Audit 20260426", lines)


def write_env_docs(audit: dict[str, Any]) -> None:
    lines = [
        f"- stwm_python: `{audit['python_policy']['stwm_python']}`",
        f"- new_conda_env_created: `{audit['python_policy']['new_conda_env_created']}`",
        f"- main_base_environment_modified: `{audit['python_policy']['main_base_environment_modified']}`",
        "",
        "| baseline | import_ok | checkpoint_ready | runnable_status | blocking_reason |",
        "|---|---:|---:|---:|---|",
    ]
    for name, e in audit["entries"].items():
        lines.append(
            f"| {name} | `{e['import_ok']}` | `{e['checkpoint_ready']}` | `{audit['baseline_runnable_status'][name]}` | {e.get('exact_blocking_reason') or ''} |"
        )
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_ENV_AUDIT_20260426.md", "STWM External Baseline Environment Audit 20260426", lines)


def write_item_docs(audit: dict[str, Any]) -> None:
    lines = [
        f"- total_items_found: `{audit['total_items_found']}`",
        f"- runnable_items: `{audit['runnable_items']}`",
        f"- exact_blocking_reason: `{audit.get('exact_blocking_reason')}`",
        "",
        "| baseline | runnable_items | skipped_items |",
        "|---|---:|---:|",
    ]
    for name, e in audit["per_baseline"].items():
        lines.append(f"| {name} | {e['runnable_items']} | {e['skipped_items']} |")
    lines.extend(["", "## Top skipped reasons", ""])
    for reason, count in sorted(audit["skipped_reason_counts"].items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"- `{reason}`: {count}")
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_ITEM_AUDIT_20260426.md", "STWM External Baseline Item Audit 20260426", lines)
