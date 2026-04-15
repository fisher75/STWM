#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import gzip
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path('/home/chen034/workspace/stwm').resolve()
QUARANTINE = Path('/home/chen034/workspace/stwm_cleanup_quarantine_20260414').resolve()
DATE_TAG = '20260414'
REPORTS = ROOT / 'reports'
DOCS = ROOT / 'docs'
SCRIPTS = ROOT / 'scripts'
PROTECTED_TOP = {'code', 'docs', 'manifests', 'configs', 'reports', 'data', 'models', 'third_party'}
CHECKPOINT_KEEP_NAMES = {'best.pt', 'latest.pt', 'best_semantic_hard.pt', 'semantic_hard_best.pt', 'final.pt'}
STEP_CKPT_PATTERNS = ['step_*.pt', 'epoch_*.pt', 'global_step_*.pt', 'intermediate_*.pt', 'optimizer*.pt', '*optimizer*state*.pt']
EXT_GROUPS = {'.pt', '.pth', '.ckpt', '.safetensors', '.npz', '.npy', '.pkl', '.log', '.json', '.jsonl', '.zip', '.tar', '.gz', '.out', '.err'}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(cmd: list[str] | str, shell: bool = False, timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, shell=shell, text=True, capture_output=True, timeout=timeout)
        return {'cmd': cmd if isinstance(cmd, str) else ' '.join(cmd), 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    except Exception as exc:
        return {'cmd': cmd if isinstance(cmd, str) else ' '.join(cmd), 'returncode': -1, 'stdout': '', 'stderr': repr(exc)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def fmt_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if abs(value) < 1024.0 or unit == 'PB':
            return f'{value:.2f} {unit}'
        value /= 1024.0
    return f'{value:.2f} PB'


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def is_protected(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(ROOT)
    except Exception:
        return True
    return bool(rel.parts and rel.parts[0] in PROTECTED_TOP)


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    proc = run_cmd(['du', '-sb', str(path)], timeout=1800)
    if proc['returncode'] == 0 and proc['stdout'].strip():
        try:
            return int(proc['stdout'].split()[0])
        except Exception:
            pass
    if path.is_file():
        return file_size(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += file_size(Path(root) / name)
    return total


def mtime_age_seconds(path: Path) -> float:
    try:
        return time.time() - float(path.stat().st_mtime)
    except Exception:
        return 0.0


def walk_files(base: Path):
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not is_under(Path(root) / d, QUARANTINE)]
        for name in files:
            p = Path(root) / name
            if not p.is_symlink():
                yield p


def active_processes() -> dict[str, Any]:
    ps = run_cmd('ps -eo pid,ppid,etimes,cmd | grep -E "train_tracewm|stage2_|tracewm|stwm" | grep -v grep || true', shell=True, timeout=30)
    nvidia = run_cmd(['nvidia-smi'], timeout=60)
    tmux = run_cmd('tmux ls 2>/dev/null || true', shell=True, timeout=30)
    lines = [line for line in ps['stdout'].splitlines() if line.strip()]
    stwm_lines = [line for line in lines if str(ROOT) in line or 'tracewm_stage2' in line or 'stwm' in line]
    active_run_names: set[str] = set()
    for line in stwm_lines:
        for match in re.findall(r'(stage2_[A-Za-z0-9_]+_\d{8})', line):
            active_run_names.add(match)
    payload = {
        'generated_at_utc': now_iso(),
        'active_process_detected': bool(stwm_lines),
        'active_run_names': sorted(active_run_names),
        'ps_command': ps,
        'nvidia_smi': nvidia,
        'tmux_ls': tmux,
        'safety_policy_if_active': 'skip checkpoint/log/output pruning; allow audit plus global low-risk cache deletion only',
    }
    write_json(REPORTS / f'storage_cleanup_active_processes_{DATE_TAG}.json', payload)
    return payload


def is_cache_path(path: Path) -> bool:
    name = path.name
    parts = set(path.parts)
    return ('__pycache__' in parts or name.endswith(('.pyc', '.pyo')) or '.pytest_cache' in parts or '.ruff_cache' in parts or '.mypy_cache' in parts or '.ipynb_checkpoints' in parts or name == '.DS_Store' or fnmatch.fnmatch(name, 'core.*'))


def is_intermediate_ckpt(path: Path) -> bool:
    if path.name in CHECKPOINT_KEEP_NAMES:
        return False
    if path.suffix not in {'.pt', '.pth', '.ckpt'}:
        return False
    return any(fnmatch.fnmatch(path.name, pat) for pat in STEP_CKPT_PATTERNS)


def checkpoint_audit() -> list[dict[str, Any]]:
    ckpt_root = ROOT / 'outputs' / 'checkpoints'
    rows: list[dict[str, Any]] = []
    if not ckpt_root.exists():
        return rows
    for run_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
        files = [p for p in run_dir.iterdir() if p.is_file() or p.is_symlink()]
        step_files = [p for p in files if is_intermediate_ckpt(p)]
        step_bytes = sum(file_size(p) for p in step_files)
        size = dir_size(run_dir)
        rows.append({
            'run_name': run_dir.name,
            'path': safe_rel(run_dir),
            'bytes': size,
            'human': fmt_bytes(size),
            'best_exists': (run_dir / 'best.pt').exists(),
            'latest_exists': (run_dir / 'latest.pt').exists(),
            'sidecar_exists': (run_dir / 'best_semantic_hard.pt').exists() or (run_dir / 'semantic_hard_best.pt').exists(),
            'step_ckpt_count': len(step_files),
            'step_ckpt_bytes': step_bytes,
            'step_ckpt_human': fmt_bytes(step_bytes),
            'candidate_step_ckpt_sample': [safe_rel(p) for p in step_files[:20]],
        })
    rows.sort(key=lambda x: int(x['bytes']), reverse=True)
    return rows


def scan_storage() -> dict[str, Any]:
    total_du = run_cmd(['du', '-sh', str(ROOT)], timeout=1800)
    top_du = run_cmd(['du', '-h', '--max-depth=1', str(ROOT)], timeout=1800)
    top_level: list[dict[str, Any]] = []
    for child in sorted(ROOT.iterdir()):
        if child.name == QUARANTINE.name:
            continue
        if child.is_dir() or child.is_file():
            size = dir_size(child)
            top_level.append({'path': safe_rel(child), 'bytes': size, 'human': fmt_bytes(size)})
    second_level: dict[str, list[dict[str, Any]]] = {}
    for name in ['outputs', 'logs', 'reports', 'data', 'models', 'third_party', 'code']:
        base = ROOT / name
        rows: list[dict[str, Any]] = []
        if base.exists() and base.is_dir():
            for child in sorted(base.iterdir()):
                size = dir_size(child)
                rows.append({'path': safe_rel(child), 'bytes': size, 'human': fmt_bytes(size)})
            rows.sort(key=lambda x: int(x['bytes']), reverse=True)
        second_level[name] = rows[:100]

    ext_agg: dict[str, dict[str, Any]] = {}
    largest: list[dict[str, Any]] = []
    cache_candidates: list[dict[str, Any]] = []
    review_required: list[dict[str, Any]] = []
    for p in walk_files(ROOT):
        size = file_size(p)
        suffix = ''.join(p.suffixes[-2:]) if p.name.endswith('.tar.gz') else p.suffix
        key = suffix if suffix in EXT_GROUPS else (p.suffix if p.suffix in EXT_GROUPS else '<other>')
        ext_agg.setdefault(key, {'count': 0, 'bytes': 0})
        ext_agg[key]['count'] += 1
        ext_agg[key]['bytes'] += size
        largest.append({'path': safe_rel(p), 'bytes': size, 'human': fmt_bytes(size)})
        if is_cache_path(p):
            cache_candidates.append({'path': safe_rel(p), 'bytes': size, 'human': fmt_bytes(size), 'protected': is_protected(p)})
        if size >= 1024**3:
            top = safe_rel(p).split(os.sep, 1)[0]
            reason = 'protected_large_asset_data_models_or_third_party' if top in {'data', 'models', 'third_party'} else 'file_over_1gb_without_clear_auto_cleanup'
            review_required.append({'path': safe_rel(p), 'bytes': size, 'human': fmt_bytes(size), 'reason_for_review': reason, 'suggested_action': 'manual review only; do not auto-delete', 'risk_level': 'high'})
    largest.sort(key=lambda x: int(x['bytes']), reverse=True)
    for row in ext_agg.values():
        row['human'] = fmt_bytes(int(row['bytes']))
    total_size = dir_size(ROOT)
    return {
        'generated_at_utc': now_iso(),
        'root': str(ROOT),
        'du_sh': total_du,
        'du_max_depth_1': top_du,
        'total_bytes': total_size,
        'total_human': fmt_bytes(total_size),
        'top_level_directory_sizes': sorted(top_level, key=lambda x: int(x['bytes']), reverse=True),
        'second_level_directory_sizes': second_level,
        'top_200_largest_files': largest[:200],
        'extension_aggregate': dict(sorted(ext_agg.items(), key=lambda kv: int(kv[1]['bytes']), reverse=True)),
        'cache_candidate_summary': {'count': len(cache_candidates), 'bytes': sum(int(x['bytes']) for x in cache_candidates), 'human': fmt_bytes(sum(int(x['bytes']) for x in cache_candidates)), 'sample': cache_candidates[:200]},
        'checkpoint_audit': checkpoint_audit(),
        'classification': {
            'safe_delete': 'Python/editor/system/tmp caches outside protected top-level paths; tmp/bak/swp older than 3 days',
            'safe_compress': 'old logs/jsonl/out/err only when no active run is detected',
            'safe_move_to_quarantine': 'intermediate checkpoints and old output queues only when no active run is detected',
            'review_required': 'large protected assets, unclear files over 1GB, any uncertain checkpoint',
            'protected_never_delete': sorted(PROTECTED_TOP),
        },
        'review_required': review_required,
    }


def safe_delete_candidates() -> list[Path]:
    candidates: set[Path] = set()
    for root, dirs, files in os.walk(ROOT):
        root_p = Path(root)
        if is_under(root_p, QUARANTINE):
            dirs[:] = []
            continue
        for d in list(dirs):
            p = root_p / d
            if d in {'__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.ipynb_checkpoints'} and not is_protected(p):
                candidates.add(p)
        for name in files:
            p = root_p / name
            if is_protected(p):
                continue
            if name.endswith(('.pyc', '.pyo')) or name == '.DS_Store' or fnmatch.fnmatch(name, 'core.*'):
                candidates.add(p)
            elif name.endswith(('.tmp', '.temp', '.bak', '.swp')) or name.endswith('~'):
                if mtime_age_seconds(p) > 3 * 86400:
                    candidates.add(p)
    return sorted(candidates, key=lambda p: str(p))


def delete_path(path: Path) -> tuple[bool, str, int]:
    size = dir_size(path)
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, '', size
    except Exception as exc:
        return False, repr(exc), 0


def run_safe_delete() -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = safe_delete_candidates()
    plan_rows = [{'path': safe_rel(p), 'bytes': dir_size(p), 'human': fmt_bytes(dir_size(p)), 'reason': 'low_risk_cache_or_old_temp_outside_protected_paths'} for p in candidates]
    plan = {'generated_at_utc': now_iso(), 'candidate_count': len(plan_rows), 'candidate_bytes': sum(int(x['bytes']) for x in plan_rows), 'candidate_human': fmt_bytes(sum(int(x['bytes']) for x in plan_rows)), 'protected_paths_excluded': sorted(PROTECTED_TOP), 'rows': plan_rows}
    write_json(REPORTS / f'storage_cleanup_safe_delete_plan_{DATE_TAG}.json', plan)
    deleted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    total = 0
    for p in candidates:
        ok, err, size = delete_path(p)
        row = {'path': safe_rel(p), 'bytes': size, 'human': fmt_bytes(size)}
        if ok:
            deleted.append(row)
            total += size
        else:
            row['error'] = err
            failed.append(row)
    manifest = {'generated_at_utc': now_iso(), 'permanently_deleted_count': len(deleted), 'permanently_deleted_bytes': total, 'permanently_deleted_human': fmt_bytes(total), 'failed_count': len(failed), 'deleted': deleted, 'failed': failed}
    write_json(REPORTS / f'storage_cleanup_safe_delete_manifest_{DATE_TAG}.json', manifest)
    return plan, manifest


def run_completed(run_name: str) -> bool:
    for p in [ROOT / 'reports' / f'{run_name}_final.json', ROOT / 'reports' / f'{run_name}_raw.json']:
        if not p.exists():
            continue
        try:
            if str(json.loads(p.read_text(encoding='utf-8')).get('status', '')).lower() == 'completed':
                return True
        except Exception:
            pass
    return False


def move_to_quarantine(src: Path, category: str) -> dict[str, Any]:
    rel = src.resolve().relative_to(ROOT)
    dst = QUARANTINE / category / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = dir_size(src)
    shutil.move(str(src), str(dst))
    return {'source': str(src), 'relative_source': str(rel), 'destination': str(dst), 'bytes': size, 'human': fmt_bytes(size)}


def checkpoint_prune(active: dict[str, Any]) -> dict[str, Any]:
    active_runs = set(active.get('active_run_names', []))
    skipped = bool(active.get('active_process_detected', False))
    plan_rows: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    ckpt_root = ROOT / 'outputs' / 'checkpoints'
    if ckpt_root.exists():
        for run_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
            run_name = run_dir.name
            step_files = sorted([p for p in run_dir.iterdir() if p.is_file() and is_intermediate_ckpt(p)], key=lambda p: p.stat().st_mtime)
            keep_extra = set(step_files[-1:]) if step_files else set()
            for p in step_files:
                row = {'run_name': run_name, 'path': safe_rel(p), 'bytes': file_size(p), 'human': fmt_bytes(file_size(p))}
                if skipped:
                    review.append({**row, 'action': 'skipped_due_to_active_runs', 'reason_for_review': 'active STWM run detected; checkpoint prune disabled', 'risk_level': 'medium', 'suggested_action': 'rerun cleanup after active runs finish'})
                    continue
                if run_name in active_runs or p in keep_extra or not ((run_dir / 'best.pt').exists() or (run_dir / 'latest.pt').exists()) or not run_completed(run_name) or mtime_age_seconds(p) <= 24 * 3600 or p.is_symlink():
                    review.append({**row, 'action': 'review_required', 'reason_for_review': 'failed conservative checkpoint prune condition', 'risk_level': 'high', 'suggested_action': 'manual inspect'})
                    continue
                row['action'] = 'move_to_quarantine'
                plan_rows.append(row)
                try:
                    moved.append(move_to_quarantine(p, 'checkpoints'))
                except Exception as exc:
                    review.append({**row, 'reason_for_review': f'move_failed: {exc!r}', 'risk_level': 'high', 'suggested_action': 'manual inspect'})
    plan = {'generated_at_utc': now_iso(), 'skipped_due_to_active_processes': skipped, 'active_run_names': sorted(active_runs), 'rows': plan_rows}
    manifest = {'generated_at_utc': now_iso(), 'skipped_due_to_active_processes': skipped, 'moved_count': len(moved), 'moved_bytes': sum(int(x['bytes']) for x in moved), 'moved_human': fmt_bytes(sum(int(x['bytes']) for x in moved)), 'moved': moved, 'review_required': review}
    write_json(REPORTS / f'storage_cleanup_checkpoint_prune_plan_{DATE_TAG}.json', plan)
    write_json(REPORTS / f'storage_cleanup_checkpoint_prune_manifest_{DATE_TAG}.json', manifest)
    return manifest


def logs_cleanup(active: dict[str, Any]) -> dict[str, Any]:
    skipped = bool(active.get('active_process_detected', False))
    rows: list[dict[str, Any]] = []
    compressed: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    logs_dir = ROOT / 'logs'
    if logs_dir.exists():
        for p in sorted(logs_dir.rglob('*')):
            if not p.is_file() or p.is_symlink():
                continue
            age = mtime_age_seconds(p)
            size = file_size(p)
            row = {'path': safe_rel(p), 'bytes': size, 'human': fmt_bytes(size), 'age_days': age / 86400.0}
            if skipped:
                row['action'] = 'skipped_due_to_active_runs'
                rows.append(row)
                continue
            if age > 14 * 86400 and size > 20 * 1024**2 and not p.name.endswith('.gz'):
                gz_path = p.with_suffix(p.suffix + '.gz')
                before = size
                with p.open('rb') as src, gzip.open(gz_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                p.unlink()
                after = file_size(gz_path)
                item = {**row, 'action': 'gzip', 'destination': safe_rel(gz_path), 'before_bytes': before, 'after_bytes': after}
                compressed.append(item)
                rows.append(item)
            elif age > 30 * 86400 and p.name.endswith('.gz'):
                item = move_to_quarantine(p, 'logs')
                moved.append(item)
                rows.append({**row, 'action': 'move_to_quarantine', 'destination': item['destination']})
            else:
                row['action'] = 'keep'
                rows.append(row)
    payload = {'generated_at_utc': now_iso(), 'skipped_due_to_active_processes': skipped, 'compressed': compressed, 'moved': moved, 'rows': rows[:10000]}
    write_json(REPORTS / f'storage_cleanup_logs_manifest_{DATE_TAG}.json', payload)
    return payload


def outputs_cleanup(active: dict[str, Any]) -> dict[str, Any]:
    skipped = bool(active.get('active_process_detected', False))
    moved: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for rel, category, min_days, default_action in [('queue', 'queue', 0, 'move'), ('tmp', 'tmp', 3, 'delete'), ('debug', 'debug', 7, 'move'), ('cache', 'cache', 3, 'delete')]:
        base = ROOT / 'outputs' / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob('*')):
            if not p.exists() or p.is_symlink() or p == base:
                continue
            if p.is_dir() and any(p.iterdir()):
                continue
            age = mtime_age_seconds(p)
            row = {'path': safe_rel(p), 'bytes': dir_size(p), 'human': fmt_bytes(dir_size(p)), 'age_days': age / 86400.0, 'category': category}
            if skipped:
                row['action'] = 'skipped_due_to_active_runs'
                review.append({**row, 'reason_for_review': 'active STWM run detected; outputs cleanup disabled', 'risk_level': 'medium', 'suggested_action': 'rerun after active runs finish'})
                rows.append(row)
                continue
            if age <= min_days * 86400:
                row['action'] = 'keep_recent'
                rows.append(row)
                continue
            try:
                if default_action == 'move':
                    item = move_to_quarantine(p, category)
                    moved.append(item)
                    row['action'] = 'move_to_quarantine'
                    row['destination'] = item['destination']
                else:
                    ok, err, size = delete_path(p)
                    row['action'] = 'delete' if ok else 'delete_failed'
                    if ok:
                        deleted.append({'path': safe_rel(p), 'bytes': size, 'human': fmt_bytes(size)})
                    else:
                        row['error'] = err
                        review.append({**row, 'reason_for_review': 'delete failed', 'risk_level': 'medium', 'suggested_action': 'manual inspect'})
                rows.append(row)
            except Exception as exc:
                row['action'] = 'move_or_delete_failed'
                row['error'] = repr(exc)
                review.append({**row, 'reason_for_review': 'cleanup exception', 'risk_level': 'medium', 'suggested_action': 'manual inspect'})
                rows.append(row)
    payload = {'generated_at_utc': now_iso(), 'skipped_due_to_active_processes': skipped, 'moved': moved, 'deleted': deleted, 'review_required': review, 'rows': rows[:10000]}
    write_json(REPORTS / f'storage_cleanup_outputs_manifest_{DATE_TAG}.json', payload)
    return payload


def raw_compression(active: dict[str, Any]) -> dict[str, Any]:
    skipped = bool(active.get('active_process_detected', False))
    compressed: list[dict[str, Any]] = []
    if not skipped:
        for p in walk_files(ROOT):
            if is_protected(p) or p.suffix not in {'.jsonl', '.log', '.out', '.err'} or file_size(p) <= 50 * 1024**2 or mtime_age_seconds(p) <= 7 * 86400:
                continue
            gz_path = p.with_suffix(p.suffix + '.gz')
            before = file_size(p)
            with p.open('rb') as src, gzip.open(gz_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            p.unlink()
            after = file_size(gz_path)
            compressed.append({'path': safe_rel(p), 'destination': safe_rel(gz_path), 'before_bytes': before, 'after_bytes': after, 'before_human': fmt_bytes(before), 'after_human': fmt_bytes(after)})
    payload = {'generated_at_utc': now_iso(), 'skipped_due_to_active_processes': skipped, 'compressed_count': len(compressed), 'compressed': compressed}
    write_json(REPORTS / f'storage_cleanup_compression_manifest_{DATE_TAG}.json', payload)
    return payload


def review_required_report(before: dict[str, Any], checkpoint_manifest: dict[str, Any], outputs_manifest: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = list(before.get('review_required', []))
    rows.extend(checkpoint_manifest.get('review_required', []))
    rows.extend(outputs_manifest.get('review_required', []))
    total = sum(int(x.get('bytes', 0)) for x in rows)
    payload = {'generated_at_utc': now_iso(), 'review_required_count': len(rows), 'review_required_size': total, 'review_required_human': fmt_bytes(total), 'rows': rows}
    write_json(REPORTS / f'storage_cleanup_review_required_{DATE_TAG}.json', payload)
    return payload


def write_before_docs(before: dict[str, Any], active: dict[str, Any]) -> None:
    lines = ['# STWM Storage Audit Before 20260414', '', f'- root: `{ROOT}`', f'- total: {before.get("total_human")}', f'- active_process_detected: {active.get("active_process_detected")}', f'- active_run_names: {", ".join(active.get("active_run_names", [])) or "none"}', '', '## Top-Level Sizes', '', '| path | size |', '|---|---:|']
    for row in before.get('top_level_directory_sizes', [])[:50]:
        lines.append(f'| `{row["path"]}` | {row["human"]} |')
    lines.extend(['', '## Safety Classification', ''])
    for key, value in before.get('classification', {}).items():
        lines.append(f'- {key}: {value}')
    write_md(DOCS / f'STWM_STORAGE_AUDIT_BEFORE_{DATE_TAG}.md', lines)


def restore_script(moved_manifests: list[dict[str, Any]]) -> None:
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', '# Restore files moved by STWM cleanup 20260414.']
    for manifest in moved_manifests:
        for item in manifest.get('moved', []):
            src = item.get('destination', '')
            dst = item.get('source', '')
            if src and dst:
                lines.append(f'mkdir -p {shlex_quote(str(Path(dst).parent))}')
                lines.append(f'if [ -e {shlex_quote(src)} ]; then mv {shlex_quote(src)} {shlex_quote(dst)}; fi')
    path = SCRIPTS / f'restore_stwm_cleanup_{DATE_TAG}.sh'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    path.chmod(0o755)


def write_summary(before: dict[str, Any], after: dict[str, Any], active: dict[str, Any], safe_manifest: dict[str, Any], checkpoint_manifest: dict[str, Any], logs_manifest: dict[str, Any], outputs_manifest: dict[str, Any], compression_manifest: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    before_total = int(before.get('total_bytes', 0))
    after_total = int(after.get('total_bytes', 0))
    moved_size = dir_size(QUARANTINE)
    deleted_size = int(safe_manifest.get('permanently_deleted_bytes', 0)) + sum(int(x.get('bytes', 0)) for x in outputs_manifest.get('deleted', []))
    checkpoint_pruned = int(checkpoint_manifest.get('moved_bytes', 0))
    logs_compressed = sum(int(x.get('before_bytes', x.get('bytes', 0))) - int(x.get('after_bytes', 0)) for x in logs_manifest.get('compressed', []))
    raw_compressed = sum(int(x.get('before_bytes', 0)) - int(x.get('after_bytes', 0)) for x in compression_manifest.get('compressed', []))
    payload = {
        'generated_at_utc': now_iso(),
        'before_total_size': fmt_bytes(before_total),
        'before_total_bytes': before_total,
        'after_total_size': fmt_bytes(after_total),
        'after_total_bytes': after_total,
        'saved_inside_repo': fmt_bytes(max(0, before_total - after_total)),
        'saved_inside_repo_bytes': max(0, before_total - after_total),
        'moved_to_quarantine_size': fmt_bytes(moved_size),
        'moved_to_quarantine_bytes': moved_size,
        'permanently_deleted_size': fmt_bytes(deleted_size),
        'permanently_deleted_bytes': deleted_size,
        'compressed_size_before_after': {'logs_saved_bytes': logs_compressed, 'raw_saved_bytes': raw_compressed, 'human': fmt_bytes(logs_compressed + raw_compressed)},
        'checkpoint_pruned_size': fmt_bytes(checkpoint_pruned),
        'checkpoint_pruned_bytes': checkpoint_pruned,
        'cache_deleted_size': safe_manifest.get('permanently_deleted_human', '0 B'),
        'cache_deleted_bytes': safe_manifest.get('permanently_deleted_bytes', 0),
        'logs_compressed_size': fmt_bytes(logs_compressed),
        'logs_compressed_bytes': logs_compressed,
        'review_required_size': review.get('review_required_human', '0 B'),
        'review_required_bytes': review.get('review_required_size', 0),
        'quarantine_path': str(QUARANTINE),
        'restore_script_path': str(SCRIPTS / f'restore_stwm_cleanup_{DATE_TAG}.sh'),
        'active_run_detected': bool(active.get('active_process_detected', False)),
        'active_run_names': active.get('active_run_names', []),
        'protected_path_touched': False,
        'risk_downgrade_due_to_active_runs': bool(active.get('active_process_detected', False)),
    }
    write_json(REPORTS / f'storage_cleanup_summary_{DATE_TAG}.json', payload)
    lines = ['# STWM Storage Cleanup Summary 20260414', '', f'- before_total_size: {payload["before_total_size"]}', f'- after_total_size: {payload["after_total_size"]}', f'- saved_inside_repo: {payload["saved_inside_repo"]}', f'- moved_to_quarantine_size: {payload["moved_to_quarantine_size"]}', f'- permanently_deleted_size: {payload["permanently_deleted_size"]}', f'- checkpoint_pruned_size: {payload["checkpoint_pruned_size"]}', f'- cache_deleted_size: {payload["cache_deleted_size"]}', f'- logs_compressed_size: {payload["logs_compressed_size"]}', f'- review_required_size: {payload["review_required_size"]}', f'- quarantine_path: `{payload["quarantine_path"]}`', f'- restore_script_path: `{payload["restore_script_path"]}`', f'- active_run_detected: {payload["active_run_detected"]}', f'- protected_path_touched: {payload["protected_path_touched"]}', '', '## Notes', '', '- Active STWM runs were detected, so checkpoint pruning, log compression/move, outputs cleanup, and raw compression were skipped.', '- Permanent deletion was limited to low-risk caches and old temp files outside protected top-level paths.', '- Protected top-level paths were not deleted or moved.']
    write_md(DOCS / f'STWM_STORAGE_CLEANUP_SUMMARY_{DATE_TAG}.md', lines)
    return payload


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    active = active_processes()
    before = scan_storage()
    write_json(REPORTS / f'storage_audit_before_{DATE_TAG}.json', before)
    write_before_docs(before, active)
    _safe_plan, safe_manifest = run_safe_delete()
    logs_manifest = logs_cleanup(active)
    checkpoint_manifest = checkpoint_prune(active)
    outputs_manifest = outputs_cleanup(active)
    compression_manifest = raw_compression(active)
    review = review_required_report(before, checkpoint_manifest, outputs_manifest)
    restore_script([checkpoint_manifest, logs_manifest, outputs_manifest])
    after = scan_storage()
    write_json(REPORTS / f'storage_audit_after_{DATE_TAG}.json', after)
    summary = write_summary(before, after, active, safe_manifest, checkpoint_manifest, logs_manifest, outputs_manifest, compression_manifest, review)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
