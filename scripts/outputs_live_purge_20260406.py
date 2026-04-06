#!/usr/bin/env python3
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path('/home/chen034/workspace/stwm')
OUTPUTS_ROOT = REPO_ROOT / 'outputs'
ARCHIVES_ROOT = REPO_ROOT / 'archives'
REPORTS_ROOT = REPO_ROOT / 'reports'
DOCS_ROOT = REPO_ROOT / 'docs'

RESTORE_TMP_ROOT = Path('/tmp/stwm_archive_restore_check_20260406')

RESTORE_MD = DOCS_ROOT / 'OUTPUTS_ARCHIVE_RESTORE_CHECK_20260406.md'
RESTORE_JSON = REPORTS_ROOT / 'outputs_archive_restore_check_20260406.json'

PURGE_MD = DOCS_ROOT / 'OUTPUTS_LIVE_PURGE_REPORT_20260406.md'
PURGE_JSON = REPORTS_ROOT / 'outputs_live_purge_report_20260406.json'

LOG_FILE = REPORTS_ROOT / 'outputs_live_purge_20260406.run.log'

BASE_ARCHIVES = [
    ARCHIVES_ROOT / 'outputs_training_archive_20260406.tar.zst',
    ARCHIVES_ROOT / 'outputs_audits_archive_20260406.tar.zst',
    ARCHIVES_ROOT / 'outputs_benchmarks_archive_20260406.tar.zst',
    ARCHIVES_ROOT / 'outputs_visualizations_archive_20260406.tar.zst',
    ARCHIVES_ROOT / 'outputs_smoke_tests_archive_20260406.tar.zst',
    ARCHIVES_ROOT / 'outputs_queue_archive_20260406.tar.zst',
    ARCHIVES_ROOT / 'outputs_background_jobs_archive_20260406.tar.zst',
    ARCHIVES_ROOT / 'outputs_eval_archive_20260406.tar.zst',
]

RECLASSIFIED_KEEP = [
    'outputs/training/stwm_v4_2_real_1b',
    'outputs/training/stwm_v4_2_real_220m',
    'outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_semteacher_mainline_seed42_v1',
    'outputs/eval/stwm_v4_2_completed_protocol_eval_20260403',
    'outputs/eval/stwm_v4_2_completed_protocol_eval_real_evalonly_20260403',
    'outputs/queue/stwm_protocol_v2',
    'outputs/queue/stwm_protocol_v2_frontend_default_v1',
    'outputs/monitoring/stwm_hourly_push',
    'outputs/baselines',
]

EXTRA_ARCHIVE = ARCHIVES_ROOT / 'outputs_keep_reclassified_archive_20260406.tar.zst'
EXTRA_SHA = Path(str(EXTRA_ARCHIVE) + '.sha256')
EXTRA_MANIFEST = ARCHIVES_ROOT / 'outputs_keep_reclassified_archive_20260406.contents_manifest.txt'
EXTRA_LIST = REPORTS_ROOT / 'archive_lists_20260406' / 'keep_reclassified_paths.txt'

SKELETON_DIRS = [
    'audits',
    'background_jobs',
    'baselines',
    'benchmarks',
    'eval',
    'monitoring',
    'queue',
    'smoke_tests',
    'training',
    'visualizations',
]

TEXT_EXTS = {
    '.txt', '.md', '.json', '.jsonl', '.yaml', '.yml', '.csv', '.log', '.py', '.sh'
}


def log(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_cmd(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')


def human_size(n: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            if u == 'B':
                return f'{int(v)}{u}'
            return f'{v:.2f}{u}'
        v /= 1024.0
    return f'{n}B'


def du_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dp, _, files in os.walk(path):
        for fn in files:
            fp = Path(dp) / fn
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def archive_manifest_path(archive_path: Path) -> Path:
    return archive_path.with_suffix('').with_suffix('.contents_manifest.txt')


def is_path_covered_by_archives(rel_path: str, manifest_files: List[Path]) -> bool:
    prefix = rel_path.rstrip('/') + '/'
    for mf in manifest_files:
        if not mf.exists():
            continue
        try:
            with open(mf, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    if p == rel_path or p.startswith(prefix):
                        return True
        except OSError:
            continue
    return False


def ensure_reclassified_keep_archive() -> Tuple[bool, List[str], List[str]]:
    manifest_files = [archive_manifest_path(a) for a in BASE_ARCHIVES]
    uncovered = []
    covered = []

    for rel in RECLASSIFIED_KEEP:
        if is_path_covered_by_archives(rel, manifest_files):
            covered.append(rel)
        else:
            uncovered.append(rel)

    existing_uncovered = [rel for rel in uncovered if (REPO_ROOT / rel).exists()]

    if not existing_uncovered:
        log('No reclassified KEEP paths require extra archive packaging.')
        return False, uncovered, covered

    # Reuse already-created extra archive if all companion files exist.
    if (
        EXTRA_ARCHIVE.exists()
        and EXTRA_ARCHIVE.stat().st_size > 0
        and EXTRA_SHA.exists()
        and EXTRA_SHA.stat().st_size > 0
        and EXTRA_MANIFEST.exists()
        and EXTRA_MANIFEST.stat().st_size > 0
    ):
        log('Reusing existing extra keep-reclassified archive (already present with sha + manifest).')
        return True, existing_uncovered, covered

    EXTRA_LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTRA_LIST, 'w', encoding='utf-8') as f:
        for rel in existing_uncovered:
            f.write(rel + '\n')

    log(f'Creating extra keep-reclassified archive for {len(existing_uncovered)} paths.')

    cmd_pack = (
        f"cd {REPO_ROOT} && "
        f"tar -cvf - -T {EXTRA_LIST} 2> {EXTRA_MANIFEST} | zstd -T0 -1 > {EXTRA_ARCHIVE}"
    )
    run_cmd(cmd_pack)
    run_cmd(f"sha256sum {EXTRA_ARCHIVE} > {EXTRA_SHA}")

    log('Extra keep-reclassified archive created with sha256 and manifest.')
    return True, existing_uncovered, covered


def load_manifest_candidates(archive_path: Path) -> List[str]:
    mf = archive_manifest_path(archive_path)
    out = []
    if mf.exists():
        with open(mf, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                p = line.strip()
                if not p or p.endswith('/'):
                    continue
                # Prefer files that currently exist in live outputs (before purge stage).
                abs_p = REPO_ROOT / p
                if abs_p.exists() and abs_p.is_file():
                    out.append(p)
                else:
                    # Keep as fallback; may still be present in archive.
                    out.append(p)
    return list(dict.fromkeys(out))


def check_text_readability(file_path: Path) -> Tuple[bool, str]:
    ext = file_path.suffix.lower()
    if ext not in TEXT_EXTS:
        return True, 'not_applicable'

    try:
        text = file_path.read_text(encoding='utf-8', errors='strict')
    except Exception as e:
        return False, f'utf8_read_failed: {e}'

    if ext == '.json':
        try:
            json.loads(text)
        except Exception as e:
            return False, f'json_parse_failed: {e}'
    elif ext == '.jsonl':
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            try:
                json.loads(lines[0])
            except Exception as e:
                return False, f'jsonl_first_line_parse_failed: {e}'

    return True, 'ok'


def restore_spot_check_archive(archive_path: Path, seed: int) -> Dict:
    random.seed(seed)
    name = archive_path.name
    target_dir = RESTORE_TMP_ROOT / name.replace('.tar.zst', '')
    target_dir.mkdir(parents=True, exist_ok=True)

    sha_file = Path(str(archive_path) + '.sha256')
    manifest_file = archive_manifest_path(archive_path)

    result = {
        'archive': str(archive_path),
        'sha256_file': str(sha_file),
        'manifest_file': str(manifest_file),
        'sha256_file_exists': sha_file.exists(),
        'sha256_file_nonempty': sha_file.exists() and sha_file.stat().st_size > 0,
        'manifest_exists': manifest_file.exists(),
        'selected_files': [],
        'passed': False,
        'errors': [],
    }

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        result['errors'].append('archive_missing_or_empty')
        return result

    candidates = load_manifest_candidates(archive_path)
    if not candidates:
        result['errors'].append('no_file_candidates_from_manifest')
        return result

    random.shuffle(candidates)
    target_n = 3 if len(candidates) >= 3 else len(candidates)
    target_n = max(1, target_n)

    # Prefer small non-zero files for quick recoverability checks.
    small_pool = []
    non_zero_pool = []
    for c in candidates:
        abs_p = REPO_ROOT / c
        if abs_p.exists() and abs_p.is_file():
            try:
                sz = abs_p.stat().st_size
            except OSError:
                sz = 0
            if sz > 0:
                non_zero_pool.append(c)
                if sz <= 50 * 1024 * 1024:
                    small_pool.append(c)

    pool = small_pool if len(small_pool) >= target_n else (non_zero_pool if non_zero_pool else candidates)
    random.shuffle(pool)
    selected = pool[:target_n]

    extract_targets = ' '.join([f"'{x}'" for x in selected])
    cmd_extract = (
        f"cd {REPO_ROOT} && "
        f"tar --use-compress-program='zstd -d -T0' -xf '{archive_path}' -C '{target_dir}' {extract_targets}"
    )

    try:
        run_cmd(cmd_extract)
    except Exception as e:
        result['errors'].append(f'extract_failed: {e}')
        return result

    for rel in selected:
        extracted = target_dir / rel
        rec = {
            'path': rel,
            'exists': extracted.exists(),
            'size_bytes': 0,
            'non_zero': False,
            'readability_check': 'not_applicable',
            'readability_ok': True,
        }

        if not extracted.exists():
            rec['readability_ok'] = False
            rec['readability_check'] = 'missing'
            result['selected_files'].append(rec)
            result['errors'].append(f'missing_after_extract: {rel}')
            continue

        try:
            rec['size_bytes'] = extracted.stat().st_size
        except OSError:
            rec['size_bytes'] = 0
        rec['non_zero'] = rec['size_bytes'] > 0
        if not rec['non_zero']:
            result['errors'].append(f'zero_size_file: {rel}')

        ok, detail = check_text_readability(extracted)
        rec['readability_ok'] = ok
        rec['readability_check'] = detail
        if not ok:
            result['errors'].append(f'readability_failed: {rel}: {detail}')

        result['selected_files'].append(rec)

    result['passed'] = len(result['errors']) == 0 and len(result['selected_files']) >= 1
    return result


def write_restore_reports(payload: Dict) -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)

    with open(RESTORE_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append('# Outputs Archive Restore Check 20260406')
    lines.append('')
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Restore tmp root: {payload['restore_tmp_root']}")
    lines.append(f"- Overall pass: {'YES' if payload['overall_pass'] else 'NO'}")
    lines.append('')

    lines.append('## Archive Results')
    lines.append('')
    lines.append('| Archive | Spot-check pass | Selected file count | Errors |')
    lines.append('|---|---|---:|---|')
    for e in payload['archives']:
        err = '; '.join(e['errors']) if e['errors'] else ''
        lines.append(
            f"| {e['archive']} | {'YES' if e['passed'] else 'NO'} | {len(e['selected_files'])} | {err.replace('|', '/')} |"
        )
    lines.append('')

    for e in payload['archives']:
        lines.append(f"### {e['archive']}")
        lines.append(f"- Passed: {'YES' if e['passed'] else 'NO'}")
        lines.append(f"- sha256 file exists/non-empty: {e['sha256_file_exists']}/{e['sha256_file_nonempty']}")
        lines.append(f"- manifest exists: {e['manifest_exists']}")
        if e['errors']:
            lines.append('- Errors:')
            for err in e['errors']:
                lines.append(f"  - {err}")
        lines.append('- Extracted samples:')
        for s in e['selected_files']:
            lines.append(
                f"  - {s['path']} | exists={s['exists']} | size={s['size_bytes']} | non_zero={s['non_zero']} | readable={s['readability_ok']} ({s['readability_check']})"
            )
        lines.append('')

    with open(RESTORE_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).rstrip() + '\n')


def purge_outputs(archives_used: List[str], reclassified_extra: List[str]) -> Dict:
    before_total = du_bytes(OUTPUTS_ROOT)
    before_map = {d: du_bytes(OUTPUTS_ROOT / d) for d in SKELETON_DIRS}

    for d in SKELETON_DIRS:
        p = OUTPUTS_ROOT / d
        if not p.exists():
            continue
        for child in p.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except FileNotFoundError:
                pass

    # Recreate skeleton and normalize perms.
    for d in SKELETON_DIRS:
        p = OUTPUTS_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        os.chmod(p, 0o775)

    after_total = du_bytes(OUTPUTS_ROOT)
    after_map = {d: du_bytes(OUTPUTS_ROOT / d) for d in SKELETON_DIRS}

    skeleton_status = []
    for d in SKELETON_DIRS:
        p = OUTPUTS_ROOT / d
        entries = []
        try:
            entries = list(p.iterdir())
        except Exception:
            entries = []
        skeleton_status.append(
            {
                'dir': f'outputs/{d}',
                'exists': p.exists(),
                'is_empty': len(entries) == 0,
                'mode_octal': oct(p.stat().st_mode & 0o777) if p.exists() else None,
            }
        )

    freed = max(0, before_total - after_total)

    payload = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'outputs_before_bytes': before_total,
        'outputs_before_human': human_size(before_total),
        'outputs_after_bytes': after_total,
        'outputs_after_human': human_size(after_total),
        'freed_bytes': freed,
        'freed_human': human_size(freed),
        'per_top_level': [
            {
                'dir': f'outputs/{d}',
                'before_bytes': before_map[d],
                'before_human': human_size(before_map[d]),
                'after_bytes': after_map[d],
                'after_human': human_size(after_map[d]),
            }
            for d in SKELETON_DIRS
        ],
        'archives_covering_purge': archives_used,
        'reclassified_keep_extra_archived': reclassified_extra,
        'skeleton_status': skeleton_status,
        'notes': [
            'live workspace has purged old STWM/V4.2 outputs content under outputs/',
            'historical content is retained in docs/reports/archives',
            'new research line can start from clean outputs skeleton',
        ],
    }
    return payload


def write_purge_reports(payload: Dict) -> None:
    with open(PURGE_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append('# Outputs Live Purge Report 20260406')
    lines.append('')
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- outputs before: {payload['outputs_before_human']} ({payload['outputs_before_bytes']} bytes)")
    lines.append(f"- outputs after: {payload['outputs_after_human']} ({payload['outputs_after_bytes']} bytes)")
    lines.append(f"- freed: {payload['freed_human']} ({payload['freed_bytes']} bytes)")
    lines.append('')

    lines.append('## Per Top-Level Size Delta')
    lines.append('')
    lines.append('| Directory | Before | After |')
    lines.append('|---|---:|---:|')
    for row in payload['per_top_level']:
        lines.append(f"| {row['dir']} | {row['before_human']} | {row['after_human']} |")
    lines.append('')

    lines.append('## Archives Used For Purge Coverage')
    lines.append('')
    for a in payload['archives_covering_purge']:
        lines.append(f"- {a}")
    lines.append('')

    lines.append('## Reclassified KEEP Supplement Archive Paths')
    lines.append('')
    if payload['reclassified_keep_extra_archived']:
        for p in payload['reclassified_keep_extra_archived']:
            lines.append(f"- {p}")
    else:
        lines.append('- none')
    lines.append('')

    lines.append('## Skeleton Status')
    lines.append('')
    lines.append('| Directory | Exists | Empty | Mode |')
    lines.append('|---|---|---|---|')
    for s in payload['skeleton_status']:
        lines.append(f"| {s['dir']} | {s['exists']} | {s['is_empty']} | {s['mode_octal']} |")
    lines.append('')

    lines.append('## Notes')
    lines.append('')
    for n in payload['notes']:
        lines.append(f"- {n}")

    with open(PURGE_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).rstrip() + '\n')


def main() -> int:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log('START outputs live purge workflow 20260406')

    created_extra_archive, extra_archived_paths, _covered_keep = ensure_reclassified_keep_archive()
    archives_for_check = list(BASE_ARCHIVES)
    if created_extra_archive:
        archives_for_check.append(EXTRA_ARCHIVE)

    if RESTORE_TMP_ROOT.exists():
        shutil.rmtree(RESTORE_TMP_ROOT)
    RESTORE_TMP_ROOT.mkdir(parents=True, exist_ok=True)

    restore_entries = []
    overall_pass = True
    for i, arch in enumerate(archives_for_check, start=1):
        log(f'Restore spot-check {i}/{len(archives_for_check)}: {arch.name}')
        entry = restore_spot_check_archive(arch, seed=20260406 + i)
        restore_entries.append(entry)
        if not entry['passed']:
            overall_pass = False

    restore_payload = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'restore_tmp_root': str(RESTORE_TMP_ROOT),
        'overall_pass': overall_pass,
        'archives': restore_entries,
        'extra_keep_archive_created': created_extra_archive,
        'extra_keep_archive_paths': extra_archived_paths,
    }
    write_restore_reports(restore_payload)
    log(f"Restore spot-check overall pass: {overall_pass}")

    if not overall_pass:
        log('STOP: restore check failed; purge not executed.')
        return 2

    archives_used = [str(x) for x in BASE_ARCHIVES]
    if created_extra_archive:
        archives_used.append(str(EXTRA_ARCHIVE))

    purge_payload = purge_outputs(archives_used=archives_used, reclassified_extra=extra_archived_paths)
    write_purge_reports(purge_payload)
    log('DONE purge and report generation.')
    log(f"outputs before={purge_payload['outputs_before_human']} after={purge_payload['outputs_after_human']} freed={purge_payload['freed_human']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
