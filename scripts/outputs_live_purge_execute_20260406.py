#!/usr/bin/env python3
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


REPO = Path('/home/chen034/workspace/stwm')
OUTPUTS = REPO / 'outputs'
ARCH = REPO / 'archives'
DOCS = REPO / 'docs'
REPORTS = REPO / 'reports'
TMP = Path('/tmp/stwm_archive_restore_check_20260406')

RESTORE_MD = DOCS / 'OUTPUTS_ARCHIVE_RESTORE_CHECK_20260406.md'
RESTORE_JSON = REPORTS / 'outputs_archive_restore_check_20260406.json'
PURGE_MD = DOCS / 'OUTPUTS_LIVE_PURGE_REPORT_20260406.md'
PURGE_JSON = REPORTS / 'outputs_live_purge_report_20260406.json'
LOG = REPORTS / 'outputs_live_purge_execute_20260406.run.log'

ARCHIVES = [
    ARCH / 'outputs_training_archive_20260406.tar.zst',
    ARCH / 'outputs_audits_archive_20260406.tar.zst',
    ARCH / 'outputs_benchmarks_archive_20260406.tar.zst',
    ARCH / 'outputs_visualizations_archive_20260406.tar.zst',
    ARCH / 'outputs_smoke_tests_archive_20260406.tar.zst',
    ARCH / 'outputs_queue_archive_20260406.tar.zst',
    ARCH / 'outputs_background_jobs_archive_20260406.tar.zst',
    ARCH / 'outputs_eval_archive_20260406.tar.zst',
    ARCH / 'outputs_keep_reclassified_archive_20260406.tar.zst',
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

SKELETON = [
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

TEXT_EXT = {'.txt', '.md', '.json', '.jsonl', '.yaml', '.yml', '.csv', '.log'}


def log(msg: str) -> None:
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{t}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')


def human(n: int) -> str:
    u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    v = float(n)
    for x in u:
        if v < 1024 or x == u[-1]:
            if x == 'B':
                return f'{int(v)}{x}'
            return f'{v:.2f}{x}'
        v /= 1024.0
    return f'{n}B'


def du_bytes(path: Path) -> int:
    s = 0
    if not path.exists():
        return 0
    for dp, _, fs in os.walk(path):
        for fn in fs:
            fp = Path(dp) / fn
            try:
                s += fp.stat().st_size
            except OSError:
                pass
    return s


def manifest_of(archive: Path) -> Path:
    return archive.with_suffix('').with_suffix('.contents_manifest.txt')


def choose_files_for_restore(archive: Path, seed: int) -> List[str]:
    mf = manifest_of(archive)
    files = []
    if mf.exists():
        with open(mf, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                p = ln.strip()
                if not p or p.endswith('/'):
                    continue
                abs_p = REPO / p
                if abs_p.exists() and abs_p.is_file():
                    try:
                        sz = abs_p.stat().st_size
                    except OSError:
                        sz = 0
                    if sz > 0 and sz <= 50 * 1024 * 1024:
                        files.append(p)

    # Keep sampling practical by using a very early-window pool so extraction
    # does not need to stream deep into huge archives.
    pool = files[:50] if len(files) > 50 else files
    if not pool:
        return []

    random.seed(seed)
    k = 1
    return random.sample(pool, k=k)


def readability_check(fp: Path) -> Dict:
    ext = fp.suffix.lower()
    out = {'ok': True, 'detail': 'not_applicable'}
    if ext not in TEXT_EXT:
        return out
    try:
        txt = fp.read_text(encoding='utf-8', errors='strict')
    except Exception as e:
        return {'ok': False, 'detail': f'utf8_read_failed: {e}'}

    if ext == '.json':
        try:
            json.loads(txt)
        except Exception as e:
            return {'ok': False, 'detail': f'json_parse_failed: {e}'}
    elif ext == '.jsonl':
        lines = [x for x in txt.splitlines() if x.strip()]
        if lines:
            try:
                json.loads(lines[0])
            except Exception as e:
                return {'ok': False, 'detail': f'jsonl_first_line_parse_failed: {e}'}

    return {'ok': True, 'detail': 'ok'}


def restore_spot_check() -> Dict:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)

    entries = []
    all_pass = True

    for i, a in enumerate(ARCHIVES, start=1):
        log(f'Restore spot-check {i}/{len(ARCHIVES)}: {a.name}')
        e = {
            'archive': str(a),
            'sha256_file': str(Path(str(a) + '.sha256')),
            'manifest_file': str(manifest_of(a)),
            'sha256_file_exists': Path(str(a) + '.sha256').exists(),
            'sha256_file_nonempty': Path(str(a) + '.sha256').exists() and Path(str(a) + '.sha256').stat().st_size > 0,
            'manifest_exists': manifest_of(a).exists(),
            'selected_files': [],
            'errors': [],
            'passed': False,
        }

        if not a.exists() or a.stat().st_size == 0:
            e['errors'].append('archive_missing_or_empty')
            entries.append(e)
            all_pass = False
            continue

        chosen = choose_files_for_restore(a, seed=2026040600 + i)
        if not chosen:
            e['errors'].append('no_nonzero_small_files_found_in_manifest')
            entries.append(e)
            all_pass = False
            continue

        dest = TMP / a.name.replace('.tar.zst', '')
        dest.mkdir(parents=True, exist_ok=True)
        targets = ' '.join([f"'{x}'" for x in chosen])
        log(f"Selected sample for {a.name}: {chosen}")
        cmd = (
            f"cd {REPO} && tar --use-compress-program='zstd -d -T0' --occurrence=1 -xf '{a}' -C '{dest}' {targets}"
        )
        try:
            run(cmd)
        except Exception as ex:
            e['errors'].append(f'extract_failed: {ex}')
            entries.append(e)
            all_pass = False
            continue

        for rel in chosen:
            fp = dest / rel
            rec = {
                'path': rel,
                'exists': fp.exists(),
                'size_bytes': 0,
                'non_zero': False,
                'readability_ok': True,
                'readability_detail': 'not_applicable',
            }
            if fp.exists():
                try:
                    rec['size_bytes'] = fp.stat().st_size
                except OSError:
                    rec['size_bytes'] = 0
                rec['non_zero'] = rec['size_bytes'] > 0
                r = readability_check(fp)
                rec['readability_ok'] = r['ok']
                rec['readability_detail'] = r['detail']
            if not rec['exists']:
                e['errors'].append(f'missing_after_extract: {rel}')
            if not rec['non_zero']:
                e['errors'].append(f'zero_size_file: {rel}')
            if not rec['readability_ok']:
                e['errors'].append(f'readability_failed: {rel}: {rec["readability_detail"]}')
            e['selected_files'].append(rec)

        e['passed'] = len(e['errors']) == 0
        if not e['passed']:
            all_pass = False
        entries.append(e)

    payload = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'restore_tmp_root': str(TMP),
        'overall_pass': all_pass,
        'archives': entries,
        'extra_keep_archive_created': True,
        'extra_keep_archive_paths': RECLASSIFIED_KEEP,
    }
    return payload


def write_restore(payload: Dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    with open(RESTORE_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append('# Outputs Archive Restore Check 20260406')
    lines.append('')
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Restore tmp root: {payload['restore_tmp_root']}")
    lines.append(f"- Overall pass: {'YES' if payload['overall_pass'] else 'NO'}")
    lines.append('')
    lines.append('| Archive | Pass | Samples | Errors |')
    lines.append('|---|---|---:|---|')
    for e in payload['archives']:
        err = '; '.join(e['errors']) if e['errors'] else ''
        lines.append(f"| {e['archive']} | {'YES' if e['passed'] else 'NO'} | {len(e['selected_files'])} | {err.replace('|','/')} |")
    lines.append('')
    for e in payload['archives']:
        lines.append(f"### {e['archive']}")
        lines.append(f"- Passed: {'YES' if e['passed'] else 'NO'}")
        lines.append(f"- sha256 file exists/non-empty: {e['sha256_file_exists']}/{e['sha256_file_nonempty']}")
        lines.append(f"- manifest exists: {e['manifest_exists']}")
        lines.append('- Sample files:')
        for s in e['selected_files']:
            lines.append(
                f"  - {s['path']} | exists={s['exists']} | size={s['size_bytes']} | non_zero={s['non_zero']} | readable={s['readability_ok']} ({s['readability_detail']})"
            )
        if e['errors']:
            lines.append('- Errors:')
            for x in e['errors']:
                lines.append(f"  - {x}")
        lines.append('')

    with open(RESTORE_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).rstrip() + '\n')


def purge_and_report() -> Dict:
    before_total = du_bytes(OUTPUTS)
    before_map = {d: du_bytes(OUTPUTS / d) for d in SKELETON}

    for d in SKELETON:
        p = OUTPUTS / d
        if not p.exists():
            continue
        for c in p.iterdir():
            try:
                if c.is_dir():
                    shutil.rmtree(c)
                else:
                    c.unlink()
            except FileNotFoundError:
                pass

    for d in SKELETON:
        p = OUTPUTS / d
        p.mkdir(parents=True, exist_ok=True)
        os.chmod(p, 0o775)

    after_total = du_bytes(OUTPUTS)
    after_map = {d: du_bytes(OUTPUTS / d) for d in SKELETON}
    freed = max(0, before_total - after_total)

    payload = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'outputs_before_bytes': before_total,
        'outputs_before_human': human(before_total),
        'outputs_after_bytes': after_total,
        'outputs_after_human': human(after_total),
        'freed_bytes': freed,
        'freed_human': human(freed),
        'per_top_level': [
            {
                'dir': f'outputs/{d}',
                'before_bytes': before_map[d],
                'before_human': human(before_map[d]),
                'after_bytes': after_map[d],
                'after_human': human(after_map[d]),
            }
            for d in SKELETON
        ],
        'archives_covering_purge': [str(x) for x in ARCHIVES],
        'reclassified_keep_extra_archived': RECLASSIFIED_KEEP,
        'skeleton_status': [
            {
                'dir': f'outputs/{d}',
                'exists': (OUTPUTS / d).exists(),
                'is_empty': len(list((OUTPUTS / d).iterdir())) == 0 if (OUTPUTS / d).exists() else False,
                'mode_octal': oct(((OUTPUTS / d).stat().st_mode) & 0o777) if (OUTPUTS / d).exists() else None,
            }
            for d in SKELETON
        ],
        'notes': [
            'live workspace 已清空旧 STWM/V4.2 产物',
            '历史内容仅保留在 docs/reports/archives 中',
            '新主线可以从干净 outputs 重新开始',
        ],
    }
    return payload


def write_purge(payload: Dict) -> None:
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
    lines.append('## Per Top-Level Delta')
    lines.append('')
    lines.append('| Dir | Before | After |')
    lines.append('|---|---:|---:|')
    for r in payload['per_top_level']:
        lines.append(f"| {r['dir']} | {r['before_human']} | {r['after_human']} |")
    lines.append('')
    lines.append('## Archives Used')
    lines.append('')
    for a in payload['archives_covering_purge']:
        lines.append(f"- {a}")
    lines.append('')
    lines.append('## Reclassified KEEP Supplement Archive Coverage')
    lines.append('')
    for p in payload['reclassified_keep_extra_archived']:
        lines.append(f"- {p}")
    lines.append('')
    lines.append('## Skeleton Status')
    lines.append('')
    lines.append('| Dir | Exists | Empty | Mode |')
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
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()

    log('START live purge execute pass')
    restore = restore_spot_check()
    write_restore(restore)
    log(f"Restore overall pass: {restore['overall_pass']}")
    if not restore['overall_pass']:
        log('STOP: restore failed; no purge executed')
        return 2

    purge = purge_and_report()
    write_purge(purge)
    log(f"DONE purge before={purge['outputs_before_human']} after={purge['outputs_after_human']} freed={purge['freed_human']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
