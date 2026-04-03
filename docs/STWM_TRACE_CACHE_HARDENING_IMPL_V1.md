# STWM Trace Cache Hardening Implementation V1

Date: 2026-04-03
Status: Phase 1 implemented and verified (runtime rerun in progress)

## 1) Scope

This implementation targets the minimal hardening required to stop trace-cache corruption from terminating D1 runs:

1. atomic cache write
2. corruption quarantine
3. read-fail auto rebuild
4. concurrent rebuild protection
5. cache metadata version/hash validation

## 2) Code Changes

### 2.1 Trace cache hardening

File:

- `code/stwm/modules/trace_adapter.py`

Implemented:

1. `tmp -> fsync(file) -> os.replace -> fsync(dir)` atomic write path for `.npz`.
2. lock-file (`.lock`) + `fcntl.flock(LOCK_EX)` around rebuild/write path.
3. recoverable cache error handling (`BadZipFile`/`EOFError`-like corruption, metadata parse/validation failures) with auto rebuild.
4. quarantine move to `trace_summaries/quarantine/*.bad_*.npz` before rebuild.
5. cache metadata blob (`cache_metadata_json`) persisted and validated on read.
6. metadata includes:
   - `cache_version`
   - `manifest_hash`
   - `frontend_hash`
7. sample cache key now incorporates version/hash context to avoid stale cache collisions.

### 2.2 Manifest hash injection

File:

- `code/stwm/datasets/stwm_dataset.py`

Implemented:

1. one-time manifest content hash (`sha1`) at dataset load.
2. per-sample metadata now carries `manifest_hash` for trace cache metadata/keying.

### 2.3 Recovery validation tool

File:

- `code/stwm/tools/run_trace_cache_recovery_test_v1.py`

Implemented checks:

1. corruption -> quarantine -> rebuild (must succeed)
2. cache-version key isolation (v2/v3 should not collide)
3. concurrent same-key encode sanity (2 processes, single `.npz` output)

## 3) Verification Artifacts

Primary report:

- `reports/stwm_trace_cache_recovery_test_v1.json`

Observed results:

1. `corruption_rebuild.ok = true`
2. `cache_version_key_isolation.ok = true`
3. `concurrent_encode_same_key.ok = true`

## 4) D1 Failed Job Targeted Rerun (Phase 1 contract)

Requeued exactly two failed jobs:

1. `full_v4_2_seed42_fixed_warmup_lambda1`
2. `wo_semantics_v4_2_seed42`

Queue evidence:

- new status files:
  - `outputs/queue/stwm_protocol_v2/d1_train/status/1775206819994_full_v4_2_seed42_fixed_warmup_lambda1.status.json`
  - `outputs/queue/stwm_protocol_v2/d1_train/status/1775206820381_wo_semantics_v4_2_seed42.status.json`
- both currently `state=running`

Runtime evidence (resume outputs receiving new rows):

- `outputs/training/stwm_v4_2_220m_protocol_diag_v1/seed_42/full_v4_2_seed42_fixed_warmup_lambda1/train_log.jsonl`
- `outputs/training/stwm_v4_2_220m_protocol_diag_v1/seed_42/wo_semantics_v4_2_seed42/train_log.jsonl`

No immediate repeat of prior `BadZipFile/EOFError` at submission window.

## 5) Notes

1. Existing run outputs are preserved (`--auto-resume` behavior unchanged).
2. The Phase 1 patch is intentionally localized; model/trainer architecture is not changed.
