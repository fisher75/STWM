# STWM Repo Redundancy Cleanup V13

- execute: `True`
- shard_report_count: `295`
- legacy_log_count: `20`
- python_cache_dir_count: `27`
- python_bytecode_file_count: `390`
- scratch_path_count: `3`
- deleted_file_count: `705`
- deleted_dir_count: `30`
- archived_file_count: `315`
- archived_size_bytes: `12215890`
- reclaimed_raw_bytes_estimate: `720657209`

## Archives
- shard_report_archive: `artifacts/stwm_redundant_shard_reports_v13_20260502.tar.gz`
- legacy_log_archive: `artifacts/stwm_legacy_large_logs_v13_20260502.tar.gz`

## Preserved Core Areas
- `data/`
- `models/`
- `outputs/cache/`
- `outputs/checkpoints/`
- `assets/`
- `artifacts/stwm_fstf_visualization_pack_v13_20260502.tar.gz`
- `reports/*summary*.json and final V8/V10/V12/V13 reports`
- `logs/fstf_* and logs/stwm_fstf_*`

## Policy
- Preserve data, models, outputs/cache, outputs/checkpoints, final assets, scripts, source code, and key summary reports.
- Archive shard-level reports/docs before removing scattered originals.
- Archive large legacy logs before removing originals; keep FSTF proof logs in place.
- Delete Python bytecode/cache and scratch smoke/profiler outputs because they are reproducible derived state.
