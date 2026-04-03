# STWM Frontend Cache Pilot Implementation V1

Date: 2026-04-03
Status: Phase 2 implemented

## 1) Scope

This phase adds a pilot-only frontend cache path without changing model architecture.

Implemented goals:

1. add `data-mode` switch: `raw` vs `frontend_cache`
2. support fixed-slice prebuilt frontend records
3. keep raw path behavior unchanged
4. avoid full-data rollout

## 2) Code Changes

### 2.1 Trainer data-mode switch

File:

- `code/stwm/trainers/train_stwm_v4_2_real.py`

Added CLI:

1. `--data-mode {raw,frontend_cache}`
2. `--frontend-cache-dir`
3. `--frontend-cache-index`
4. `--frontend-cache-max-shards-in-memory`

Added runtime path:

1. `_FrontendCacheReader` for shard/index lookup with small in-memory shard LRU.
2. `_build_features_for_sample(...)` now has frontend-cache branch.
3. frontend-cache branch returns the same tensor contract as raw path:
   - `trace_features`
   - `semantic_features`
   - `prior_features`
   - `teacher_objectness`
   - `target_trajectory`
   - `target_visibility`
   - `target_semantic_probs`
4. summary output now records `data_mode` and frontend cache paths.

### 2.2 Frontend cache builder

File:

- `code/stwm/tools/build_frontend_cache_pilot.py`

Build behavior:

1. input fixed manifest slice (`--slice-offset`, `--slice-size`)
2. outputs shard records (`--shard-size`) + index JSON
3. writes pilot manifest for strict A/B reuse

Record content includes:

1. trace frontend outputs
2. semantic frontend outputs
3. clip metadata
4. target/query metadata
5. mask/visibility metadata

## 3) Pilot Artifacts

Generated cache root:

- `data/cache/frontend_cache_pilot_v1`

Artifacts:

1. `data/cache/frontend_cache_pilot_v1/index.json`
2. `data/cache/frontend_cache_pilot_v1/shards/shard_00000.pt`
3. `data/cache/frontend_cache_pilot_v1/shards/shard_00001.pt`
4. `data/cache/frontend_cache_pilot_v1/shards/shard_00002.pt`
5. `data/cache/frontend_cache_pilot_v1/shards/shard_00003.pt`
6. `data/cache/frontend_cache_pilot_v1/pilot_manifest.json`
7. `data/cache/frontend_cache_pilot_v1/build_summary.json`

Build summary:

1. selected clips: 256
2. shard count: 4
3. schema: `frontend_cache_pilot_v1`

## 4) Notes

1. No model structure changes were made.
2. No 1B path changes were made.
3. This is pilot-only and does not force full pre-extraction rollout.
