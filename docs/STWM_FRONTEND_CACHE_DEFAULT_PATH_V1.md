# STWM Frontend Cache Default Path V1

Date: 2026-04-03
Status: Activated for protocol-frozen 220M mainline

## Decision

For protocol-frozen 220M mainline runs, `frontend_cache` is now the default training data path.
`raw` is retained only as explicit fallback/debug mode.

## Effective Changes

1. D1 clean-matrix launcher now defaults to frontend cache mode.
   - File: `scripts/enqueue_stwm_protocol_v2_d1_matrix_v1.sh`
   - Default envs:
     - `STWM_D1_DATA_MODE=frontend_cache`
     - `STWM_D1_FRONTEND_CACHE_DIR=/home/chen034/workspace/stwm/data/cache/frontend_cache_protocol_v2_full_v1`
     - `STWM_D1_FRONTEND_CACHE_INDEX=$STWM_D1_FRONTEND_CACHE_DIR/index.json`
2. Mainline output root switched to frontend-default namespace:
   - `outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1`
3. Raw fallback remains available by override:
   - `STWM_D1_DATA_MODE=raw`

## Queue/Execution Policy

1. Long jobs are submitted only through detached queue workflows (protocol_v2 queue + tmux worker).
2. No blocking foreground execution for long runs.
3. New clean matrix queue root:
   - `outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train`

## Rationale

1. Confirm benchmark GO remains valid:
   - Step-time mean reduction: `82.94%`
   - Data-time mean reduction: `98.79%`
   - Data-wait mean absolute drop: `0.7765`
2. No new numerical stability regressions detected in confirm run.
3. Frontend cache integrity checks passed (no miss in confirm scope).
