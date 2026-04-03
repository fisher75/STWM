# STWM V4.2 Shared Cluster Scheduling Policy V3

Date: 2026-04-03
Scope: Shared 8xB200 cluster, protocol_v2 queue/tmux/lease stack

## Goals

1. Keep the existing protocol_v2 architecture unchanged (queue + tmux workers + lease lock).
2. Prevent starvation caused by an overly strict single global threshold.
3. Preserve D0 -> D1 ordering and keep D1 blocked until D0 success.
4. Keep future 1B/critical training isolation stricter than detached eval jobs.

## Why V2 Was Too Conservative

Previous gating effectively required very high free memory for every class.
In shared conditions, this created long waits even when a card was operationally safe.
Result: repeated `waiting_for_gpu` loops and delayed protocol closure.

## Class Policy (Default + Timed Fallback)

Configured in `scripts/select_protocol_v2_gpu.py`.

### Class A (Detached protocol eval / D0 style)

- Default thresholds:
  - `min_free_mem_gib = 28`
  - `max_gpu_util = 95`
  - `max_active_apps = 3`
- Fallback after `600s`:
  - `min_free_mem_gib = 22`
  - `max_gpu_util = 97`
  - `max_active_apps = 4`

Rationale: detached eval is lower risk and should not starve behind idealized idle-card requirements.

### Class B (Main training / D1 style)

- Default thresholds:
  - `min_free_mem_gib = 24`
  - `max_gpu_util = 90`
  - `max_active_apps = 3`
- Fallback after `1200s`:
  - `min_free_mem_gib = 20`
  - `max_gpu_util = 93`
  - `max_active_apps = 4`

Rationale: keep training quality stable while permitting controlled progress under moderate contention.

### Class C (Future high-priority training, including 1B-conservative mode)

- Default thresholds:
  - `min_free_mem_gib = 49`
  - `max_gpu_util = 85`
  - `max_active_apps = 2`
- Fallback after `600s`:
  - `min_free_mem_gib = 40`
  - `max_gpu_util = 90`
  - `max_active_apps = 3`

Rationale: preserve more headroom and lower interference for large-scale or high-stakes runs.

## Lease Co-location Rules

Lease checks are still enforced in addition to thresholds.

- Class A and Class B:
  - up to 2 STWM tasks total on one card
  - sharing is allowed only when the GPU is still the best eligible choice
- Class C:
  - isolation mode (no A/B/C co-card)

## Wait-Time Fallback Execution

Implemented via worker-to-selector `--wait-seconds` passthrough.

- Worker records elapsed wait time for each running job in `waiting_for_gpu`.
- Selector switches `policy.mode` from `default` to `fallback` once class-specific `after_seconds` is reached.
- Worker logs:
  - active `policy` JSON (`mode`, thresholds, fallback boundary)
  - `nearest_candidate` JSON (closest GPU and threshold gap)

This provides direct observability for why a job is still waiting and what is blocking it.

## Operational Notes

1. Architecture remains protocol_v2 queue/tmux/lease; no system rebuild.
2. Current D0 job identity and queue lineage are preserved.
3. D1 remains blocked until D0 protocol-best closure succeeds.

## Files

- Scheduler policy: `scripts/select_protocol_v2_gpu.py`
- Worker fallback/logging bridge: `scripts/protocol_v2_queue_worker.sh`
- Queue roots:
  - `outputs/queue/stwm_protocol_v2/d0_eval`
  - `outputs/queue/stwm_protocol_v2/d1_train`
