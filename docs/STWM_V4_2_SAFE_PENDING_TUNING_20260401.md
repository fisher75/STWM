# STWM V4.2 Safe Pending Tuning (2026-04-01)

## Scope And Intent

This tuning pass is a safe, parameter-only update for pending/rerun jobs in the
current 2-lane real matrix queue.

Out of scope:

- no interruption of currently running jobs
- no train-loop refactor
- no dataset refactor
- no lane expansion to 4

## Why We Do Not Hot-Stop Running Jobs

Current running jobs already hold GPU slots and have valid progress logs.
Hot-stopping would:

1. waste already-computed optimizer steps,
2. add queue churn under shared-resource contention,
3. increase operational risk while mandatory matrix is still incomplete.

Therefore running jobs continue unchanged; only not-yet-started jobs adopt new tuning.

## Why We Do Not Refactor Main Thread/Data Pipeline Now

Read-only profiling indicates the bottleneck is mostly data-side latency with
significant main-thread feature-build time. A structural pipeline rewrite is
higher risk and would delay mandatory matrix completion.

At this phase, we prefer conservative parameter tuning first, then revisit
structural optimization after mandatory evidence is secured.

## Selected Safe Parameters

### Checkpoint Policy (pending/rerun)

- latest overwrite interval: every 300 optimizer steps
- best checkpoint: unchanged (metric-driven)
- milestone checkpoints: disabled (`milestone_interval=0`)

Reason for `latest=300`:

- lower I/O overhead than 100-step frequency,
- still provides practical restart granularity,
- safer balance for long runs under shared storage.

### DataLoader Policy (pending/rerun)

- 1B: `num_workers=14`
- 220M: `num_workers=12`
- `prefetch_factor=2`
- `persistent_workers=true`
- `pin_memory=true`

Why not more aggressive values:

- host is highly contended (multi-tenant CPU/GPU pressure),
- very large worker counts can increase scheduling overhead,
- this step is intentionally moderate; evaluate 150-200 steps before further ramp.

### Thread Isolation

Applied early in launcher and trainer entrypoint:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `cv2.setNumThreads(0)` (best effort)

Purpose:

- reduce CPU oversubscription and thread-pool contention under shared loads.

## Jobs Affected By This Safe Tuning

At doc creation time, pending/rerun queue includes:

1. `1B seed42 rerun` (lane0 pending)
2. `220M seed123` (lane1 pending)

Any future submission via
`scripts/run_stwm_v4_2_real_train_seed.sh` (including a future seed456 queue item)
will inherit the same defaults unless explicitly overridden by environment variables.
