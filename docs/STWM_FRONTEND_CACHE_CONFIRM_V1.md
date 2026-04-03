# STWM Frontend Cache Confirm V1

Date: 2026-04-03
Status: Completed

## 1) Goal

Run a longer confirm benchmark (beyond 12-step pilot) under the same seed/batch/protocol knobs, comparing `raw` vs `frontend_cache`.

## 2) Run Configuration

- Repo: `/home/chen034/workspace/stwm`
- Script: `code/stwm/tools/benchmark_frontend_cache_ab.py`
- Manifest: `data/cache/frontend_cache_pilot_v1/pilot_manifest.json`
- Frontend cache index: `data/cache/frontend_cache_pilot_v1/index.json`
- Seed: `42`
- Steps: `120`
- Device: GPU `3`
- Batch settings: unchanged from prior pilot (`micro_batch_per_gpu=2`, `grad_accum=8`, same worker/prefetch/pin settings)

Primary machine-readable output:

- `reports/stwm_frontend_cache_confirm_v1.json`

## 3) Confirm Results (120-step)

### 3.1 Throughput and data-path

- Step time mean:
  - raw: `10.0715 s`
  - frontend_cache: `1.7180 s`
  - reduction: `82.94%`
- Data time mean:
  - raw: `8.3191 s`
  - frontend_cache: `0.1007 s`
  - reduction: `98.79%`
- Data wait ratio mean:
  - raw: `0.8339`
  - frontend_cache: `0.0574`
  - absolute drop: `0.7765`

### 3.2 Our-process GPU behavior

- Our-process GPU SM util mean:
  - raw: `1.9690`
  - frontend_cache: `19.5455`
  - delta: `+17.5764`

Interpretation: frontend cache removes data bottleneck enough to materially increase useful compute occupancy.

### 3.3 Stability and numerical behavior

- Raw: no NaN/Inf in monitored losses/grad_norm (`bad_value_count=0`)
- Frontend cache: no NaN/Inf in monitored losses/grad_norm (`bad_value_count=0`)
- Both runs exited cleanly (`exit_code=0`)

### 3.4 Cache integrity signals

- Frontend cache coverage check:
  - manifest clips: `256`
  - index entries: `256`
  - miss count: `0`
- Runtime tail scan (raw/frontend): no `miss/rebuild/corrupt/BadZipFile/EOFError/KeyError` signals detected in collected output tails.

## 4) Confirm Conclusion

Frontend cache remains strongly positive under longer-horizon confirm run:

1. Large end-to-end step-time improvement (`82.94%`).
2. Data path bottleneck nearly removed (`98.79%` data-time reduction).
3. No newly observed numeric instability or cache integrity regression in this confirm scope.
