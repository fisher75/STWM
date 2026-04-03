# STWM Phase 4 Decision V1

Date: 2026-04-03
Decision: GO (promote frontend cache path toward standard training path with staged rollout)

## 1) Decision Rule

Promotion rule (from pilot plan):

1. `data_wait_ratio` absolute drop >= 0.15, or
2. `step_time` reduction >= 20%,

and

3. no stability regression.

## 2) Result Against Rule

Evidence source:

- `reports/stwm_frontend_cache_ab_report_v1.json`

Observed:

1. `data_wait_ratio_p50` drop: 0.8712 (pass)
2. `step_time_p50` reduction: 89.06% (pass)
3. monitored A/B exits: both success (pass)

All gate conditions pass.

## 3) Promotion Scope

Promote as staged standard path, not immediate full blast:

1. default to `raw` in global configs for now
2. add controlled rollout profile using `--data-mode frontend_cache`
3. keep raw fallback switch available for instant rollback

## 4) Rollout Guardrails

1. keep cache schema/version hash checks enabled
2. keep trace cache hardening enabled (atomic/quarantine/rebuild/lock)
3. monitor per-run:
   - data wait ratio
   - step time p50/p95
   - cache read/rebuild warnings
4. rollback trigger:
   - new cache corruption failures
   - sustained throughput regression vs raw baseline

## 5) Next Upgrade Actions

1. extend pilot from 256 clips to larger slices in stages
2. build queue-ready helper scripts for frontend-cache mode jobs
3. only after staged stability is confirmed, consider broader default enablement
