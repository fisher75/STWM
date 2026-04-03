# STWM Frontend Cache Promotion Decision V1

Date: 2026-04-03
Decision: GO (Promote to default path for protocol-frozen 220M multiseed)

## 1) Inputs

- Hardening and recovery evidence:
  - `reports/stwm_trace_cache_recovery_test_v1.json`
- Pilot and earlier A/B evidence:
  - `reports/stwm_frontend_cache_ab_report_v1.json`
- This round longer confirm evidence:
  - `reports/stwm_frontend_cache_confirm_v1.json`

## 2) Promotion Gate Check

From `reports/stwm_frontend_cache_confirm_v1.json`:

1. `speedup_significant = true`
   - step-time mean reduction `82.94%`
2. `stability_ok = true`
   - raw/frontend both no NaN/Inf anomaly in monitored loss/grad metrics
3. `cache_integrity_ok = true`
   - frontend manifest/index coverage complete; miss count `0`

Final gate:

- `go = true`

## 3) Policy Update

1. `frontend_cache` is promoted as the default training data path for upcoming protocol-frozen 220M multiseed runs.
2. `raw` path is retained as fallback/debug path only.
3. Existing fallback principle remains: if cache integrity or numerical stability regresses in future runs, temporarily switch affected run back to `raw` and open an incident note.

## 4) Operational Notes

1. Keep frontend cache artifacts versioned (manifest/index/schema_version) and tied to exact protocol manifest slice.
2. Preserve trace-cache hardening protections already merged for reliability even when frontend cache is default.
3. Continue periodic process-level GPU/CPU telemetry in multi-tenant environments to avoid whole-GPU utilization misinterpretation.
