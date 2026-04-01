# STWM V4.2 1B Confirmation (2026-04-01) Paper/Report Brief

## Status: Lightweight Staged Sanity Only (Invalid For Main Claim)

This document summarizes a lightweight staged confirmation run and is kept for pipeline sanity record only.

- Not allowed as main paper 1B scale-up evidence.
- Not allowed as final 220M vs 1B decision evidence.
- Replaced by real 1B flow rooted at `outputs/training/stwm_v4_2_1b_real_confirmation/`.

## 1) Why The Queue Finished Very Fast (Expected)

This run was configured as a lightweight staged confirmation, not a full long-train round.

- Queue runtime evidence from `outputs/queue/stwm_1b/queue_events.log`:
  - `stwm_1b_smoke`: `13s`
  - `stwm_1b_confirmation_phase1_seed42`: `105s`
  - `stwm_1b_confirmation_phase2_seed42_123`: `107s`
  - `stwm_1b_confirmation_phase3_seed42_123_456`: `107s`
- Command profile from done jobs in `outputs/queue/stwm_1b/done/`:
  - Smoke used `STWM_V4_2_1B_SMOKE_RUN_TRIO=0`, `STWM_V4_2_1B_SMOKE_STEPS=8`, `STWM_V4_2_1B_SMOKE_SAMPLE_LIMIT=8`.
  - Confirmation phases used `STWM_V4_2_1B_STAGE_RUNS=full_v4_2` and cumulative seeds with `SKIP_EXISTING=1`, so each phase only added one new seed workload.
  - State-identifiability stage is `--eval-only` on resumed checkpoints.
  - `BUILD_FIGURES=0` during staged confirmation.

Conclusion: the short runtime was caused by intentionally small workload knobs plus incremental skip logic, not by silent crash.

## 2) Scope And Protocol

- Base protocol root: `outputs/training/stwm_v4_2_1b_confirmation_staged/base`
- Harder protocol root: `outputs/training/stwm_v4_2_1b_confirmation_staged/state`
- Seeds: `42, 123, 456`
- Runs: `full_v4_2`, `wo_semantics_v4_2`, `wo_object_bias_v4_2`
- Protocol coverage report: `reports/stwm_v4_2_1b_state_identifiability_protocol_v1.json`

## 3) Core Quantitative Results (mean +- std)

### Base protocol

Source: `outputs/training/stwm_v4_2_1b_confirmation_staged/base/comparison_multiseed.md`

| run | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate |
|---|---:|---:|---:|---:|
| full_v4_2 | 0.406253 +- 0.012013 | 0.436565 +- 0.018064 | 0.030311 +- 0.008723 | 0.102778 +- 0.019642 |
| wo_semantics_v4_2 | 0.488988 +- 0.052532 | 0.488326 +- 0.018365 | -0.000662 +- 0.034243 | 0.044444 +- 0.051069 |
| wo_object_bias_v4_2 | 0.396091 +- 0.028969 | 0.429387 +- 0.036371 | 0.033295 +- 0.007485 | 0.116667 +- 0.000000 |

### Harder state-identifiability protocol

Source: `outputs/training/stwm_v4_2_1b_confirmation_staged/state/comparison_state_identifiability.md`

| run | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate |
|---|---:|---:|---:|---:|
| full_v4_2 | 0.494497 +- 0.030433 | 0.487315 +- 0.006262 | -0.007182 +- 0.027207 | 0.166667 +- 0.023570 |
| wo_semantics_v4_2 | 0.527424 +- 0.008088 | 0.524835 +- 0.012672 | -0.002589 +- 0.004757 | 0.094444 +- 0.062854 |
| wo_object_bias_v4_2 | 0.484196 +- 0.044852 | 0.512931 +- 0.033658 | 0.028735 +- 0.018458 | 0.200000 +- 0.023570 |

## 4) Stability, Decoupling, And Boundary-Accurate Reading

- Base full-better counts across seeds (`outputs/training/stwm_v4_2_1b_confirmation_staged/base/comparison_multiseed.md`):
  - vs `wo_semantics_v4_2`: trajectory `3/3`, query `3/3`.
  - vs `wo_object_bias_v4_2`: trajectory `1/3`, query `2/3` (mixed).
- Harder protocol sign consistency (`outputs/training/stwm_v4_2_1b_confirmation_staged/state/comparison_state_identifiability.json`):
  - vs `wo_semantics_v4_2`: overall query `3/3`.
  - vs `wo_object_bias_v4_2`: overall query `2/3`.
- Decoupling (non-proxy-like maintained):
  - Base: `reports/stwm_v4_2_1b_query_decoupling_multiseed.json`
    - full decoupling_score mean `0.566300`, proxy_like_count `0`
  - State: `reports/stwm_v4_2_1b_state_identifiability_decoupling_v1.json`
    - full decoupling_score mean `0.565324`, proxy_like_count `0`

## 5) 220M vs 1B Comparison And Go/No-Go

Source: `reports/stwm_v4_2_220m_vs_1b_confirmation_staged.json`

- Decision: `NO-GO` for 3B escalation under current confirmation setting.
- Key gates:
  - Q1 (full vs wo_semantics stability): `YES`
  - Q2 (full vs wo_object_bias stability): `NO`
  - Q3 (harder protocol still better than both ablations): `YES`
  - Q4 (harder decoupling better than 220M): `NO`
  - Q5 (overall 3B gate): `NO-GO`
- Full delta (1B - 220M):
  - Base trajectory_l1: `+0.355444` (lower is better)
  - Base query_localization_error: `+0.385389` (lower is better)
  - State trajectory_l1: `+0.239574` (lower is better)
  - State query_localization_error: `+0.215461` (lower is better)

## 6) Paste-Ready Executive Text

Under the current lightweight 1B confirmation setup, full_v4_2 shows stable gains over wo_semantics_v4_2 across seeds on core query/trajectory indicators, while comparisons against wo_object_bias_v4_2 are mixed rather than uniformly dominant. On the harder state-identifiability protocol, full_v4_2 remains stronger than wo_semantics_v4_2 and is directionally better than wo_object_bias_v4_2 on query error in 2/3 seeds. Decoupling remains non-proxy-like (proxy_like_count=0), but the staged 220M-vs-1B comparison does not yet satisfy scale-gain and go/no-go gates, so the current decision is NO-GO for immediate 3B escalation.
