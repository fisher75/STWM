# STWM V4.2 Completed Runs Protocol-Eval Results (Executable Proxy Set)

Date: 2026-04-03  
Scope: four completed real-matrix runs  
Runs:

- 1b seed123 full_v4_2_1b
- 1b seed123 wo_semantics_v4_2_1b
- 220m seed42 full_v4_2
- 220m seed42 wo_semantics_v4_2

Source artifacts:

- `outputs/training/stwm_v4_2_real_1b/seed_123/full_v4_2_1b/mini_val_summary.json`
- `outputs/training/stwm_v4_2_real_1b/seed_123/wo_semantics_v4_2_1b/mini_val_summary.json`
- `outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/mini_val_summary.json`
- `outputs/training/stwm_v4_2_real_220m/seed_42/wo_semantics_v4_2/mini_val_summary.json`

## Metric Family Used

Because detached identity-consistency evaluator is currently checkpoint-incompatible for real runs, this report uses the strongest executable shared protocol-proxy metrics:

- `trajectory_l1`
- `query_localization_error`
- `query_traj_gap`
- `reconnect_success_rate`
- `reappearance_event_ratio`

Interpretation rule:

- Lower is better: `trajectory_l1`, `query_localization_error`, `query_traj_gap`
- Higher is better: `reconnect_success_rate`, `reappearance_event_ratio` (event coverage indicator)

## Results Table

| Scale | Run | disable_semantics | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate | reappearance_event_ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| 1b | full_v4_2_1b | 0 | 0.012208 | 0.011966 | -0.000243 | 0.000000 | 0.000000 |
| 1b | wo_semantics_v4_2_1b | 1 | 0.007319 | 0.006936 | -0.000384 | 0.000000 | 0.000000 |
| 220m | full_v4_2 | 0 | 0.012766 | 0.012503 | -0.000264 | 0.000000 | 0.000000 |
| 220m | wo_semantics_v4_2 | 1 | 0.014397 | 0.014098 | -0.000300 | 0.000000 | 0.000000 |

## Within-Scale Delta (wo_semantics - full)

### 1b (seed123)

- `trajectory_l1`: -0.004889
- `query_localization_error`: -0.005030
- `query_traj_gap`: -0.000141
- `reconnect_success_rate`: +0.000000
- `reappearance_event_ratio`: +0.000000

Read: for this 1b seed, removing semantics improved motion/query proxy errors.

### 220m (seed42)

- `trajectory_l1`: +0.001631
- `query_localization_error`: +0.001595
- `query_traj_gap`: -0.000036
- `reconnect_success_rate`: +0.000000
- `reappearance_event_ratio`: +0.000000

Read: for this 220m seed, removing semantics worsened motion/query localization proxies.

## Hard-Boundary Interpretation

1. Cross-scale semantic effect is not sign-stable under current completed-run evidence (1b and 220m show opposite direction).
2. Reconnect metrics are uninformative in this slice (`reappearance_event_ratio = 0` for all four), so no reconnect superiority claim is justified here.
3. These numbers support bounded protocol-proxy comparison only; they do not substitute for standard MOT metrics or detached identity-consistency protocol scores.

## Direct Answers (Results-Oriented)

1. Do we have executable, comparable protocol-level numbers now?  
   Yes, for the proxy family above, from completed-run summaries.

2. Is semantics universally helping under this evidence slice?  
   No. Direction flips by scale/seed.

3. Can we claim reconnect robustness superiority from these four runs?  
   No. Event coverage is zero in all four runs.

4. What is the strongest defensible statement now?  
   On available executable proxy metrics, performance differences are scale-dependent, and reconnect conclusions are currently underdetermined.
