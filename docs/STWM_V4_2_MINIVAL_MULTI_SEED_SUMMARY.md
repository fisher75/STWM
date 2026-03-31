# STWM V4.2 Mini-Val Multi-Seed Summary

## Scope And Frozen Boundaries

This round is strictly for stability validation only.

- architecture frozen:
  - dense 4D field substrate
  - object-biased learned tokenizer
  - factorized heads
  - single retrieval/reconnect memory
- no new module
- no loss composition change
- no evaluator churn
- no 1B scaling

## Run Setup

- runs:
  - `full_v4_2`
  - `wo_semantics_v4_2`
  - `wo_identity_v4_2`
- seeds: `42, 123, 456`
- train steps: `120`
- sample limit: `18`
- model preset: `prototype_220m_v4_2`

Artifacts root:

- `outputs/training/stwm_v4_2_minival_multiseed`

Primary summary files:

- `outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.json`
- `outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.md`

## Aggregate Table (mean +- std)

| Run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.050809 +- 0.003275 | 0.051175 +- 0.002910 | 1.038251 +- 0.002731 | 2.358817 +- 0.000377 | 0.000366 +- 0.001522 | 0.807672 +- 0.168442 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 |
| wo_semantics_v4_2 | 0.171720 +- 0.025330 | 0.172328 +- 0.025847 | 0.000000 +- 0.000000 | 2.367549 +- 0.002794 | 0.000608 +- 0.000902 | 0.481930 +- 0.259599 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 |
| wo_identity_v4_2 | 0.052933 +- 0.016863 | 0.053115 +- 0.016916 | 1.055313 +- 0.028668 | 0.000000 +- 0.000000 | 0.000182 +- 0.000412 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| Run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | +0.120911 +- 0.026661 | +0.121153 +- 0.025737 | -1.038251 +- 0.002731 | +0.008732 +- 0.002792 | +0.000242 +- 0.001156 | -0.325742 +- 0.395056 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |
| wo_identity_v4_2 | +0.002124 +- 0.013697 | +0.001940 +- 0.014791 | +0.017062 +- 0.030958 | -2.358817 +- 0.000377 | -0.000184 +- 0.001279 | -0.807672 +- 0.168442 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |

## Pairwise Per-Seed Check (trajectory/query)

`wo_semantics_v4_2 - full_v4_2`:

- seed42: `d_traj=+0.090688`, `d_query=+0.091411`
- seed123: `d_traj=+0.116501`, `d_query=+0.117854`
- seed456: `d_traj=+0.155546`, `d_query=+0.154194`

`wo_identity_v4_2 - full_v4_2`:

- seed42: `d_traj=+0.021225`, `d_query=+0.022706`
- seed123: `d_traj=-0.010218`, `d_query=-0.010623`
- seed456: `d_traj=-0.004635`, `d_query=-0.006264`

## Stability Answers

1. `full_v4_2 > wo_semantics_v4_2` 是否跨 seed 稳定成立:
   - **Yes** (trajectory/query on 3/3 seeds).
2. `full_v4_2 > wo_identity_v4_2` 是否跨 seed 稳定成立:
   - **No** (trajectory/query only 1/3 seed).
3. reconnect/reappearance metrics:
   - all zero in this slice across all runs and seeds.
   - cannot support reconnect claim in this round.

## Decision Boundary For Next Round

- keep current V4.2 architecture frozen
- continue longer 220M validation before any 1B discussion
- prioritize supervision-side hardening for identity/reconnect under the same split/scope
