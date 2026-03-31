# STWM V4.2 Mini-Val Seed42 Summary

## Run Setup

- runs:
  - `full_v4_2`
  - `wo_semantics_v4_2`
  - `wo_identity_v4_2`
- seed: `42`
- train steps: `120`
- sample limit: `18`
- model preset: `prototype_220m_v4_2`
- parameter count: `207,543,382`

Artifacts root:

- `outputs/training/stwm_v4_2_minival_seed42`

Primary summary files:

- `outputs/training/stwm_v4_2_minival_seed42/comparison_seed42.json`
- `outputs/training/stwm_v4_2_minival_seed42/comparison_seed42.md`

## Main Table (Average)

| Run | trajectory_l1 | query_localization_error | query_traj_gap | semantic_loss | reid_loss | memory_gate_mean |
|---|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.054970 | 0.053609 | -0.001360 | 1.035185 | 2.359350 | 0.889223 |
| wo_semantics_v4_2 | 0.145658 | 0.145020 | -0.000637 | 0.000000 | 2.367757 | 0.645087 |
| wo_identity_v4_2 | 0.076194 | 0.076315 | 0.000121 | 1.095802 | 0.000000 | 0.000000 |

Delta vs `full_v4_2`:

- `wo_semantics_v4_2`
  - trajectory_l1: `+0.090688`
  - query_localization_error: `+0.091411`
- `wo_identity_v4_2`
  - trajectory_l1: `+0.021225`
  - query_localization_error: `+0.022706`

## First/Last Trend Snapshot

### full_v4_2

- total loss: `3.296833 -> 2.029532`
- trajectory_l1: `0.176732 -> 0.018367`
- query_localization_error: `0.191972 -> 0.024827`
- semantic_loss: `4.164250 -> 0.939680`
- reid_loss: `0.000000 -> 2.771586`

### wo_semantics_v4_2

- total loss: `1.150372 -> 1.559741`
- trajectory_l1: `0.171415 -> 0.024331`
- query_localization_error: `0.126980 -> 0.008036`
- reid_loss: `0.000000 -> 2.774180`

### wo_identity_v4_2

- total loss: `3.294899 -> 1.335428`
- trajectory_l1: `0.176732 -> 0.035857`
- query_localization_error: `0.191972 -> 0.043548`
- semantic_loss: `4.160384 -> 0.937524`

## Risk Flags

- full_v4_2:
  - `tokenizer_collapse_risk=false`
  - `background_bias_risk=false`
  - `memory_inactive_risk=false`
  - `semantic_decorative_risk=false`
  - `identity_decorative_risk=true`
- wo_semantics_v4_2:
  - `identity_decorative_risk=true`
- wo_identity_v4_2:
  - no decorative risk flag triggered

## Seed42 Interpretation

1. `full_v4_2 > wo_semantics_v4_2` has a clear directional signal on trajectory and query localization.
2. `full_v4_2 > wo_identity_v4_2` is also directional on trajectory and query localization.
3. Identity branch remains unstable as an optimization target (`reid_loss` behavior + decorative risk).
4. This is stronger than smoke-only evidence, but still single-seed and not yet load-bearing multi-seed evidence.

## Decision For Next Round

- keep current V4.2 architecture fixed
- continue 220M validation (multi-seed) before any 1B decision
- prioritize identity signal hardening and reconnect-event coverage
