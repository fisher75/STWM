# STWM V4.2 3B Go/No-Go (220M vs 1B)

## Status: Lightweight Staged Decision Snapshot Only

This decision note is derived from lightweight staged 1B artifacts and is not final evidence for scale-up.

- Not valid as final main-paper 1B conclusion.
- Not valid as final 3B go/no-go gate.
- Final gate must come from real 1B confirmation outputs.

## Full Run Delta (1B - 220M)

### Base Protocol

| metric | delta_1b_minus_220m | expected_direction |
|---|---:|---|
| trajectory_l1 | +0.355444 | lower_better |
| query_localization_error | +0.385389 | lower_better |
| semantic_loss | +0.096860 | lower_better |
| reid_loss | +0.190483 | lower_better |
| query_traj_gap | +0.029945 | lower_better |
| memory_gate_mean | +0.158179 | higher_better |
| reconnect_success_rate | +0.102778 | higher_better |

### State-Identifiability Protocol

| metric | delta_1b_minus_220m | expected_direction |
|---|---:|---|
| trajectory_l1 | +0.239574 | lower_better |
| query_localization_error | +0.215461 | lower_better |
| query_traj_gap | -0.024113 | lower_better |
| reconnect_success_rate | +0.033333 | higher_better |

### Harder-Protocol Decoupling

- decoupling_score_220m: 0.634395
- decoupling_score_1b: 0.565324
- delta_1b_minus_220m: -0.069071
- proxy_like_count_220m: 0
- proxy_like_count_1b: 0

## Final 5 Questions

- Q1: 1B上，full 相比 wo_semantics 的主效应是否跨seed稳定？
  - answer: YES
  - evidence: {"full_better_count_query": 3, "full_better_count_traj": 3}
- Q2: 1B上，full 相比 wo_object_bias 的主效应是否跨seed稳定？
  - answer: NO
  - evidence: {"full_better_count_query": 2, "full_better_count_traj": 1}
- Q3: 在 state-identifiability harder protocol 上，full 是否仍优于两组消融？
  - answer: YES
  - evidence: {"full_better_count_vs_wo_semantics_query": 3, "full_better_count_vs_wo_object_bias_query": 2}
- Q4: harder-protocol decoupling 是否优于220M且保持非proxy-like？
  - answer: NO
  - evidence: {"decoupling_score_220m": 0.634395341392448, "decoupling_score_1b": 0.565324178160565, "delta_1b_minus_220m": -0.069071163231883, "proxy_like_count_220m": 0, "proxy_like_count_1b": 0}
- Q5: 是否满足进入3B训练的 go/no-go 门槛？
  - answer: NO-GO
  - evidence: {"base_delta_trajectory_l1": 0.355444154712475, "base_delta_query_localization_error": 0.385389385266333, "state_delta_trajectory_l1": 0.23957402035593983, "state_delta_query_localization_error": 0.21546056788582874, "all_gates": {"q1_sem_stability": true, "q2_objbias_stability": false, "q3_state_ident": true, "q4_harder_decoupling": false, "q5_scale_gain": false}}

## 3B Decision

- decision: NO-GO
- ready_for_3b: false
