# STWM V4.2 3B Go/No-Go (220M vs 1B)

## Full Run Delta (1B - 220M)

### Base Protocol

| metric | delta_1b_minus_220m | expected_direction |
|---|---:|---|
| trajectory_l1 | +0.208648 | lower_better |
| query_localization_error | +0.206143 | lower_better |
| semantic_loss | +1.326311 | lower_better |
| reid_loss | +0.405680 | lower_better |
| query_traj_gap | -0.002505 | lower_better |
| memory_gate_mean | -0.233665 | higher_better |
| reconnect_success_rate | +0.028889 | higher_better |

### State-Identifiability Protocol

| metric | delta_1b_minus_220m | expected_direction |
|---|---:|---|
| trajectory_l1 | +0.000779 | lower_better |
| query_localization_error | -0.005374 | lower_better |
| query_traj_gap | -0.006154 | lower_better |
| reconnect_success_rate | +0.029167 | higher_better |

### Harder-Protocol Decoupling

- decoupling_score_220m: 0.634395
- decoupling_score_1b: 0.573726
- delta_1b_minus_220m: -0.060669
- proxy_like_count_220m: 0
- proxy_like_count_1b: 0

## Final 5 Questions

- Q1: 1B上，full 相比 wo_semantics 的主效应是否跨seed稳定？
  - answer: NO
  - evidence: {"full_better_count_query": 1, "full_better_count_traj": 1}
- Q2: 1B上，full 相比 wo_object_bias 的主效应是否跨seed稳定？
  - answer: YES
  - evidence: {"full_better_count_query": 2, "full_better_count_traj": 2}
- Q3: 在 state-identifiability harder protocol 上，full 是否仍优于两组消融？
  - answer: NO
  - evidence: {"full_better_count_vs_wo_semantics_query": 1, "full_better_count_vs_wo_object_bias_query": 1}
- Q4: harder-protocol decoupling 是否优于220M且保持非proxy-like？
  - answer: NO
  - evidence: {"decoupling_score_220m": 0.634395341392448, "decoupling_score_1b": 0.5737263880861634, "delta_1b_minus_220m": -0.060668953306284634, "proxy_like_count_220m": 0, "proxy_like_count_1b": 0}
- Q5: 是否满足进入3B训练的 go/no-go 门槛？
  - answer: NO-GO
  - evidence: {"base_delta_trajectory_l1": 0.20864786343752512, "base_delta_query_localization_error": 0.20614289418276813, "state_delta_trajectory_l1": 0.0007791908988211693, "state_delta_query_localization_error": -0.0053743311824897555, "all_gates": {"q1_sem_stability": false, "q2_objbias_stability": true, "q3_state_ident": false, "q4_harder_decoupling": false, "q5_scale_gain": false}}

## 3B Decision

- decision: NO-GO
- ready_for_3b: false
