# STWM V4.2 Mini-Val Seed42 Comparison

Runs root: `outputs/training/stwm_v4_2_minival_seed42`

## Average Loss/Metric Table

| run | total | traj_l1 | query_loc_err | semantic | reid | query | memory_gate | reconnect_success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 1.992070 | 0.054970 | 0.053609 | 1.035185 | 2.359350 | 3.513054 | 0.889223 | 0.000000 |
| wo_semantics_v4_2 | 1.482042 | 0.145658 | 0.145020 | 0.000000 | 2.367757 | 3.447270 | 0.645087 | 0.000000 |
| wo_identity_v4_2 | 1.431226 | 0.076194 | 0.076315 | 1.095802 | 0.000000 | 3.489465 | 0.000000 | 0.000000 |

## First/Last Trend

- full_v4_2
  - first total/traj_l1/query_err: 3.296833 / 0.176732 / 0.191972
  - last total/traj_l1/query_err: 2.029532 / 0.018367 / 0.024827
- wo_semantics_v4_2
  - first total/traj_l1/query_err: 1.150372 / 0.171415 / 0.126980
  - last total/traj_l1/query_err: 1.559741 / 0.024331 / 0.008036
- wo_identity_v4_2
  - first total/traj_l1/query_err: 3.294899 / 0.176732 / 0.191972
  - last total/traj_l1/query_err: 1.335428 / 0.035857 / 0.043548

## Risk Flags

- full_v4_2: {"tokenizer_collapse_risk": false, "background_bias_risk": false, "memory_inactive_risk": false, "semantic_decorative_risk": false, "identity_decorative_risk": true}
- wo_semantics_v4_2: {"tokenizer_collapse_risk": false, "background_bias_risk": false, "memory_inactive_risk": false, "semantic_decorative_risk": false, "identity_decorative_risk": true}
- wo_identity_v4_2: {"tokenizer_collapse_risk": false, "background_bias_risk": false, "memory_inactive_risk": false, "semantic_decorative_risk": false, "identity_decorative_risk": false}

## Delta vs full_v4_2

| run | d_traj_l1 | d_query_err | d_sem | d_reid | d_memory_gate | d_reconnect_success |
|---|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | 0.090688 | 0.091411 | -1.035185 | 0.008407 | -0.244136 | 0.000000 |
| wo_identity_v4_2 | 0.021225 | 0.022706 | 0.060617 | -2.359350 | -0.889223 | 0.000000 |
