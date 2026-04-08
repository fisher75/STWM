# Stage1-v2 State Ablation

- generated_at_utc: 2026-04-08T10:19:10.154465+00:00
- objective: Is multi-token state beneficial vs legacy mean-5d under matched GRU backbone?
- best_variant: stage1_v2_state_multitoken_gru
- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval
- total_loss is reference only and not used as main selection key

| variant | status | teacher_forced_coord_loss | free_rollout_coord_mean_l2 | free_rollout_endpoint_l2 | tapvid_eval | tapvid3d_limited_eval | total_loss_ref |
|---|---|---:|---:|---:|---|---|---:|
| stage1_v2_state_legacy_mean5d_gru | ok | 0.490615 | 0.675824 | 0.690000 | unavailable | unavailable | 0.972991 |
| stage1_v2_state_multitoken_gru | ok | 0.469488 | 0.660218 | 0.678815 | unavailable | unavailable | 1.066337 |
