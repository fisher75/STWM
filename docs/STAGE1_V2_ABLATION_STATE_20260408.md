# Stage1-v2 State Ablation

- generated_at_utc: 2026-04-08T11:02:33.904242+00:00
- objective: Is multi-token state beneficial vs legacy mean-5d under matched GRU backbone?
- best_variant: stage1_v2_state_multitoken_gru
- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval_or_tapvid3d_limited_eval
- clear_winner: True (margin=0.011184622844060299, required>=0.005)
- total_loss is reference only and not used as main selection key

| variant | status | primary_endpoint_l2 | secondary_mean_l2 | tertiary_external_l2 | total_loss_ref | tapvid_status | tapvid3d_status | winner_reason |
|---|---|---:|---:|---:|---:|---|---|---|
| stage1_v2_state_legacy_mean5d_gru | ok | 0.690000 | 0.675824 | 0.594070 | 0.972991 | available_and_run | available_and_run | loses on primary by +0.011185 |
| stage1_v2_state_multitoken_gru | ok | 0.678815 | 0.660218 | 0.590521 | 1.066337 | available_and_run | available_and_run | winner: best primary, then secondary, then tertiary |
