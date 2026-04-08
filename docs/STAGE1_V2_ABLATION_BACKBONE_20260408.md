# Stage1-v2 Backbone Ablation

- generated_at_utc: 2026-04-08T11:02:48.538937+00:00
- objective: Under fixed multi-token state, compare GRU vs Transformer and debug_small vs prototype_220m.
- best_variant: stage1_v2_backbone_transformer_debugsmall
- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval_or_tapvid3d_limited_eval
- clear_winner: True (margin=0.2754451433817545, required>=0.005)
- total_loss is reference only and not used as main selection key

| variant | status | primary_endpoint_l2 | secondary_mean_l2 | tertiary_external_l2 | total_loss_ref | tapvid_status | tapvid3d_status | winner_reason |
|---|---|---:|---:|---:|---:|---|---|---|
| stage1_v2_backbone_multitoken_gru | ok | 0.671317 | 0.647286 | 0.563574 | 0.982168 | available_and_run | available_and_run | loses on primary by +0.275445 |
| stage1_v2_backbone_transformer_debugsmall | ok | 0.395871 | 0.406978 | 0.346700 | 1.128954 | available_and_run | available_and_run | winner: best primary, then secondary, then tertiary |
| stage1_v2_backbone_transformer_prototype220m | ok | 1.274472 | 1.269138 | 1.170281 | 11.092827 | available_and_run | available_and_run | loses on primary by +0.878601 |
