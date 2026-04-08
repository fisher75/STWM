# Stage1-v2 Backbone Ablation

- generated_at_utc: 2026-04-08T10:19:16.133041+00:00
- objective: Under fixed multi-token state, compare GRU vs Transformer and debug_small vs prototype_220m.
- best_variant: stage1_v2_backbone_transformer_debugsmall
- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval
- total_loss is reference only and not used as main selection key

| variant | status | teacher_forced_coord_loss | free_rollout_coord_mean_l2 | free_rollout_endpoint_l2 | tapvid_eval | tapvid3d_limited_eval | total_loss_ref |
|---|---|---:|---:|---:|---|---|---:|
| stage1_v2_backbone_multitoken_gru | ok | 0.436387 | 0.647286 | 0.671317 | unavailable | unavailable | 0.982168 |
| stage1_v2_backbone_transformer_debugsmall | ok | 0.208219 | 0.406978 | 0.395871 | unavailable | unavailable | 1.128954 |
| stage1_v2_backbone_transformer_prototype220m | ok | 1.628312 | 1.269139 | 1.274473 | unavailable | unavailable | 11.092827 |
