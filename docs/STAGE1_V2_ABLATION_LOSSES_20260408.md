# Stage1-v2 Losses Ablation

- generated_at_utc: 2026-04-08T10:19:20.174954+00:00
- objective: Under fixed state/backbone, compare loss families using external rollout metrics.
- best_variant: stage1_v2_loss_coord_visibility
- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval
- total_loss is reference only and not used as main selection key

| variant | status | teacher_forced_coord_loss | free_rollout_coord_mean_l2 | free_rollout_endpoint_l2 | tapvid_eval | tapvid3d_limited_eval | total_loss_ref |
|---|---|---:|---:|---:|---|---|---:|
| stage1_v2_loss_coord_only | ok | 0.192707 | 0.418521 | 0.530423 | unavailable | unavailable | 0.390546 |
| stage1_v2_loss_coord_visibility | ok | 0.165333 | 0.381416 | 0.397214 | unavailable | unavailable | 0.898364 |
| stage1_v2_loss_coord_visibility_residual_velocity | ok | 0.743995 | 0.805689 | 0.773782 | unavailable | unavailable | 1.127964 |
| stage1_v2_loss_coord_visibility_residual_velocity_endpoint | ok | 0.286735 | 0.497043 | 0.474754 | unavailable | unavailable | 1.055887 |
