# Stage1-v2 Losses Ablation

- generated_at_utc: 2026-04-08T11:02:53.954065+00:00
- objective: Under fixed state/backbone, compare loss families using external rollout metrics.
- best_variant: stage1_v2_loss_coord_visibility
- selection_policy: primary=free_rollout_endpoint_l2, secondary=free_rollout_coord_mean_l2, tertiary=tapvid_eval_or_tapvid3d_limited_eval
- clear_winner: True (margin=0.07754007478555042, required>=0.0)
- total_loss is reference only and not used as main selection key

| variant | status | primary_endpoint_l2 | secondary_mean_l2 | tertiary_external_l2 | total_loss_ref | tapvid_status | tapvid3d_status | winner_reason |
|---|---|---:|---:|---:|---:|---|---|---|
| stage1_v2_loss_coord_only | ok | 0.530423 | 0.418521 | 0.775875 | 0.390546 | available_and_run | available_and_run | loses on primary by +0.133210 |
| stage1_v2_loss_coord_visibility | ok | 0.397214 | 0.381416 | 0.512318 | 0.898364 | available_and_run | available_and_run | winner: best primary, then secondary, then tertiary |
| stage1_v2_loss_coord_visibility_residual_velocity | ok | 0.773782 | 0.805689 | 0.588082 | 1.127964 | available_and_run | available_and_run | loses on primary by +0.376568 |
| stage1_v2_loss_coord_visibility_residual_velocity_endpoint | ok | 0.474754 | 0.497043 | 0.537177 | 1.055887 | available_and_run | available_and_run | loses on primary by +0.077540 |
