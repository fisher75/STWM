# Stage1-v2 220M Mainline Freeze Results

- generated_at_utc: 2026-04-08T12:49:04.396402+00:00
- freeze_220m_run: stage1_v2_freeze_220m_mainline
- freeze_debugsmall_ref_run: stage1_v2_freeze_debugsmall_ref
- winner_by_ranked_policy: stage1_v2_freeze_220m_mainline
- is_220m_mainline_still_better_than_debugsmall: True
- final_stage1_backbone_decision: freeze_220m_as_stage1_backbone
- next_step_choice: freeze_stage1_and_prepare_stage2

## Training Budget
- optimizer_steps: 192
- epochs: 1
- eval_steps: 16

## Ranked Metrics
| run | primary_endpoint_l2 | secondary_mean_l2 | tertiary_tapvid | quaternary_tapvid3d_limited | teacher_forced_coord_loss | parameter_count | effective_batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1_v2_freeze_220m_mainline | 0.246459 | 0.251255 | 0.301853 | 3.572357 | 0.085847 | 207613450 | 2 |
| stage1_v2_freeze_debugsmall_ref | 0.819004 | 0.484446 | 0.879724 | 3.550781 | 0.033419 | 3213066 | 2 |
