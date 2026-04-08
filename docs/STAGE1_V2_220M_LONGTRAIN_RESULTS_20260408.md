# Stage1-v2 220M Long-Train Results

- generated_at_utc: 2026-04-08T14:35:17.179064+00:00
- run_name: stage1_v2_longtrain_220m_mainline_continue_10000
- final_stage1_backbone_decision_source: freeze_220m_as_stage1_backbone (from mainline freeze)
- next_step_choice: freeze_stage1_and_prepare_stage2

## Training Budget
- optimizer_steps: 10000
- effective_batch: 2
- epochs: 156
- eval_interval: 1000
- save_every_n_steps: 1000

## Best Ranked Metrics
| metric | value |
|---|---:|
| free_rollout_endpoint_l2 (primary) | 0.210244 |
| free_rollout_coord_mean_l2 (secondary) | 0.216247 |
| tapvid_endpoint_l2 (tertiary) | 0.328698 |
| tapvid3d_limited_endpoint_l2 (quaternary) | 3.600426 |
| teacher_forced_coord_loss | 0.064875 |

## Checkpoint Inventory
- checkpoint_dir: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408
- best: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/best.pt
- latest: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/latest.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0001000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0002000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0003000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0004000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0005000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0006000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0007000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0008000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0009000.pt
- step: /home/chen034/workspace/stwm/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408/step_0010000.pt
