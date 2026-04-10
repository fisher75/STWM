# Stage2 Fullscale Wave2 Results

- wave2_status: 0_running_5_completed_0_failed
- current_strongest_candidate_mainline: core-only + crop_visual_encoder
- next_step_choice: summarize_wave2_after_completion

| run_name | gpu | lease_id | train_counts | val_counts | batch_size | train_steps | eval_interval | save_every_n_steps | status | best_step | best_endpoint_l2 | best_coord_mean_l2 | teacher_forced_coord_loss |
|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| stage2_fullscale_core_legacysem_seed123_wave2_20260409 | 2 | 1e32f36a-18a9-4552-80da-c146f3e6789b | VIPSeg=2806,VSPW=2806 | VIPSeg=343,VSPW=343 | 8 | 10000 | 1000 | 1000 | completed | 10000 | 0.000650 | 0.000631 | 0.00002802 |
| stage2_fullscale_core_legacysem_seed456_wave2_20260409 | 0 | 6073dc4f-d862-4886-af2d-e336bcd6ad0a | VIPSeg=2806,VSPW=2806 | VIPSeg=343,VSPW=343 | 8 | 10000 | 1000 | 1000 | completed | 10000 | 0.000633 | 0.000610 | 0.00002987 |
| stage2_fullscale_coreplusburst_cropenc_seed123_wave2_20260409 | 1 | 75300b90-322a-4570-af9d-d3401a5a23f3 | BURST=329,VIPSeg=2806,VSPW=2806 | BURST=657,VIPSeg=343,VSPW=343 | 8 | 10000 | 1000 | 1000 | completed | 9000 | 0.001111 | 0.001098 | 0.00001698 |
| stage2_fullscale_coreplusburst_cropenc_seed456_wave2_20260409 | 5 | 684b0265-ec49-48f1-8357-91eee65db45d | BURST=329,VIPSeg=2806,VSPW=2806 | BURST=657,VIPSeg=343,VSPW=343 | 8 | 10000 | 1000 | 1000 | completed | 10000 | 0.000853 | 0.000837 | 0.00001895 |
| stage2_fullscale_core_cropenc_seed789_wave2_20260409 | 3 | b38f05f7-46a8-4787-99ad-c68a8cef98a9 | VIPSeg=2806,VSPW=2806 | VIPSeg=343,VSPW=343 | 8 | 10000 | 1000 | 1000 | completed | 4000 | 0.001272 | 0.001241 | 0.00002846 |

## Aggregate Metrics (Mean/Std)

- mainline: count=4
  free_rollout_endpoint_l2 mean=0.000946 std=0.000318
  free_rollout_coord_mean_l2 mean=0.000918 std=0.000316
  teacher_forced_coord_loss mean=0.000030 std=0.000001
- legacysem: count=3
  free_rollout_endpoint_l2 mean=0.000760 std=0.000205
  free_rollout_coord_mean_l2 mean=0.000738 std=0.000204
  teacher_forced_coord_loss mean=0.000031 std=0.000003
- coreplusburst: count=3
  free_rollout_endpoint_l2 mean=0.000951 std=0.000140
  free_rollout_coord_mean_l2 mean=0.000937 std=0.000141
  teacher_forced_coord_loss mean=0.000017 std=0.000001
