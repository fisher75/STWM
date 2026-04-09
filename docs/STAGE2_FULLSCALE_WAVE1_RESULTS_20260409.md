# Stage2 Fullscale Wave1 Results

- wave1_status: 0_running_5_completed_0_failed
- current_strongest_candidate_mainline: core-only + crop_visual_encoder
- next_step_choice: summarize_wave1_after_completion

| run_name | gpu | batch_size | train_steps | eval_interval | save_every_n_steps | status | best_endpoint_l2 |
|---|---:|---:|---:|---:|---:|---|---:|
| stage2_fullscale_core_cropenc_seed42_20260409 | 2 | 8 | 10000 | 1000 | 1000 | completed | 0.001137 |
| stage2_fullscale_core_cropenc_seed123_20260409 | 7 | 8 | 10000 | 1000 | 1000 | completed | 0.000803 |
| stage2_fullscale_core_cropenc_seed456_20260409 | 3 | 8 | 10000 | 1000 | 1000 | completed | 0.000572 |
| stage2_fullscale_core_legacysem_seed42_20260409 | 5 | 8 | 10000 | 1000 | 1000 | completed | 0.000996 |
| stage2_fullscale_coreplusburst_cropenc_seed42_20260409 | 1 | 8 | 10000 | 1000 | 1000 | completed | 0.000888 |
