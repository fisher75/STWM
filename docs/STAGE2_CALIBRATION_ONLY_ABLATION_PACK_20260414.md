# Stage2 Calibration-Only Ablation Pack

- generated_at_utc: 2026-04-14T17:28:31.841360+00:00
- ablation_pack_status: 0_running_3_completed_0_failed
- reference_mainline_run_name: stage2_calonly_topk1_seed123_wave1_20260413
- alignment_sparse_gating_delay_load_bearing: True

| ablation | run_name | completed | endpoint_l2 | hard_score | gate_ratio | expected_direction_observed |
|---|---|---|---:|---:|---:|---|
| noalign | stage2_calonly_noalign_seed123_ablate_20260414 | True | 0.000537 | 0.000117 | 0.0000 | True |
| densegate | stage2_calonly_densegate_seed123_ablate_20260414 | True | 0.000535 | 0.000130 | 1.0000 | True |
| nodelay | stage2_calonly_nodelay_seed123_ablate_20260414 | True | 0.000570 | 0.000155 | 0.1250 | True |
