# Stage2 Calibration-Only Fullscale Wave1 Results

- generated_at_utc: 2026-04-14T09:27:01.983690+00:00
- calibration_only_wave1_status: 0_running_6_completed_0_failed
- failed/incomplete runs: metrics suppressed when no valid completed raw+checkpoint artifact exists
- overall_best_run_name: stage2_calonly_topk1_seed123_wave1_20260413
- semantic_hard_best_run_name: stage2_calonly_topk1_seed123_wave1_20260413
- best_family: topk1
- stage2_calibration_only_is_true_mainline: True
- calibration_only_improved_vs_current_cropenc_baseline: True
- calibration_only_improved_vs_v7_alignment_only: True
- true_new_best_not_warm_start_inherited: True
- cross_seed_support_present: True
- semantic_hard_signal_preserved: True
- overall_best_and_semantic_hard_best_diverged: False
- partial_unfreeze_beats_frozen_calibration: False
- forgetting_or_instability_detected: True
- next_step_choice: stage2_calibration_only_is_true_mainline

| run_name | family | seed | status | global_step | endpoint_l2 | hard_score | gate_ratio | sidecar_diverged |
|---|---|---:|---|---:|---:|---:|---:|---|
| stage2_calonly_topk1_seed42_wave1_20260413 | topk1 | 42 | completed | 7000 | 0.000675 | 0.000250 | 0.1250 | False |
| stage2_calonly_topk1_seed123_wave1_20260413 | topk1 | 123 | completed | 12000 | 0.000528 | 0.000125 | 0.1250 | False |
| stage2_calonly_topk1_seed456_wave1_20260413 | topk1 | 456 | completed | 14000 | 0.000572 | 0.000512 | 0.1250 | True |
| stage2_calonly_qcap15_seed42_wave1_20260413 | qcap15 | 42 | completed | 7000 | 0.000676 | 0.000250 | 0.1250 | False |
| stage2_calonly_qcap15_seed123_wave1_20260413 | qcap15 | 123 | completed | 12000 | 0.000528 | 0.000125 | 0.1250 | False |
| stage2_calonly_qcap15_seed456_wave1_20260413 | qcap15 | 456 | completed | 14000 | 0.000572 | 0.000512 | 0.1250 | True |
