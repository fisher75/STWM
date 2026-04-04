# STWM Replacement Clean Matrix Seed42 Final Decision V1

Generated: 2026-04-04 12:54:03
Run resolution source: /home/chen034/workspace/stwm/reports/stwm_replacement_clean_matrix_seed42_submit_v1_20260404_114420.tsv

## A. Completion Verification

all_done_and_complete: True

## B. Official Result Comparison

| role | run_name | selected_best_step | qloc | qtop1 | l1 | mask_iou | id_consistency | id_switch_rate | eval_summary | selection_sidecar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| current full rerun baseline | full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2 | 2000 | 0.0066951184934028836 | 0.926208651399491 | 0.006538258073969955 | 0.1604957929660223 | 0.9990458015267175 | 0.0 | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2/checkpoints/protocol_eval/protocol_val_main_step_002000.json | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2/checkpoints/best_protocol_main_selection.json |
| alpha050 replacement | full_v4_2_seed42_objbias_alpha050_replacement_v1 | 2000 | 0.005061522543278662 | 0.9567430025445293 | 0.005183082604757095 | 0.16049538122634638 | 0.9990458015267175 | 0.0 | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/full_v4_2_seed42_objbias_alpha050_replacement_v1/checkpoints/protocol_eval/protocol_val_main_step_002000.json | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/full_v4_2_seed42_objbias_alpha050_replacement_v1/checkpoints/best_protocol_main_selection.json |
| wo_semantics control | wo_semantics_v4_2_seed42_control_v1 | 2000 | 0.008400639429043875 | 0.8956743002544529 | 0.00847303056876168 | 0.16048321318491163 | 0.9990458015267175 | 0.0 | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/wo_semantics_v4_2_seed42_control_v1/checkpoints/protocol_eval/protocol_val_main_step_002000.json | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/wo_semantics_v4_2_seed42_control_v1/checkpoints/best_protocol_main_selection.json |
| wo_object_bias control | wo_object_bias_v4_2_seed42_control_v1 | 2000 | 0.0022589355403837053 | 0.9796437659033079 | 0.0024298117105059952 | 0.16049471405638766 | 0.9990458015267175 | 0.0 | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/wo_object_bias_v4_2_seed42_control_v1/checkpoints/protocol_eval/protocol_val_main_step_002000.json | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42/wo_object_bias_v4_2_seed42_control_v1/checkpoints/best_protocol_main_selection.json |

official_answers:
- replacement_beats_baseline_official: True
- replacement_improvement_not_noise: True
- best_full_still_loses_to_wo_object_bias: True
- supports_alpha050_as_new_default_full: True

## C. Efficiency And Stability

| run_name | full_mean_step_time_s | full_mean_data_time_s | full_mean_data_wait_ratio | recent500_mean_step_time_s | recent500_mean_data_wait_ratio | p50_step_time_s | p95_step_time_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2 | 1.1680843300879502 | 0.2510276020052902 | 0.225901415626169 | 1.152534817230073 | 0.2327348551729391 | 1.1320209839614108 | 1.2855717379192355 |
| full_v4_2_seed42_objbias_alpha050_replacement_v1 | 1.1549443638232406 | 0.24783701886373916 | 0.2256822371912225 | 1.131257232638658 | 0.2297661390600275 | 1.1310514629876707 | 1.2564721049711807 |
| wo_semantics_v4_2_seed42_control_v1 | 1.2919478150744081 | 0.28394837209416335 | 0.23343406946803677 | 1.0772668772850884 | 0.23443552825805078 | 1.1340383189963177 | 3.0220920998021024 |
| wo_object_bias_v4_2_seed42_control_v1 | 1.34014876418657 | 0.29136838341823545 | 0.23149429171313893 | 1.1345572250651894 | 0.23594589688333772 | 1.1521811459679157 | 3.172622158197918 |

stability_answers:
- frontend_cache_stable_and_fast: True
- cache_miss_rebuild_corruption_signal: False
- nan_or_instability_signal: False
- replacement_slower_than_baseline: False
- replacement_more_unstable_than_baseline: False

## D. Seed123 Promotion Gate

gate_pass: True
conditions:
- official_rule_better_than_baseline: True
- query_localization_error_substantial_improvement: True
- query_top1_acc_non_degrade: True
- future_trajectory_l1_acceptable: True
- no_new_stability_issues: True
seed123_launched: True
seed123_reused_existing: False
seed123_submit_tsv: /home/chen034/workspace/stwm/reports/stwm_seed123_replication_clean_matrix_submit_v1_20260404_125402.tsv

| seed123_run | job_id | state | status_file | main_log | output_dir |
|---|---|---|---|---|---|
| full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2 | 20260404_125402_4566 | waiting_for_gpu | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775278442700_full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775278442700_full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2.log | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123/full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2 |
| full_v4_2_seed123_objbias_alpha050_replacement_v1 | 20260404_125402_2764 | waiting_for_gpu | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775278442887_full_v4_2_seed123_objbias_alpha050_replacement_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775278442887_full_v4_2_seed123_objbias_alpha050_replacement_v1.log | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123/full_v4_2_seed123_objbias_alpha050_replacement_v1 |
| wo_semantics_v4_2_seed123_control_v1 | 20260404_125403_6954 | queued | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775278443248_wo_semantics_v4_2_seed123_control_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775278443248_wo_semantics_v4_2_seed123_control_v1.log | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123/wo_semantics_v4_2_seed123_control_v1 |
| wo_object_bias_v4_2_seed123_control_v1 | 20260404_125403_229 | queued | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775278443599_wo_object_bias_v4_2_seed123_control_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775278443599_wo_object_bias_v4_2_seed123_control_v1.log | /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1/seed_123/wo_object_bias_v4_2_seed123_control_v1 |

