# Stage1-v2 Mainline Confirmation Results

- generated_at_utc: 2026-04-08T12:24:17.763883+00:00
- current_best_small_run: stage1_v2_confirm_debugsmall_mainline
- current_best_220m_run: stage1_v2_confirm_220m_bestrecipe
- does_220m_now_surpass_small: True
- final_mainline_decision: promote_220m_as_mainline
- next_step_choice: promote_220m_as_mainline

## Training Budget
- optimizer_steps: 96
- epochs: 1
- eval_steps: 12

## Ranked Metrics
| run | primary_endpoint_l2 | secondary_mean_l2 | tertiary_tapvid | quaternary_tapvid3d_limited | teacher_forced_coord_loss | parameter_count | effective_batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1_v2_confirm_220m_bestrecipe | 0.233802 | 0.242064 | 0.367235 | 3.611966 | 0.074894 | 207613450 | 2 |
| stage1_v2_confirm_220m_ref | 0.376601 | 0.386483 | 0.576739 | 3.724907 | 0.176755 | 207613450 | 2 |
| stage1_v2_confirm_debugsmall_mainline | 0.457273 | 0.428146 | 0.618383 | 3.730382 | 0.143580 | 3213066 | 2 |

## Promotion Justification
- 220m is not worse on any ranked metric and surpasses/ties small across all four tiers.
