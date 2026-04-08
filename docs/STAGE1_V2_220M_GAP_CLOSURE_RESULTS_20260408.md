# Stage1-v2 220M Gap Closure Results

- generated_at_utc: 2026-04-08T12:07:05.666492+00:00
- small_ref_run: stage1_v2_gap_debugsmall_ref
- run_220m_ref: stage1_v2_gap_220m_ref
- best_220m_run: stage1_v2_gap_220m_ref
- best_220m_optimization_run: stage1_v2_gap_220m_opt_lossweights
- should_promote_220m_now: False
- next_step_choice: stop_220m_for_now

## Key Answer
- small_ref vs 220m_ref primary gap: -0.105036
- best 220m remaining primary gap to small_ref: -0.105036

## Ranked Metrics Table
| run | primary_endpoint_l2 | secondary_mean_l2 | tertiary_tapvid | quaternary_tapvid3d_limited | total_loss_ref | effective_batch | params_est | winner_reason_vs_best_overall |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| stage1_v2_gap_220m_ref | 0.266829 | 0.270792 | 0.301695 | 3.561830 | 2.430206 | 2 | 207613450 | winner by primary->secondary->tertiary->quaternary |
| stage1_v2_gap_220m_opt_lossweights | 0.305260 | 0.315271 | 0.496231 | 3.685571 | 3.057269 | 2 | 207613450 | loses on primary by +0.038431 |
| stage1_v2_gap_debugsmall_ref | 0.371865 | 0.411761 | 0.429104 | 3.553100 | 0.358909 | 2 | 3213066 | loses on primary by +0.105036 |
| stage1_v2_gap_220m_opt_batch | 0.444309 | 0.451683 | 0.424086 | 3.596328 | 2.410107 | 4 | 207613450 | loses on primary by +0.177480 |
| stage1_v2_gap_220m_opt_lr | 0.509802 | 0.505656 | 0.474912 | 3.564613 | 0.733922 | 2 | 207613450 | loses on primary by +0.242973 |

## 220M Optimization Effect
- best optimization run: stage1_v2_gap_220m_opt_lossweights
- delta primary vs 220m_ref: 0.038431
- delta secondary vs 220m_ref: 0.044480
- delta tertiary vs 220m_ref: 0.194536
- delta quaternary vs 220m_ref: 0.123741
