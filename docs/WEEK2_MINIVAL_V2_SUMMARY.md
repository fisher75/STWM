# Week2 Mini-Val V2 Summary

## Key Outcome

V2 successfully changed the evaluation behavior (query/identity no longer saturated exactly as in v1), but the current V2 implementation does **not** yet yield a stable "full wins all ablations" result.

## Final Metrics (Step 80)

| Run | future_mask_iou | future_trajectory_l1 | query_localization_error | query_top1_acc | query_hit_rate | identity_consistency | identity_switch_rate | occlusion_recovery_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 0.006243 | 0.035330 | 0.035353 | 1.000000 | 1.000000 | 0.013889 | 0.986111 | 0.000000 |
| wo_semantics | 0.007378 | 0.054658 | 0.054487 | 0.888889 | 0.111111 | 0.000000 | 1.000000 | 0.000000 |
| wo_trajectory | 0.008025 | 0.043202 | 0.043278 | 0.944444 | 0.055556 | 0.027778 | 0.972222 | 0.000000 |
| wo_identity_memory | 0.009160 | 0.060368 | 0.060225 | 0.944444 | 0.055556 | 0.013889 | 0.986111 | 0.000000 |

## Delta vs Full (Step 80)

- `wo_semantics`: trajectory `+0.019328`, query error `+0.019134`, query_top1 `-0.111111`
- `wo_trajectory`: trajectory `+0.007872`, query error `+0.007925`, query_top1 `-0.055556`
- `wo_identity_memory`: trajectory `+0.025038`, query error `+0.024872`, query_top1 `-0.055556`

## Interpretation

1. Full still beats all ablations on trajectory and query error at final step.
2. Identity-memory ablation is now the largest degradation on trajectory/query metrics in this run.
3. Mask IoU ranking is inverted (ablations slightly higher than full), indicating mask proxy may not align with trajectory/query objective under V2 target extraction.
4. Identity metrics changed from v1 saturation, but remain harsh:
   - consistency near `0.0`
   - switch near `1.0`
   This suggests target-point-on-label criterion is now sensitive but likely over-penalizing.
5. Query branch now has independent metrics (`query_top1_acc`, `query_hit_rate`) and no longer equals trajectory mean error; however, it still needs calibration for stronger realism.

## Which Ablation Hurts Most in V2

- By trajectory/query error: `wo_identity_memory` hurts most.
- By query retrieval/hit metrics: `wo_semantics` hurts most.
- By mask IoU: `full` is currently lowest (undesired behavior to investigate).

## Stepwise Stability Notes

- All runs reached best mask IoU at step 80 in this run.
- Query metrics are high early and drop by step 80 for ablations, indicating stronger separation than v1 but also instability.

## Bottom Line

V2 fixed the evaluation blind spots enough to reveal new behavior, but protocol tuning is still needed before treating these numbers as paper-ready headline evidence.
