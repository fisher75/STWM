# STWM Reappearance/Visibility Joint V1 Eval

Controlled joint training was evaluated with full-model teacher-forced and free-rollout FutureSemanticTraceState exports, each at 256 valid items. The eval consumed raw FutureSemanticTraceState export only; no old association report was used.

## Free-Rollout Metrics

- event_AP_joint: `0.803574`
- event_AUROC_joint: `0.713056`
- per_horizon_AP_joint: `0.160716`
- per_horizon_AUROC_joint: `0.284099`
- visibility_AP: `0.999653`
- visibility_AUROC: `0.995395`
- output_degenerate: `False`
- trace_rollout_regression_detected: `False`

## Comparison

- event_AP_headonly_v2: `0.798104`
- event_AUROC_headonly_v2: `0.706111`
- per_horizon_AP_headonly_v2: `0.152947`
- event_AP_random_mean: `0.619263`
- event_AP_random_max: `0.848802`

Joint training is marked signal-positive for this controlled validation, but paper-level world-model claim remains `unclear` because the run is still small-scale and per-horizon signal remains weak.
