# Week2 Mini-Val Summary

## Executive Summary

- Four short runs completed under a unified protocol: `full`, `wo_semantics`, `wo_trajectory`, `wo_identity_memory`.
- All four have checkpoints, per-step mini-val summaries, run reports, and case-level JSONs.
- By step 60, `full` is best on trajectory/query errors, while mask IoU differences are small.
- Largest degradation appears when removing semantics or trajectory (mainly on trajectory and query metrics).

## Run Configuration Differences

Shared:

- `steps=60`, `eval_interval=20`, `obs_steps=8`, `pred_steps=8`
- `train_max_clips=32`, `val_max_clips=20`, `num_train_samples=8`, `num_val_samples=18`
- model preset `prototype_220m`, seed `42`

Ablation switches:

- `full`: semantics on, trajectory on, identity memory on
- `wo_semantics`: semantics off
- `wo_trajectory`: trajectory off
- `wo_identity_memory`: identity memory off

## Final Metrics at Step 60

| Run | future_mask_iou | future_trajectory_l1 (lower is better) | query_localization_error (lower is better) |
|---|---:|---:|---:|
| full | 0.158949 | 0.017848 | 0.017848 |
| wo_semantics | 0.158337 | 0.052370 | 0.052370 |
| wo_trajectory | 0.157876 | 0.048166 | 0.048166 |
| wo_identity_memory | 0.158725 | 0.027327 | 0.027327 |

Delta vs full at step 60:

- `wo_semantics`: mask IoU `-0.000612`, trajectory L1 `+0.034522`, query error `+0.034522`
- `wo_trajectory`: mask IoU `-0.001073`, trajectory L1 `+0.030318`, query error `+0.030318`
- `wo_identity_memory`: mask IoU `-0.000224`, trajectory L1 `+0.009478`, query error `+0.009478`

## Stepwise Trend (20/40/60)

Observations from `mini_val_summary_step_00020/40/60.json`:

- Step 40 is best mask IoU point for all runs (`run_report` also reports best IoU at step 40).
- Trajectory/query metrics are non-monotonic across steps (short-run instability is visible).
- Even with short-run variance, step-60 ranking on trajectory/query remains:
  - best: `full`
  - second: `wo_identity_memory`
  - then: `wo_trajectory`
  - worst: `wo_semantics`

## Which Ablation Hurts Most

Answer depends on metric:

- By mask IoU: `wo_trajectory` hurts most.
- By trajectory/query error: `wo_semantics` hurts most (very close to `wo_trajectory`).
- By identity-related reported metrics: no separability yet (all saturated).

## Story Support Status

Current support level for the paper narrative:

- Semantics contribution: supported on trajectory/query metrics.
- Trajectory contribution: supported on mask and trajectory/query metrics.
- Identity memory contribution: weakly supported in this protocol; only modest gain vs ablation on trajectory/query.

Why identity evidence is weak in this stage:

- Identity metrics are saturated (`identity_consistency=1.0`, `switch_rate=0.0` for all runs).
- Occlusion metric stays `0.0` across runs, indicating this probe is not informative under current setup.

## Artifact Completeness

All four run roots contain:

- `checkpoints/step_00020.pt`, `step_00030.pt`, `step_00040.pt`, `step_00060.pt`
- `config_snapshot.json`
- `eval/mini_val_summary_step_00020.json`
- `eval/mini_val_summary_step_00040.json`
- `eval/mini_val_summary_step_00060.json`
- `eval/mini_val_summary_last.json`
- `run_report.json`
- `train_log.jsonl`

No global rerun is needed for this stage; analysis can continue from existing artifacts.
