# Week2 Unified Mini-Val Protocol

## Scope

- Stage: Week-2 unified short-training plus periodic mini-val.
- Runs:
  - `outputs/training/week2_minival/full`
  - `outputs/training/week2_minival/wo_semantics`
  - `outputs/training/week2_minival/wo_trajectory`
  - `outputs/training/week2_minival/wo_identity_memory`
- Dataset tag in summaries: `vspw`.
- Model preset: `prototype_220m`.

## Fixed Evaluation Setup

- Observation horizon: `obs_steps = 8`.
- Prediction horizon: `pred_steps = 8`.
- Training length: `steps = 60`.
- Eval interval: `eval_interval = 20`.
- Checkpoints present at step 20/30/40/60.
- Data budget:
  - `train_max_clips = 32`
  - `val_max_clips = 20`
  - `num_train_samples = 8`
  - `num_val_samples = 18`
- Seed: `42`.

## Validation Clip Set (18 clips)

- `1041_kIXALP9plU0`
- `1061_hWl2HQh1MG8`
- `107_tQA8kJXlTwc`
- `1125_5Eplp7nV12E`
- `1133_1dG6RUos-TQ`
- `1208_1nNkCuPWNlg`
- `1214_7uQUZ3GP568`
- `1340_MOKoxBoEvd4`
- `142_TrvLK24VlEs`
- `1722_E4mqlPuc-Is`
- `201_MRpIKGg_Y3c`
- `2183_9Jq-7HnWa_0`
- `43_-LB7cp3_mqY`
- `476_mNnY0LY1Tq0`
- `62_gmxfXIvSBNs`
- `747_ILU2occQNYQ`
- `857_NQKL_UzQCd8`
- `90_bPui3fxQq4k`

## Metrics and Current Reliability

Metrics recorded in `mini_val_summary_*.json`:

- `future_mask_iou`
- `future_trajectory_l1`
- `visibility_accuracy`
- `visibility_f1`
- `identity_consistency`
- `identity_switch_rate`
- `occlusion_recovery_acc`
- `query_localization_error`

Current practical interpretation:

- Primary useful metrics (non-saturated in this run):
  - `future_mask_iou`
  - `future_trajectory_l1`
  - `query_localization_error`
- Saturated or weak-proxy metrics in this run:
  - `visibility_accuracy = 1.0` for all runs and clips
  - `visibility_f1 = 1.0` for all runs and clips
  - `identity_consistency = 1.0` for all runs and clips
  - `identity_switch_rate = 0.0` for all runs and clips
  - `occlusion_recovery_acc = 0.0` for all runs and clips

These saturated values indicate the current protocol does not yet stress identity/occlusion behavior strongly enough for decisive claims.

## Optimization Objective (from config)

- `loss_formula = trajectory + 0.5*visibility + 0.2*semantic + 0.1*temporal_consistency`

Implication:

- Identity behavior is not explicitly supervised by a dedicated identity loss term in this protocol.
- Identity metrics are currently downstream probes and may not reflect strong learning signal.

## Logging and Audit Note

- `logs/week2_minival_master.log` confirms start/done for all four runs.
- Per-run logs exist but are empty files in the current artifact set:
  - `logs/week2_minival_full.log`
  - `logs/week2_minival_wo_semantics.log`
  - `logs/week2_minival_wo_trajectory.log`
  - `logs/week2_minival_wo_identity_memory.log`

This does not block metric analysis because `run_report.json`, `mini_val_summary_*.json`, checkpoints, and case JSONs are complete.
