# STWM V4.2 Protocol Best Dryrun (D0)

Date: 2026-04-03
Status: completed

## Scope

- checkpoint_dir: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints
- dryrun_dir: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints/protocol_eval_dryrun_20260403_142929
- evaluator: code/stwm/evaluators/eval_mini_val.py
- protocol manifest: manifests/protocol_v2/protocol_val_main_v1.json
- protocol version request: v2_4_detached_frozen

## Candidate Checkpoints

- best.pt
- latest.pt

## Protocol-Main Metrics Per Candidate

| candidate | query_localization_error | query_top1_acc | future_trajectory_l1 | action | improved |
|---|---:|---:|---:|---|---:|
| best.pt | 0.009459 | 0.875000 | 0.009240 | updated | 1 |
| latest.pt | 0.004150 | 1.000000 | 0.003709 | updated | 1 |

## Final Official Best

- best_protocol_main_exists: true
- selection_sidecar_exists: true
- selected_candidate_checkpoint: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints/latest.pt
- selected_metrics: query_localization_error=0.004150, query_top1_acc=1.000000, future_trajectory_l1=0.003709

Selection rationale follows protocol_best_rule_v2 exactly:
1. primary: query_localization_error (lower)
2. tie-break 1: query_top1_acc (higher)
3. tie-break 2: future_trajectory_l1 (lower)

## Toolchain Readiness For Future 500-Step Updates

This dryrun validates end-to-end chain on a real completed run:
1. detached eval summary generation
2. protocol best updater invocation
3. best_protocol_main.pt materialization
4. sidecar selection record persistence

Conclusion: the chain is executable for periodic future updates every 500 steps when integrated in D1 training jobs.

## Artifacts

- dryrun_report_json: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints/protocol_eval_dryrun_20260403_142929/d0_protocol_best_dryrun_report.json
- official_best_checkpoint: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints/best_protocol_main.pt
- official_best_sidecar: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints/best_protocol_main_selection.json
