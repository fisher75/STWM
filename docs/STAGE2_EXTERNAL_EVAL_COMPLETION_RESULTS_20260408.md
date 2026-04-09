# Stage2 External Eval Completion Results

> This document is claim-boundary-hardened completion status only. The frozen Stage2 mainline was not retrained in this round.

## Locked Facts
- current_stage2_mainline_checkpoint: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt
- secondary_checkpoint_reference: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/latest.pt
- datasets_bound_for_eval: ['vspw', 'vipseg']
- current_mainline_semantic_source: crop_visual_encoder
- frozen_boundary_kept_correct: True

## Claim Boundary
- tap_style_eval_status: partially_bridged
- official_evaluator_invoked: True
- official_task_faithfully_instantiated: False
- paper_official_benchmark: False
- current_metric_scope: adapter-based TAP-style probe: official TAP-Vid evaluator run on an adapter-converted, non-benchmark-faithful payload exported from the frozen Stage2 core-only VSPW+VIPSeg binding
- why_partially_bridged_not_proxy_only: Kept as partially_bridged because the official TAP-Vid evaluator was actually invoked on an adapter-converted payload, returned metric tensors, and therefore exceeds a pure proxy-only bridge. It is still not an official benchmark result because task faithfulness remains false.

## Adapter-Based TAP-Style Probe
- claim_scope_name: adapter_based_tap_style_probe
- adapter_probe_only: True
- paper_official_benchmark: False
- primary_checkpoint_path: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt
- average_jaccard: 0.900000
- average_pts_within_thresh: 0.900000
- occlusion_accuracy: 1.000000

## TAP-Style Remaining Gaps
- current frozen stage2 bridge exports future-only core-eval trajectories rather than benchmark-native full TAP-Vid episodes
- current frozen stage2 mainline does not expose a predicted occlusion head; pred_occluded is evaluator-side all-visible adapter output
- current evaluation binding remains VSPW+VIPSeg core-only rather than the official TAP-Vid dataset family

## Best vs Latest Reference
- comparison_scope: adapter_based_tap_style_probe_only
- adapter_probe_only: True
- paper_official_benchmark: False
- free_rollout_coord_mean_l2_best: 0.004344
- free_rollout_coord_mean_l2_latest: 0.008631
- tapvid_average_jaccard_best: 0.900000
- tapvid_average_jaccard_latest: 0.662500

## Secondary Reference Label
- secondary_exists: True
- secondary numbers remain adapter-probe-only and must not be treated as official benchmark values.

## Mandatory Answers
1. current mainline checkpoint is still `best.pt`: True
2. TAP-style is currently: `partially_bridged`
3. official TAP evaluator connected: True
4. TAP3D-style progressed to: `not_yet_implemented`
5. project readiness is: `training_ready_but_eval_gap_remains`
