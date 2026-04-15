# Stage2 External Eval Completion Results

> This document is completion-round status only. The frozen Stage2 mainline was not retrained in this round.

## Locked Facts
- current_stage2_mainline_checkpoint: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_fullscale_core_cropenc_seed456_20260409/best.pt
- secondary_checkpoint_reference: 
- datasets_bound_for_eval: ['vspw', 'vipseg']
- current_mainline_semantic_source: crop_visual_encoder
- frozen_boundary_kept_correct: True

## Completion Status
- tap_style_eval_status: partially_bridged
- tap_style_proxy_bridge_connected: True
- official_evaluator_invoked: True
- official_tapvid_evaluator_connected: True
- official_task_faithfully_instantiated: False
- tap3d_style_eval_status: not_yet_implemented
- external_eval_readiness: training_ready_but_eval_gap_remains
- next_step_choice: do_one_targeted_external_eval_fix

## TAP-Style Primary Result
- primary_checkpoint_path: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_fullscale_core_cropenc_seed456_20260409/best.pt
- proxy_payload_npz: /home/chen034/workspace/stwm/reports/stage2_aux_external_probe_batch_20260414_assets/cropenc_baseline_best_proxy.npz
- official_payload_npz: /home/chen034/workspace/stwm/reports/stage2_aux_external_probe_batch_20260414_assets/cropenc_baseline_best_official.npz
- average_jaccard: 1.000000
- average_pts_within_thresh: 1.000000
- occlusion_accuracy: 1.000000

## TAP-Style Remaining Gaps
- current frozen stage2 bridge exports future-only core-eval trajectories rather than benchmark-native full TAP-Vid episodes
- current frozen stage2 mainline does not expose a predicted occlusion head; pred_occluded is evaluator-side all-visible adapter output
- current evaluation binding remains VSPW+VIPSeg core-only rather than the official TAP-Vid dataset family

## TAP3D Remaining Gaps
- tap3d_status: not_yet_implemented
- current frozen stage2 external eval binding is fixed to VSPW+VIPSeg, which does not provide TAPVid-3D aligned XYZ ground truth for the checkpoint under test
- current stage2 dataset/bridge path does not export intrinsics, extrinsics, projection, or lifting utilities needed to convert 2D rollout states into camera-consistent 3D trajectories
- current evaluator-side completion round does not yet include a verified adapter that emits official TAPVid-3D prediction files with tracks_XYZ and visibility for the frozen stage2 checkpoint

## Mandatory Answers
1. current mainline checkpoint is still `best.pt`: True
2. TAP-style is currently: `partially_bridged`
3. official TAP evaluator connected: True
4. TAP3D-style progressed to: `not_yet_implemented`
5. project readiness is: `training_ready_but_eval_gap_remains`
