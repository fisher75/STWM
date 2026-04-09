# Stage2 External Eval Fidelity Audit

- current_stage2_mainline_checkpoint: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt
- official_evaluator_invoked: True
- official_tapvid_evaluator_connected: True
- official_task_faithfully_instantiated: False
- tap_style_eval_status: partially_bridged
- tap3d_style_eval_status: not_yet_implemented

## TAP-Style Checks
- benchmark_native_full_tap_episode: False
- query_time_matches_official_task: False
- pred_visibility_from_model_output: False
- dataset_binding_is_official_tap_dataset_family: False

## TAP3D Checks
- aligned_3d_gt_for_current_binding: False
- camera_geometry_projection_or_lifting_path_available: False
- verified_exporter_to_tracks_xyz_visibility: False
- official_tapvid3d_metric_importable: True

## Blocking Reasons
- current frozen stage2 bridge exports future-only core-eval trajectories rather than benchmark-native full TAP-Vid episodes
- current frozen stage2 mainline does not expose a predicted occlusion head; pred_occluded is evaluator-side all-visible adapter output
- current evaluation binding remains VSPW+VIPSeg core-only rather than the official TAP-Vid dataset family
- current frozen stage2 external eval binding is fixed to VSPW+VIPSeg, which does not provide TAPVid-3D aligned XYZ ground truth for the checkpoint under test
- current stage2 dataset/bridge path does not export intrinsics, extrinsics, projection, or lifting utilities needed to convert 2D rollout states into camera-consistent 3D trajectories
- current evaluator-side completion round does not yet include a verified adapter that emits official TAPVid-3D prediction files with tracks_XYZ and visibility for the frozen stage2 checkpoint

## Packaging Note
- completion_log_exists: True
- packaged_in_repo_snapshot: False
- note: exists locally but logs/** are gitignored
