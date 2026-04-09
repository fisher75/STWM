# Stage2 External Eval Fidelity Audit

- current_stage2_mainline_checkpoint: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt
- tap_style_eval_status: partially_bridged
- tap3d_style_eval_status: not_yet_implemented
- official_evaluator_invoked: True
- official_task_faithfully_instantiated: False
- paper_official_benchmark: False
- official_benchmark_equivalent: False
- why_partially_bridged_not_proxy_only: Kept as partially_bridged because the official TAP-Vid evaluator was actually invoked on an adapter-converted payload, returned metric tensors, and therefore exceeds a pure proxy-only bridge. It is still not an official benchmark result because task faithfulness remains false.

## TAP-Style Checks
- dataset_family_match: False
- query_protocol_match: False
- visibility_protocol_match: False

## Status Boundary Rule
- fully_implemented_and_run: Requires official_evaluator_invoked=true and official_task_faithfully_instantiated=true, meaning the official evaluator ran on a benchmark-faithful task instantiation.
- partially_bridged: Allowed only when the official TAP-Vid evaluator successfully ran on an adapter-converted payload and returned metrics, but official_task_faithfully_instantiated=false remains true.
- proxy_only: Required when only proxy bridge outputs exist, or when no successful official TAP-Vid evaluator invocation/result is available on adapter payload.
- not_yet_implemented: Use when neither proxy bridge nor official-evaluator-side adapter path is operational enough to produce a meaningful probe.

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
