# TRACEWM Stage2 External-Eval Completion Protocol (2026-04-08)

## 1. Frozen Facts

1. The current Stage2 mainline is already frozen.
2. The frozen mainline semantic source remains `crop_visual_encoder`.
3. The current frozen boundary must stay intact:
   - Stage1 220m backbone remains frozen.
   - Stage1 tokenizer / core rollout backbone remains frozen.
4. `best.pt` is the readiness anchor for this round.
5. `latest.pt` may be read only as a secondary reference.

## 2. What This Round Is

1. This round is `Stage2 external-eval completion round`.
2. This round is evaluator-side completion only.
3. This round is not a trainer-side continuation.
4. This round exists to make the frozen `stage2_core_cropenc` line more defendable from an external-eval perspective.

## 3. What This Round Does Not Change

1. No Stage1 backbone changes.
2. No Stage2 core mainline changes.
3. No Stage2 semantic-source changes.
4. No dataset-binding expansion beyond `VSPW + VIPSeg` for the core eval path.
5. No WAN / MotionCrafter work.
6. No BURST / TAO / VISOR mainline expansion.

## 4. Hard Prohibitions

1. No new Stage2 training.
2. No new Stage1 training.
3. No architecture search.
4. No new long training.
5. No backtracking to old Stage1 work.

## 5. Allowed Operations

1. Read the frozen checkpoint.
2. Run evaluator bridge / payload export / metric adapter code.
3. Run very short evaluator-side utility jobs if required for completion.
4. Write completion-only docs and reports.

## 6. Fixed Evaluation Target

1. Primary checkpoint:
   - `outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt`
2. Secondary reference:
   - `outputs/checkpoints/stage2_core_mainline_train_20260408/latest.pt`
3. Data binding remains core-only:
   - `VSPW + VIPSeg`

## 7. TAP-Style Completion Contract

1. Do not stop at a proxy-only payload unless the official evaluator truly cannot be connected.
2. The output must explicitly record:
   - `proxy_bridge_connected`
   - `official_tapvid_evaluator_connected`
3. If the official evaluator is still not directly usable, the report must record:
   - exact blocking reason
   - exact missing component
4. If the official evaluator can run, the report must still separate:
   - what is genuinely official now
   - what remains only partially bridged

## 8. TAP3D-Style Completion Contract

1. Do not overclaim full implementation.
2. TAP3D-style status must be one of:
   - `fully_implemented_and_run`
   - `partially_bridged`
   - `not_yet_implemented`
3. If TAP3D is not fully runnable, the report must explicitly isolate:
   - 3D GT alignment gap
   - camera geometry / projection / lifting gap
   - metric adapter gap

## 9. Required Completion Outputs

1. `reports/stage2_external_eval_completion_20260408.json`
2. `docs/STAGE2_EXTERNAL_EVAL_COMPLETION_RESULTS_20260408.md`

The completion json must include at least:

1. `current_stage2_mainline_checkpoint`
2. `datasets_bound_for_eval`
3. `current_mainline_semantic_source`
4. `frozen_boundary_kept_correct`
5. `tap_style_eval_status`
6. `tap_style_proxy_bridge_connected`
7. `official_tapvid_evaluator_connected`
8. `tap3d_style_eval_status`
9. `tap3d_missing_components`
10. `external_eval_readiness`
11. `exact_blocking_reasons`
12. `next_step_choice`

## 10. Allowed Status Values

1. `tap_style_eval_status ∈ {fully_implemented_and_run, partially_bridged, proxy_only, not_yet_implemented}`
2. `tap3d_style_eval_status ∈ {fully_implemented_and_run, partially_bridged, not_yet_implemented}`
3. `external_eval_readiness ∈ {paper_eval_ready, training_ready_but_eval_gap_remains, eval_not_ready}`

## 11. End-Of-Round Questions That Must Be Answered

1. Is the current mainline checkpoint still `best.pt`?
2. Is TAP-style now `proxy_only`, `partially_bridged`, or `fully_implemented_and_run`?
3. Is the official TAP evaluator really connected?
4. How far did TAP3D-style actually advance?
5. Is the project now `paper_eval_ready` or still `training_ready_but_eval_gap_remains`?

## 12. Runtime Contract

1. Fixed tmux session:
   - `tracewm_stage2_external_eval_completion_20260408`
2. Fixed log:
   - `/home/chen034/workspace/stwm/logs/tracewm_stage2_external_eval_completion_20260408.log`
