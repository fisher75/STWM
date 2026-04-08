# TRACEWM Stage1-v2 220M Competitiveness Gap Closure Protocol (2026-04-08)

## Frozen Current Fact
1. Latest Stage1-v2 scientific rigor fix has been completed.
2. TraceWM v2 scientific protocol, TAP-Vid evaluation, TAPVid-3D limited evaluation, and mainline replay artifacts already exist.
3. Current final comparison conclusion is:
   - best_small_model = stage1_v2_backbone_transformer_debugsmall
   - best_220m_model = stage1_v2_backbone_transformer_prototype220m
   - should_promote_220m_now = false
   - next_step_choice = run_220m_competitiveness_gap_closure
4. The remaining problem is no longer protocol missing. The real issue is 220M competitiveness gap versus small model.

## Round Objective
Answer only one question in the same scientific protocol:
How to let prototype_220m approach or surpass debug_small without changing task definition?

## Scope Lock
1. Do not modify P0 trace cache.
2. Do not modify perf tooling.
3. Do not enter Stage2.
4. Do not touch WAN or MotionCrafter VAE.
5. Do not touch joint training.
6. Do not expand dataset scope.

## Allowed Change Classes
1. Training budget upgrade (longer train, short eval, clearly above 8/12-step level).
2. 220M-only optimization on optimizer/lr/warmup/wd, batch/accum/clip, and small loss-weight tuning.
3. Strict debug_small reference + prototype_220m main group comparison.

## Fixed Experiment Matrix (exactly five runs)
1. stage1_v2_gap_debugsmall_ref
2. stage1_v2_gap_220m_ref
3. stage1_v2_gap_220m_opt_lr
4. stage1_v2_gap_220m_opt_batch
5. stage1_v2_gap_220m_opt_lossweights

## Unified Selection Policy
1. Primary: free_rollout_endpoint_l2
2. Secondary: free_rollout_coord_mean_l2
3. Tertiary: TAP-Vid free_rollout_endpoint_l2
4. Quaternary: TAPVid-3D limited free_rollout_endpoint_l2
5. total_loss is reference-only and never the winner selector.
