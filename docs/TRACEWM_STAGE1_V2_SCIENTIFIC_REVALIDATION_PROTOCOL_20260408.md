# TRACEWM Stage1-v2 Scientific Revalidation Protocol (2026-04-08)

## Round Objective
This round is only for Stage1-v2 scientific revalidation.
It replaces mixed G1-G5 conclusions with defendable three-axis ablations:
state, backbone, and loss family.

## Frozen Current Problems
1. Existing G1-G5 only stacks losses on one multi-token transformer and is not a complete state/backbone/loss three-group ablation.
2. `selected_backbone` in current final comparison is still `debug_small`, not a scientific 220M mainline decision.
3. `ablation_backbone` has not completed a real backbone comparison.
4. Different loss families cannot be compared only by `total_loss`; a unified external evaluation metric is required.

## Scope Rules For This Round
1. Do not modify trace cache.
2. Do not modify GPU selector or perf tooling.
3. Do not enter Stage2.
4. Do not touch WAN or MotionCrafter VAE.
5. Do not touch DynamicReplica, video reconstruction, or joint training.
6. Continue using current first-wave contract.
7. Continue using current recommended runtime.
8. Continue using single-GPU mode only.

## Scientific Axes To Revalidate
1. State ablation: legacy mean-5d GRU vs multi-token GRU.
2. Backbone ablation: multi-token GRU vs transformer debug_small vs transformer prototype_220m.
3. Loss ablation: coord_only, coord+visibility, coord+visibility+residual+velocity, and endpoint-enabled variant.

## Unified Evaluation Policy
1. Primary ranking metric: free-rollout endpoint L2.
2. Secondary ranking metric: free-rollout mean L2.
3. Tertiary ranking metric: TAP-Vid eval (if supported).
4. TAPVid-3D limited eval is supplementary (if supported).
5. `total_loss` is reference-only and is not the primary model selection key.

## Expected Final Decisions
Final comparison must explicitly provide:
1. best_state_variant
2. best_backbone_variant
3. best_loss_variant
4. final_mainline_model
5. final_mainline_parameter_count
6. final_mainline_target_220m_range_pass
7. whether_v2_is_scientifically_validated
8. next_step_choice (only from allowed options)
