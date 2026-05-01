# STWM Same Output Baseline Suite V6 20260501

## Status
- baseline_suite_completed: `False`
- STWM_beats_same_output_baselines: `False`
- strongest_same_output_baseline: `copy_semantic_memory_baseline`

## Mixed Main Result vs Copy
- changed_gain_vs_copy: `0.08164124822622099`
- overall_top5_delta: `0.036400633624272816`
- stable_drop: `0.00020451294277912258`

## Missing Baselines
- trace_only_AR_transformer_baseline
- semantic_only_memory_transition_baseline
- simple_semantic_plus_trace_transformer_baseline
- slotformer_like_trace_unit_slot_dynamics_baseline
- dino_wm_like_latent_feature_dynamics_baseline

## Note
- The audit is complete, but the evidence is not. We cannot claim a same-output baseline suite is complete until these baselines are implemented and evaluated under the frozen free-rollout STWM-FSTF protocol.
