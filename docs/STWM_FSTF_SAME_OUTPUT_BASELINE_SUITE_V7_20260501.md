# STWM FSTF Same-Output Baseline Suite V7

- baseline_suite_completed: `True`
- completed_baseline_count: `5`
- new_checkpoint_count: `17`
- new_eval_summary_count: `17`
- strongest_same_output_baseline: `trace_semantic_transformer` / `plain_trace_semantic_transformer`
- strongest_baseline_seed_count: `5`
- STWM_beats_same_output_baselines: `True`
- next_step_choice: `run_scaling_laws`

## Seed Mean/Std
- trace_only_ar_transformer: changed_gain_mean=-0.181285, overall_delta_mean=-0.328950, stable_drop_mean=0.448429, seeds=3
- semantic_only_memory_transition: changed_gain_mean=-0.179684, overall_delta_mean=-0.174027, stable_drop_mean=0.169450, seeds=3
- trace_semantic_transformer: changed_gain_mean=-0.040441, overall_delta_mean=-0.084000, stable_drop_mean=0.119245, seeds=5
- slotformer_like_trace_unit_dynamics: changed_gain_mean=-0.056815, overall_delta_mean=-0.107594, stable_drop_mean=0.148681, seeds=3
- dino_wm_like_latent_dynamics_proxy: changed_gain_mean=-0.357767, overall_delta_mean=-0.524782, stable_drop_mean=0.659918, seeds=3

## Table Placement
- Main FSTF table: copy lower bound plus controlled ablations only.
- Appendix proxy table: slot_ar_trace_unit_proxy and dinov2_latent_dynamics_proxy.
- External boundary table: SAM2, CoTracker, Cutie only.

## Officiality Guardrail
- Do not write: STWM beats SlotFormer.
- Do not write: STWM beats DINO-WM.
- Allowed: STWM outperforms controlled slot-AR trace-unit proxy / DINOv2 latent dynamics proxy if supported by appendix proxy table.
