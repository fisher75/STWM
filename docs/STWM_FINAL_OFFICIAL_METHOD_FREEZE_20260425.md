# STWM Final Official Method Freeze 20260425

- official_method_name: `STWM / TUSB-v3.1 + trace_belief_assoc`
- official_checkpoint: `best_semantic_hard.pt`
- official_readout: `trace_belief_assoc`
- frozen_components: Stage1 backbone, Stage2 TUSB-v3.1 backbone, semantic teacher/cache inputs, protocol panels/splits, matched 6 seeds, best_semantic_hard.pt checkpoint
- trainable_components: none in this freeze step
- why belief: final belief validation is the current source of truth and supersedes hybrid/residual/gallery variants.
- no_longer_mainline_variants: hybrid_light, clean_residual_v2, trace_gallery_assoc, trace_prototype_only, coord_only, tusb_semantic_target-only, frozen_external_teacher_only
