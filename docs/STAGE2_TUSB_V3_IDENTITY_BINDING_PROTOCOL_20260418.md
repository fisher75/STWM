# Stage2 TUSB-V3 Identity-Binding Protocol 20260418

- Stage1 remains frozen. No training, no unfreeze, no backbone swap.
- Current TUSB-v2 already landed.
- anti-collapse is load-bearing.
- z_sem slower_than_z_dyn = true.
- multi-entity data path, cache, and context-preserving eval already exist.
- core unresolved issue: semantic_instance_id_* reaches dataset/batch, but current training mostly does not use true instance identity as supervision.
- current protocol flatness is no longer primarily attributable to eval mismatch; context-preserving eval exists and remains flat.
- this round only repairs identity binding. No protocol v4, no persistence, no Stage1 edits, no calibration-only micro-fix.
