# Stage2 TUSB-V3.1 Hard-Subset Conversion Protocol 20260418

- Stage1 remains frozen. No training, no unfreeze, no backbone swap.
- TUSB-v3 already learned identity binding and z_sem slower-than-z_dyn.
- current unresolved issue is not identity entry anymore; it is hard-subset conversion failure.
- occlusion / long-gap already benefit, but crossing ambiguity and appearance change still lag.
- best.pt is no longer sufficient as the only development checkpoint for TUSB-family.
- this round focuses on ambiguity-aware separation, appearance-drift-aware teacher refinement, hard-subset curriculum, and protocol-aligned checkpoint choice.
