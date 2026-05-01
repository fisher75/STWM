# STWM Final Prototype Vocab Scaling 20260428

## What C Means
- C is semantic prototype vocabulary size controlling granularity vs stability.

## Selection
- selected_C: `32`
- selected_seed: `456`
- selected_C_justified: `True`
- missing_requested_C: `[16]`

## Reason
- Early prototype sweep showed C256 becomes long-tailed and C128 is finer but less stable; final mixed fullscale val-only selection over the complete 10-run matrix chose C32 seed456 as the best changed-gain tradeoff with low stable drop and low trace error.
