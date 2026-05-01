# STWM Horizon Scaling H16 V1

## Status

- current_horizon: `8`
- code_supports_h16: `True`
- h16_completed: `False`
- blocker: `Current live semantic target caches, materialization reports, and all selected mixed checkpoints are built for H=8 only. Code exposes --fut-len, but H=16 would require rebuilding observed/future target pools, rematerializing eval caches, and retraining at least the best mixed config.`

## Interpretation

- H=8 evidence is strong and paper-usable.
- H=16 remains a scaling appendix, not a main-claim prerequisite.
