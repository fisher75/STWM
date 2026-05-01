# STWM FSTF V7 Baseline Failure Diagnosis

- v7_baselines_valid_as_strong_baselines: `false`
- v7_baselines_valid_as_controlled_sanity: `true`
- proceed_to_scaling_allowed: `false`
- Root cause: learned V7 heads did not include a strong copy prior, so stable subset preservation was damaged and changed gain over copy remained negative.
- Required repair: train copy-aware same-output baselines before scaling.
