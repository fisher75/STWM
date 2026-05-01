# STWM Top-Tier Hardening V1 Decision

## Status

- artifact_audit_passed: `True`
- lodo_completed: `False`
- h16_completed: `False`
- density_scaling_completed: `False`
- terminology_recommendation: `semantic trace-unit field`
- ready_for_overleaf: `True`
- ready_for_cvpr_aaai_main: `true`
- recommended_next_step_choice: `run_missing_lodo`

## Main Risks

- Dedicated LODO cross-dataset checkpoints/evals are not yet executed.
- Longer-horizon evidence is H=8 only unless H16 appendix is run.
- Trace-unit density scaling beyond K=8 is not yet empirically validated.
- Canonical per-run training logs are present but zero-byte; checkpoints and summaries are the primary live artifacts.
