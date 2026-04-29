# STWM World Model No-Drift Guardrail V19

## Allowed
- Future GT semantic features may be used as training targets.
- Candidate features are evaluation measurements only.
- Rollout input remains observed video/trace/target context only.
- The output contract is future semantic trace state / future semantic trajectory field.

## Forbidden
- Future candidate input leakage.
- SAM2/CoTracker plugin framing.
- Appearance-only result as STWM success.
- Post-hoc scorer tuning as the main contribution.
- Paper world-model claim before feature prediction signal.
