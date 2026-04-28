# STWM World Model No Drift Guardrail V17

- Allowed: Alignment probe is a measurement-space calibration for evaluating FutureSemanticTraceState.
- Allowed: Listwise loss is allowed because candidate identity is an evaluation measurement, not rollout input.
- Allowed: Candidate features are never used in rollout.
- Forbidden: Updating STWM backbone with future candidates.
- Forbidden: Training on heldout.
- Forbidden: Claiming weak AP gain as paper proof without bootstrap.
- Forbidden: Hiding appearance-only dominance.
- Forbidden: Calling STWM a SAM2/CoTracker plugin.
- Forbidden: Claiming external overall SOTA.
