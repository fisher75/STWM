# STWM World Model No-Drift Guardrail V13

## Allowed
- Candidate-expanded external query eval tests FutureSemanticTraceState as a world-state utility.
- Future candidates may be used only for eval scoring, not rollout input.
- Association/reacquisition remains utility, not method definition.

## Forbidden
- Feeding future candidate geometry into model input.
- Fallback to Stage2 val split in external query mode.
- Reading old association reports for semantic-state eval.
- Claiming feedback adapter effect.
- Claiming STWM as a SAM2/CoTracker plugin.
- Continuing semantic-state training before external query bridge is valid and positive.
