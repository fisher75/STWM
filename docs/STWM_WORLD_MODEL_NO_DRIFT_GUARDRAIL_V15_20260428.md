# STWM World Model No-Drift Guardrail V15

## Allowed
- Future candidate crops are measurement observations used only for posterior scoring.
- Rollout remains candidate-free.
- Candidate measurement features test whether FutureSemanticTraceState is externally useful.

## Forbidden
- Feeding future candidates into rollout input.
- Calling weak bbox/RGB stats strong semantic evidence.
- Claiming semantic-state branch if only target-candidate appearance helps.
- Tuning weights on heldout.
- Claiming SAM2/CoTracker plugin.
- Continuing training before measurement problem is solved.
