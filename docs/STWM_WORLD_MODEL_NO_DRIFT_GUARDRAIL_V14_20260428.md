# STWM World Model No-Drift Guardrail V14

## Allowed
- Candidate observations are used only as measurement likelihood for posterior scoring.
- Rollout remains candidate-free.
- Semantic/identity compatibility is part of world-state utility evaluation.

## Forbidden
- Feeding future candidates into rollout input.
- Claiming distance-only scoring as semantic world-model evidence.
- Stopping semantic-state branch before scoring autopsy.
- Tuning weights on heldout.
- Calling STWM a SAM2/CoTracker plugin.
- Claiming external overall SOTA.
