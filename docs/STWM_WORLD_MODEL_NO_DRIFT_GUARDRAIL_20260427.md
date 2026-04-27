# STWM World Model No-Drift Guardrail 20260427

## Allowed
- STWM outputs future semantic trace state.
- SAM2/CoTracker are external consumers or baselines.
- Future identity association is one utility of the world state.

## Forbidden
- STWM is a SAM2 plugin.
- STWM is only a candidate reranker.
- STWM beats SAM2 overall.
- Future semantic query eval is valid if it only reads old association reports.
- A default-off untrained head is enough to claim semantic trajectory world model.
