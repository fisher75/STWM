# STWM World Model No-Drift Guardrail V2 20260427

## Allowed
- FutureSemanticTraceState is the main output contract.
- External association/reacquisition is a utility probe.
- SAM2/CoTracker are external consumers/baselines.
- Medium training validates semantic trajectory world-state output only after reality-check and non-degeneracy gates pass.

## Forbidden
- Calling STWM a SAM2/CoTracker plugin.
- Claiming external overall SOTA.
- Claiming semantic trajectory world model from default-off or degenerate head.
- Claiming semantic-state eval when only old association report is used.
- Moving to 1B/longrun before medium semantic-state signal is positive.

- v3_guardrail_decision: `block_medium_training_until_export_eval_hardened`
