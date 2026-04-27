# STWM World Model I/O Audit 20260427

## Current Level
- current_world_model_level: `['trace_future_state_backbone', 'semantic_conditioned_rollout_adapter']`
- Official checkpoint still does not emit a trained future semantic trace field.
- Current live repo now contains an optional default-off FutureSemanticTraceState head/schema.

## Missing For True Semantic Trajectory World Model
- trained future_semantic_embedding targets and nonzero loss weights
- trained future_visibility/reappearance logits with calibrated labels
- calibrated future_uncertainty and uncertainty ECE from explicit head outputs
- multi-hypothesis semantic rollout training beyond K=1 default
- free-rollout semantic state feedback so future semantic predictions influence later steps
- optional action/ego-motion conditioning interface beyond placeholder framing
- full evaluation that consumes FutureSemanticTraceState rather than association-only reports

## SAM2 / CoTracker Boundary
- SAM2/CoTracker remain external baselines/utility probes, not STWM core method.
