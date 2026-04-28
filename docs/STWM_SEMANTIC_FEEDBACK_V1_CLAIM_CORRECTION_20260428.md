# STWM Semantic Feedback V1 Claim Correction

- current feedback mode: readout_only
- true_autoregressive_semantic_feedback: false
- semantic_state_feedback_in_free_rollout: false
- current mechanism: semantic_state_informed_readout_refinement
- paper_world_model_claimable: unclear

The current feedback adapter is applied after the rollout loop and therefore does not affect autoregressive `state_seq`. Feedback V1 improved metrics slightly, but that improvement cannot yet be attributed solely to the adapter because training also updated the FutureSemanticTraceState head, semantic projection, and readout head.

## Forbidden Claims
- semantic-state feedback rollout is validated
- semantic state affects autoregressive rollout
- feedback adapter alone improves results

## Allowed Claim
- readout-only semantic feedback setting gives a small feasibility signal without trace regression
