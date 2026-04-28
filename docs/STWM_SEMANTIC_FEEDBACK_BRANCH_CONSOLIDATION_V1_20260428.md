# STWM Semantic Feedback Branch Consolidation V1

- readout_only feedback adapter has measurable same-checkpoint effect: false
- enabled = disabled = zero_delta: true
- enabled_minus_disabled_event_AP: 0.0
- enabled_minus_zero_delta_event_AP: 0.0
- enabled_minus_disabled_per_horizon_AP: 0.0
- alpha_sensitivity_observed: false

The feedback adapter should not be claimed as load-bearing. The current best semantic-state checkpoint should be treated as **semantic_state_joint_readout_refinement**, not as a feedback-adapter method.

Allowed: event-level reappearance supervision gives preliminary positive signal. Forbidden: semantic-state feedback rollout works, or feedback adapter improves results.
