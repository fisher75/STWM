# STWM True Semantic Rollout Feedback Feasibility Audit

Current readout_only feedback is not true rollout feedback: it is applied after the free-rollout loop and does not affect `state_seq` or next-step hidden dynamics.

A true rollout feedback implementation should insert a small gated residual inside the rollout loop, ideally first in teacher-forced one-step form, then as a free-rollout smoke with alpha <= 0.02. Risks include trace drift, feedback collapse, error accumulation, and loss entanglement.

Immediate implementation is not recommended because same-checkpoint readout-only attribution found no measurable adapter residual effect.
