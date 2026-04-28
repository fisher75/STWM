# STWM World Model No-Drift Guardrail V10

## Allowed
- readout_only feedback can be studied as semantic-state readout refinement.
- true rollout feedback requires separate implementation and smoke.
- attribution is required before expanding training.

## Forbidden
- calling readout_only feedback true rollout feedback.
- claiming feedback adapter effect without same-checkpoint enabled/disabled ablation.
- expanding feedback training before attribution.
- moving to 1B before feedback mechanism is proven.
- turning STWM into SAM2/CoTracker plugin.
