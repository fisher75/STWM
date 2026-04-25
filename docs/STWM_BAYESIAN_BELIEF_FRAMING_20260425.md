# STWM Bayesian Belief Framing 20260425

## State Variables
- z_dyn: dynamic trace state / future-state substrate; partially available through TUSB/state diagnostics, not claimed as full generative video state
- z_sem: slow semantic state / semantic trace representation where readable; used conceptually for identity belief
- trace_units: identity-bound semantic trace units from TUSB-v3.1
- semantic_identity_belief: observed target semantic belief mean/variance proxy from target history and trace-conditioned semantic/unit evidence
- candidate_observation: future candidate crop/mask/coordinate observations in context-preserving eval

## Probabilistic Framing
- trace_transition_prior: p(z_t | z_<t, trace units) provides dynamic/identity continuity prior over future association
- semantic_measurement_likelihood: p(o_candidate | z_sem, target belief) scores candidate semantic consistency with observed target belief
- belief_update_posterior_association: p(candidate is target | trace prior, semantic likelihood, unit identity, coord plausibility) is approximated by trace_belief_assoc
- readout: future identity association, not full video generation or teacher-guided retrieval

## Claim
- trace defines dynamic prior; semantic evidence defines measurement likelihood; belief readout performs posterior future identity association.

## Not Claimed
- full video generation
- teacher-guided retrieval as official method
- protocol v4
- new backbone
