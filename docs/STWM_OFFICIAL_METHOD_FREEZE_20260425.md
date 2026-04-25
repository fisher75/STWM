# STWM Official Method Freeze 20260425

- official_method_name: `STWM / TUSB-v3.1 + trace_belief_assoc`
- Stage1: frozen trace-first future-state backbone
- Stage2: TUSB-v3.1 identity-bound semantic trace units
- Readout: `trace_belief_assoc`
- Checkpoint: `best_semantic_hard.pt`
- why belief over hybrid_light: belief uses observed target semantic history / uncertainty and trace-conditioned unit/semantic consistency; final belief reports supersede hybrid_light as official source-of-truth.
- why not teacher-only: tusb_semantic_target is trace-conditioned; trace_belief_assoc combines observed belief, semantic trace evidence, unit identity consistency, and motion plausibility rather than pure teacher crop retrieval.
