# STWM Final Paper Outline

## Structure
1. Introduction
2. Related Work
3. Method
4. Experiments
5. Limitations

## Method Blocks
- Stage1 frozen trace-first rollout backbone
- TUSB semantic trace units
- observed semantic memory
- copy-gated residual semantic transition
- future semantic prototype field
- trace_belief_assoc utility readout

## Experiments
- semantic trace field prediction
- stable/changed analysis
- mixed and per-dataset breakdown
- trace guardrail
- belief association / reacquisition / planning-lite / counterfactual utilities
- ablations and limitations

- audit_name: `stwm_final_paper_outline`
- preferred_title: `STWM: A Semantic-Trace World Model with Belief Filtering for Future Identity Association`
- abstract_skeleton: `We model future video state as trace-conditioned semantic fields rather than RGB generation or tracker post-processing. STWM rolls out future trace units and semantic prototype fields from observed video/trace/semantic memory, using a copy-gated residual transition for semantic changes and a belief readout for identity association utilities.`
