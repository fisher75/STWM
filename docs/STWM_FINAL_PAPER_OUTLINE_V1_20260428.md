# STWM Final Paper Outline V1

## Introduction
- Trace dynamics alone are insufficient for future semantic identity reasoning.
- Observed semantic memory plus future trace rollout yields a better world-model state.
- Copy-gated residual transition preserves stable semantics while correcting changed states.

## Related Work
- trajectory field / point tracking
- object-centric world models
- future instance prediction
- semantic world models

## Method
- trace backbone
- semantic trace units
- observed semantic memory
- copy-gated residual semantic transition
- belief readout utility

## Experiments
- mixed/VSPW/VIPSeg free-rollout semantic trace field prediction
- stable vs changed analysis
- trace guardrail
- belief utility / counterfactual evidence
- optional LODO appendix
- optional horizon and density appendices

## Limitations
- VIPSeg effect smaller than VSPW
- LODO appendix not yet executed
- H=8 evidence stronger than longer-horizon evidence
- current field is semantic trace-unit field, not dense pixel field
