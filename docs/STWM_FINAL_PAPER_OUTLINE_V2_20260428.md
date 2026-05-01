# STWM Final Paper Outline V2 20260428

## Introduction
- Trace dynamics alone are insufficient for future semantic identity reasoning.
- Observed semantic memory plus future trace rollout yields a better world-model state.
- Copy-gated residual transition preserves stable semantics while correcting changed states.

## Related Work
- trajectory fields / Trace Anything
- object-centric dynamics / SlotFormer
- real-world object-centric learning / SAVi++
- future instance prediction / FIERY
- latent feature world models / DINO-WM
- video diffusion is not same-output baseline / MotionCrafter distinction

## Method
- trace backbone
- semantic trace units
- observed semantic memory
- copy-gated residual semantic transition
- belief readout utility

## Experiments
- mixed/VSPW/VIPSeg free-rollout semantic trace field
- stable/changed analysis
- prototype vocabulary scaling
- trace guardrail
- optional LODO/scaling appendices
- belief utility evidence

## Limitations
- LODO/scaling/baseline/video evidence pack still incomplete in live repo
- semantic trace-unit field wording unless density scaling is completed
