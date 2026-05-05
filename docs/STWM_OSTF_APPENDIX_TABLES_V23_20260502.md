# STWM OSTF Appendix Tables V23

## Scope
- This appendix material is diagnostic and forward-looking.
- It must not be used to claim that `OSTF` beats `CV` or that object-dense semantic trajectory world modeling is solved.

## Appendix Table A: Real Teacher Cache Statistics
- Source: `V16`
- Include:
  - processed clips
  - point count
  - valid point ratio
  - teacher source
  - persistent identity validity
  - fake-dense check

## Appendix Table B: V17-V22 Progression
- `V17`: naive multi-trace baseline
- `V18`: physics-prior residual improves over V17
- `V19`: semantic fairness fix
- `V20`: context-aware deterministic failure
- `V21`: oracle multimodal diagnostic
- `V22`: calibrated top-1 no-go

## Appendix Table C: CV Saturation and Oracle Diagnostic
- Show:
  - V21 best-of-K / minFDE improvements over CV
  - hard-subset gains
  - deterministic no-harm failure
  - V22 top-1 selection collapsing back to CV

## Appendix Figure/Video Usage
- Use V16 real-teacher cache visualizations to demonstrate that the teacher cache is real.
- Use selected V21/V22 qualitative cases to show:
  - oracle alternative hypotheses
  - top-1 CV-mode selection
  - why current setup is appendix-only

## Recommended Future Work
- Move to a harder trajectory-field teacher and/or harder benchmark.
- Do not continue same-setup small-model iteration as the main path.
