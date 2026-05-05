# STWM Overleaf Outline V23

## Main Paper Mode
- Main paper = `FSTF + TUSB/trace_belief utility`
- OSTF = `appendix / future-work redesign`

## Title Direction
- Semantic trace-unit world model over frozen video-derived trace/semantic states

## Core Sections
1. Introduction
2. Related Work
3. STWM-FSTF Task and Benchmark Protocol
4. Method
   - observed semantic memory
   - copy-gated residual transition
   - future rollout hidden
   - semantic trace-unit field output
5. Main Results
   - mixed / VSPW / VIPSeg
   - stable vs changed
   - same-output controlled baselines
6. Mechanism and Scaling Boundary
   - future-hidden load-bearing
   - C32 vocabulary tradeoff
   - horizon gains
   - density/model-size caveats
7. Utility / Trace-Belief Association
   - TUSB light readout
   - trace_belief association
   - continuity-heavy utility
8. External Boundary
   - SAM2 / CoTracker / Cutie as consumer/boundary comparisons
9. Limitations
   - no raw-video end-to-end claim
   - no dense semantic trace field claim
   - no universal OOD claim
10. Conclusion

## Appendix Placement
- Appendix A: LODO negative/domain-shift limitation
- Appendix B: FSTF scaling and valid-unit audits
- Appendix C: OSTF real-teacher cache and V17-V22 no-go progression
- Appendix D: Visualization gallery
