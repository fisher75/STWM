# STWM Tables And Figures Plan V23

## Main Tables
### Table 1
- FSTF main result on mixed / VSPW / VIPSeg
- include copy baseline, STWM residual, changed gain, stable preservation, CE, trace guardrail

### Table 2
- controlled same-output baselines
- include strongest copy-aware baseline `copy_residual_mlp`
- keep proxy-inspired SlotFormer/DINO-WM style rows out of the main table

### Table 3
- stable vs changed breakdown
- emphasize changed improvement and stable preservation

### Table 4
- utility / trace_belief association
- STWM trace_belief_assoc
- frozen external teacher only
- legacy semantic baselines

### Table 5
- external consumer boundary
- STWM vs SAM2 / CoTracker / Cutie
- clearly marked as boundary / downstream utility, not same-output FSTF baselines

## Main Figures
### Figure 1
- method overview
- observed trace + observed semantic memory -> future semantic trace-unit field

### Figure 2
- FSTF rollout qualitative strip
- VSPW changed success
- VIPSeg changed success
- copy failure fixed by STWM
- stable preservation

### Figure 3
- changed vs stable analysis
- residual gain over copy with stable preservation

### Figure 4
- boundary/utility figure
- long-gap and occlusion-heavy comparison narrative

## Appendix Tables
- LODO negative / domain-shift summary
- scaling boundary summary
- OSTF V16-V22 progression and no-go table

## Appendix Figures
- FSTF extra rollout gallery
- OSTF real-teacher cache gallery
- OSTF V21 oracle multimodal diagnostic
- OSTF V22 calibrated top-1 no-go cases
