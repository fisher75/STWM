# STWM Final LODO Domain Shift Diagnosis V5 20260428

## Main Finding
- LODO negative is best explained as cross-dataset domain shift in prototype priors and transition statistics, not as a failure of the mixed-protocol semantic world model.

## Prototype Shift
- C32 JS divergence: `0.160388`
- C64 JS divergence: `0.169380`

## Coverage / Ratio Shift
- VSPW observed coverage: `0.4015`
- VIPSEG observed coverage: `0.5438`
- VSPW changed ratio: `0.6242`
- VIPSEG changed ratio: `0.3970`

## Interpretation
- VIPSEG does not fail because observed memory is missing anymore. The harder part is cross-dataset transfer of semantic transition statistics and prototype priors.
