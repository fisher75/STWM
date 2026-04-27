# STWM External Baseline Full Eval Summary 20260426

| method | top1 | top5 | MRR | false confuser | long-gap top1 | OOD top1 |
|---|---:|---:|---:|---:|---:|---:|
| SAM2 | 0.671 | 0.925 | 0.775 | 0.329 | 0.538 | 0.662 |
| Cutie | 0.488 | 0.825 | 0.626 | 0.512 | 0.462 | 0.503 |
| CoTracker | 0.602 | 0.895 | 0.729 | 0.398 | 0.327 | 0.603 |
| STWM trace_belief_assoc | 0.509 | 0.865 | 0.660 | 0.491 | 0.548 | 0.518 |
| calibration-only | 0.151 | 0.644 | 0.357 | 0.849 | 0.192 | 0.154 |
| cropenc | 0.149 | 0.646 | 0.356 | 0.851 | 0.221 | 0.150 |
| legacysem | 0.147 | 0.650 | 0.358 | 0.853 | 0.221 | 0.150 |
| frozen_external_teacher_only | 0.520 | 0.931 | 0.695 | 0.480 | 0.590 | NA |
