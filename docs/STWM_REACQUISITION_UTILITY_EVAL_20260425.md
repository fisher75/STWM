# STWM Reacquisition Utility Eval 20260425

Metrics are recomputed from rank fields on the held-out test split; protocol top1 summary is not copied.

| method | count | top1 | top5 | MRR | false reacquisition | false confuser |
|---|---:|---:|---:|---:|---:|---:|
| STWM trace_belief_assoc | 1038 | 0.4990 | 0.9094 | 0.6680 | 0.5010 | 0.5010 |
| frozen_external_teacher_only | 1038 | 0.4682 | 0.8902 | 0.6456 | 0.5318 | 0.5318 |
| legacysem | 1038 | 0.2004 | 0.7919 | 0.4384 | 0.7996 | 0.7996 |
| calibration-only | 1038 | 0.1869 | 0.7919 | 0.4345 | 0.8131 | 0.8131 |
| cropenc | 1038 | 0.1888 | 0.7967 | 0.4354 | 0.8112 | 0.8112 |
| stage1 frozen | 0 | n/a | n/a | n/a | n/a | n/a |
