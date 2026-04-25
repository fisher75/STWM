# STWM Reacquisition Utility Bootstrap 20260425

Positive delta means STWM improvement.

## STWM vs frozen_external_teacher_only

| metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---:|---:|---:|---:|---:|---:|
| top1 | 1038 | 0.0308 | 0.0000 | 0.0636 | False | 0.1522 |
| top5 | 1038 | 0.0193 | 0.0058 | 0.0328 | True | 0.0347 |
| MRR | 1038 | 0.0224 | 0.0033 | 0.0421 | True | 0.3006 |
| false_reacquisition_rate | 1038 | 0.0308 | 0.0000 | 0.0636 | False | 0.1522 |
| false_confuser_rate | 1038 | 0.0308 | 0.0000 | 0.0636 | False | 0.1522 |

## STWM vs legacysem

| metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---:|---:|---:|---:|---:|---:|
| top1 | 1038 | 0.2987 | 0.2620 | 0.3343 | True | 0.3690 |
| top5 | 1038 | 0.1175 | 0.0925 | 0.1435 | True | 0.1541 |
| MRR | 1038 | 0.2296 | 0.2051 | 0.2549 | True | 0.6108 |
| false_reacquisition_rate | 1038 | 0.2987 | 0.2620 | 0.3343 | True | 0.3690 |
| false_confuser_rate | 1038 | 0.2987 | 0.2620 | 0.3343 | True | 0.3690 |

## STWM vs calibration-only

| metric | count | mean_delta | ci95_low | ci95_high | zero_excluded | win_rate |
|---|---:|---:|---:|---:|---:|---:|
| top1 | 1038 | 0.3121 | 0.2775 | 0.3478 | True | 0.3757 |
| top5 | 1038 | 0.1175 | 0.0925 | 0.1445 | True | 0.1561 |
| MRR | 1038 | 0.2336 | 0.2092 | 0.2579 | True | 0.6146 |
| false_reacquisition_rate | 1038 | 0.3121 | 0.2775 | 0.3478 | True | 0.3757 |
| false_confuser_rate | 1038 | 0.3121 | 0.2775 | 0.3478 | True | 0.3757 |

claim_level = `strong_claim`
