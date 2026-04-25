# STWM Planning-Lite Risk Eval 20260425

| method | count | risk AUC | false-safe | false-alarm | top1 safe-path acc | top-k safe-path acc |
|---|---:|---:|---:|---:|---:|---:|
| STWM trace_belief_assoc risk | 14580 | 0.7595 | 0.3212 | 0.1606 | 0.8255 | 1.0000 |
| frozen_external_teacher_only risk | 14580 | 0.7467 | 0.3407 | 0.1704 | 0.8543 | 1.0000 |
| calibration-only risk | 14580 | 0.5933 | 0.5759 | 0.2687 | 0.7029 | 1.0000 |
| cropenc risk | 14580 | 0.5949 | 0.5720 | 0.2684 | 0.7128 | 1.0000 |
| legacysem risk | 14580 | 0.5898 | 0.5233 | 0.3223 | 0.7173 | 1.0000 |
| stage1 trace-only risk | 0 | n/a | n/a | n/a | n/a | n/a |

## Occlusion / Long-Gap Breakdown

See JSON for full per-method breakdown.
