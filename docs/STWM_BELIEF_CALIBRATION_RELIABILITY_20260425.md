# STWM Belief Calibration Reliability 20260425

Reliability uses `rank_confidence_proxy = MRR = 1 / target_rank`; the source reports do not include calibrated probabilities or raw candidate score maps.

## Summary

| series | count | ECE | Brier proxy | mean confidence | mean accuracy |
|---|---:|---:|---:|---:|---:|
| teacher_all | 3480 | 0.1559 | 0.0608 | 0.6403 | 0.4845 |
| belief_all | 3480 | 0.1567 | 0.0622 | 0.6573 | 0.5006 |
| teacher_id | 894 | 0.1694 | 0.0638 | 0.5922 | 0.4228 |
| belief_id | 894 | 0.1646 | 0.0637 | 0.6277 | 0.4631 |
| teacher_ood | 2586 | 0.1512 | 0.0598 | 0.6570 | 0.5058 |
| belief_ood | 2586 | 0.1540 | 0.0616 | 0.6676 | 0.5135 |

calibration_improved = `False`
