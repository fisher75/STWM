# STWM Counterfactual Planning-Lite Risk 20260425

| variant | risk_AUC | false_safe_rate | false_alarm_rate |
|---|---:|---:|---:|
| original | 0.7595 | 0.3212 | 0.1606 |
| no_trace_prior | 0.7801 | 0.2961 | 0.1480 |
| object_removed | 0.5000 | 1.0000 | 0.0000 |
| object_shifted | 0.5930 | 0.0000 | 1.0000 |

- removing_dangerous_object_lowers_risk = `True`
- shifting_object_near_path_raises_risk = `True`
- counterfactual_risk_direction_consistent = `True`
