# STWM Real Visibility/Reappearance V1 Export/Eval Audit 20260427

- export_eval_audit_passed: `True`
- old_association_report_used: `False`
- visibility_metric_status: `calibrated_visibility_available`

## Metrics Added
- future_visibility_accuracy
- future_visibility_AUROC
- future_visibility_AP
- future_reappearance_accuracy
- future_reappearance_AUROC
- future_reappearance_AP
- both_class_visibility_available
- both_class_reappearance_available

Calibrated visibility target is now slot-aligned, but model reappearance quality remains untrained/weak until visibility-reappearance training is run.
