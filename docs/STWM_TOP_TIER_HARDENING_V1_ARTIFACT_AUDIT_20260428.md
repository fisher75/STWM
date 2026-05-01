# STWM Top-Tier Hardening V1 Artifact Audit

## Summary

- expected_checkpoint_count: `10`
- existing_checkpoint_count: `10`
- expected_log_count: `10`
- existing_log_count: `10`
- zero_byte_training_log_count: `10`
- report_only_completion_detected: `False`
- artifact_audit_passed: `True`

## Notes

- Per-run checkpoint artifacts exist for the full 10/10 matrix.
- Canonical per-run tmux stdout logs are present but zero-byte; driver/materialization logs remain available.
