# STWM Storage Cleanup Summary 20260414

- before_total_size: 674.48 GB
- after_total_size: 674.48 GB
- saved_inside_repo: 0.00 B
- moved_to_quarantine_size: 0.00 B
- permanently_deleted_size: 76.64 KB
- checkpoint_pruned_size: 0.00 B
- cache_deleted_size: 76.64 KB
- logs_compressed_size: 0.00 B
- review_required_size: 343.79 GB
- quarantine_path: `/raid/chen034/workspace/stwm_cleanup_quarantine_20260414`
- restore_script_path: `/raid/chen034/workspace/stwm/scripts/restore_stwm_cleanup_20260414.sh`
- active_run_detected: True
- protected_path_touched: False

## Notes

- Active STWM runs were detected, so checkpoint pruning, log compression/move, outputs cleanup, and raw compression were skipped.
- Permanent deletion was limited to low-risk caches and old temp files outside protected top-level paths.
- Protected top-level paths were not deleted or moved.
