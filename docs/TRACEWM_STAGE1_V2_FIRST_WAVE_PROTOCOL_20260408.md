# TRACEWM Stage1 v2 First-Wave Protocol (2026-04-08)

## 1) Blocking Point
The Stage1-v2 full-cache path is blocked by very large PointOdyssey anno.npz files that can stall P0 for a long time.
The first-wave protocol exists to remove this runtime bottleneck for initial scientific decision making.

## 2) Objective Shift
Current objective is changed from full-cache completion to fast first-wave scientific judgment under explicit, auditable partial-cache constraints.
No implicit fallback is allowed.

## 3) First-Wave Principles
1. PointOdyssey is first-wave train/val only.
2. PointOdyssey skipped scenes must be explicitly recorded in a manifest with fields:
   - scene_id
   - split
   - anno_path
   - anno_size_bytes
   - skip_reason
3. Kubric must run in exactly one declared first-wave mode: panning_raw_first_wave.
4. Contract and audit must explicitly carry first-wave declarations and pass/fail checks.
5. Session/log naming must remain fixed:
   - tmux session: tracewm_stage1_v2_20260408
   - run log: /home/chen034/workspace/stwm/logs/tracewm_stage1_v2_20260408.log

## 4) Prohibitions
1. No PointOdyssey test split in first-wave cache build.
2. No silent scene skipping.
3. No mixed Kubric modes in one first-wave run.
4. No synthetic or deterministic trajectory substitution for Stage1-v2 real-trace first-wave claims.

## 5) Required First-Wave Artifacts
1. /home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_20260408.json
2. /home/chen034/workspace/stwm/reports/stage1_v2_trace_cache_audit_20260408.json
3. /home/chen034/workspace/data/_manifests/stage1_v2_pointodyssey_skipped_20260408.json
4. /home/chen034/workspace/stwm/reports/stage1_v2_ablation_state_20260408.json
5. /home/chen034/workspace/stwm/reports/stage1_v2_ablation_backbone_20260408.json
6. /home/chen034/workspace/stwm/reports/stage1_v2_ablation_losses_20260408.json
7. /home/chen034/workspace/stwm/reports/stage1_v2_final_comparison_20260408.json
8. /home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_RESULTS_20260408.md
