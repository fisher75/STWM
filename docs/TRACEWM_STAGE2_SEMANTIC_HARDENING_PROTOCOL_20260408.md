# TRACEWM Stage2 Semantic Hardening Protocol (2026-04-08)

## 1. Frozen Facts

1. Stage2 bootstrap and Stage2 small-train have completed.
2. Frozen boundary has been verified as correct in completed rounds.
3. The current bottleneck is not Stage1.
4. The current bottleneck is Stage2 semantic source quality.

## 2. Why Hardening Is Needed

Current Stage2 semantic branch still relies mainly on hand-crafted semantic statistics:
1. mean_rgb
2. std_rgb
3. bbox ratio
4. fg_ratio

This is sufficient for bootstrap/small-train bring-up but not sufficient as paper-mainline semantic branch.

## 3. Scope Of This Round

This round upgrades Stage2 semantic source from hand-crafted statistics to crop-based visual encoding.

Current trainer status note:
1. TAP-Vid style eval is still not connected in Stage2 trainer.
2. TAPVid-3D limited eval is still not connected in Stage2 trainer.
3. Therefore this round prioritizes semantic source hardening only and does not expand the evaluator system.

## 4. Locked Constraints

This round does not change:
1. Stage1 frozen backbone
2. Stage2 data binding (core VSPW + VIPSeg, BURST optional)
3. Stage2 overall task definition
4. WAN / MotionCrafter
5. TAO / VISOR mainline

## 5. Semantic Source Mainline Rule

Mainline semantic source must be:
1. current_mainline_semantic_source = crop_visual_encoder

Legacy route may remain only as non-mainline compatibility/ablation:
1. legacy_semantic_source = hand_crafted_stats

## 6. Freeze/Trainable Boundary (Unchanged)

Frozen:
1. Stage1 220m backbone
2. Stage1 tokenizer / core rollout backbone

Trainable:
1. semantic encoder / crop encoder
2. semantic fusion / adapter
3. optional lightweight readout head

Mandatory checks:
1. frozen/trainable parameter counts are recorded
2. boundary_ok is reported
3. no hidden Stage1 unfreeze is allowed

## 7. Allowed Runs (Exactly Two)

1. stage2_smalltrain_cropenc_core
   - datasets: VSPW + VIPSeg
2. stage2_smalltrain_cropenc_core_plus_burst
   - datasets: VSPW + VIPSeg + BURST

Forbidden:
1. TAO / VISOR mainline
2. new backbone family
3. extra architecture sweep
4. full Stage2 longtrain
5. new dataset onboarding

## 8. Required Outputs

1. reports/stage2_semantic_hardening_core_20260408.json
2. reports/stage2_semantic_hardening_core_plus_burst_20260408.json
3. reports/stage2_semantic_hardening_comparison_20260408.json
4. docs/STAGE2_SEMANTIC_HARDENING_RESULTS_20260408.md

## 9. Fixed Runtime Envelope

1. single-GPU recommended runtime + selector/lease
2. fixed tmux session: tracewm_stage2_semantic_hardening_20260408
3. fixed log: /home/chen034/workspace/stwm/logs/tracewm_stage2_semantic_hardening_20260408.log
