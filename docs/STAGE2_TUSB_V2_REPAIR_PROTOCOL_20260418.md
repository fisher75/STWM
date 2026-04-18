# Stage2 TUSB-V2 Repair Protocol 20260418

- generated_at_utc: 2026-04-17T19:03:48.849793+00:00
- stage1_backbone_status: frozen and untouchable for this round
- stage1_training_allowed: false
- stage1_unfreeze_allowed: false
- current_stage2_interpretation: calibration-only remains the current reasonable mainline, but it is semantic calibration/readout alignment rather than semanticized trace state
- current_tusb_lite_status: landed but not freeze-ready
- key_blockers: eval blind risk; single-entity training body; weak real instance signal usage; active-unit collapse; old cache incompatibility
- repair_target: multi-entity TUSB-V2 with real instance-aware path, anti-collapse unitization, and stronger frozen semantic prior
- forbidden: Stage1 changes; persistence revival; protocol v4; qualitative expansion; video/render head; codec/VAE; full architecture search
