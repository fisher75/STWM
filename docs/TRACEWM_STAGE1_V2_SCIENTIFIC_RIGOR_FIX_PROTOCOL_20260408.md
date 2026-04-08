# TRACEWM Stage1-v2 Scientific Rigor Fix Protocol (2026-04-08)

## Round Goal
This round upgrades existing Stage1-v2 scientific revalidation from "pipeline completed" to "defendable scientific validation" quality.

## Current State Acknowledgement
The scientific revalidation pipeline already exists, but its current validation standard is still too weak for paper-level defense.

## Frozen Core Problems
1. TAP-Vid and TAPVid-3D evaluation are still hardcoded as unavailable in the Stage1-v2 revalidation reports.
2. `whether_v2_is_scientifically_validated` currently uses an overly weak criterion.
3. `final_mainline_model` is currently concatenated from three independently selected winners, which may not correspond to one truly trained configuration.
4. The current final mainline is still `debug_small`, not a 220M-class target backbone.

## Non-Goals In This Round
1. Do not modify P0 trace cache.
2. Do not modify performance tooling.
3. Do not enter Stage2.
4. Do not touch WAN or MotionCrafter VAE.
5. Do not expand dataset scope.

## Required Rigor Upgrades
1. Connect TAP-Vid and TAPVid-3D limited evaluation when interfaces and data are available.
2. If unavailable, report explicit reason categories:
   - `available_and_run`
   - `not_implemented_yet`
   - `data_not_ready`
3. Tighten validation criteria with explicit winner margins and real replay evidence.
4. Replace stitched mainline with one real replayed mainline configuration.
5. Explicitly report small-vs-220M competitiveness and promotion decision.

## Validation Policy
1. Primary: `free_rollout_endpoint_l2`
2. Secondary: `free_rollout_coord_mean_l2`
3. Tertiary: TAP-Vid / TAPVid-3D limited endpoint metric when available
4. `total_loss` is reference-only and must not be the primary selector.

## Mandatory Final Fields
Final comparison must include:
1. `validation_status`
2. `validation_gaps`
3. `why_not_fully_validated` (if not fully validated)
4. `best_small_model`
5. `best_220m_model`
6. `should_promote_220m_now`
