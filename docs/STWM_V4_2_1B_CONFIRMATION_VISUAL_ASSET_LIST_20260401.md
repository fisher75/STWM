# STWM V4.2 1B Confirmation Visualization Asset List (2026-04-01)

## Status: Lightweight Staged Visual Bundle (Invalid For Main Claim)

This asset list is tied to lightweight staged 1B runs and must not be cited as main paper evidence.

- Allowed: pipeline sanity, visual wiring checks.
- Not allowed: final paper-scale qualitative evidence.
- Replacement target: real 1B visual outputs under `outputs/visualizations/stwm_v4_2_real_1b_*`.

## 1) Build Status

- Pipeline command: `scripts/build_stwm_v4_2_1b_visualization.sh` with staged roots
  - base root: `outputs/training/stwm_v4_2_1b_confirmation_staged/base`
  - state root: `outputs/training/stwm_v4_2_1b_confirmation_staged/state`
- Final status: `SUCCESS`
  - demo frames: `38`
  - demo video: `created=true`

## 2) Primary Deliverables

1. Base casebook manifest
   - `outputs/visualizations/stwm_v4_2_1b_confirmation_multiseed_casebook/figure_manifest.json`
2. State-identifiability casebook manifest
   - `outputs/visualizations/stwm_v4_2_1b_confirmation_state_identifiability_figures/figure_manifest.json`
3. Demo manifest
   - `outputs/visualizations/stwm_v4_2_1b_confirmation_demo/demo_manifest.json`
4. Demo video (mp4)
   - `outputs/visualizations/stwm_v4_2_1b_confirmation_demo/stwm_v4_2_1b_demo.mp4`
5. Storyboard frames directory
   - `outputs/visualizations/stwm_v4_2_1b_confirmation_demo/storyboard_frames/`

## 3) Artifact Counts (Verified)

- Base casebook (`outputs/visualizations/stwm_v4_2_1b_confirmation_multiseed_casebook/figure_manifest.json`)
  - semantic_sensitive: `8`
  - identity_reconnect: `0`
  - query_grounding: `8`
- State casebook (`outputs/visualizations/stwm_v4_2_1b_confirmation_state_identifiability_figures/figure_manifest.json`)
  - semantic_sensitive: `8`
  - instance_disambiguation: `8`
  - future_grounding: `6`
- Demo package (`outputs/visualizations/stwm_v4_2_1b_confirmation_demo/demo_manifest.json`)
  - total_unique_images: `38`
  - rendered_frames: `38`
  - video.created: `true`
  - video.fps: `2`

## 4) File Size Snapshot

- `outputs/visualizations/stwm_v4_2_1b_confirmation_demo/stwm_v4_2_1b_demo.mp4`: `2.7M`

## 5) Notes For Usage

- The base casebook currently has no `identity_reconnect` entries because this staged 1B confirmation uses `wo_object_bias_v4_2` as the second ablation branch, while the base multi-seed visual selector expects identity-specific contrast for that group.
- Semantic-sensitive and query-grounding groups are populated and ready for report/main/supplement use.
- State-identifiability three groups are populated and directly usable.
