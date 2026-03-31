# STWM V4.2 State-Identifiability Figures

## Scope

Main-text figure pack for second contribution.

Artifacts:

- root: `outputs/visualizations/stwm_v4_2_state_identifiability_figures`
- manifest: `outputs/visualizations/stwm_v4_2_state_identifiability_figures/figure_manifest.json`

Seeds used:

- `42, 123, 456`

## Required Three Groups

1. `semantic_sensitive`
2. `instance_disambiguation`
3. `future_grounding`

Current selected counts:

- semantic_sensitive: `7`
- instance_disambiguation: `8`
- future_grounding: `6`

## Group Intent

1. semantic_sensitive
   - comparator: `wo_semantics_v4_2`
   - show cases where semantic state is needed for correct grounding.
2. instance_disambiguation
   - comparator: `wo_object_bias_v4_2`
   - show same-category/spatial/relation disambiguation gains from object-biased representation.
3. future_grounding
   - comparator: `wo_object_bias_v4_2`
   - show future-conditioned/reappearance-aware grounding cases.

## Notes For Main Text

1. `instance_disambiguation` already reaches target (`8`).
2. `semantic_sensitive` and `future_grounding` are below `8`; this reflects strict cross-seed filtering on available clips.
3. Use all selected cases in main/appendix split:
   - main text: strongest 6 (2 per group)
   - appendix: remaining selected cases.
