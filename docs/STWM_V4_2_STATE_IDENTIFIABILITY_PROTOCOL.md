# STWM V4.2 State-Identifiability Protocol

## Scope

This protocol formalizes the second contribution target:

- semantic trajectory state supports instance-grounded future-state identification.

Boundaries preserved:

- no new model module
- no loss composition change
- no 1B scaling

## Taxonomy

Source report:

- `reports/stwm_v4_2_state_identifiability_protocol_v1.json`

Query taxonomy:

1. `same_category_distractor`
2. `spatial_disambiguation`
3. `relation_conditioned_query`
4. `future_conditioned_reappearance_aware`

## Coverage And Sample Counts

- selected_count: `18`
- coverage_insufficient: `False`

Per-type count:

- same_category_distractor: `18`
- spatial_disambiguation: `18`
- relation_conditioned_query: `17`
- future_conditioned_reappearance_aware: `11`

Per-type ratio:

- same_category_distractor: `1.0000`
- spatial_disambiguation: `1.0000`
- relation_conditioned_query: `0.9444`
- future_conditioned_reappearance_aware: `0.6111`

## Difficulty Definitions

Difficulty is normalized to `[0, 1]` and bucketed to `easy/medium/hard`.

- same_category_distractor: `0.55*ambiguity + 0.25*crowd + 0.20*area_cv`
- spatial_disambiguation: `0.50*spatial_ambiguity + 0.30*motion + 0.20*crowd`
- relation_conditioned_query: `0.40*crowd + 0.35*ambiguity + 0.25*motion`
- future_conditioned_reappearance_aware: `0.35*missing_span + 0.30*reappearance + 0.20*reconnect + 0.15*motion`

## Difficulty Statistics

- same_category_distractor
  - mean/std: `0.5641 / 0.1877`
  - bucket: easy `2`, medium `11`, hard `5`
- spatial_disambiguation
  - mean/std: `0.3543 / 0.2102`
  - bucket: easy `10`, medium `7`, hard `1`
- relation_conditioned_query
  - mean/std: `0.4755 / 0.1922`
  - bucket: easy `7`, medium `6`, hard `4`
- future_conditioned_reappearance_aware
  - mean/std: `0.4307 / 0.1476`
  - bucket: easy `3`, medium `8`, hard `0`

## Artifacts

- manifest: `manifests/minisplits/stwm_v4_2_state_identifiability_v1.json`
- clip ids: `manifests/minisplits/stwm_v4_2_state_identifiability_clip_ids_v1.json`
- report: `reports/stwm_v4_2_state_identifiability_protocol_v1.json`
