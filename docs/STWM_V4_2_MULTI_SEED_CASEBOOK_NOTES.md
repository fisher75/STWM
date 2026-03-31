# STWM V4.2 Multi-Seed Shared Casebook Notes

Outputs:

- root: `outputs/visualizations/stwm_v4_2_multiseed_casebook`
- manifest: `outputs/visualizations/stwm_v4_2_multiseed_casebook/figure_manifest.json`

## Goal

Build one shared cross-seed casebook instead of large per-seed packs.

Groups:

1. `semantic_sensitive`
2. `identity_reconnect`
3. `query_grounding`

Selection constraints:

- min consistent seeds: `2`
- cases per group target: `8`

## Generated Counts

- semantic_sensitive: `7`
- identity_reconnect: `8`
- query_grounding: `8`

## Cross-Seed Delta Summary (selected cases)

- semantic_sensitive
  - consistency count range: `2..2`
  - mean avg_delta: `0.009179`
  - mean event_seed_count: `0.0`
- identity_reconnect
  - consistency count range: `2..3`
  - mean avg_delta: `0.006881`
  - mean event_seed_count: `0.0`
- query_grounding
  - consistency count range: `2..3`
  - mean avg_delta: `0.017748`
  - mean event_seed_count: `0.0`

Each selected case includes:

- representative seed
- representative ablation
- per-seed delta summary
- output artifact path

## What This Casebook Supports

1. There are cross-seed-consistent semantic-sensitive and query-grounding cases.
2. The package can be used as qualitative support for multi-seed tables.

## What This Casebook Cannot Support

1. It cannot provide reconnect hard evidence because event coverage is zero in this slice.
2. It cannot replace quantitative multi-seed statistical conclusions.
