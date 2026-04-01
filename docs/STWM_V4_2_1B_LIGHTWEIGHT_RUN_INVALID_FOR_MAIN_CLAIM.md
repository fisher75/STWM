# STWM V4.2 Lightweight 1B Run Is Invalid For Main Claim

## Status

The current staged 1B run is explicitly downgraded to pipeline sanity only.

- Lightweight run root: `outputs/training/stwm_v4_2_1b_confirmation_staged/`
- Smoke run root: `outputs/training/stwm_v4_2_1b_smoke/`
- Queue evidence: `outputs/queue/stwm_1b/queue_events.log`

## Why It Is Invalid As Main Scale-Up Evidence

1. Smoke-style tiny budget was used in this round (`steps=8`, tiny `sample_limit`).
2. Staged confirmation used incremental execution with `SKIP_EXISTING=1`.
3. Staged phases were configured as `full-only` for key segments.
4. State-identifiability stage was eval-oriented and not a full new training budget.
5. Visualization bundle was generated from the same lightweight staged artifacts.

## Allowed / Disallowed Usage

- Allowed:
  - queue and pipeline sanity checks
  - script wiring and artifact path validation
  - rough runtime profiling
- Disallowed:
  - main paper scale-up claim support
  - headline 220M vs 1B conclusion support
  - 3B go/no-go justification

## Affected Artifacts (Do Not Cite As Main Evidence)

- `docs/STWM_V4_2_1B_CONFIRMATION_PAPER_BRIEF_20260401.md`
- `docs/STWM_V4_2_1B_CONFIRMATION_VISUAL_ASSET_LIST_20260401.md`
- `docs/STWM_V4_2_3B_GO_NO_GO_CONFIRMATION_STAGED.md`
- `reports/stwm_v4_2_220m_vs_1b_confirmation_staged.json`

## Replacement Path

Main-claim-ready 1B evidence must come from the real confirmation flow:

- Training root: `outputs/training/stwm_v4_2_1b_real_confirmation/`
- Budget policy: `docs/STWM_V4_2_REAL_1B_BUDGET_PLAN.md`
- Visualization policy: `docs/STWM_V4_2_REAL_1B_VIS_PLAN.md`
