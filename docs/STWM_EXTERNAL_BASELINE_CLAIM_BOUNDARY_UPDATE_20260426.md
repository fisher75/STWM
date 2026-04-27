# STWM External Baseline Claim Boundary Update 20260426

## Allowed claims

- STWM outperforms Cutie overall on this materialized external-baseline protocol, although the overall bootstrap CI is not strongly separated.
- STWM outperforms CoTracker on occlusion_reappearance and long_gap_persistence continuity-heavy subsets with zero-excluded bootstrap intervals.
- STWM outperforms internal baselines and legacysem on the official setting.
- STWM provides complementary trace-belief advantage over strong VOS/point trackers in continuity-heavy future identity association cases.

## Moderate claims

- STWM is competitive with strong external baselines in selected hard continuity cases, but not overall.
- STWM offers utility beyond pure VOS/point tracking under reacquisition/risk/counterfactual tasks, subject to the already documented claim boundary.
- STWM can be positioned as a structured belief mechanism that complements SAM2/CoTracker rather than replacing them as universal trackers.

## Forbidden claims

- STWM beats SAM2 overall.
- STWM beats CoTracker overall.
- STWM is universal external-baseline SOTA.
- Cutie/SAM2/CoTracker are weak baselines.
- External baseline failure proves STWM superiority.
- Smoke metrics are full eval metrics.
- 389-item diagnostic set is a full video benchmark.
