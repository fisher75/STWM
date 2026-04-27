# STWM External Baseline Result Interpretation 20260426

## Overall ranking

1. SAM2 is strongest external overall: top1 `0.671`, MRR `0.775`.
2. CoTracker is second: top1 `0.602`, MRR `0.729`.
3. STWM is below SAM2/CoTracker overall but above Cutie/internal baselines: top1 `0.509`.

## Hard subsets

- STWM vs CoTracker occlusion: `0.552` vs `0.417`, zero_excluded `True`.
- STWM vs CoTracker long-gap: `0.548` vs `0.327`, zero_excluded `True`.
- STWM vs SAM2 long-gap: `0.548` vs `0.538`, not significant.
- STWM weakness: SAM2 is stronger on crossing `0.689` vs STWM `0.472`, and OOD `0.662` vs STWM `0.518`.

## Paper implication

- Do not claim external overall SOTA.
- Claim continuity-heavy trace-belief advantage and complementary utility.

```json
{
  "created_at": "2026-04-27T08:30:02.409893+00:00",
  "false_confuser_interpretation": {
    "cotracker": 0.39845758354755784,
    "cutie": 0.5115681233933161,
    "interpretation": "SAM2 and CoTracker have lower overall false-confuser rates than STWM; STWM only improves over Cutie overall and over CoTracker on selected continuity subsets.",
    "sam2": 0.32904884318766064,
    "stwm": 0.4910025706940874
  },
  "hard_subset_interpretation": {
    "STWM_vs_CoTracker_long_gap": {
      "bootstrap_mean_delta": 0.22115384615384615,
      "cotracker": 0.3269230769230769,
      "interpretation": "STWM has significant long-gap advantage over CoTracker.",
      "stwm": 0.5480769230769231,
      "zero_excluded": true
    },
    "STWM_vs_CoTracker_occlusion": {
      "bootstrap_mean_delta": 0.13517060367454067,
      "cotracker": 0.41732283464566927,
      "interpretation": "STWM has significant continuity-heavy occlusion advantage over CoTracker.",
      "stwm": 0.5524934383202099,
      "zero_excluded": true
    },
    "STWM_vs_SAM2_long_gap": {
      "bootstrap_mean_delta": 0.009615384615384616,
      "interpretation": "STWM is numerically slightly above SAM2 on long-gap but not significant.",
      "sam2": 0.5384615384615384,
      "stwm": 0.5480769230769231,
      "zero_excluded": false
    },
    "STWM_weakness_OOD_vs_SAM2": {
      "bootstrap_mean_delta": -0.14414414414414414,
      "interpretation": "SAM2 is significantly stronger on OOD_hard.",
      "sam2": 0.6621621621621622,
      "stwm": 0.5180180180180181,
      "zero_excluded": true
    },
    "STWM_weakness_crossing_vs_SAM2": {
      "interpretation": "SAM2 is much stronger on crossing/ambiguity.",
      "sam2": 0.6887417218543046,
      "stwm": 0.4718543046357616
    }
  },
  "overall_ranking": [
    {
      "MRR": 0.7754042623766482,
      "interpretation": "strongest external overall",
      "method": "SAM2",
      "rank": 1,
      "top1": 0.6709511568123393
    },
    {
      "MRR": 0.7289178924174189,
      "interpretation": "second strongest external overall",
      "method": "CoTracker",
      "rank": 2,
      "top1": 0.6015424164524421
    },
    {
      "MRR": 0.6599690243984163,
      "interpretation": "below SAM2/CoTracker overall, above Cutie and internal baselines",
      "method": "STWM trace_belief_assoc",
      "rank": 3,
      "top1": 0.5089974293059126
    },
    {
      "MRR": 0.6255134137899792,
      "interpretation": "weaker than STWM overall on this protocol but not dramatically",
      "method": "Cutie",
      "rank": 4,
      "top1": 0.4884318766066838
    }
  ],
  "paper_writing_implication": [
    "Do not claim STWM is external overall SOTA.",
    "Do not hide that SAM2 and CoTracker are stronger overall.",
    "Claim STWM continuity-heavy trace-belief advantage, especially against CoTracker on occlusion/long-gap.",
    "Frame external baselines as strong comparison that clarifies the boundary of STWM, not as defeated weak baselines."
  ]
}
```
