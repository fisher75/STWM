# STWM External Baseline Full Eval Bootstrap 20260426

```json
{
  "bootstrap_n": 2000,
  "comparisons": {
    "STWM_vs_cotracker": {
      "MRR": {
        "bootstrap_win_rate": 0.29562982005141386,
        "ci95_high": -0.02760026670776575,
        "ci95_low": -0.1111915861766334,
        "count": 389,
        "mean_delta": -0.06894886801900255,
        "zero_excluded": true
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.23783783783783785,
        "ci95_high": -0.022972972972972978,
        "ci95_low": -0.15045045045045047,
        "count": 370,
        "mean_delta": -0.0846846846846847,
        "zero_excluded": true
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.03299057412167952,
        "ci95_low": -0.15295629820051415,
        "count": 389,
        "mean_delta": -0.09254498714652958,
        "zero_excluded": true
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.02999143101970866,
        "ci95_low": -0.1525278491859469,
        "count": 389,
        "mean_delta": -0.09254498714652958,
        "zero_excluded": true
      },
      "long_gap_persistence_top1": {
        "bootstrap_win_rate": 0.36538461538461536,
        "ci95_high": 0.3685897435897436,
        "ci95_low": 0.07371794871794872,
        "count": 52,
        "mean_delta": 0.22115384615384615,
        "zero_excluded": true
      },
      "occlusion_reappearance_top1": {
        "bootstrap_win_rate": 0.3543307086614173,
        "ci95_high": 0.22703412073490814,
        "ci95_low": 0.034120734908136476,
        "count": 127,
        "mean_delta": 0.13517060367454067,
        "zero_excluded": true
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.031705227077977724,
        "ci95_low": -0.15252784918594686,
        "count": 389,
        "mean_delta": -0.09254498714652956,
        "zero_excluded": true
      },
      "top1": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.03470437017994859,
        "ci95_low": -0.15295629820051415,
        "count": 389,
        "mean_delta": -0.09254498714652956,
        "zero_excluded": true
      }
    },
    "STWM_vs_cutie": {
      "MRR": {
        "bootstrap_win_rate": 0.39588688946015427,
        "ci95_high": 0.0758152353334643,
        "ci95_low": -0.00963447048694198,
        "count": 389,
        "mean_delta": 0.0344556106084372,
        "zero_excluded": false
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.27837837837837837,
        "ci95_high": 0.07072072072072072,
        "ci95_low": -0.041891891891891894,
        "count": 370,
        "mean_delta": 0.015315315315315313,
        "zero_excluded": false
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.07626392459297343,
        "ci95_low": -0.03513281919451586,
        "count": 389,
        "mean_delta": 0.020565552699228787,
        "zero_excluded": false
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.07669237360754069,
        "ci95_low": -0.03256212510711226,
        "count": 389,
        "mean_delta": 0.020565552699228787,
        "zero_excluded": false
      },
      "long_gap_persistence_top1": {
        "bootstrap_win_rate": 0.28846153846153844,
        "ci95_high": 0.2467948717948718,
        "ci95_low": -0.0641025641025641,
        "count": 52,
        "mean_delta": 0.08653846153846154,
        "zero_excluded": false
      },
      "occlusion_reappearance_top1": {
        "bootstrap_win_rate": 0.3700787401574803,
        "ci95_high": 0.2532808398950131,
        "ci95_low": 0.06692913385826771,
        "count": 127,
        "mean_delta": 0.15879265091863518,
        "zero_excluded": true
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.07369323050556983,
        "ci95_low": -0.031276778063410456,
        "count": 389,
        "mean_delta": 0.02056555269922879,
        "zero_excluded": false
      },
      "top1": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.07455012853470437,
        "ci95_low": -0.03470437017994859,
        "count": 389,
        "mean_delta": 0.02056555269922879,
        "zero_excluded": false
      }
    },
    "STWM_vs_sam2": {
      "MRR": {
        "bootstrap_win_rate": 0.2442159383033419,
        "ci95_high": -0.07603336886941059,
        "ci95_low": -0.15676585381675603,
        "count": 389,
        "mean_delta": -0.11543523797823184,
        "zero_excluded": true
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.1783783783783784,
        "ci95_high": -0.08828828828828829,
        "ci95_low": -0.20405405405405405,
        "count": 370,
        "mean_delta": -0.14414414414414414,
        "zero_excluded": true
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10539845758354756,
        "ci95_low": -0.2189374464438732,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10154241645244216,
        "ci95_low": -0.21979434447300772,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "long_gap_persistence_top1": {
        "bootstrap_win_rate": 0.23076923076923078,
        "ci95_high": 0.16666666666666666,
        "ci95_low": -0.15064102564102563,
        "count": 52,
        "mean_delta": 0.009615384615384616,
        "zero_excluded": false
      },
      "occlusion_reappearance_top1": {
        "bootstrap_win_rate": 0.25984251968503935,
        "ci95_high": 0.07874015748031496,
        "ci95_low": -0.12467191601049869,
        "count": 127,
        "mean_delta": -0.022309711286089242,
        "zero_excluded": false
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10411311053984576,
        "ci95_low": -0.2215081405312768,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.1049700085689803,
        "ci95_low": -0.21979434447300772,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      }
    },
    "STWM_vs_strongest_external_baseline": {
      "MRR": {
        "bootstrap_win_rate": 0.2442159383033419,
        "ci95_high": -0.07603336886941059,
        "ci95_low": -0.15676585381675603,
        "count": 389,
        "mean_delta": -0.11543523797823184,
        "zero_excluded": true
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.1783783783783784,
        "ci95_high": -0.08828828828828829,
        "ci95_low": -0.20405405405405405,
        "count": 370,
        "mean_delta": -0.14414414414414414,
        "zero_excluded": true
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10539845758354756,
        "ci95_low": -0.2189374464438732,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10154241645244216,
        "ci95_low": -0.21979434447300772,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "long_gap_persistence_top1": {
        "bootstrap_win_rate": 0.23076923076923078,
        "ci95_high": 0.16666666666666666,
        "ci95_low": -0.15064102564102563,
        "count": 52,
        "mean_delta": 0.009615384615384616,
        "zero_excluded": false
      },
      "occlusion_reappearance_top1": {
        "bootstrap_win_rate": 0.25984251968503935,
        "ci95_high": 0.07874015748031496,
        "ci95_low": -0.12467191601049869,
        "count": 127,
        "mean_delta": -0.022309711286089242,
        "zero_excluded": false
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10411311053984576,
        "ci95_low": -0.2215081405312768,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.1049700085689803,
        "ci95_low": -0.21979434447300772,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      }
    }
  },
  "created_at": "2026-04-27T17:03:56+0800",
  "positive_delta_policy": "positive means STWM better; false-rate metrics are negated before delta",
  "strongest_external_baseline": "sam2"
}
```
