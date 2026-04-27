# STWM External Baseline Full Eval Bootstrap 20260426

```json
{
  "bootstrap_n": 2000,
  "comparisons": {
    "STWM_vs_cotracker": {
      "MRR": {
        "bootstrap_win_rate": 0.29562982005141386,
        "ci95_high": -0.023909147741381944,
        "ci95_low": -0.11322847025741782,
        "count": 389,
        "mean_delta": -0.06894886801900255,
        "zero_excluded": true
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.23783783783783785,
        "ci95_high": -0.022522522522522525,
        "ci95_low": -0.1445945945945946,
        "count": 370,
        "mean_delta": -0.0846846846846847,
        "zero_excluded": true
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.03213367609254499,
        "ci95_low": -0.1520994001713796,
        "count": 389,
        "mean_delta": -0.09254498714652958,
        "zero_excluded": true
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.02999143101970866,
        "ci95_low": -0.15167095115681234,
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
        "ci95_high": 0.23753280839895013,
        "ci95_low": 0.036745406824146974,
        "count": 127,
        "mean_delta": 0.13517060367454067,
        "zero_excluded": true
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.03341902313624679,
        "ci95_low": -0.1533847472150814,
        "count": 389,
        "mean_delta": -0.09254498714652956,
        "zero_excluded": true
      },
      "top1": {
        "bootstrap_win_rate": 0.23393316195372751,
        "ci95_high": -0.03213367609254499,
        "ci95_low": -0.1520994001713796,
        "count": 389,
        "mean_delta": -0.09254498714652956,
        "zero_excluded": true
      }
    },
    "STWM_vs_cutie": {
      "MRR": {
        "bootstrap_win_rate": 0.39588688946015427,
        "ci95_high": 0.07698811912319546,
        "ci95_low": -0.007231403867971499,
        "count": 389,
        "mean_delta": 0.0344556106084372,
        "zero_excluded": false
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.27837837837837837,
        "ci95_high": 0.07207207207207207,
        "ci95_low": -0.0427927927927928,
        "count": 370,
        "mean_delta": 0.015315315315315313,
        "zero_excluded": false
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.0754070265638389,
        "ci95_low": -0.03384747215081406,
        "count": 389,
        "mean_delta": 0.020565552699228787,
        "zero_excluded": false
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.0754070265638389,
        "ci95_low": -0.036846615252784924,
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
        "ci95_low": 0.06430446194225721,
        "count": 127,
        "mean_delta": 0.15879265091863518,
        "zero_excluded": true
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.0741216795201371,
        "ci95_low": -0.03341902313624679,
        "count": 389,
        "mean_delta": 0.02056555269922879,
        "zero_excluded": false
      },
      "top1": {
        "bootstrap_win_rate": 0.2827763496143959,
        "ci95_high": 0.07583547557840617,
        "ci95_low": -0.03213367609254499,
        "count": 389,
        "mean_delta": 0.02056555269922879,
        "zero_excluded": false
      }
    },
    "STWM_vs_sam2": {
      "MRR": {
        "bootstrap_win_rate": 0.2442159383033419,
        "ci95_high": -0.07323056350424632,
        "ci95_low": -0.15545647160419293,
        "count": 389,
        "mean_delta": -0.11543523797823184,
        "zero_excluded": true
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.1783783783783784,
        "ci95_high": -0.08513513513513514,
        "ci95_low": -0.1990990990990991,
        "count": 370,
        "mean_delta": -0.14414414414414414,
        "zero_excluded": true
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10411311053984576,
        "ci95_low": -0.21722365038560412,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10282776349614396,
        "ci95_low": -0.21936589545844046,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "long_gap_persistence_top1": {
        "bootstrap_win_rate": 0.23076923076923078,
        "ci95_high": 0.16666666666666666,
        "ci95_low": -0.14743589743589744,
        "count": 52,
        "mean_delta": 0.009615384615384616,
        "zero_excluded": false
      },
      "occlusion_reappearance_top1": {
        "bootstrap_win_rate": 0.25984251968503935,
        "ci95_high": 0.08136482939632546,
        "ci95_low": -0.12598425196850394,
        "count": 127,
        "mean_delta": -0.022309711286089242,
        "zero_excluded": false
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10796915167095116,
        "ci95_low": -0.2185089974293059,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.1036846615252785,
        "ci95_low": -0.21636675235646957,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      }
    },
    "STWM_vs_strongest_external_baseline": {
      "MRR": {
        "bootstrap_win_rate": 0.2442159383033419,
        "ci95_high": -0.07323056350424632,
        "ci95_low": -0.15545647160419293,
        "count": 389,
        "mean_delta": -0.11543523797823184,
        "zero_excluded": true
      },
      "OOD_hard_top1": {
        "bootstrap_win_rate": 0.1783783783783784,
        "ci95_high": -0.08513513513513514,
        "ci95_low": -0.1990990990990991,
        "count": 370,
        "mean_delta": -0.14414414414414414,
        "zero_excluded": true
      },
      "false_confuser_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10411311053984576,
        "ci95_low": -0.21722365038560412,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "false_reacquisition_rate": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10282776349614396,
        "ci95_low": -0.21936589545844046,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "long_gap_persistence_top1": {
        "bootstrap_win_rate": 0.23076923076923078,
        "ci95_high": 0.16666666666666666,
        "ci95_low": -0.14743589743589744,
        "count": 52,
        "mean_delta": 0.009615384615384616,
        "zero_excluded": false
      },
      "occlusion_reappearance_top1": {
        "bootstrap_win_rate": 0.25984251968503935,
        "ci95_high": 0.08136482939632546,
        "ci95_low": -0.12598425196850394,
        "count": 127,
        "mean_delta": -0.022309711286089242,
        "zero_excluded": false
      },
      "reacquisition_top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.10796915167095116,
        "ci95_low": -0.2185089974293059,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      },
      "top1": {
        "bootstrap_win_rate": 0.17737789203084833,
        "ci95_high": -0.1036846615252785,
        "ci95_low": -0.21636675235646957,
        "count": 389,
        "mean_delta": -0.16195372750642675,
        "zero_excluded": true
      }
    }
  },
  "created_at": "2026-04-27T16:09:55+0800",
  "positive_delta_policy": "positive means STWM better; false-rate metrics are negated before delta",
  "strongest_external_baseline": "sam2"
}
```
