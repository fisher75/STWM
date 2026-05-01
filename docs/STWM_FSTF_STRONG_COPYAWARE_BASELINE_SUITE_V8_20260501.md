# STWM FSTF Strong Copy-Aware Baseline Suite V8

- Baseline suite completed: `True`
- New checkpoints: `25`
- New eval summaries: `25`
- Strongest copy-aware baseline: `copy_residual_mlp`
- Any learned copy-aware baseline beats copy: `True`
- Proceed to scaling allowed: `True`
- Next step: `run_scaling_laws`

## Paired Bootstrap

- STWM minus strongest changed top5: `{'item_count': 610, 'mean_delta': 0.06527920483565722, 'ci95': [0.04726071874626347, 0.0831896554397755], 'zero_excluded': True, 'bootstrap_win_rate': 1.0}`
- STWM minus strongest overall top5: `{'item_count': 647, 'mean_delta': 0.04174363051283488, 'ci95': [0.03289333287125577, 0.050528337183470336], 'zero_excluded': True, 'bootstrap_win_rate': 1.0}`
- STWM minus strongest stable drop: `{'item_count': 632, 'mean_delta': -0.03839903106983704, 'ci95': [-0.047379059429410134, -0.03001479049083553], 'zero_excluded': True, 'bootstrap_win_rate': 0.0}`

Visibility/reappearance remain `metric_invalid_or_untrained` and are not used as positive evidence.
