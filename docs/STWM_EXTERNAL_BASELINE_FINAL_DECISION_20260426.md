# STWM External Baseline Final Decision 20260426

| baseline | cloned | import_ok | smoke_pass | full_eval |
|---|---:|---:|---:|---:|
| cutie | `True` | `True` | `False` | `False` |
| sam2 | `True` | `True` | `False` | `False` |
| cotracker | `True` | `True` | `False` | `False` |

- best_external_baseline: `None`
- stwm_improved_vs_best_external: `None`
- next_step_choice: `do_not_use_external_baselines`
- exact_blocking_reason: `External repos cloned and importable in the isolated stwm conda env, but no baseline can enter full eval because the current STWM reports do not expose raw frame/video paths plus observed prompts and future candidate masks/boxes needed for frozen VOS/tracking adaptation.`
