# STWM V4.2 Real 1B Visualization Plan

## Purpose

Regenerate 1B visualization assets only from real confirmation outputs.

Input root (required):

- `outputs/training/stwm_v4_2_1b_real_confirmation/`

Lightweight staged visualization artifacts are not used as main evidence.

## Required Figure Targets

1. semantic-sensitive main figure set
2. instance-disambiguation main figure set
3. future-grounding main figure set
4. side-by-side demo video with readable GT/full/ablation comparison

## Quality Gates (Must Pass)

For selected cases used in main narrative:

1. GT trajectory and predicted trajectory are both visible.
2. Query target marker is visible and not occluded by text.
3. Full and ablation panels are clearly labeled.
4. Temporal continuity is preserved (no random frame jumps inside one case).
5. Each selected case has an interpretable failure/success story in one screen.

## Generation Workflow

1. Ensure real 1B summaries are finalized:
   - `outputs/training/stwm_v4_2_1b_real_confirmation/base/comparison_multiseed.json`
   - `outputs/training/stwm_v4_2_1b_real_confirmation/state/comparison_state_identifiability.json`
2. Build casebooks and demo package from real roots:

```bash
bash scripts/build_stwm_v4_2_real_1b_visualization.sh \
  outputs/training/stwm_v4_2_1b_real_confirmation
```

3. Verify output manifests are non-empty and readable.

## Output Paths

- semantic-sensitive (base casebook):
  - `outputs/visualizations/stwm_v4_2_real_1b_multiseed_casebook/semantic_sensitive_cases/`
- instance-disambiguation:
  - `outputs/visualizations/stwm_v4_2_real_1b_state_identifiability_figures/instance_disambiguation_cases/`
- future-grounding:
  - `outputs/visualizations/stwm_v4_2_real_1b_state_identifiability_figures/future_grounding_cases/`
- demo package:
  - `outputs/visualizations/stwm_v4_2_real_1b_demo/demo_manifest.json`
  - `outputs/visualizations/stwm_v4_2_real_1b_demo/stwm_v4_2_1b_demo.mp4`

## Acceptance Checklist For Paper Use

- `comparison_multiseed.json` and `comparison_state_identifiability.json` are generated from real root.
- All three visual groups are non-empty.
- Demo manifest references only real-root artifacts.
- Video is readable at normal playback speed and labels are visible.
