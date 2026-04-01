# STWM V4.2 1B Visualization And Video Pipeline

## Goal

Produce reusable visualization artifacts for the 1B confirmation round:

1. base multi-seed casebook
2. state-identifiability casebook
3. stitched storyboard frames
4. demo manifest and mp4 (if ffmpeg is available)

## Entry Script

```bash
cd /home/chen034/workspace/stwm
bash scripts/build_stwm_v4_2_1b_visualization.sh
```

## Pipeline Stages

1. `code/stwm/evaluators/build_stwm_v4_2_multiseed_casebook.py`
2. `code/stwm/evaluators/build_stwm_v4_2_state_identifiability_figures.py`
3. `code/stwm/tools/package_stwm_v4_2_video_demo.py`

## Main Outputs

1. `outputs/visualizations/stwm_v4_2_1b_multiseed_casebook/figure_manifest.json`
2. `outputs/visualizations/stwm_v4_2_1b_state_identifiability_figures/figure_manifest.json`
3. `outputs/visualizations/stwm_v4_2_1b_demo/demo_manifest.json`
4. `outputs/visualizations/stwm_v4_2_1b_demo/stwm_v4_2_1b_demo.mp4`
5. `outputs/visualizations/stwm_v4_2_1b_demo/storyboard_frames/`

## Tunable Knobs

1. `STWM_V4_2_1B_VIS_SEEDS` (default `42,123,456`)
2. `STWM_V4_2_1B_DEMO_FPS` (default `2`)
3. `STWM_V4_2_1B_DEMO_MAX_FRAMES` (default `120`)
4. `STWM_V4_2_1B_BASE_VIS_OUT` / `STWM_V4_2_1B_STATE_VIS_OUT` / `STWM_V4_2_1B_DEMO_OUT`

## Notes

1. The packager auto-scans figure manifests first, then falls back to recursive image discovery.
2. If `ffmpeg` is unavailable, the script still writes `demo_manifest.json` and storyboard frames.
3. This pipeline is analysis-only and does not alter model or protocol definitions.
