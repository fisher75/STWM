# MILESTONE AUDIT 2026-03-31

## Scope

- Audit root: `/home/chen034/workspace/stwm`
- Audit time: 2026-03-31
- Audit mode: read-only verification + artifact integrity checks

## What Is Confirmed Completed

1. Project status and download docs are updated to real state.
2. Storage report exists and matches expected capacity snapshot.
3. Week-1 mini split manifest exists and sample composition is correct:
   - total 70 samples
   - vspw 20 / vipseg 20 / burst 20 / visor 10
4. Real-data code path exists for:
   - dataset loader
   - trace adapter
   - semantic adapter
   - smoke test entry
   - minimal train entry
5. Real outputs exist for smoke/minimal-train/prototype-220m.
6. DEVA smoke test output exists and is complete:
   - 45 annotation masks
   - 45 visualizations
   - pred.json

## Results That Are Officially Passed

1. STWM real-data smoke path:
   - `outputs/smoke_tests/smoke_test_one_clip.json`
2. STWM minimal training step:
   - `outputs/training/minimal_train_step.json`
3. Week-2 prototype smoke (`prototype_220m`):
   - `outputs/training/prototype_220m_minimal_train_step.json`
   - parameter count around 227.9M
4. DEVA baseline smoke:
   - `outputs/smoke_tests/deva_vspw`

## Results That Are Fallback Only

1. YOLO-World currently has fallback success via ultralytics runtime path.
2. Official `third_party/YOLO-World` path is not yet stable in main environment.
3. Current fallback output is usable for temporary open-vocab baseline, but not equivalent to declaring official path closed.

## Current Technical Debt

1. Main `stwm` environment has prior manual patch history in site-packages (mmengine path needs explicit audit record and isolation policy).
2. Official YOLO-World dependency closure is not isolated yet.
3. Cutie and XMem smoke tests are still missing.
4. Week-2 ablation runs are not fully executed yet (full/wo_semantics/wo_trajectory/wo_identity_memory all required).

## Files Worth Immediate Commit Freeze

Core code and configs:
- `code/stwm/datasets/stwm_dataset.py`
- `code/stwm/modules/trace_adapter.py`
- `code/stwm/modules/semantic_adapter.py`
- `code/stwm/tools/smoke_test_one_clip.py`
- `code/stwm/trainers/train_stwm.py`
- `code/stwm/models/stwm_1b.py`
- `code/stwm/configs/model_presets.json`
- `scripts/run_week2_prototype_smoke.sh`

Core docs:
- `docs/STATUS.md`
- `docs/DOWNLOAD_STATUS.md`
- `docs/BASELINE_SMOKE_TESTS.md`
- `manifests/storage_report_20260331_004839.txt`

Core artifacts:
- `outputs/smoke_tests/smoke_test_one_clip.json`
- `outputs/training/minimal_train_step.json`
- `outputs/training/prototype_220m_minimal_train_step.json`
- `outputs/smoke_tests/deva_vspw/`

## Audit Verdict

- Project is past bootstrap and in experiment-ready state.
- Baseline chain is partially closed (SAM2 + DEVA + YOLO fallback).
- Highest priority now is environment boundary control and week-2 ablation execution, not further ad-hoc patching in main env.
