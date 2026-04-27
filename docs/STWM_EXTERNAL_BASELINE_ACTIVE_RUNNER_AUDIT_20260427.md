# STWM External Baseline Active Runner Audit 20260427

## Active Path
- Full eval uses `CutieSmokeRunner`, `SAM2SmokeRunner`, and `CoTrackerSmokeRunner` from `run_external_baseline_smoke_20260426.py` with `STWM_EXTERNAL_OUTPUT_PHASE=full_eval`.
- The conservative `cutie_adapter_20260426.py`, `sam2_adapter_20260426.py`, and `cotracker_adapter_20260426.py` files are not the active full-eval inference path.

## Stub Adapter Status
- `code/stwm/tools/external_baselines/cutie_adapter_20260426.py`: conservative_stub=True, used_by_full_eval_runner=False
- `code/stwm/tools/external_baselines/sam2_adapter_20260426.py`: conservative_stub=True, used_by_full_eval_runner=False
- `code/stwm/tools/external_baselines/cotracker_adapter_20260426.py`: conservative_stub=True, used_by_full_eval_runner=False

## Validity Impact
No numeric validity impact for current reports: the completed full eval path is the SmokeRunner-derived active inference path, not the conservative stub adapter files. The naming is confusing and should be cleaned later.

## TODO
Rename SmokeRunner classes to FullEvalRunner or unify adapter API so the active inference path and adapter filenames match.
