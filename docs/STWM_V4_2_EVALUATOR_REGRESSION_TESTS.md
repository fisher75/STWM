# STWM V4.2 Evaluator Regression Tests (Phase A)

Date: 2026-04-03
Status: PASSED

## Test Suite

- Test file: code/stwm/tests/test_eval_mini_val_regression.py
- Command:

```bash
cd /home/chen034/workspace/stwm
PYTHONPATH=/home/chen034/workspace/stwm/code conda run -n stwm pytest -q code/stwm/tests/test_eval_mini_val_regression.py
```

- Result: 4 passed in 59.22s

## Covered Regression Cases

1. legacy checkpoint smoke
   - verifies evaluator can run stwm_1b checkpoint
   - verifies requested_protocol_version = v2_4_detached_frozen
   - verifies canonical protocol_version = v2_3
2. v4.2 real checkpoint smoke
   - verifies evaluator can run stwm_v4_2 checkpoint
   - verifies evaluator_version and protocol alias behavior
3. tiny manifest schema regression
   - verifies stable metric keys exactly match frozen set
   - verifies summary.protocol.stable_comparable_metrics consistency
4. same checkpoint twice determinism
   - same checkpoint, same seed, same tiny manifest
   - max absolute metric diff <= 1e-12

## Operational Meaning

- Evaluator freeze contract is executable for both model families.
- Protocol aliasing and metric schema are regression-guarded.
- Deterministic behavior on fixed tiny inputs is validated.
