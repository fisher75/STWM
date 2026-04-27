# STWM Future Semantic Trace State Validation Audit 20260427

- validation_passed: `True`
- coverage: valid 2D, valid 3D, invalid scalar rank, invalid hypothesis coord rank

| case | expected | observed | coord_dim | errors |
|---|---:|---:|---:|---|
| valid_2d_case | True | True | 2 | 0 |
| valid_3d_case | True | True | 3 | 0 |
| invalid_scalar_rank_case | False | False | 2 | 1 |
| invalid_hypothesis_coord_rank_case | False | False | 2 | 1 |
