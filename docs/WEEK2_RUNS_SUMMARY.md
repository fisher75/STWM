# Week-2 Runs Summary (2026-03-31)

## Scope

- Objective: run minimal 4-way ablation on the same setup used for `prototype_220m` smoke.
- Runner script: `scripts/run_week2_ablations.sh`
- Trainer entry: `code/stwm/trainers/train_stwm.py`
- Manifest: `manifests/minisplits/stwm_week1_mini.json`
- Sample limit: `--limit 1`

## Run Outputs

- `outputs/training/week2_ablations/full.json`
- `outputs/training/week2_ablations/wo_semantics.json`
- `outputs/training/week2_ablations/wo_trajectory.json`
- `outputs/training/week2_ablations/wo_identity_memory.json`

Master log:
- `logs/week2_ablations_master.log`

## Result Table

| Run | Loss | Params | Input Dim | Notes |
|---|---:|---:|---:|---|
| full | 0.270403 | 227,899,459 | 45 | Includes trajectory + semantics + identity memory features. |
| wo_semantics | 0.323654 | 227,899,459 | 45 | Semantic token slice zeroed. |
| wo_trajectory | 0.292344 | 227,899,459 | 45 | Center/velocity token slices zeroed. |
| wo_identity_memory | 0.104229 | 227,891,267 | 37 | Identity memory append disabled (token dim reduced by 8). |

## Implementation Notes

- Ablation toggles added to trainer:
  - `--disable-semantics`
  - `--disable-trajectory`
  - `--disable-identity-memory`
  - `--identity-memory-dim` (default `8`)
- Token layout used by the trainer report:
  - center: `[0, 2)`
  - velocity: `[2, 4)`
  - visibility: `[4, 5)`
  - semantics: `[5, 37)`
- Identity memory appends deterministic sinusoidal/cosine features over frame index.

## Quick Reproduce

```bash
cd /home/chen034/workspace/stwm
/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
  bash scripts/run_week2_ablations.sh
```
