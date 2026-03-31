# ENV PATCH AUDIT (stwm)

## Audit Target

- Environment: `stwm`
- Path: `/home/chen034/miniconda3/envs/stwm`
- Purpose of this audit: record package state and detect local manual patch traces

## Installed Package Snapshot

- torch==2.11.0+cu128
- torchvision==0.26.0+cu128
- mmengine==0.10.3
- mmcv==NOT_INSTALLED
- mmcv-lite==2.0.0
- mmdet==3.0.0
- mmyolo==0.6.0
- ultralytics==8.4.32
- transformers==5.4.0
- segment-anything==1.0
- timm==0.6.13

## Local Patch Evidence

### mmengine

Observed file:
- `/home/chen034/miniconda3/envs/stwm/lib/python3.10/site-packages/mmengine/optim/optimizer/builder.py`

Observed local modification marker:
- `if 'Adafactor' not in OPTIMIZERS:`

Interpretation:
- This is a manual defensive patch to avoid duplicate Adafactor registration with the current torch/transformers stack.
- File timestamp indicates local modification on 2026-03-31 00:59:53 +0800.

### mmcv / mmdet / mmyolo

- No direct source-level patch evidence was found in this audit step.
- However, current stack is mixed (`mmcv-lite` + mmdet/mmyolo), which is functional for some paths but insufficient for official YOLO-World path requiring `mmcv._ext` ops.

## What Must Be Recorded and Preserved

1. The mmengine local patch exists in main environment and must be documented as non-reproducible manual state.
2. Any result produced under this environment should annotate that the runtime is patched.
3. Main training environment should no longer be used as the place to debug official YOLO-World dependency closure.

## What Should Not Be Relied On

1. Do not rely on this patched mmengine behavior as a stable baseline.
2. Do not continue adding site-packages patches under `stwm`.
3. Do not claim official YOLO-World closure from this environment.

## Policy Going Forward

1. Freeze `stwm` as main STWM training environment.
2. Isolate official YOLO-World verification into a dedicated env (`stwm_yolo_official`).
3. Keep fallback baseline (ultralytics YOLOWorld) available for short-term experimentation until official path is closed in isolated env.
