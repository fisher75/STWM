# YOLO-World Official Env Report (2026-03-31)

## Scope

- Objective: verify official `third_party/YOLO-World` smoke inference without patching the main `stwm` environment.
- Isolated env: `/home/chen034/miniconda3/envs/stwm_yolo_official`
- Main env policy: no further site-packages patching in `stwm`.

## Environment Snapshot

- python: 3.10
- torch: 2.0.1+cpu
- numpy: 2.2.6
- mmengine: 0.10.3
- mmcv: 2.0.0
- mmdet: 3.0.0
- mmyolo: 0.6.0
- transformers: 4.36.2
- tokenizers: 0.15.2
- lvis: 0.5.3

Evidence logs:
- `logs/yolo_official_env_setup.log`
- `logs/yolo_official_env_setup_cpu.log`
- `logs/yolo_world_official_smoke.log`

## Attempt Timeline

1. Created `stwm_yolo_official` with Python 3.10.
2. Initial GPU wheel path was abandoned due large `torch` download stall.
3. Switched to CPU-compatible isolated stack and installed core OpenMMLab packages.
4. Installed `third_party/YOLO-World` with `--no-deps` after dependency resolver/build failures.
5. Added missing runtime dependencies iteratively (`transformers`, `tokenizers`, `lvis`).
6. Initialized missing submodule dependency (`third_party/YOLO-World/third_party/mmyolo`).
7. Applied one local syntax hotfix in third-party code:
   - file: `third_party/YOLO-World/yolo_world/models/detectors/yolo_world.py`
   - line: `self.text_feats, _ = self.backbone.forward_text(texts)`

## Final Official Smoke Command

```bash
cd /home/chen034/workspace/stwm/third_party/YOLO-World
/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm_yolo_official \
  python demo/image_demo.py \
  configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py \
  /home/chen034/workspace/stwm/models/checkpoints/yolo_world/yolo_world_s_stage2.pth \
  /home/chen034/workspace/stwm/tmp/deva_smoke/videos/43_-LB7cp3_mqY \
  person,hand,cup,bowl,knife,plate \
  --topk 30 --threshold 0.05 --device cpu \
  --output-dir /home/chen034/workspace/stwm/outputs/smoke_tests/yolo_world_official_vspw
```

## Final Failure State

Primary blocking error:
- `FileNotFoundError: data/coco/lvis/lvis_v1_minival_inserted_image_name.json`

Additional risk observed:
- NumPy ABI warning (`module compiled using NumPy 1.x cannot be run in NumPy 2.2.6`).

Interpretation:
- This official config expects LVIS minival bootstrap files not available in this workspace.
- Current isolated env also has ABI instability risk after `lvis` installation upgraded NumPy to 2.x.

## Verdict

- Official YOLO-World smoke in isolated env: **blocked** (data/config dependency + ABI risk).
- Main `stwm` env remains protected from further YOLO official dependency churn.
- Temporary baseline remains valid via ultralytics fallback (`outputs/smoke_tests/yolo_world_ultralytics/vspw_43`).

## Suggested Next Retry Boundary

1. Keep retries inside `stwm_yolo_official` only.
2. Pin `numpy<2` in the isolated env before next official run.
3. Either provide expected LVIS minival annotation files under YOLO-World expected path, or use a config that does not require LVIS minival dataset bootstrap.
