# STWM OSTF V35.52 Reproducibility Package and Benchmark Card

## 中文总结
V35.52 已把 V35.49 full 325 M128/H32 raw-video closure 整理成可复验 package manifest 和 benchmark card。当前可支持 bounded full M128/H32 video-system benchmark claim；仍不支持任意尺度/full CVPR-scale/open-vocabulary semantic field claim。

## Package 状态
- reproducibility_package_ready: True
- selected_clip_count: 325
- frontend_npz_count: 325
- unified_npz_count: 325
- figure_png_count: 12
- missing_artifact_count: 0

## Benchmark Card 核心口径
- 输入：raw video / predecode frame paths，经 frontend 重新生成 observed dense trace。
- 输出：frozen V30 M128/H32 future trace、V35 semantic state / transition / uncertainty、real-instance pairwise identity retrieval。
- 旧 trace cache：只做 drift comparison，不作为输入结果。
- pseudo identity：diagnostic-only，不进入 identity claim gate。
- teacher / DINO / CLIP / SAM2 / CoTracker：只能作为 frontend、measurement 或 supervision source。

## Claim 边界
- m128_h32_full_325_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- 不允许外推到 H64/H96、M512/M1024、任意 horizon、任意分辨率或 full open-vocabulary dense segmentation。

- package_manifest: reports/stwm_ostf_v35_52_reproducibility_package_manifest_20260516.json
- benchmark_card: reports/stwm_ostf_v35_52_benchmark_card_20260516.json
- recommended_next_step: run_v35_53_reproducibility_dry_run_from_package_manifest
