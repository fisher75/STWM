# STWM CoTracker Object-Dense Teacher V16

- teacher_run_success: `True`
- combo: `M128_H32`
- teacher_source: `cotracker_official`
- processed_clip_count: `289`
- skipped_existing_clip_count: `192`
- failed_clip_count: `127`
- object_count: `1605`
- point_count: `205440`
- valid_point_ratio: `0.6920003593119233`
- runtime_seconds: `110.33419561386108`
- next_step_if_failed: `fix_window_selection`

## 中文总结
本次只构建 video-derived object-dense trace cache；CoTracker 作为 video trace teacher 生成观测/未来 trace 监督，STWM 主模型输入仍限制为 observed trace。
- 已处理 clip 数: `289`
- 已跳过已有 cache 数: `192`
- 失败 clip 数: `127`
- 输出 cache: `outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32`
