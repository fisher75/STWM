# STWM TraceAnything Teacher Pilot V2

- traceanything_teacher_runnable: `True`
- traceanything_object_tracks_valid: `True`
- processed_clip_count: `40`
- failed_clip_count: `14`
- processed_split_counts: `{'train': 20, 'val': 10, 'test': 10}`
- processed_dataset_counts: `{'VIPSEG': 12, 'VSPW': 28}`
- mean_valid_point_ratio: `0.7074883252098447`
- mean_same_trajectory_fraction: `0.09065560477120535`
- comparison_to_cotracker_same_clips: `{'M128': {'matched_clip_count': 40, 'mean_point_consistency_l2_px': 84.31767590045929, 'mean_endpoint_consistency_l2_px': 167.48230998516084, 'mean_visibility_agreement': 0.6579661196754092, 'mean_traceanything_trajectory_variance': 8843.286486625671, 'mean_cotracker_trajectory_variance': 5945.928779625892}, 'M512': {'matched_clip_count': 20, 'mean_point_consistency_l2_px': 120.30586748123169, 'mean_endpoint_consistency_l2_px': 202.32999768257142, 'mean_visibility_agreement': 0.6352235485258557, 'mean_traceanything_trajectory_variance': 12360.08047094345, 'mean_cotracker_trajectory_variance': 8120.700392428041}}`
- recommended_next_step: `train_ostf_v2_on_traceanything_hardbench`
