# STWM TraceAnything Target Semantics Audit V27

- track_coordinate_semantics: `image_plane_xy_in_original_raw_frame_pixels_in_cache; V26 loader normalizes by max(raw_width, raw_height).`
- target_side_box_search_used_ratio: `1.0`
- query_frame_only_comparison: `{'sample_count': 602, 'mean_point_disagreement_px': 0.9155762988732953, 'passes_50_clip_requirement': True}`
- why_last_observed_copy_beats_cv: `TraceAnything hardbench targets are strongly low-displacement under the target-box-constrained trajectory-field extraction. Observed one-step velocity is noisy and CV extrapolation compounds that noise over H32/H64, so CV overshoots while last-observed remains near the teacher target.`
- future_tracks_represent_physical_point_motion_or_reassociation: `trajectory_field_derived_object_internal_points_with_target_side_object_support_reassociation; valid pseudo-targets but not pure unconstrained physical point tracks.`
- last_observed_strength_expected_or_bug: `expected_under_current_teacher_target_semantics_not_a_direct_extraction_bug`
- target_semantics_valid: `True`
- target_extraction_bug_detected: `False`
