# STWM FSTF Video Input Pipeline Audit V9

- video_input_claim_allowed: `True`
- raw_video_end_to_end_training: `False`
- frozen_frontend_pipeline: `True`
- cache_training_disclosed: `True`
- future_leakage_detected: `False`

Training/evaluation use a frozen video-derived trace and observed semantic-memory cache. The system pipeline starts from raw video frames, but the FSTF transition is trained on materialized video-derived trace/semantic states, not end-to-end raw RGB.
