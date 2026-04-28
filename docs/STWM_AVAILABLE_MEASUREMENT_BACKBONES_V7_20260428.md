# STWM Available Measurement Backbones V7

- selected_backbone: `local_openai_clip_vit_b_32`
- selection_reason: `highest-priority locally available frozen measurement backbone`
- no_internet_download_attempted: `True`
- fallback_if_none: `crop_encoder_feature_only`
- available_backbones: `{'local_openai_clip_vit_b_32': {'available': True, 'cache_or_weight_paths': ['/home/chen034/.cache/clip/ViT-B-32.pt', '/home/chen034/.cache/clip/ViT-B-16.pt'], 'python_module_available': True}, 'huggingface_clip_cache': {'available': True, 'cache_or_weight_paths': ['/home/chen034/.cache/huggingface/hub/models--openai--clip-vit-base-patch32'], 'python_module_available': True}, 'local_sam2_code_or_cache': {'available': True, 'cache_or_weight_paths': ['/raid/chen034/workspace/stwm/third_party/sam2'], 'python_module_available': False}, 'stwm_crop_visual_encoder': {'available': True, 'cache_or_weight_paths': [], 'python_module_available': True}}`
- unavailable_backbones: `{'huggingface_dinov2_cache': {'available': False, 'cache_or_weight_paths': [], 'python_module_available': True}, 'huggingface_siglip_cache': {'available': False, 'cache_or_weight_paths': [], 'python_module_available': True}}`
