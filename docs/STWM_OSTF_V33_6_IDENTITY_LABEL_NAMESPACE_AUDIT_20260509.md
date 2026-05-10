# STWM OSTF V33.6 Identity Label Namespace Audit

- fut_instance_id_global_unique: `False`
- cross_sample_label_collision_detected: `True`
- contrastive_loss_uses_global_identity: `False`
- identity_training_label_safe: `False`
- exact_risk: `PointOdyssey fut_instance_id values are sample/video-local mask ids; V33.3 contrastive loss used them directly, so equal numeric ids across unrelated samples can become false positives.`
- recommended_fix: `Build fut_global_instance_id keyed by dataset+split+sample_uid+fut_instance_id and train contrastive/retrieval losses on global labels.`
