# STWM World Model No Drift Guardrail V6

- generated_at_utc: `2026-04-27T17:21:57Z`
- allowed: `["Visibility/reappearance training is world-model state supervision.", "Reappearance is a FutureSemanticTraceState prediction task.", "Future association is utility, not method definition.", "A failed head-only signal gate should block joint training."]`
- forbidden: `["Using visibility logit as reappearance logit.", "Claiming paper-level world model from random-init or negative-signal head.", "Skipping head-only warmup and directly joint-training everything.", "Moving to 1B before reappearance signal is positive.", "Claiming external hard-case result from Stage2 val split."]`
- this_round_joint_training_started: `False`
- this_round_reason: `head-only warmup did not improve reappearance AUROC/AP over pre-warmup baseline; joint training blocked.`
