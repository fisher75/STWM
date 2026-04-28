# STWM World-Model No-Drift Guardrail V8

## Allowed

- Controlled joint training is only a small validation of FutureSemanticTraceState supervision.
- Reappearance/visibility remain world-state predictions.
- Association/reacquisition remains utility, not method definition.
- Frozen recurrent trace modules may remain in train mode only to satisfy cuDNN backward; their parameters remain frozen.

## Forbidden

- Large training before joint signal is verified.
- Stage1 unfreeze.
- Calling STWM a SAM2 plugin.
- Claiming paper-level world model if only headonly event-level result is positive.
- Ignoring weak per-horizon reappearance result.
- Treating this Stage2 validation split export as external hard-case full-model export.

Current paper_world_model_claimable remains `unclear`.
