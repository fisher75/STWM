# TRACEWM Evidence Hardening Protocol (2026-04-09)

## 1. Frozen Facts

1. Stage1 is frozen and must not be resumed, retrained, or unfrozen.
2. The current Stage2 mainline remains `stage2_core_cropenc`.
3. The current semantic source remains `crop_visual_encoder`.
4. The current mainline checkpoint is `best.pt`; `latest.pt` is secondary reference only.
5. The current readiness baseline is `training_ready_but_eval_gap_remains`, not `paper_eval_ready`.
6. BURST / TAO / VISOR must not be pulled back into the mainline, and no architecture search is allowed.

## 2. This Round Only Does Evidence Hardening

1. This round does not add training.
2. This round does not revive parked queues.
3. This round does not continue long training.
4. This round does not change Stage1 or the Stage2 core training path.
5. This round only strengthens:
   - source-of-truth evidence inside the live repo
   - data-state evidence bundles
   - external-eval fidelity evidence
   - final readiness judgment

## 3. Required Distinctions

Every output in this round must distinguish:

1. engineering-usable state
2. paper-defendable state
3. repo-internal real audit support
4. contract / summary / doc-only support
5. live-path-exists-but-not-packaged cases
6. missing-paths-that-require-regeneration cases

## 4. Dataset Evidence Contract

Stage1 evidence must explicitly answer:

1. whether PointOdyssey is still complete under the current hard-complete rule
2. whether Kubric is still complete under the movi_e-only rule
3. whether TAP-Vid is eval-ready
4. whether TAPVid-3D is limited-eval-ready only
5. which supporting manifests really exist
6. which live-root checks were regenerated in this round

Stage2 evidence must explicitly answer:

1. whether VSPW / VIPSeg are current core-ready
2. whether BURST is only optional-extension-ready
3. whether TAO is only access-ready
4. whether VISOR remains manual-gate
5. whether the current core train/eval binding has zero missing frame/mask paths
6. whether split files, local paths, sample counts, and semantic crop prerequisites hold

## 5. External-Eval Fidelity Contract

The external-eval audit must separate:

1. `official_evaluator_invoked`
2. `official_task_faithfully_instantiated`

The TAP-style task is only `fully_implemented_and_run` when both are true.
If the official evaluator can run but the task is not instantiated faithfully, the status must stay below full-ready.

TAP3D-style must stay strict:

1. aligned 3D GT for the current binding
2. camera geometry / projection / lifting path
3. verified Stage2 exporter to `tracks_XYZ + visibility`

If those are not all present, TAP3D must not be inflated.

## 6. Runtime Discipline

1. Use one fixed tmux session:
   - `tracewm_evidence_hardening_20260409`
2. Use one fixed log:
   - `/home/chen034/workspace/stwm/logs/tracewm_evidence_hardening_20260409.log`
3. No front-desk blocking work.
4. No training jobs.

## 7. Required Outputs

1. `reports/tracewm_evidence_hardening_audit_20260409.json`
2. `reports/stage1_dataset_evidence_bundle_20260409.json`
3. `docs/STAGE1_DATA_EVIDENCE_BUNDLE_20260409.md`
4. `reports/stage2_dataset_evidence_bundle_20260409.json`
5. `docs/STAGE2_DATA_EVIDENCE_BUNDLE_20260409.md`
6. `reports/stage2_external_eval_fidelity_audit_20260409.json`
7. `docs/STAGE2_EXTERNAL_EVAL_FIDELITY_AUDIT_20260409.md`
8. `reports/tracewm_project_readiness_20260409.json`
9. `docs/TRACEWM_PROJECT_READINESS_20260409.md`
