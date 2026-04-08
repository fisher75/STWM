# TRACEWM Stage2 Bootstrap Protocol (2026-04-08)

## 1. Frozen Facts

1. Stage1-v2 220M backbone has been frozen from the completed Stage1 round:
   - `whether_stage1_backbone_is_now_fully_ready = true`
   - `next_step_choice = freeze_stage1_and_prepare_stage2`
2. Stage1 is no longer the main engineering target for this phase:
   - do not continue 15000-step Stage1 long-train as the primary task,
   - do not continue new Stage1 architecture search.
3. Stage2 objective is to introduce semantics on top of a frozen Stage1 backbone.
4. This round is bootstrap-only and explicitly excludes full Stage2 long-train.

## 2. Round Scope

This Stage2 bootstrap round is limited to:
- Stage2 data interface specification and contract mapping,
- Stage2 semantic input/output definition,
- Stage2 freeze/trainable module policy,
- Stage2 runnable code skeleton and smoke validation.

## 3. Hard Exclusions

This round does not modify:
- Stage1 backbone family,
- Stage1 data contract framework,
- WAN / MotionCrafter,
- video reconstruction objectives.

## 4. Stage2 Bootstrap Completion Conditions

Bootstrap is considered complete only when:
- Stage2 I/O spec and freeze policy are documented,
- Stage2 bootstrap data contract is generated,
- Stage2 trainer can load frozen Stage1 backbone,
- semantic branch receives non-empty semantic inputs,
- frozen/trainable boundary behaves as specified,
- smoke report explicitly decides `bootstrap_ready` or `not_ready`.

## 5. Runtime Envelope

- tmux session: `tracewm_stage2_bootstrap_20260408`
- fixed log: `/home/chen034/workspace/stwm/logs/tracewm_stage2_bootstrap_20260408.log`
- this round must not launch full Stage2 long-train.
