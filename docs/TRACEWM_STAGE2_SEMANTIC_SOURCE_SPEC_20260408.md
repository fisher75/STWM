# TRACEWM Stage2 Semantic Source Spec (2026-04-08)

## 1. Mainline Semantic Definition

Stage2 semantic signal is defined as visual semantic state from:
1. object region crop
2. mask crop when mask is available

## 2. Explicit Disallow Rules

The following are not allowed as Stage2 semantic mainline:
1. fake hash label
2. CLIP teacher distillation as mainline source

## 3. Engineering Interface

1. semantic encoder is a trainable lightweight branch.
2. semantic fusion explicitly injects semantic token into frozen Stage1 token space.
3. semantic source mode in bootstrap artifacts must be object_region_or_mask_crop_visual_state.

## 4. Bootstrap Validation Requirement

Bootstrap smoke must show:
1. semantic branch receives non-empty inputs
2. semantic fusion forward path is valid
3. no violation of Stage1 freeze boundary