# TRACEWM Stage2 Freeze Policy (2026-04-08)

## Frozen Modules

1. Stage1 220M backbone (all backbone parameters frozen)
2. Stage1 tokenizer / decoder path for this bootstrap round (no Stage1 structural edits)

## Trainable Modules

1. Stage2 semantic encoder
2. Stage2 semantic fusion / adapter block
3. optional lightweight Stage2 readout head

## Enforcement Notes

- No ambiguous mixed-state policy is allowed.
- Any parameter outside the explicit trainable set must remain frozen.
- Smoke report must verify boundary behavior (frozen grads absent, trainable grads present).
