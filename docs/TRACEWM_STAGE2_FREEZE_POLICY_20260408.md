# TRACEWM Stage2 Freeze Policy (2026-04-08)

## 1. Frozen Modules (Must Stay Frozen)

1. Stage1 220m backbone
2. Stage1 tokenizer
3. Stage1 core rollout backbone path

These modules are frozen for the entire bootstrap round.

## 2. Trainable Modules (Only These)

1. semantic encoder
2. semantic fusion or semantic adapter
3. optional lightweight readout head

No other module is trainable in this round.

## 3. Enforcement

1. Any parameter outside the trainable module list must keep requires_grad=False.
2. Bootstrap smoke must verify:
	- Stage1 gradient leakage is absent
	- semantic branch gradient is present
3. Any accidental backbone unfreeze is treated as protocol violation.
