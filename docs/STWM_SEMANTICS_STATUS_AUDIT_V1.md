# STWM V4.2 Semantics Status Audit V1

Generated: 2026-04-04 23:54:14 +0800
Workspace: /home/chen034/workspace/stwm
Mode: read-only evidence audit

## Scope

This audit answers five fixed questions about semantics status in V4.2 using only existing artifacts:

- seed42 clean matrix final decision
- seed123 clean replication final decision
- wo_semantics sidecar and eval summary
- QSTR and QTSA seed42 reports
- object bias autopsy
- V4.2 design and trainer definitions

## Key Evidence

1. V4.2 design and supervision definitions include a semantic head and semantic supervision.
   - docs/STWM_V4_2_DESIGN.md
   - docs/STWM_V4_2_SUPERVISION_PLAN.md

2. Trainer contains explicit semantics controls and loss terms.
   - code/stwm/trainers/train_stwm_v4_2_real.py
   - includes --disable-semantics, --lambda-sem, semantic adapter/features/loss paths

3. Seed42 D1 clean matrix (reports/stwm_d1_clean_matrix_final_report_v1.json):
   - full_nowarm beats wo_semantics on all 3 official metrics
     - qloc: 0.006695 < 0.008401
     - qtop1: 0.926209 > 0.895674
     - l1: 0.006538 < 0.008473
   - wo_object_bias beats full_nowarm on all 3 official metrics
     - qloc: 0.002259 < 0.006695
     - qtop1: 0.979644 > 0.926209
     - l1: 0.002430 < 0.006538

4. Seed123 replication clean final decision (reports/stwm_seed123_replication_final_decision_v1.json):
   - baseline full: qloc 0.008338, qtop1 0.898219, l1 0.008264
   - wo_semantics: qloc 0.006680, qtop1 0.928753, l1 0.006663 (better than full)
   - wo_object_bias: qloc 0.008152, qtop1 0.908397, l1 0.008287 (better qloc/qtop1 than full)

5. wo_semantics controls are real ablations in training commands and run summaries:
   - done/*.job commands for seed42 and seed123 include --disable-semantics
   - mini_val_summary.json for both runs has ablation.disable_semantics = true

6. QSTR/QTSA seed42 matrices:
   - best run in both reports is trace_sem_baseline control
   - qstr/qtsa challenge variants are worse than trace_sem_baseline
   - QTSA trace_sem_baseline summary shows:
     - disable_semantics = false
     - qtsa_disable_semantic_transition = true
   - this means strongest current baseline keeps semantics but removes semantics from transition path

7. Object bias autopsy (reports/stwm_object_bias_autopsy_v1.json) gives direct failure evidence:
   - query full_worse_rate: 0.984733
   - trajectory full_worse_rate: 0.982188
   - early/late query gap ratio: 279.279138
   - diagnosis: over-strong and over-early object bias injection

## Direct Answers

A. In V4.2 definition, semantics is a native system component (semantic head + semantic supervision): YES.

B. Current strongest baseline includes semantics, but semantics is blocked from transition in that strongest QTSA control path: YES.

C. The line more directly falsified by clean evidence is object prior line, not semantics line.

D. Semantics final project status: unresolved / high-risk.

E. Standard sentence for paper/report:
"In STWM V4.2, semantics remains a core modeled component, but current clean evidence is cross-seed inconsistent for net gain, while object-prior over-injection is the more directly falsified failure mode; therefore semantics is presently classified as unresolved/high-risk rather than proved or disproved."

## Final Status Label

- semantics_status: unresolved_high_risk
- object_prior_status: directly_falsified_in_current_clean_evidence
- strongest_baseline_semantics_presence: true
- strongest_baseline_semantics_in_transition: false