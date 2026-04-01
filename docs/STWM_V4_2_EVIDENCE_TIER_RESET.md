# STWM V4.2 Evidence Tier Reset

## Purpose

This document resets claim boundaries and evidence tiers for STWM V4.2.

## Tier Definitions

Tier A (Main Claim Evidence):

1. real 220M runs on true train split (VSPW train + VIPSeg train)
2. real 1B runs on true train split (VSPW train + VIPSeg train)
3. fixed hard budget and fixed batch/precision/checkpoint/resume policy satisfied

Tier B (Supporting But Not Main Claim):

1. lightweight or mini-split training runs
2. staged or smoke 1B runs
3. partial-coverage warmups used only for engineering diagnostics

Tier C (Protocol Evaluation Views):

1. base protocol reports
2. state-identifiability reports
3. decoupling reports

Important boundary:

- Tier C defines evaluation protocols and analysis views.
- Tier C does not define or replace main training-set provenance.

## Explicit Non-Claim Rules

The following cannot be used as headline proof of final model capability:

1. old 220M lightweight evidence alone
2. old staged 1B lightweight evidence alone
3. any run with positive sample truncation on real manifest

## Main-Claim Readiness Conditions

Main claims become publishable only when all are true:

1. real 220M mandatory matrix complete (seeds 42/123 first; 456 conditional)
2. real 1B mandatory matrix complete (seeds 42/123 first; 456 conditional)
3. each run meets fixed budget policy (`>=3 epochs` and `>=5000 steps`, capped at `8000`)
4. phase0/1 scale-out gate passed on compliant full-train audit
5. checkpoint retention and resume policies validated

## Reporting Requirement

All final summaries must clearly label evidence source tier.

Recommended tags:

- `MAIN_EVIDENCE_REAL_220M`
- `MAIN_EVIDENCE_REAL_1B`
- `SUPPORTING_LIGHTWEIGHT_ONLY`
- `EVAL_PROTOCOL_ONLY`
