# TRACEWM Stage2 Bootstrap Protocol (2026-04-08)

## 1. Frozen Facts

1. Stage1 220m backbone has been frozen and accepted before this round.
2. Stage1 is not the focus of this round and must not be continued here.
3. Latest Stage2 data audit status is fixed as:
   - VSPW = complete
   - VIPSeg = complete
   - BURST = complete
   - TAO = access_ready
   - VISOR = manual_gate
   - final_decision = STAGE2_CORE_READY_WITH_EXTENSION_GAPS

## 2. Stage2 Binding for This Round

This Stage2 bootstrap round is only allowed to bind:
- core datasets: VSPW + VIPSeg
- optional extension: BURST

TAO and VISOR are explicitly not part of current bootstrap train mainline.

## 3. Explicit Prohibitions

The following are forbidden in this round:
- continue Stage1 training
- modify Stage1 architecture/backbone
- include TAO in Stage2 training mainline
- include VISOR in Stage2 training mainline
- Stage2 full longtrain
- WAN / MotionCrafter
- video reconstruction targets

## 4. Round Objective

This round is bootstrap-ready engineering only:
- lock protocol/spec/freeze/semantic-source documents,
- produce Stage2 bootstrap data contract,
- verify frozen Stage1 loading + Stage2 semantic/fusion forward path,
- run only small smoke training/evaluation,
- output bootstrap-ready decision.

## 5. Bootstrap Completion Conditions

Bootstrap is complete only when all conditions are satisfied:
1. Stage2 I/O spec exists and is explicit.
2. Stage2 freeze policy exists and is explicit.
3. Stage2 semantic source spec exists and is explicit.
4. Stage2 bootstrap data contract is generated and references current audit decision.
5. Stage2 smoke report answers:
   - frozen Stage1 backbone loadability
   - semantic branch input acceptance
   - semantic fusion forward correctness
   - core dataset input readiness
   - bootstrap_ready or not_ready

## 6. Fixed Runtime Envelope

- tmux session: tracewm_stage2_bootstrap_20260408
- fixed log: /home/chen034/workspace/stwm/logs/tracewm_stage2_bootstrap_20260408.log
- no full Stage2 longtrain in this round
