# VARC query-blind ARC-TGI development anchor v1

## Decision question

Can the pinned VARC checkpoint produce a nontrivial, query-gold-free visual
posterior on the 50 development families, sufficient to choose and freeze one
object/AST proposal family before opening the 100 confirmatory tasks?

This run characterizes a static visual candidate distribution.  It does not
train a router, test residual control, or count a posterior pixel consensus as
a repair.

## Frozen identities

- VARC source commit:
  `bd478ecf362e6499a988b05f33223e5c5fc6a6be`;
- checkpoint SHA-256:
  `c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3`;
- ARC-TGI source cohort ID:
  `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`;
- development blind-cohort ID:
  `aea6cb1ad136d17688774a40b9aa07a174580a5e35857484053187063410b33c`;
- confirmatory blind-cohort ID, not yet augmented or executed:
  `f631e91243529984cca65b2463d3dcbbb5e2265f1f998d44f265b7780dbfcd5f`;
- development tasks: 50 generator families;
- confirmatory tasks: 100 disjoint generator families.

Every provider-visible query output is an exact copy of its corresponding
query input.  This sentinel is required by the upstream loader but is not the
ARC solution.  Its identity and every augmented file are hashed before the run.
The upstream post-prediction `analyze_data` routine compares candidates with
this sentinel and may print a pass/oracle score.  Those numbers are identity
diagnostics, not ARC accuracy, and are excluded from every project result.

## Frozen numerical contract

- ViT depth 10, embedding width 512, eight heads;
- image size 64, patch size 2, twelve color tokens;
- command argument `--epochs 100`; the pinned upstream inclusive loop executes
  epochs 0 through 100, hence 101 actual epochs per task;
- batch size 8, learning rate 3e-4, cosine schedule, zero weight decay;
- ten attempts, `ttt_num_each=1`, seed 42;
- resume the pinned checkpoint while skipping a new task token;
- per-task timeout 2,400 seconds;
- one currently idle GPU only; no use of GPUs occupied by other jobs.

No numerical fallback, task skip, retry with a different seed, or partial
cohort result is valid.

## Exposure and action boundary

The development sequence is fixed:

1. run all 50 blind tasks and close a native-cost receipt;
2. freeze every valid unique output grid and its sample multiplicity;
3. replay the freeze byte-identically;
4. only then read development solutions and compute rank-1, rank-2, and raw
   oracle coverage;
5. diagnose failures using demonstrations, frozen posterior samples, and the
   now-exposed development outcomes;
6. implement exactly one coherent typed object/AST proposal family;
7. freeze that family, its unguided equal-cost restart, and all native costs
   before augmenting or executing the 100 confirmatory blind tasks.

The proposal family may use the visual posterior to choose a component,
relation, canvas, mask, AST node, or typed parameter.  It may not emit a
pixelwise marginal consensus as the final candidate, add task-ID clauses,
modify the frozen raw candidate order, or train a controller.

## Gates

The development run is operationally valid only if all 50 task processes exit
zero, all prediction files are nonempty, source/checkpoint/data hashes close,
and independent freezing reports exactly the manifest task/query set.

Development accuracy is diagnostic and cannot pass the final method claim.
The only confirmatory bridge gate remains:

- at least 5/100 strict task recoveries unique over all frozen anchors;
- positive novel-frontier count for every claimed repair;
- demonstration-exact typed execution;
- higher recovery than an unguided cold restart with the same native proposal
  and execution budget;
- zero task-specific tuning after confirmatory execution begins.

If that gate fails with a valid implementation, the next action is a
theory-level reformulation of failure certificates and functional switching,
not another router or another hand-written relation.
