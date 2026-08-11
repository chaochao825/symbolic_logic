# Visual relational-transducer development result — 2026-08-11

## Decision

Retain exactly one frozen bridge family for confirmation. Do not train or
extend the router, and do not add task-specific relations.

The development result is positive but deliberately non-confirmatory:

> Under equal incremental native cost, the composite query-blind allocation
> found all three tasks for which the frozen relational-transducer grammar adds
> a correct candidate beyond the VARC/NVARC union, while the pre-existing cold
> allocation found none. A direct outcome ablation does not attribute this
> advantage to the visual-posterior term alone.

This is evidence that the visual posterior can allocate a small typed search
budget on this exposed development cohort. It is not evidence for a general
dynamic functional-switching mechanism.

## Frozen representation

The program is a two-node AST:

1. one of eight D4 coordinate correspondences;
2. one of three color relations: a learned global homomorphism, row foreground
   count plus an inferred offset, or column foreground count plus an inferred
   offset.

Exactly 24 structural hypotheses are tried per selected task. The provider is
same-canvas only and cannot crop, compose, insert or erase objects, infer a new
canvas, or use extra hand-written relations. Only demo-exact candidates that
are novel relative to the content-addressed VARC/NVARC pool enter the frontier.

## Development result

| Quantity | Value |
|---|---:|
| eligible development tasks | 49 |
| raw VARC/NVARC union | 46/49 |
| visual allocation size | 15/49 |
| cold allocation size | 15/49 |
| visual unique recoveries | 3 |
| cold unique recoveries | 0 |
| static-family recoveries | 3 |
| visual − cold | +3 |

Both allocated arms are padded to 360 program trials, 2,160 demonstration
executions, and 1,080 query executions. The visual arm changes three task
frontiers; the cold arm changes none. Two independent freezes and two scores
are byte-identical.

A separate query-gold-free allocation intervention is also byte-identically
replayed. Removing only the posterior preservation term while retaining the
static candidate-count and overlap descriptors replaces 7/15 selected tasks.
Removing every certificate feature replaces 9/15. The posterior therefore has
a real causal effect on development actions. However, scoring the already
frozen counterfactual allocations gives 3 recoveries for the observed rule, 3
for the posterior-cleared rule, and 1 for the all-features-cleared rule. The
posterior term has no measured development performance effect; the positive
comparison with cold is attributable only to the composite static descriptors
or their interaction.

The recoveries have clean typed explanations:

- same coordinates with a demo-inferred global color homomorphism;
- same coordinates with each row's foreground count plus two as its color;
- left/right reflection followed by a demo-inferred global color homomorphism.

## What this result does and does not test

The static-family audit shows that the candidate language itself adds three
regions. The visual-versus-cold comparison tests whether the visual posterior
and static pool descriptors jointly allocate the fixed AST-search budget to
those regions. It does not show that posterior preservation mass improves
recovery, and it does not test an online controller, residual-dependent second
action, or final pass@2 selection. Because development outcomes were already
exposed, it cannot be used as a paper-level causal conclusion.

The untouched 100-task gate remains unchanged: visual typed proposals must
produce at least five unique recoveries and strictly exceed the equal-cost
cold allocation. The family, visual ranking rule, cold seed, 30% allocation,
native-cost padding, and novelty rule are all frozen before confirmation.
