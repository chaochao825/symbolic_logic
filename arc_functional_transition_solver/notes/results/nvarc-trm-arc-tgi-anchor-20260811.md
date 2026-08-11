# NVARC/TRM ARC-TGI anchor result — 2026-08-11

## Decision

The static reference-scale NVARC/TRM provider qualifies as a strong candidate
anchor on the frozen 100-task ARC-TGI mechanism cohort. It does not establish
competitive ARC-AGI performance, cross-representation repair, residual
causality, or functional switching.

This changes one part of the project status:

> A strong static candidate generator now exists for a synthetic,
> generator-family-disjoint cohort. Strong generation on ARC-AGI, executable
> visual-to-symbolic repair, and causal functional switching remain
> unestablished.

## Protocol integrity

The released NVARC checkpoint was not eligible on official ARC evaluation
tasks because its identifier inventory overlaps them. The replacement cohort
uses 100 post-checkpoint ARC-TGI instances from mutually disjoint generator
family IDs and has no identifier overlap with the released pretraining index.

The compiler necessarily stores query labels in the test arrays for the
upstream evaluator. A separate query-semantic-blind audit established:

- 48,498/48,498 compiled training pairs invert exactly to source
  demonstrations;
- 13,728/13,728 compiled test inputs invert exactly to source query inputs;
- no training pair falls outside the demonstration set;
- no demo/query input collision occurs;
- all 100 augmentation groups correspond one-to-one with base tasks;
- test-label bytes were content-addressed but their semantics were not read.

The upstream source path was pinned. Test labels contribute to loss/metric
comparison, while fixed ten-step candidate logits and rank construction use
inputs, puzzle identifiers, and model state. Project-side candidates were
content-addressed before solutions were opened. Two receipt, candidate-freeze,
and aggregate-score constructions were byte-identical.

## Result

| Endpoint | pass@1 | pass@2 | pass@5 | pass@10 |
|---|---:|---:|---:|---:|
| strict whole-task | 87/100 | 90/100 | 92/100 | 92/100 |
| query | 99/113 | 102/113 | 104/113 | 105/113 |
| upstream mean pair | 0.890 | 0.915 | 0.935 | 0.940 |

The distinction matters: upstream `ARC/pass@k` averages per-task query hit
fractions, whereas the project primary endpoint requires every query in a task
to be correct. The independent scorer reconstructs both and matches upstream
exactly on its metric definition.

The top ten collapse to 437 unique grids over 113 queries. No additional
strict task is recovered after rank five, so the anchor is accurate on this
cohort but not especially diverse.

## Why the score is high without demonstrated leakage

The most plausible explanation is cohort structure, not ARC-level general
intelligence. The 100 tasks have a mean of 4.01 demonstrations; 354/401
demonstrations preserve canvas shape, and the instances are generated from
systematic ARC-Mini programs. Family IDs are disjoint across development and
confirmation, but the generators still share ARC primitives, regular visual
statistics, and an overall synthesis process. A pretrained visual recursive
model with 2,000-epoch per-cohort adaptation can exploit that regularity.

Therefore the result supports only:

> reference-scale test-time adaptation can provide a high-coverage static
> candidate pool on this synthetic distribution without query-label-dependent
> model updates or project-side reranking.

It does not support claiming 92% on ARC-AGI, nor does it validate the original
dynamic brain-region-switching motivation.

## Consequence for the bridge experiment

The anchor leaves only eight strict top-10 failures. The frozen requirement of
at least five unique recoveries is therefore stringent: the bridge would need
to recover at least five of those eight while beating an equal-native-cost
cold restart. This is useful as a falsification gate but creates a ceiling and
small-denominator problem. Task-level anchor failures stay hidden until the
visual proposal family and all confirmatory bridge candidates are frozen.

No router training is opened. The next admissible step remains exactly one
visual-posterior-to-typed-proposal family, chosen on the 50-task development
split, followed by the unchanged 100-task confirmation gate.
