# ARC-TGI family-disjoint anchor cohort v1

## Purpose

This cohort replaces the contaminated public ARC-AGI-2 development tasks for
the mechanism-track experiment. It asks one narrow question:

> Can a frozen visual posterior create typed object/AST proposals that recover
> at least 5 of 100 tasks uniquely, under the same native cost as a cold
> restart?

It does not replace solver-track evaluation on ARC-AGI-2 private tasks.

## Source freeze

- ARC-TGI commit: `a614132ff5b2cb3628063d541e7cbd74a2cd2edb`
- source collection: `Generators/ARC-Mini`
- expected upstream files: 180
- canonical families: 170 after conservatively merging only `_1`, `_new`, and
  `-new` revision suffixes
- NVARC pretraining identifier map: Kaggle
  `cpmpml/arc-prize-trm-training-data`, version 1

ARC-TGI was released after the November 2025 NVARC checkpoint and emits fresh
episodes from executable task-family generators. ARC-Mini is used because its
opaque generator IDs do not overlap the NVARC pretraining identifiers and its
families do not inherit an ARC-AGI-2 source task that the checkpoint saw.

## Partition and exposure contract

For each canonical family, choose exactly one upstream source variant using
the minimum `(source_sha256, relative_path)` tuple. Rank families by SHA-256 of
the fixed partition seed, canonical family ID, and family source ID.

- development: 50 families
- confirmatory: 100 families
- reserve: 20 families, not materialized

The one generator source inspected manually while designing this protocol,
`task2fJ984g27gSFKHfq53RTVH`, is forced into development before hash ranking.
No confirmatory generator implementation, reasoning chain, witness, grid, or
query output may be manually inspected before the experiment is frozen.

Each materialized family produces one episode. Python `random` and NumPy are
seeded from a SHA-256 digest over a separate frozen episode seed, the canonical
family ID, and family source ID. A second generation must be byte-identical.

## Integrity conditions

Every episode must satisfy all of the following or cohort construction stops:

1. all grids are rectangular, 1–30 cells per axis, and use colors 0–9;
2. train and test sets are non-empty;
3. the generator's direct transformation reproduces every stored output;
4. the partially evaluated executable witness reproduces every stored output;
5. same-seed replay is byte-identical;
6. blind task content is unique across development and confirmatory families;
7. development and confirmatory canonical family IDs are disjoint;
8. ARC-TGI family IDs have zero overlap with NVARC pretraining identifiers.

## Gold and witness boundary

For each partition, construction writes three immutable artifacts:

- `*_challenges.json`: demonstrations and query inputs only;
- `*_solutions.json`: evaluator-only query outputs;
- `*_witnesses.json`: evaluator-only reasoning, task variables, and executable
  transformation code.

The anchor and every proposal policy may read only the challenge artifact.
Solution and witness hashes are bound into the cohort manifest but their
contents are not inputs to candidate generation, ranking, repair, or stopping.
Any run violating this boundary is invalid rather than recoverable.

## Anchor and bridge gates

The released NVARC checkpoint is first treated as a fixed, reference-scale
candidate generator on this cohort. Its architecture, checkpoint, 128
augmentations, 2,000 test-time-training epochs, global batch size 128, ten
recursive inference steps, voting, and candidate budget stay frozen. Hardware
portability changes are logged separately and may not change those numerical
settings.

Only after anchor candidates, costs, and content hashes are frozen may the
visual-to-structure bridge run. The confirmatory success gate is conjunctive:

- at least 5/100 unique exact task recoveries over the frozen anchor;
- strictly more unique recoveries than equal-native-cost cold restart;
- every claimed repair is demo-exact;
- every claimed repair has `novel_frontier_count > 0`;
- no query output or witness was read before candidate freeze;
- no family appears in both development and confirmatory sets.

Failure at this gate stops controller work and triggers a new theory analysis
of what information and action space causal functional switching requires.
