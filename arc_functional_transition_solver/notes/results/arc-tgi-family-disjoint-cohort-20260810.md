# ARC-TGI family-disjoint anchor cohort: 2026-08-10

## Decision

A post-checkpoint, query-gold-separated mechanism cohort is frozen and replayed.
It contains 50 development families and 100 confirmatory families from
ARC-TGI's ARC-Mini inventory. Eighteen additional families remain unmaterialized
reserve. Two source-hash-bound generators are quarantined because their frozen
episodes violate ARC's 30x30 canvas limit.

This cohort is suitable for testing candidate generation and typed repair. It
is synthetic and therefore does **not** replace an ARC-AGI-2 solver score.

## Source and partition

- ARC-TGI commit: `a614132ff5b2cb3628063d541e7cbd74a2cd2edb`;
- inventory considered: 180 ARC-Mini generator files;
- conservative canonical families after merging `_1`, `_new`, and `-new`
  suffix variants: 170;
- partition: 50 development, 100 confirmatory, 18 reserve, 2 quarantine;
- episode construction: deterministic source-hash/family-bound seeds;
- generated witness checked by direct execution and partial application.

The partition is over canonical generator families, not sampled grids. No
canonical family ID overlaps NVARC's pretraining identifiers.

## Preserved failed preflight

The v1 preflight stopped before publishing task artifacts when two frozen
families produced out-of-contract canvases:

- `taskGqqVrV4CnAUNRgZGQZCUQK`, source SHA-256
  `6050e25b52f8c4f669b7ddb0cdcc519ec25a1b5d42a59835292e0abe3353558e`,
  with outputs up to 10x45;
- `taskaasAJ4e5NPRnnWF5HTmp35`, source SHA-256
  `3159f8fab3c1fdeeec3aebc91b4d4b021f655a8e586af573ff7c184491e02726`,
  with a 22x50 output.

v2 quarantines exactly those two source hashes before partitioning. It does not
resample a convenient episode, relax the ARC canvas rule, or silently drop a
failed generated pair. The failed logs remain under
`results/arc_tgi_arcmini_cohort_v1_failed_20260810/`.

## Leakage and replay boundary

Challenge files contain demonstrations and query inputs only. Query solutions
and executable witnesses are separate evaluator-only artifacts. The
confirmatory challenge file was frozen before model execution. A complete
second construction in an external replay directory made all eight published
JSON artifacts byte-identical.

The 100 confirmatory tasks contain 113 query pairs. The TRM-native compiler
created 12,051 non-blank augmented puzzle identifiers from the same frozen
episodes. A base-identifier audit verified all 100 demo groups and all 113 query
pairs exactly, with zero demo/query exact-pair collisions. Thirteen highly
symmetric families naturally produced only 36 or 72 distinct augmentations;
this lower native cost is retained rather than padded with duplicates.

## Evidence identity

- cohort ID:
  `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`;
- partition ID:
  `bcd033de79dbcc32cd15f8cf62a6c3abf2d3f55242c5727f4ef8c52b14005bde`;
- cohort manifest SHA-256:
  `eb17f2fd4a7786d0bba58e9fd92804da4b222b8f77c24888446f322a7d7525e6`;
- partition manifest SHA-256:
  `a3b6713d62fbd04adb2491910905486ca9669e10a8562a219d9f576ddb70d635`;
- confirmatory challenges SHA-256:
  `b6789009213705a68cd7a1067eaf06da7f95b12bf1c7b58532d6dd87f6098ebb`;
- confirmatory solutions SHA-256:
  `0f1f6ee910bd222dd85950cc0d7305bd5439b3594b3a38330070669d780053b5`;
- confirmatory witnesses SHA-256:
  `70cdf795da2eea3d703e4808b9b894fea3bce0f2c9b18afc8ca339841d333b1a`;
- development challenges SHA-256:
  `d2099b588f2cfc4485cfc7b3849e2983ff1429f2c59b2bd3f9da47b84ca3329b`.

Machine-readable artifacts are under
`results/arc_tgi_arcmini_cohort_v2_20260810/`.
