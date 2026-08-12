# EXP-004 pre-execution commitment

This directory contains the compact, query-gold-free commitment for the
collision-free prospective reserve experiment.

- Eligible cohort ID:
  `9706b3af57b19564e4579a20c29f7127271d253bab4e2e5c58657216681ffe15`
- Eligible seal ID:
  `d9c3c71ca6eb10289e3a64eeff3fedbec3107c6c8f947160ff50afc2edd9fe60`
- Eligible challenge file SHA-256:
  `60682d2ca38496987ddd85033ce9400cfb9a7838995310c1676a6e1156a6d864`
- Committed seal file SHA-256:
  `85af3256cf3b2e540238ac7252ce796e86766e11967300374b9f5027fc3ba3b5`
- Parent seal ID:
  `7c89b3e3cc3f41131b39b60b5bb5ad7d62a9032fabcf864407dc057e3b7e899b`
- Task count: 98
- Eligibility rule: `no-demo-query-input-collision/v1`
- Query gold written or opened: no
- Frozen protocol:
  `.research-control/experiments/protocols/EXP-004.md`
- Protocol commit:
  `216800f9c2e9d00e36e5ee8b458e269e58b2138a`

The two excluded tasks were selected solely because a query input exactly
equals a demonstration input.  Provider outcomes played no role.  `EXP-003`
remains an invalid protocol record and is not pooled with this experiment.

The primary ordering and gates are frozen before either provider outcome on
this cohort.  Recursive candidates are frozen before the visual run; the
visual provider cannot influence the recruitment plan.

## Pre-visual freeze

The recursive dataset boundary audit passed twice with byte-identical output:
98 represented tasks, no out-of-bound training pair or test input, and no
demonstration/query input collision.  The audit ID is
`e7e345dc8490628bcd5271bf8610403de7207d5716c06d9e2ac9c02cd7062a68`.

The frozen recursive run exited zero after 5,608 exact replayed training steps
and 104 evaluation batches.  Its receipt ID is
`279770b882819a300674f06d4bebc97a153fb0a6e706ae223cbbe14995d01b7d`;
its candidate-freeze ID is
`00f11f4eee979c245fb7554457439270117ae1a32635cb42dfee12999e34dadc`.
The upstream zero pass rates are not ARC scores: the upstream evaluator sees
opaque query sentinels by construction.  Query gold remains unopened.

The full anchor-only recruitment plan was frozen before visual data
preparation.  Its plan ID is
`65a6391381ed4425e736de736560f97c033fc8effdb11f1eae843e9d1e21c424`;
the committed task counts are 10, 20, 30, and 49.  The pre-visual commitment ID
is `9035b0ddcc1e4eb8b7d2f14be83ea726b0af9318f7f223a4aa3ba3407fea9e69`.

The visual run is partitioned into deterministic 33/33/32-task shards only to
reduce wall-clock time.  Full-cohort augmentation precedes partitioning, and a
byte-level audit must prove that every task and augmentation file is identical
to the full-cohort version.  Each shard remains a serial single-GPU run.  The
assignment ID is
`7c1cdb12b02dc2323fc9d2ecf6f880cdfac1272a8cf852bcc128bbdd102877b2`.

Full-cohort augmentation and shard projection passed two byte-identical
audits.  All 98 tasks and 4,998 evaluation/augmentation JSON files match the
full-cohort bytes; audit ID
`ae1a97a89b19e889f5f2a46dfee721c60dcd8f16e15c43e9e3f77ce903e3cbe9`.
The data-preparation ID is
`5dd4399b6fea228711e128beb50eb2c3d5895462117cd89bac3d92ee08c79e49`.

The launch assignment was frozen while GPUs 0, 1, and 3 were idle; GPU 2 was
excluded because an unrelated job was active.  Shards 0, 1, and 2 map to GPUs
0, 1, and 3 respectively.  The launch commitment ID is
`eca2577a08d9e044f8b1c590f481f46092afe39cd478f5cb2f22f4159ec983d2`.
Query gold was still unopened at this commitment.

The exact orchestration and scoring scripts are under `provenance/`.  General
content-addressed shard, receipt-merge, and equal-native-cost utilities are in
`scripts/`.  These additions change orchestration and auditing only; provider
model, checkpoint, augmentation order, per-task numerical configuration,
candidate ranking, policy order, and gate thresholds remain frozen.

## Closure-runtime incident

All 98 visual tasks completed with zero exit codes.  The first closure attempt
then stopped before oracle authorization because it invoked ARC-TGI with the
VARC environment, which lacks ARC-TGI's declared `shortuuid` dependency.  The
query solutions were neither created nor read.  `ORACLE_RUNTIME_RECOVERY.md`
records the failure hashes and the non-semantic correction to use the cohort's
already recorded ARC-TGI Python runtime during deterministic oracle
regeneration.

## Final result

`EXP-004` is valid and completed, but its preregistered outcome is
**provider-null**.  NVARC reaches 93/98 raw strict oracle coverage and VARC
88/98; their union reaches 96/98.  VARC contributes two strict exclusive tasks,
below the frozen threshold of three.  The seven recursive-exclusive tasks and
three neither-hit tasks are retained in the full population result.

The outcome is more informative than the one-task threshold gap suggests.  A
seal-grounded family audit finds 18 source families, and all three anchor-miss
union-hit tasks belong to one family: two are VARC-exclusive and one is
query-wise cross-provider composition.  Thus the marginal task count has
effective family count one and cluster concentration 3/3.

The frozen policy is still useful descriptive evidence.  Its first ten tasks
recover all three marginal tasks using 1,910/20,112 visual GPU seconds; the
30-task prefix also recovers 3/3 using 6,076/20,112 seconds, versus a frozen
random median of one.  The equal-native-cost random median is also one.  These
numbers do not rescue the failed complement gate and are not a confirmatory
recruitment result.

This experiment validates the execution and audit path, and shows that cheap
anchor uncertainty can localize the one complementary family.  It does not
establish broad provider specialization, final pass@2 improvement, ARC-AGI
competitiveness, residual-driven repair, or a biological switching mechanism.
