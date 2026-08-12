# RESULT-EXP-003: Invalid before provider execution

- Status: completed
- Outcome: invalid cohort/provenance boundary
- Date: 2026-08-12
- Claims affected: none
- GPU/provider execution: not started

## Failure

The pre-GPU NVARC dataset boundary audit deterministically failed only
`demo_query_input_collision_count=2`.  Two reserve tasks repeat a
demonstration input as a query input.  All compiled training pairs are exact
demonstrations, all test inputs are exact queries, all 100 tasks are
represented, and every remaining violation is zero.

The protocol also named seal ID
`618c4c81a9c6002b1890fac2457b08c23d7c90a06c08d70c438b07be8a29e873`,
but the committed compact seal supplied to compilation has ID
`7c89b3e3cc3f41131b39b60b5bb5ad7d62a9032fabcf864407dc057e3b7e899b`.
The challenge bytes, cohort ID, oracle hashes, and task rows agree; the IDs
differ because their source-file provenance differs.  This still violates the
frozen identity contract.

## Classification

This is a data/protocol validity failure, not an algorithmic null and not a
provider implementation failure.  The audit gate worked as intended and
prevented GPU expenditure and any oracle opening.

## Correction

Using blind inputs only, derive a new seal that excludes exactly the two tasks
with demo/query input collisions.  Give the 98-task subset a new cohort and
seal identity, preserve the parent oracle hashes, keep all provider settings
and absolute success thresholds fixed, and preregister it as `EXP-004`.
