# ARC-TGI development eligibility amendment — 2026-08-11

## Decision

The original 50-task development cohort remains immutable. A separate,
content-addressed eligibility amendment excludes one task and selects no
replacement. All development comparisons after this point use the same 49-task
intersection for every provider.

The exclusion criterion is structural and query-gold-free:

> Exclude a task when a query input is cell-for-cell equal to any demonstration
> input in the same task.

This criterion was applied after the VARC candidate pool was frozen, but it does
not depend on VARC outputs, model scores, query outputs, generator witnesses, or
the identity of the transformation. Promoting a reserve task after observing a
development result was rejected because that would make cohort membership
outcome-dependent.

## Preserved failed preflight

The first compiled 50-task NVARC development dataset is retained unchanged. Its
boundary audit failed only on:

- `demo_query_input_collision_count = 1`;
- task
  `arc_tgi_task7dwpsvzyGCxYgeSrJjQ5c3_471be35a57f78bd1`;
- demonstration index `4`, query index `0`.

All other six boundary violations were zero. This is a dataset-protocol failure,
not a model or algorithm result, and the NVARC run was not started on that
dataset.

## Frozen amendment

- schema: `afts.arc-cohort-eligibility-amendment/v1`;
- amendment ID:
  `19d28af711d514d7a05b7f2e0954209f8909c016da0ae2d5ad23701adc61e4b4`;
- eligibility-decision ID:
  `3d9fdcf9ea981c86229bcab6e23905fe6d9fd9dafb3d99f83096a2e0db5a2a54`;
- source tasks: 50;
- eligible tasks: 49;
- excluded tasks: 1;
- replacement policy: none;
- query gold used for eligibility: false.

The recompiled 49-task dataset passed two byte-identical audits:

- audit ID:
  `b1b8a1053dc4eacaf959991ccb4f2b351b54ce8c70567d3781e25a4bd75f68a7`;
- audit file SHA-256:
  `33659e45c89b68bb309fd4956ecdb44992aa97dddbd1ee6663cfb687c669f0d1`;
- augmented puzzles: 5,882;
- reconstructed training pairs: 22,974;
- reconstructed test inputs: 7,172;
- exact full batches: 2,962;
- all seven violation counts: zero.

The generalized receipt implementation was replayed twice against the already
published 100-task confirmatory run. Both outputs were byte-identical to the
original receipt and retained receipt ID
`94cbc61b67ce97f042a60ddad8b2df801a3a0b4e62ac0093f8db076d87bbe280`.
This demonstrates that parameterizing the audited workload does not change the
existing 100-task numerical result or artifact identity.
