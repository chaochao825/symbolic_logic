# Query-blind visual provider and posterior bridge: 2026-08-07 result

## Bottom line

The experiment establishes one positive Solver-track result and one narrower
Mechanism-track diagnostic:

- a static visual TTT provider enters a candidate region missed by the frozen
  portfolio and object/code provider; and
- query-blind sample disagreement reliably localizes many top-1 pixel errors.

It does **not** establish residual-driven functional switching.  The only
licensed pixel-consensus bridge produced new grids but no unique recovery, so
the controller remains frozen.

## Protocol boundary

The provider is VARC at source commit
`bd478ecf362e6499a988b05f33223e5c5fc6a6be`, using checkpoint SHA-256
`c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3`.
The released loader serializes a test output during inference.  Every query
output was therefore replaced by an exact copy of its query input before the
provider ran.  The provider could see demonstration pairs and query inputs, but
not query labels.  This is a protocol repair, not an exact reproduction of the
released paper script.

The two scored cohorts are disjoint ARC-AGI-2 training-task development sets.
They exclude all ARC-AGI-1 IDs and every task ID in the explicit exposure
registry.  Predictions, syntax policy, ranking, provider contract, baseline,
and object/code comparison were frozen before visual gold scoring.  A failed
v1 overlap cohort and strict-v2 interface failure were invalidated before gold;
neither contributes a score.

The checkpoint says it used ARC training data plus ReARC, but the original
training dataset contents were not reconstructed.  These results are therefore
an exposure-audited development estimate, not a private-evaluation score and
not proof of checkpoint-level non-contamination.

## Results

| Endpoint | v2.1 pilot | v3 confirmation | Aggregate |
|---|---:|---:|---:|
| Tasks | 20 | 11 | 31 |
| Frozen base raw/selectable | 0 / 0 | 0 / 0 | 0 / 0 |
| Visual raw coverage | 7 | 6 | 13 (41.9%) |
| Visual selectable / pass@2 | 4 | 4 | 8 (25.8%) |
| Visual pass@1 | 4 | 4 | 8 |
| Unique raw over base | 7 | 6 | 13 |
| Unique selectable over base | 4 | 4 | 8 |
| Raw candidates | 13,260 | 6,120 | 19,380 |
| Valid / rejected | 13,062 / 198 | 5,989 / 131 | 19,051 / 329 |
| Provider GPU seconds | 2,874 | 1,713 | 4,587 |

The aggregate 95% Wilson interval is 13.7%-43.2% for selectable coverage and
26.4%-59.2% for raw coverage.  The v3 gate replicated both nonzero unique
selectable coverage (at least 1/11) and the preregistered stronger small-cohort
criterion (at least 2/11).  It does not replace the required 100-task gate.

The visual raw pool contains thirteen exact query-level hits.  Eight are rank
1; the other five are ranks 3, 5, 6, 14, and 16.  Consequently pass@2 equals
pass@1 and uses only 8/13 of the task-level raw oracle coverage.  Frequency is a
good selector for high-mass solutions, but it discards every correct minority
hypothesis in this sample.

Among the 24 raw-absent queries that still have a same-shape candidate, 12 have
a best candidate within 5% Hamming distance and 14 within 10%.  This is
gold-derived diagnosis only; it is not a query-blind repair result.

## Posterior diagnosis and bridge result

Across the two cohorts, 27 incorrect same-shape queries support pixel-error
localization analysis:

- mean AUROC: 0.851;
- median AUROC: 0.942;
- fixed top-10% disagreement mask: 0.598 macro recall, 0.385 precision, and
  5.87x recall lift over a random mask; and
- fixed top-5% mask: 0.410 recall, 0.440 precision, and 7.95x lift.

The localization signal independently passed in v2.1 and v3.  This supports
using the posterior as a failure-location certificate.

The preregistered `posterior_consensus_compose` action then tested whether
per-pixel marginal voting was already a useful repair.  It triggered on 26/26
pilot queries and produced a novel content-addressed grid on 11, but produced
zero novel exact candidates and zero unique selectable recoveries.  It did not
advance to a matched-cost restart comparison.  Replay and content hashes
passed, so this is an action-semantics failure rather than evidence of a broken
implementation: marginal pixel modes discard object and program correlations.

## Relation to the original motivation

The original proposal was:

```text
masked/visual generation -> feature or hard-rule filtering -> local repair
```

The first arrow now has evidence.  The visual distribution supplies independent
coverage and a localized uncertainty field.  The second arrow remains missing:
neither frequency top-2 nor pixel consensus turns low-mass or near-miss samples
into verified solutions.  The result supports functional specialization but not
stateful switching, because no typed cross-representation action has yet
recovered a task.

Recent strong ARC systems clarify the distinction.  VARC and recurrent visual
models improve a single candidate distribution.  ARCgentica, AB-MCTS, and
modality-driven search spend large inference budgets on code or multimodal
hypothesis generation and selection.  USRL improves an internal
understanding-solving loop.  VLSR assigns vision and language different roles.
These systems raise the Solver-track bar, but none by itself validates this
project's proposed contribution: content-addressed failure certificates that
cause legal, budgeted frontier changes and beat equal-cost restart.

## Highest-value next gate

The next bridge should preserve whole-object and whole-program correlations:

```text
leave-one-demo-out visual posterior
  -> known demo error/canvas/object certificate
  -> candidate-conditioned object graph or AST holes
  -> mask-constrained object_rematch / canvas_reinfer / fill_ast_hole
  -> demo-exact verification
  -> novel frontier after parent-pool subtraction
```

Use ARC-GEN or ARC-TGI to create family-disjoint tasks with known latent factors.
Calibrate certificates only on held-out demonstration outputs, which are legal
solver inputs, and apply the fixed mapping to queries.  Cluster visual samples
by canvas, palette, connected components, topology, D4-canonical shape, and
relative-object relations.  Treat a visual grid as an abductive program target:
search for the shortest typed program that exactly fits every demonstration and
emits that candidate on the query.  The posterior proposes *where and which
representation to search*; hard execution still decides validity.

Compare against frequency top-2, structural-diversity top-2, the failed pixel
consensus action, and equal native-cost cold restart.  Retain the existing hard
gates: at least 5 unique repairs per 100 near misses, positive
`novel_frontier_count`, predictable residual intervention, and at least +3/100
before any controller training.

## Content-addressed evidence

- v2.1 score result: `e38947056dd4ed81c28c47301ca946a2e75576ed6609e114e9b47503300e3e84`;
- v3 score result: `97f605f13dad6bb9e5cea6fbefdf36eb9763c7d418cea0f74355bbcb3981491c`;
- v2.1 posterior audit: `fe5e746e07c24181ccfa5861e8ad8e8f41d8e64629ee0e084f05c9c45d745cc2`;
- v3 posterior audit: `4d2d69476b1e515931642c0ecc8ad109c53e4502679e0f886e666640120854a3`;
- pixel-consensus bridge: `944f74a7dca1a0df6288415ea64d67908c2e19841ad5f987d08b3e5374fab2de`;
- two-cohort aggregate: `7ee3983d48b9e72f46f344eddfc92eb74abeca2296943572cc091e8d68da6459`.

The full summaries, freezes, validation receipts, raw compact provider packages,
and SHA-256 manifest are under
`results/visual_provider_query_blind_20260807/`.
