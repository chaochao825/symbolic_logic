# Counterfactual Transition v4 preregistration

## Decision question

Under exactly the v3 parent prefix and one matched native reservation per arm,
can a bounded transition over one to three persistent scene-AST regions create
at least five query-blind output frontiers on the frozen ARC-TGI dev50 cohort?

This is a representation-and-search diagnostic. It is not a leaderboard
evaluation, a learned controller test, a visual-provider test, or evidence of
biological brain-region equivalence.

## Immutable parent and control

- Parent freeze ID:
  `66fb434846843584cde0198be17af637fb9af2d61521e9699ecb63c1f5712269`.
- Incumbent frontier: all output bundles stored in that v3 freeze, inherited
  from Object-Graph Rewrite v2.
- Shared parent search: first 2,000 object/code trials, at most four
  execution-valid scene near misses.
- Transition reservation: 512 trials per parent, at most 2,048 per task and
  arm, with unused trials logged as padding.
- Cold control: the already frozen v3 unconditioned continuation with the same
  2,048-trial reservation.
- Emitted candidates: at most 32 per task and arm, ranked by description length
  and transition content ID, then deduplicated without changing rank order.
- Tasks may execute in parallel, but task order is canonical and worker count is
  recorded in the freeze. Serial/parallel semantic equality is regression tested.

Regenerated certificate IDs must exactly match v3. This guard prevents a code
refactor or candidate-prefix change from silently changing the experiment.

## Typed transition contract

The dependency order is:

```text
parse -> assignment(correspond + select) -> operate -> canvas -> render
```

A transition changes one, two, or three distinct nodes. The declaration must
equal the serialized parent/child differences in canonical dependency order.
Replay starts at the first changed node; all earlier state IDs are reused and
the entire dependent suffix is re-executed. Non-contiguous declarations are
legal because every intervening node is still re-executed.

## Frozen arms

### A: `distance_tiered`

Partition the grammar by one-, two-, and three-node distance from each parent.
Draw deterministically in a `1:2:1` cyclic allocation. This arm asks whether a
generic bounded neighborhood contains useful exact transitions without using
the certificate to select a node set.

### B: `typed_semantic_bundles`

Map the diagnosed node to a frozen list of legal companion sets, such as
`assignment+operate`, `assignment+canvas`, `operate+canvas`, or
`parse+canvas`. Allocate evenly across bundles. This arm asks whether a typed
failure certificate narrows the combinatorial neighborhood productively.

### C: `residual_beam`

Spend one quarter of the per-parent reservation on deterministic single-node
probes, scheduling the diagnosed node first. Rank probes only by demonstration
exact count, shape agreement, total mismatch, validity, description length,
and content ID. Retain eight probes, then spend the remaining reservation on
two- or three-node semantic bundles that preserve one retained probe's changed
payload. Query labels never enter this ranking.

This is a fixed within-task search policy, not a learned router. Its result
cannot establish residual causality without subsequent empty, shuffled, and
injected-residual controls.

## Verification and leakage guards

1. query-gold-free construction of all arms in one candidate freeze;
2. exact match to the v3 blind task hashes and certificate sequence;
3. strict changed-node declaration and parent-trace bridge;
4. demonstration exactness before any query execution is emitted;
5. equality of suffix replay, fresh stateful replay, and legacy execution;
6. AST and transition content-address round trips;
7. query-output novelty against the complete frozen incumbent;
8. byte-identical independent freezes before scoring; and
9. explicit per-arm realized, padded, reserved, trace-reuse, node-set, and
   residual-improvement accounting.

## Outcome mapping

- Replay, hash, certificate-prefix, leakage, or reservation failure:
  **invalid engineering run**, no scientific update.
- Fewer than 5/50 novel-output opportunities for every individual arm:
  **bounded-transition null**; do not materialize reserve or train a router.
- At least 5/50 for one arm: open a separately frozen family-disjoint reserve.
- Development exact score is diagnostic only and cannot select a winning arm.
- A reserve pass additionally requires at least 2/18 novel opportunities,
  at least 2/18 unique exact recoveries, and at least one more recovery than an
  equal-cost cold restart.
- A residual-beam advantage requires later residual interventions that cause
  predicted, selective action changes; otherwise it is only a search heuristic.

## Scope of a negative result

This gate samples at most 2,048 programs from a much larger multi-node product.
A null rejects these frozen allocations over the current scene variables and
operators. It does not prove that all multi-node programs are unreachable, that
persistent state is useless, or that functional switching is impossible.
