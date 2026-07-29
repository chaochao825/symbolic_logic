# Object/code scene-AST gate v0.3 (2026-07-29)

## Decision

The representation-only gate passes narrowly: the new object/code provider adds
exactly 3/100 unique selectable tasks on the frozen offset-100 development slice.
This meets the provider-specific threshold, but it does not meet the broader
solver or repair thresholds. Baseline-plus-provider selectable union is 12/100,
below the required 25/100, and natural typed repair recovers 0/100 cases.

The controller therefore remains frozen. No router, code model, masked model, or
learned proposal prior was trained or called. A separately pre-registered static
code-proposal experiment may now be designed as candidate-language work, but this
result does not authorize a controller or masked-neural claim.

## Change under test

Source commit `6f9ca2ceb9f1e45816739fdb717f46ac0f5b3efd` adds a
connected-component scene graph and a replayable six-stage AST:

1. parse monochrome components, foreground components, or color groups;
2. correspond objects by shape, size, topology, and relative position;
3. select typed object roles;
4. apply crop, copy, count, arrange, or compose operations;
5. infer an explicit output canvas, background, size, and padding; and
6. render with a declared conflict policy.

Programs carry provenance-independent SHA256 IDs, node-level execution traces,
and flattened typed-hole paths such as `ast.select.role` and
`ast.canvas.padding`. The previous v0.1/v0.2 JSON schemas and execution behavior
remain supported, and their enumeration is an identical prefix under the same
20,000-trial per-task cap.

Before source freeze, a demonstration-only analysis inspected demonstrations and
query inputs, but not query outputs. It identified three reachable families:
`0b148d64`, `23b5c85d`, and `3f7978a0`. This was a reachability prediction, not
oracle evidence. Query outputs were used only after each content-addressed pool
had been frozen.

## Development result

| Metric on the identical offset-100 tasks | v0.2 | v0.3 | Change |
|---|---:|---:|---:|
| grammar programs | 4,300 | 2,227,846 | +2,223,546 |
| executed program trials | 4,300 | 1,430,098 | 332.58x |
| demo executions | 13,544 | 4,481,426 | 330.87x |
| query executions after pool freeze | 0 | 538 | +538 |
| emitted candidates | 76 | 764 | 10.05x |
| retained demo-exact programs | 0 | 4 | +4 |
| provider raw/selectable/pass@2 | 0/0/0 | 5/4/4 | +5/+4/+4 |
| new unique selectable coverage | 0/100 | 3/100 | +3 |
| baseline + provider selectable union | 9/100 | 12/100 | +3 |
| natural typed-repair recoveries | 0 | 0 | 0 |

The 3/100 gain is a representation-reachability result, not an efficiency result:
it required about 333 times as many provider trials as v0.2. The baseline union
comparison is a coverage audit, not a matched-cost solver comparison.

The provider raw-oracle cases are:

| Task | Frozen program behavior | Selectable | Unique over baseline |
|---|---|---:|---:|
| `0b148d64` | color groups, `least_holes`, crop, bbox canvas | yes | yes |
| `23b5c85d` | color groups, `smallest_area`, horizontal flip and crop | yes | yes |
| `3f7978a0` | color groups, `smallest_area`, crop with asymmetric padding | yes | yes |
| `a740d043` | select all, crop to bbox | yes | no |
| `ed36ccf7` | rotate-270 crop near miss | no | no |

The first three exactly match the demonstration-only reachability targets. The
four selectable hits are demo-exact and all generalize to the public query. The
`ed36ccf7` candidate happens to match the query but has only 0.805556 demo
agreement, so hard verification correctly excludes it; it is not a solve.

All four exact programs use the crop lane. Object roles and explicit canvas
inference produced the observed gain, but correspondence features and the copy,
count, arrange, and compose lanes contribute no exact program in this run. The
current AST is a typed multi-stage pipeline with one object-operation node; it is
not yet a general sequential program such as crop-then-recolor-then-arrange.

## Failure attribution

The 100 tasks decompose as follows:

| Class | Count | Interpretation |
|---|---:|---|
| solved | 4 | demo-exact, selectable, query-correct |
| representation failure | 63 | full declared grammar exhausted without a demo-exact program |
| search-budget failure | 32 | 20,000-trial prefix exhausted before an exact program |
| selectability/verifier failure | 1 | query-matching near miss rejected by demo verification |

Sixty-six tasks exhaust the grammar: 63 fail representationally and three are
the unique solves. Thirty-four tasks hit the cap: 32 remain budget-indeterminate,
`a740d043` is solved inside the prefix, and `ed36ccf7` contains only a
non-selectable query hit. It would therefore be incorrect to label every miss as
out-of-language, but it is also incorrect to attribute the aggregate result to
the router or implementation.

The narrow object-role/canvas hypothesis is supported. The broader representation
hypothesis is not: union remains 13 points below its gate, 63 exhaustive tasks
remain out of language, and the Cartesian grammar spends most of its cost on
programs that do not survive demo replay. The 32 capped tasks additionally show
that enumeration order and proposal quality are now first-order problems.

## Natural repair and frontier integrity

The audit constructs 100 natural near-miss cases. Ninety have matched typed and
cold-restart executions; ten correctly abstain because no novel program remains.
The typed actions are:

| Action | Cases | Frontier-changing | Novel recovery |
|---|---:|---:|---:|
| `canvas_reinfer` | 8 | 8 | 0 |
| `object_rematch` | 34 | 34 | 0 |
| `reparse_background` | 58 | 48 | 0 |

Across the repair audit, 16,164 raw frontier programs become 15,984 novel
content-addressed programs after parent-pool filtering; 342 retained candidates
are emitted. Nevertheless, typed repair, equal-trial cold restart, and unique
typed repair all recover zero query oracles.

The required invariant holds for every action:

```text
frontier_changed == (novel_frontier_count > 0)
```

Controlled injected faults trigger the expected typed actions and beat their
matched cold controls, so the compiler and accounting plumbing work. Their
success does not rescue the natural-repair hypothesis. On natural parents, the
current diagnoses change the search frontier but still do not reach a useful
candidate region. This is a repair-semantics/theory failure under the tested
action language, not evidence of an execution bug.

## Correct claim boundary

1. **Implementation and protocol:** supported. Full regression is 400 passed,
   9 skipped, and 167 subtests passed; targeted scene/object tests are 29/29.
   Legacy replay checks 142 frozen v0.1/v0.2 programs with zero mismatch.
2. **Artifact integrity:** supported. An independent audit recomputes all 100
   pool IDs, 764 program IDs, 13 repair-manifest IDs/references, and the summary
   result ID with zero error; every pool replays exactly.
3. **Object-role/canvas reachability:** supported narrowly by 3/100 unique
   selectable coverage, exactly the pre-registered provider gate.
4. **Broad candidate-language sufficiency:** rejected on development. Selectable
   union is 12/100 rather than 25/100.
5. **Natural typed repair:** rejected for this compiler/action set. It recovers
   0/100 and does not beat equal-trial restart.
6. **Efficiency, learned switching, and neural-provider claims:** unsupported.
   No such experiment was run, and the representation gain used roughly 333x
   the v0.2 provider trials.

## Next gate

Do not train a router. The next phase should first replace Cartesian enumeration
with a query-blind, demo-conditioned proposal policy and evaluate it on an
untouched task slice under a fixed native budget. A static code-model lane is now
eligible only as such a candidate proposal experiment. It must keep generation
query-output blind, freeze artifacts before scoring, expose token and execution
costs, and compare against the same-budget typed enumerator and cold restart.

The highest-value representation changes are sequential typed operations,
constraint propagation from node traces into AST holes, and correspondence-aware
object matching that actually enters exact programs. A follow-up should require
new provider-only unique coverage on an untouched slice, not merely rediscovery
of these three development tasks. Masked-neural work and any learned controller
remain frozen until the selectable-union and natural-repair gates pass.

## Artifact identity

- Result ID:
  `06aec779ce4db6aecc6a5737fb580cea3b1706c880ecd2b0481ff3fd7234d3d4`.
- Result directory:
  `results/object_code_gate_v3_scene_ast_arc1_train_dev_offset100_n100_20260729`.
- Source commit:
  `6f9ca2ceb9f1e45816739fdb717f46ac0f5b3efd`.
- Observed artifact-production window: approximately 67 minutes 15 seconds.

The summary records the clean source provenance, dataset selection, frozen
baseline hash, provider and DSL versions, runtime environment, native cost
ledgers, coverage gates, failure classes, repair controls, and publication claim
boundary.
