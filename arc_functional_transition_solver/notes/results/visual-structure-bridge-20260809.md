# Visual structure certificate bridge v1

## Decision

The frozen confirmation gate is **null**. The visual provider retained strong
complementary candidate coverage on the generated confirmation families, but
the hand-engineered structural contract did not improve use of that coverage.
The controller remains frozen and this result does not authorize the LODO typed
repair gate defined in the preregistration.

## Frozen endpoint

The 12 ARC-GEN families were fixed before provider execution. Each episode had
three demonstrations and one query. VARC used the previously released
checkpoint and the same 100-epoch TTT, seed-42, 10-attempt configuration as the
earlier query-blind run. Candidate 1 remained the frequency top-1. Candidate 2
was the highest-scoring different whole-grid sample under the stable
demonstration structural contract.

| Endpoint | Frequency | Hybrid | Delta |
|---|---:|---:|---:|
| Task/query pass@1 | 7/12 | unchanged candidate 1 | 0 |
| Task/query pass@2 | 7/12 | 7/12 | 0 |
| Unique recovery | -- | 0/12 | 0 |
| Regression | -- | 0/12 | 0 |

The raw visual oracle covered 8/12, while frequency pass@2 covered 7/12. The
frozen legacy portfolio covered 0/12 raw or selectable tasks, so visual raw and
selectable coverage were unique on 8/12 and 7/12 generated families,
respectively. This is evidence for complementary visual candidate generation,
not a hidden ARC-AGI-2 score.

The sole visual selection gap was `e39e9282`: the exact output was raw frequency
rank 4, but structural rank 7. The bridge did not select it.

## Why the bridge was null

This is an algorithm/representation failure, not a run or replay failure.

1. Seven of eight raw-covered tasks were already correct at frequency rank 1,
   leaving only one task on which a selector could improve pass@2.
2. The stable contracts had a median of only one field that varied across the
   query candidate pool. Consequently, the hybrid second candidate was exactly
   the frequency second candidate on 10/12 tasks.
3. On `e39e9282`, all three demonstrations had the 4-connected component-count
   relation `less`. The query gold had relation `equal`. A wrong sample at raw
   frequency rank 20 matched all ten retained fields and became the structural
   second candidate, while the exact rank-4 sample violated the overfit
   component-count invariant.
4. Certificates were absent on 10/12 tasks and emitted `object_rematch` on only
   two. Contract shuffling changed 11/12 actions, showing that the compiler is
   causally sensitive to its structural input, but action sensitivity did not
   produce action utility.
5. Four failures were outside the raw pool. Three nevertheless had natural
   same-shape near misses: `8dab14c2` differed by one pixel, `6bcdb01e` by nine,
   and `b5bb5719` by five. `57edb29d` was a canvas/shape failure. A selector over
   existing grids cannot recover any of them.

The causal summary is therefore:

```text
coarse stable object marginals
  -> median one candidate-discriminative field
  -> frequency tie-break dominates on 10/12 tasks
  -> the only selection-gap gold violates a demo-only component invariant
  -> zero pass@2 recovery
```

The result sharpens the earlier pixel- and object-consensus failures. Preserving
whole-grid samples avoids destroying joint support, but a coarse marginal
object signature is still not a sufficient model of correspondence or program
state.

## Integrity and implementation status

- 12/12 provider tasks completed; zero task failures.
- 6,120 raw samples were emitted; 6,015 were valid and 105 were rejected by the
  frozen syntax rule.
- Provider-side sentinel mismatches: 0.
- Aggregate provider time: 1,738 GPU-seconds; maximum family time: 236 seconds,
  below the 8,000/2,400-second caps.
- The pre-gold frozen-prediction and candidate artifacts replayed byte-for-byte.
- Frozen candidate ID:
  `66c99bc5a750941628bddd76a0d56bd876f0e5295b020fe6b942f77284cc6cd6`.
- Frozen result ID:
  `b43e70a8a24526bf95e4accac211aedf962f7a6e61fce61621d4824b52fc77cd`.

A post-outcome audit found that `shape_delta` was represented by a Python tuple
in memory and a list after JSON reload. Candidate construction and both frozen
replays recomputed the contract in memory, so the endpoint was unaffected. The
field now uses a JSON-native list. Replaying with the fix produced byte-identical
frozen prediction and candidate files; the metric artifact was not recomputed.

The provider was query-blind and the method, features, and tie-breaks were fixed
before cohort execution. The wider process was not analyst-blind: generation
created gold on the 210 server, and an integrity audit inspected output shapes
and color counts before the final candidate artifact was written. No algorithm
setting was changed after that inspection. This result should therefore be
treated as a preregistered generated-family confirmation, not a hidden-set
estimate.

## Research implication

The highest-value insight survives in a narrower form:

> Visual posterior disagreement is useful as evidence about where a concrete
> hypothesis fails, but neither pixel/object marginal composition nor an
> unconditioned structural similarity score supplies the missing joint
> correspondence.

The next independent gate should not reweight this ranker on the same families.
It should use fresh, family-disjoint episodes with candidate-conditioned traces:

```text
LODO visual posterior + executable parent trace
  -> earliest canvas/object/mask/AST violation
  -> typed existing-slot edit
  -> novel-frontier check
  -> demo-exact verification
  -> equal-cost cold-restart comparison
```

Before that gate, object identities must include relational correspondence
rather than only 4-connected color components, and `fill_ast_hole` must require
an actual executable AST node. Controller training remains disallowed until a
typed repair has unique recovery and residual interventions change actions with
positive endpoint utility.

Related-work reasoning and evidence boundaries are recorded in
`notes/literature/visual-posterior-to-typed-search-20260809.md`. The immutable
machine-readable evidence is under `results/visual_structure_bridge_20260809/`.
