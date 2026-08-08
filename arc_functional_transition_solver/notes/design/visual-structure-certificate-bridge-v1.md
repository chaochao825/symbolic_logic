# Visual structure certificate bridge v1

## Decision question

Can a query-blind, demo-derived structural contract use the VARC whole-grid
posterior to (a) select a complementary minority hypothesis within pass@2 and
(b) emit a deterministic typed representation diagnosis, without composing
pixels or independently composing objects?

This is a bounded bridge gate. It does not test a learned controller, novel
candidate coverage, typed repair success, or a brain-like switching claim.

## Evidence that opened the gate

The frozen 31-task development set established two distinct facts.

- The visual provider has 13/31 raw and 8/31 selectable task coverage.
- On 27 incorrect same-shape queries, posterior disagreement localizes top-1
  errors (median pixel AUROC 0.942; top-10% recall lift 5.87x).

Two marginal-composition actuators failed on outcome-exposed development data.

- Pixel consensus: 11 novel frontiers, zero new exact candidates.
- Connected-object consensus exploratory sweep: 38 queries, 380 fixed
  connectivity/support variants, 30 queries with a novel frontier, zero novel
  exact candidates and zero novel candidates with a gold-matching object
  signature.

The second result rules out the narrow explanation that pixel granularity alone
caused failure. Independent posterior marginals still discard relations even
when the marginal unit is a connected object.

A single, fixed structural-ranker recipe was then evaluated on the same exposed
development data. Pure structural top-2 merely exchanged one success for one
failure (8/38). Keeping frequency top-1 and selecting a structurally distinct
second whole-grid sample retained all eight frequency successes and recovered
one additional query (9/38). This result is hypothesis-generating only.

## Related-work-derived constraint

The candidate preserves a complete sampled grid. This follows the released
VARC inference rule, where views vote only when complete output grids are
identical, rather than using per-pixel marginal voting. It also instantiates the
division suggested by vision-language ARC work: vision supplies abstraction and
verification evidence, while an exact representation remains responsible for
execution. Execution-trace-guided program synthesis and repair motivate making
the diagnosis typed and replayable instead of storing an unstructured score.

## Frozen candidate

For each query:

1. Rank valid visual samples by the existing frequency, first-emission, and
   canonical-grid order. Candidate 1 is unchanged.
2. Extract a deterministic 4-connected object-transition signature from each
   demonstration and retain only fields identical across every demonstration.
3. Score each complete visual sample by the number of retained fields matched
   by its query input-to-candidate transition.
4. Candidate 2 is the highest-scoring complete sample different from candidate
   1. Ties use frequency, first emission, canonical signature, then canonical
   grid JSON.
5. Compare candidate 1 with the stable contract and compile one certificate:
   shape/canvas violations map to `canvas_reinfer`; background-role violations
   map to `reparse_background`; object-count, component-shape, palette, or
   foreground-transition violations map to `object_rematch`. This v1 gate does
   not fabricate `fill_ast_hole` without a candidate AST.

No grid is cropped, recolored, coerced, repaired, or synthesized. The method can
improve selectable utilization but cannot increase raw oracle coverage.

## Frozen confirmation data

The confirmatory episodes are generated from ARC-GEN commit
`a15cbdb44c776610aeeb9f487a06af875d3d0878`, with ARC-AGI submodule commit
`399030444e0ab0cc8b4e199870fb20b863846f34`.

Selection seed: `structured-visual-bridge-confirm-v1-20260809`.

The selection population is ARC-GEN families absent from all ARC-AGI-1 task
IDs and from the 31 visual-provider development families. Families are sorted
by `sha256(seed + NUL + family_id)`, ascending. The first 12 are frozen:

```text
bae5c565 9b30e358 8dab14c2 af726779 87ab05b8 9841fdad
78e78cff 14b8e18c 6bcdb01e 57edb29d e39e9282 b5bb5719
```

Each episode has three demonstrations and one query. Pair seeds are the first
64 bits of `sha256(seed + NUL + family_id + NUL + pair_index)`. Any generator
exception, non-ARC color, empty/non-rectangular grid, dimension above 30, or
duplicate pair makes the family invalid. An invalid family is reported and is
not replaced after endpoint inspection. The split is episode-disjoint and
bridge-design-family-disjoint from the 31 development tasks; it is not claimed
to be globally family-unseen by the wider project.

The query output is replaced by an exact input-copy sentinel before VARC can
read the episode. Raw prediction files, hashes, syntax validation, and the
candidate artifact are frozen before gold scoring.

## Fixed provider and cost

- VARC checkpoint SHA-256:
  `c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3`.
- The v3 configuration is reused exactly: 100 TTT epochs, seed 42, one
  replica, ten attempts, five perspectives, nine color permutations, 64x64
  canvas, 2x2 patches.
- Invalid provider samples use the frozen `reject` syntax rule and are never
  repaired.
- GPU cap: three workers, at most 2,400 seconds per family and 8,000 aggregate
  GPU-seconds. Stop on any nonzero task exit, checkpoint mismatch, sentinel
  failure, artifact overwrite attempt, or missing valid query candidate.
- The structural scorer is CPU-only. Its native cost records candidate
  signatures and feature comparisons.

## Endpoints and guards

Primary endpoint: generated-episode task pass@2 difference between the frozen
frequency top-2 and the hybrid frequency-top-1/structural-top-2 policy.

Secondary endpoints:

- query pass@1/pass@2 and raw oracle coverage;
- unique recoveries and regressions relative to frequency top-2;
- gold rank of recovered minority hypotheses;
- certificate type counts and content-address replay;
- certificate changes after deterministic cross-task contract shuffling;
- CPU feature-comparison count and unchanged provider GPU cost.

Hard guards:

- candidate 1 is byte-identical to frozen frequency top-1;
- every selected candidate is already in the frozen raw pool;
- candidate artifacts replay exactly from blind episodes and raw predictions;
- no query gold is present or read before the candidate artifact is frozen;
- the controller remains frozen.

## Outcome mapping

- **Pass:** at least one unique task recovery, positive net task pass@2, zero
  guard violations, and at least one certificate changes under the frozen
  contract shuffle. This licenses a separate LODO typed-repair gate.
- **Boundary:** query pass@2 improves but task pass@2 does not, or certificate
  interventions change as predicted without endpoint gain. Keep the mechanism
  as a diagnostic probe; do not claim solver improvement.
- **Null:** no unique query recovery and no task gain. The structural contract
  is not a useful selector under this scope.
- **Adverse:** net pass@2 decreases or a cost/validity guard fails. Park this
  candidate.
- **Invalid:** any leakage, sentinel, freeze, replay, checkpoint, generator, or
  evaluator violation. Repair infrastructure once without changing the
  candidate, then rerun within the same cost cap.

No threshold, feature, tie-break, task, or provider setting may be changed after
the candidate artifact is frozen. A null/adverse result is not followed by a
new ranker variant in this gate.
