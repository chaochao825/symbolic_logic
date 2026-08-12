# Anchor-rasterized delta v0.3 gate

## Decision question

Can one finite, query-blind `Grid x Grid x Scene -> Grid` node turn the dominant
single-color spatial residual class into exact leave-one-demo-out support
predictions and novel natural-task candidate outputs under the existing 2,048
native-trial reservation?

This is an outcome-exposed development gate.  It may select whether the node is
worth prospective confirmation, but it cannot support an ARC-AGI generalization
claim.

## Active candidate

Insert one node on the current render edge:

```text
parent render grid ----\
original input grid ---- AnchorRasterizedDelta -> output grid
persistent parse scene -/
```

The effect is factorized into a support program and a color delta:

```text
anchors = cells of one non-background color in the persistent scene
mask    = one registered rasterizer(anchors)
output  = replace source_color by target_color only inside mask
```

The mask grammar is frozen to one non-compositional primitive:

```text
Stencil  = one cardinal neighbor or one anchored 2x2 quadrant
Project  = full rows or full columns containing an anchor
BBoxFill = anchor bounding box with inset 0 or 1
```

There is no union, intersection, complement, recursion, arbitrary coordinate,
learned mask, reflection, checkerboard, copy, count, or multi-color delta.  The
same node parameters must satisfy every demonstration.  Input, parent output,
and expected output must share a canvas; otherwise the language is proved
incompatible rather than silently aligned or padded.

The source and target colors are compiled from a single shared demonstration
delta.  The only searched dimensions are ten anchor colors and twelve mask
specifications, giving exactly 120 programs per parent and at most 480 for four
parents.  The remaining reservation is explicit padding.

## Incumbent and fixed inputs

- Parent/cold freeze:
  `results/counterfactual_transition_v4_arc_tgi_dev_20260812/candidate_freeze_a.json`.
- Development challenges and solutions:
  `results/arc_tgi_arcmini_cohort_v2_20260810/development_{challenges,solutions}.json`.
- Cohort ID:
  `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`.
- Reconstruct exactly the first 2,000 object/code trials and at most four
  execution-valid parents per task.
- Preserve every parent program ID, certificate ID, blind task hash, incumbent
  output bundle, and frozen equal-cost cold candidate.

## Primary and diagnostic endpoints

The primary development endpoint is unique exact recovery over the incumbent
plus frozen cold restart.  Promotion requires at least 3/50 tasks.

Before query scoring, report:

- exact bounded abstract/concrete agreement on exhaustive controlled grids;
- strict serialization and content-address round trips;
- leave-one-demo-out predictable folds and exact folds;
- strict LODO task success, requiring every fold to infer a non-empty version
  space and predict the held-out output exactly;
- identity, global-recolor, and whole-component-recolor LODO baselines;
- reachable parents/tasks, demo-exact candidates, and content-novel query
  frontiers.

LODO scores are audit-only and cannot influence query candidate generation or
ranking.  Query targets remain structurally unavailable until the candidate
freeze has been written twice and compared byte-for-byte.

## Guards

1. The old recolor and scene-pipeline schemas and executions remain unchanged.
2. Candidate construction accepts `BlindTask` only.
3. The new node declares render, original-input, and parse-scene dependencies.
4. Every complete candidate must agree under repeated full replay and strict
   JSON reconstruction.
5. Content output, not program identity, defines frontier novelty.
6. `novel_frontier_count == 0` forbids any frontier-changing claim.
7. All mask programs, demo executions, query executions, padding, and LODO-only
   audit cost are logged separately.
8. No grammar amendment is allowed after reading development query scores.

## Outcome mapping and stop rule

- Any parent/certificate/hash/replay/schema/budget/query-blind violation:
  **invalid**, repair once without changing the grammar and rerun.
- Controlled semantics fail: **engineering failure**, no scientific update.
- Controlled semantics pass but fewer than three strict LODO task successes:
  **sensor null**; do not run or interpret query utility as a mask-language test.
- LODO passes but no novel query frontier: **boundary**, support inference exists
  but does not extend this incumbent.
- Novel frontier exists but unique recovery is 0/50: **natural-utility null**.
- Unique recovery is 1-2/50: **boundary natural utility**.
- Unique recovery is at least 3/50 and exceeds cold: freeze the implementation
  and preregister one family-disjoint 100-task confirmation requiring at least
  5/100 unique recoveries.

Stop after this one grammar and one valid development run.  Do not add another
mask primitive, train a router, or materialize a reserve in response to the
observed score.
