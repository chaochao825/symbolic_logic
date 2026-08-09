# Relational delta-program development gate v0.2

## Status and decision question

This is an outcome-exposed **development** gate.  The frozen 50-task
ARC-AGI-2 public-training cohort and the diagnostic tasks `7acdf6d3`,
`320afe60`, and `3ad05f52` have already been inspected and scored.  They may
test implementation semantics and representation reachability, but cannot
support a held-out generalization claim.  ARC-AGI-2 `data/evaluation` remains
outside this iteration.

The bounded question is:

> Can a query-blind, content-addressed delta-program AST represent at least two
> of the three observed failure modes and turn a demonstration failure into a
> legal single-slot edit whose output is exact on every demonstration, while
> rejecting background-dominated pseudo-near-misses?

Passing this gate does not authorize controller training or a visual-posterior
experiment.  A later confirmation needs a newly frozen semantic-family source.

## Frozen representation

The opt-in representation is `afts-relational-delta-dsl/v0.2`.  It is not
inserted into the legacy object/code enumeration, so historical program order,
candidate IDs, checkpoints, artifacts, and numerical results remain unchanged.

Every program has the following typed path:

```text
parse / role assignment
  -> n-ary relational correspondence
  -> target component / coordinate / region
  -> typed erase and write masks
  -> add | erase_add | recolor_component | fill_relation_region
```

The first bounded relation vocabulary is:

- `actor_area_to_row_interiors`: relate one payload color group to all
  support-component row-interior regions and retain the unique area match;
- `component_to_horizontal_border`: relate each actor component topology
  jointly to the left and right canvas anchors; solid or holed components map
  left, while components with only an open indentation map right;
- `support_bbox_centers`: derive coordinate targets from support-component
  bounding boxes;
- `support_enclosed_regions`: derive complete topological holes of support
  components as region targets.

Color roles are inferred per input (`minority_foreground`,
`majority_foreground`, or `only_foreground`) rather than fixed to ARC color
identities.  All query enumeration and execution receive a `BlindTask` with
demonstration outputs and query inputs only.

## Typed-mask and topology contract

Execution emits separate, sorted `erase_mask` and `write_mask` coordinates,
the rendered write colors, all instantiated correspondence members, and a
node-level trace.  A successful execution must satisfy the operation-specific
topology contract:

- `add`: no source erasure and every written coordinate is a complete typed
  target;
- `erase_add`: erase the complete actor role and write a complete relation
  region with equal cardinality;
- `recolor_component`: erase complete actor components and place color-changed
  copies with identical normalized shapes, without clipping or conflicting
  writes;
- `fill_relation_region`: preserve the source and fill complete relation
  regions, never a pixel subset.

An invalid or partial target produces no candidate.

## Near-miss eligibility

For each demonstration, let `G` be the input-to-gold delta mask and `P` the
input-to-parent delta mask.  A natural or controlled parent is eligible only if:

1. every execution is valid, same-canvas, non-identity, and topology-valid;
2. parent mismatches are no greater than identity mismatches on every
   demonstration, with a strict aggregate improvement;
3. micro `precision(P, G) >= 0.5` and micro `recall(P, G) >= 0.5`;
4. the compiled action names existing typed slots and every emitted child
   differs from the parent in exactly one such slot;
5. at least one child ID is absent from the frozen prefix, so
   `frontier_changed == (novel_frontier_count > 0)`.

The identity, delta, topology, and certificate checks use demonstrations only.
Query gold is unavailable until the candidate artifact has been serialized and
replayed byte-for-byte.

## Frozen development budgets

- complete relational-delta grammar cap: 256 programs per task;
- natural parent prefix: first 64 programs ordered by description length then
  program content ID;
- typed action: 16 content-novel single-existing-slot variants, ranked only by
  demonstration residual quality and then content ID;
- equal-cost cold restart: the next 16 programs after the same prefix;
- both action arms reserve 16 trials and execute every trial on every
  demonstration and query input; absent variants replay the parent as padding
  and cannot emit candidates;
- final candidates are demo-exact and ranked by description length then program
  ID; pass@2 and oracle union remain separate.

## Development outcomes and stop rules

The semantic unit gate requires exact, replayable fixtures for all four render
operations, strict JSON round trips, topology rejection, identity-normalized
near-miss rejection, and single-slot novelty accounting.

The exposed representation gate passes only if at least two of the three
diagnostic tasks have a demo-exact program.  Query-exact results are reported as
descriptive development evidence.  The exposed 50-task audit separately reports
unique coverage over the frozen legacy/v0.1 candidate pool.

The natural repair gate requires at least two eligible natural parents, at
least one typed recovery, and a positive net recovery over equal-cost restart.
Failure is classified as:

- implementation failure if a declared semantic fixture cannot replay or an
  accounting/integrity invariant fails;
- representation failure if fixtures pass but fewer than two diagnostic
  transformations are demo-exact;
- search/actuator failure if exact programs exist but no legal novel edit can
  reach them;
- generality null if the diagnostic representation gate passes but the exposed
  50-task frontier adds no unique exact candidate.

Regardless of the development outcome, GRU/MLP/bandit/diffusion controller
training and visual LODO routing remain frozen until a genuinely fresh
confirmation source passes independently.

## Pre-artifact semantic preflight amendment

The first executable preflight falsified an earlier draft interpretation of
`component_to_horizontal_border` as nearest-edge placement: several
demonstration components are geometrically nearer the left edge but move right.
Before any candidate-freeze artifact was created, the relation was corrected to
the topology-role rule above.  This amendment is part of outcome-exposed
representation development, not a confirmatory result; the failed nearest-edge
interpretation remains recorded here rather than being silently omitted.
