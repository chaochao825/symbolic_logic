# RDR-20260812: replace free masks with relational causal footprints

## Status

Accepted as the post-v0.3 research direction.  The anchor-rasterized delta
gate is closed as a valid sensor null.  This decision does not open another
experiment on the outcome-exposed dev50 cohort and does not modify any frozen
candidate, score, or conclusion.

## Decision

Keep proof-carrying representation recruitment, but replace the interface

```text
residual raster -> choose a mask primitive -> recolor
```

with

```text
executable trace
-> minimal typed counterfactual intervention
-> persistent identity/relation effects
-> derived raster footprint
-> affected-subtree replay
-> exact verification
```

The mask is an effect of an intervention, not an independently synthesized
output object.

## Formal object

Let `z_i` be the content-addressed execution state of a parent program on
demonstration `i`, including persistent entities, relations, node outputs, and
cell provenance.  For a typed intervention `a`, define its causal footprint:

```text
Phi(a, z_i) = { cell : Replay(a, z_i)[cell] != Replay(identity, z_i)[cell] }.
```

Let the observed demonstration discrepancy be:

```text
Delta_i = { cell : parent_i[cell] != target_i[cell] }.
```

An action is admissible only when its effect type is compatible with the
failure core, its replay is valid, and the same rule inferred from the other
demonstrations predicts the held-out `Delta_i` and target exactly.  A semantic
extension is recruited only when the old bounded version space is empty, the
extended version space is non-empty, and no smaller registered extension is
non-empty.  Query frontier novelty remains content-based and query-target-free.

This yields three separate gates:

1. **diagnosability** -- a typed intervention is predictable under strict
   leave-one-demo-out evaluation and beats identity/global/component baselines;
2. **frontier change** -- exact demo programs produce query content absent from
   the incumbent DAG;
3. **natural utility** -- the new frontier uniquely recovers targets at lower
   or equal native cost than a frozen restart.

Failure at an earlier gate forbids claims about later gates.

## Minimal engineering contract

The next implementation, if separately preregistered, needs only four new
auditable objects:

1. `PersistentEntityId` and `PersistentRelationId`, stable across node replay;
2. a provenance map from output cells to producing node and structural IDs;
3. a node-level `EffectSummary` over canvas, color deltas, entities, relations,
   and possible raster footprint;
4. a `RelationalFailureCore` that identifies the smallest typed cut whose
   effect summary can cover the observed discrepancy.

Candidate generation then enumerates bounded interventions at the cut, prunes
them by sound abstract effects, replays only the affected suffix, verifies the
full program, and records output-content novelty.  A visual or language model
may propose entity roles, relations, or a cut, but cannot directly write query
pixels or bypass exact replay.

The first natural gate must use a fresh, family-disjoint cohort.  It must freeze
one finite relation/effect language before outcome access, include synthetic
identifiability controls and natural near misses, require strict LODO to beat
the three simple baselines, and compare against equal native-cost cold restart.
No controller is trained until the action has both unique coverage and a
residual intervention effect.

## Relationship to prior work

Object graphs and DSL search are not the contribution: ARGA already provides a
strong, reproducible instance of that combination inside a fixed language.
Shared reflective workspaces and learned scheduling are not the contribution:
ARCANA describes that architecture with dense latent refinement.

The narrower potential novelty is **proof-carrying functional recruitment**:
an execution failure must identify a typed missing effect, the recruited
function must minimally expand semantic reachability, and its causal footprint
and output novelty must survive replay.  This is compatible with the original
brain-inspired motivation as a functional analogy -- specialized operations
are selectively recruited into a shared active computation -- without making
a biological fidelity claim.
