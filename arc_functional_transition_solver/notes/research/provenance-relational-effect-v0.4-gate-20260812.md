# Provenance-aligned relational effect v0.4 gate

## Decision question

Under a frozen controller and the existing 2,048-native-trial reservation, can
an execution failure be compiled into one typed post-render action whose support
is derived from exact source-cell provenance, and can that action provide at
least five unique recoveries per 100 fresh episodes from families disjoint from
all development families?

This gate tests one bridge mechanism.  It does not test a learned router and it
does not by itself support an ARC-AGI solver-performance claim.

## Why this candidate is open

The outcome-exposed 50-task development audit localized 48 parent executions
classified as `anchor_mask_unreachable`.  A read-only opportunity audit found:

- 46/48 parents are `crop` operations on a `bbox` canvas and therefore admit an
  exact source-to-render coordinate map, including the frozen D4 transform;
- selecting whole persistent entities recovers 0/48 parents: the residuals cut
  through an entity or originate from cells treated as parser background;
- aligning anchors only through persistent entities also recovers 0/48;
- keeping the old v0.3 twelve-effect language unchanged, but pulling every
  visible source cell (including a cell with no object entity) through exact
  provenance, makes 13/48 parents and 5/12 affected tasks jointly reachable;
- all five tasks also pass the stricter development LODO criterion below.

Adding `cross`, checkerboard, and canvas-interior effects would raise exposed
development reachability, but those effects are excluded from v0.4.  Keeping
the effect grammar identical to v0.3 makes any prospective change attributable
to the provenance bridge rather than to a larger hand-written DSL.

## Frozen representation and language

The parent execution remains unchanged.  An opt-in adapter adds:

```text
PersistentEntityId / PersistentRelationId
        +
CellProvenance(output cell -> source cell, optional entity/relation IDs,
               producer node, exactness)
        +
EffectSummary(observed single-delta support and provenance coverage)
        +
RelationalFailureCore(minimal surviving typed effect version space)
```

An entity ID is an optional semantic referent, not a prerequisite for cell
identity.  A source cell classified as parser background still has an exact
coordinate origin and may act as a marker.  This distinction is mandatory:
object-only provenance would repeat the observed 0/48 failure.

The only action is one post-render node:

```text
RelationalEffectNode(
    anchor_color,
    effect_kind,
    effect_parameter,
    source_color,
    target_color,
)
```

The finite effect domain is exactly the existing v0.3 domain:

```text
Stencil  = north | south | west | east
         | block_nw | block_ne | block_sw | block_se
Project  = row | column
BBoxFill = inset0 | inset1
```

There are ten anchor colors and twelve effects for one certified color delta:
120 native programs per parent.  There is no union, intersection, complement,
reflection, checkerboard, free mask, learned mask, arbitrary coordinate,
multi-color delta, recursive composition, or fallback to raw input alignment.

For a supported crop parent, the adapter maps each visible source coordinate
through crop translation and the declared D4 transform.  The effect is
rasterized from those output-space anchor coordinates and changes only cells
equal to `source_color`.  Unsupported operator/canvas/render combinations must
return a typed `provenance_unsupported` certificate; a coarse approximation is
forbidden.

## Fresh family-disjoint confirmation

Use a new sealed cohort of exactly 100 deterministic episodes drawn only from
the 18 ARC-TGI reserve families in the accepted v2 partition.  These families
are disjoint from the 50 development and 100 earlier confirmatory families.
Allocate six indexed episodes to the first ten lexicographically sorted reserve
families and five to the remaining eight.  Derive every episode seed from a new
frozen seed contract containing family source identity and episode index.

Before any solver reads a query target, write only:

- blind demonstrations and query inputs;
- family/source IDs and deterministic episode IDs;
- hashes of the sealed query oracle and witness;
- generator commit, source hashes, protocol hash, and cohort hash.

All task-level metrics must also report the number of distinct successful
families.  One generator family producing several recoveries cannot alone pass
a gate.

## Sequential gates

### G0: compatibility and exact provenance

- historical scene outputs, execution IDs, schemas, and serialized artifacts
  remain byte-for-byte unchanged;
- exhaustive controlled crop/D4 cases reproduce the historical render and the
  source coordinate of every rendered cell exactly;
- background-origin and object-origin cells are both represented, while only
  object-origin cells carry entity IDs;
- serialization, canonical IDs, repeated execution, and worker-count replay
  are identical;
- invalid provenance is explicit and never falls back.

Failure is an engineering failure.  One implementation-only repair is allowed
without changing the language or thresholds.

### G1: strict LODO sensor gate

Run only on demonstrations.  For every held-out demonstration:

1. compile the version space using all other demonstrations;
2. require a non-empty version space;
3. require **every** surviving program to produce the same held-out grid;
4. require that grid to equal the held-out target.

A task passes only when every fold passes.  Report identical strict task/family
metrics for identity, global recolor, and whole-component recolor baselines.

Promotion requires both:

- at least 5/100 strict-LODO tasks from at least three reserve families; and
- at least three more strict-LODO tasks than the strongest baseline, with an
  advantage present in at least two families.

If G1 fails, do not execute query candidates or inspect query targets.

### G2: query-blind frontier gate

Only after G1 passes, execute the frozen version space on query inputs.  A task
counts only if at least one demo-exact candidate output bundle is content-new
relative to both the incumbent parent pool and the equal-cost cold restart.

Promotion requires at least 5/100 novel-frontier tasks from at least three
families.  `novel_frontier_count == 0` forbids a frontier-changing claim.

### G3: natural utility gate

Only after G2 passes may the sealed query oracle be deterministically
regenerated and hash-verified.  Promotion requires:

- at least 5/100 unique exact recoveries from at least three families;
- strictly more unique recoveries than the equal-native-cost cold restart;
- no reservation, replay, data-boundary, or content-address violation.

The main endpoint is unique recovery over the frozen incumbent plus cold pool,
not raw candidate accuracy.

### G4: residual causality

Run only after G3 passes.  On recovered tasks, compare the intact system with:

- cleared `RelationalFailureCore` (no action is legal);
- family-shuffled cores (typed compatibility is checked, not coerced);
- a bridge lesion replacing provenance pull-forward with the old raw-input
  alignment;
- an entity lesion removing entity/relation IDs while retaining source-cell
  coordinates.

The preregistered prediction is selective: clearing or shuffling the core and
lesioning the provenance bridge must remove at least three of the five required
recoveries; the entity lesion may have a smaller effect because v0.4 permits
background-origin anchors.  Action IDs and footprints must change in the
predicted direction.  Controller training remains disabled throughout.

## Cost and trace contract

- Reconstruct exactly the first 2,000 object/code trials and at most four valid
  parents per task.
- Reserve 2,048 native trials per parent/action comparison.
- Log realized and reserved program trials, demonstration executions, query
  executions, padding, provenance builds, and separate LODO-only cost.
- Count content-new output bundles, not distinct program IDs.
- Record source commit, Python version, worker count, command line, input hashes,
  protocol hash, candidate-language hash, and output hashes.
- `controller_training_started` must be `false`; no router checkpoint may be
  read or written.

## Outcome mapping and persistence rule

- G0 failure: implementation failure; repair once without semantic change.
- G0 passes and G1 fails: the frozen twelve-effect provenance sensor is null.
- G1 passes and G2 fails: the diagnosis is identifiable but does not change the
  live candidate frontier.
- G2 passes and G3 fails: the bridge creates new candidates but has no natural
  utility under this parent pool and budget.
- G3 passes and G4 fails: solver utility exists, but residual-driven functional
  switching is not causally supported.
- G3 and G4 pass: retain the bridge and only then preregister the next relation
  family or controller comparison.

A negative result closes only the tuple `(crop provenance, v0.3 twelve-effect
language, frozen parent pool, frozen cost, reserve-v2 cohort)`.  It must not be
reported as disproving persistent workspaces, relation/effect programs, or the
broader functional-recruitment hypothesis.  The failure matrix and implemented
provenance layer remain reusable evidence for selecting the next bounded
candidate.
