# Provenance-aligned relational effect v0.4: reserve-100 result

## Status

This is a valid, prospective negative result for the frozen tuple

```text
(exact crop cell provenance,
 v0.3 twelve-effect language,
 one post-render color delta,
 frozen parent pool and native cost,
 ARC-TGI reserve-v2 cohort)
```

It is not evidence against persistent workspaces, relation/effect programs, or
functional recruitment in general.  G1 failed, so query candidates and query
targets were not read, G2--G4 were not run, and controller training remained
disabled.

## Pre-execution judgment

The candidate was worth a prospective test for a narrow reason.  In the
outcome-exposed development audit, 46/48 `anchor_mask_unreachable` parents were
crop executions with an exact source-to-output coordinate map.  Keeping the old
twelve-effect language fixed while replacing raw-coordinate alignment with
exact cell provenance made 13/48 parents and 5/12 affected tasks reachable and
strict-LODO exact.  Whole-entity selection and entity-only anchor alignment
recovered 0/48.  This isolated cell lineage as a plausible bridge without
crediting a larger hand-written DSL.

The main risks were preregistered:

1. the twelve effects might fit an enriched development failure cluster but not
   the unconditioned reserve-family distribution;
2. a single shared color delta might be too weak for natural multi-effect tasks;
3. persistent entity identity might not resolve cross-demonstration role
   binding;
4. successful tasks might be explainable by identity, global recolor, or
   whole-component recolor.

The fresh result realizes risks 1, 2, 3, and 4.  It does not show that exact
provenance was implemented incorrectly.

## Frozen implementation

The implementation adds an opt-in layer and leaves historical scene execution
and serialization unchanged:

- `PersistentEntityId` and `PersistentRelationId` are typed identifiers whose
  runtime strings and existing hashes remain compatible;
- `CellProvenance` records exact source/output coordinates, optional entity
  identity, producer node, and exactness;
- `EffectSummary` records the observed delta support, source/entity coverage,
  and a content-addressed affected relation subgraph;
- `RelationalFailureCore` is the minimal surviving typed effect version space;
- `RelationalEffectNode` contains exactly one source/target color delta and one
  of the twelve frozen stencil, row/column projection, or bbox-fill effects.

Exact provenance is implemented only for supported crop + bbox +
`source_crop`/`selected_only` traces through every D4 transform.  Unsupported
traces return an explicit typed certificate.  There is no coordinate fallback.

One correctness repair was made before the valid run: demonstrations on which
the parent is already exact now constrain a learned delta/effect program by
requiring it to be a no-op, rather than making a mixed exact/inexact task
unidentifiable.  Runs made before this repair are marked invalid and are not
used below.  A later serialization correction removed non-causal incident
relations from crop-cell producer lineage and stores a hash/count plus a bounded
sample of the affected relation subgraph.  It changes neither candidates nor
metrics.

## Data and replay boundary

The valid confirmation used 100 deterministic indexed episodes from 18 sealed
ARC-TGI reserve-v2 families, disjoint from all development families.  The two
independently generated blind cohorts and seals were byte-identical.  They
contain demonstrations and query inputs plus hashes of sealed query oracles and
witnesses; no query target is present.

| Artifact | Value |
|---|---|
| cohort ID | `9250415ec1a58f03a8557cf96cdf2cc0693d82130edbb82fd2ab50031c139e6c` |
| seal ID | `618c4c81a9c6002b1890fac2457b08c23d7c90a06c08d70c438b07be8a29e873` |
| protocol SHA-256 | `1e96d4bce8764919665afbb8fac053634f503124317aa4e8d8fe13239ab4cb5a` |
| compact freeze ID | `1d6eeecc4c0aab4d6332ecafc49d4e853ff50de0f5b0e251ee3929d38f786fac` |
| compact freeze SHA-256, A and B | `26e75dc52a8142a147da65bc1739e7c8bcd35f9bdf55758952220139171a6db6` |
| failure-audit ID | `4f78389c97137732d2d5d002747a519b45e940082cda9b0a4000ee0aa2c890d0` |
| failure-audit SHA-256, A and B | `8f4f6538baa1790130725a32ce5474172c59ff488b42778388e95f7a9d600f7d` |

Both valid freezes are byte-identical across eight-worker executions.  The
candidate freeze is 34,810,877 bytes, versus 386 MB for the superseded
over-serialized diagnostic run.

The compact freeze can be replayed from the repository root with:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python scripts/afts_provenance_relational_effect_gate.py freeze \
  --challenges results/provenance_relational_effect_reserve100_v1_20260812/cohort_a/reserve100_challenges.json \
  --seal results/provenance_relational_effect_reserve100_v1_20260812/cohort_a/reserve100_seal_manifest.json \
  --protocol notes/research/provenance-relational-effect-v0.4-gate-20260812.md \
  --max-first-stage-trials 2000 --max-exact-programs 32 \
  --max-parents 4 --reserved-native-trials 2048 --workers 8 \
  --output results/provenance_relational_effect_reserve100_v1_20260812/replay/candidate_freeze.json

PYTHONHASHSEED=0 PYTHONPATH=src \
python scripts/afts_provenance_relational_effect_gate.py audit \
  --freeze results/provenance_relational_effect_reserve100_v1_20260812/replay/candidate_freeze.json \
  --output results/provenance_relational_effect_reserve100_v1_20260812/replay/failure_audit.json
```

The 34.8 MB freezes remain content-addressed on the 210 experiment workspace;
the repository result bundle keeps their hashes, the compact summary, the seal,
and one 52 KB failure audit instead of duplicating large deterministic pools.

## Gate result

| G1 metric | Method | Identity | Global recolor | Whole component |
|---|---:|---:|---:|---:|
| strict-LODO tasks / 100 | 2 | 0 | 2 | 2 |
| successful families / 18 | 1 | 0 | 1 | 1 |
| unique over all baselines | 0 | -- | -- | -- |
| advantage over strongest baseline | 0 | -- | -- | -- |

The two passing episodes are both from family
`taskVUoN2Euv3vJadQ35Fb3ery`.  Both are also solved by global recolor and
whole-component recolor.  The promotion rule required at least 5/100 tasks,
three families, an advantage of three tasks, and unique advantage in two
families.  Therefore G1 fails unambiguously.

Consequences required by the protocol:

- `frontier_evaluated = false`;
- `query_gold_read = false`;
- `controller_training_started = false`;
- G2 novel frontier, G3 unique recovery/cold restart, and G4 residual
  interventions are not statistically or procedurally eligible.

## Complete failure matrix

### Task-terminal causes

| Terminal cause | Tasks / 100 | Interpretation |
|---|---:|---|
| `delta_semantics_inconsistent` | 54 | A single shared source/target color delta does not describe all demonstrations. |
| `cross_demo_delta_inconsistent` | 13 | Each demo admits a delta, but there is no stable cross-demo delta/role binding. |
| `relation_effect_language_unreachable` | 24 | Delta semantics are coherent, but none of the twelve frozen effects explains the support. |
| `no_scene_parent` | 5 | The frozen first-stage provider yields no supported parent trace. |
| `shape_incompatible` | 2 | Parent and target canvas shapes are incompatible with the post-render action. |
| `strict_but_baseline_redundant` | 2 | Strict LODO succeeds, but a simpler baseline succeeds too. |

### Parent-level causes

Across 380 supported parent attempts:

| Parent cause | Count |
|---|---:|
| `delta_semantics_inconsistent` | 220 |
| `relation_effect_language_unreachable` | 94 |
| `cross_demo_delta_inconsistent` | 52 |
| `shape_incompatible` | 8 |
| `identified` | 4 |
| `provenance_unsupported` | 2 |

### Strict-LODO fold outcomes

Across 1,428 held-out folds, 15 were unanimously exact, 12 were unanimously
wrong, and the remaining folds were unidentified: 837 delta-inconsistent, 184
cross-demo-delta-inconsistent, 341 language-unreachable, 16 shape-incompatible,
7 provenance-unsupported, and 16 without an eligible failure core.

Observed changed cells have 97.76% exact source-coordinate provenance but only
63.32% entity-backed provenance.  Therefore the principal failure is not loss
of pixel lineage.  Entity IDs alone are also insufficient: marker/background
cells and object-internal subregions remain semantically relevant.

The terminal causes are not driven by one reserve generator: delta-semantic
failures span ten families, relation/effect-language failures six,
cross-demo-delta failures four, and no-parent and shape failures one each.

A post-outcome, demonstration-only diagnostic further refines the next design
choice.  These counts are over repeated parent/demo summaries, not independent
tasks, and are not promotion evidence.  Among 816 summaries attached to
delta-semantic terminal tasks, 587 contain exactly two delta pairs.  Of those,
488 map one source color to two target colors, 80 contain two source and two
target colors, and only 19 map two source colors to one target.  Across the 52
cross-demo-inconsistent parent cores, 43 keep one source color but require two
or three different target colors across demonstrations.  The dominant missing
variable is therefore a relationally bound **target role**, not merely an
additional geometric mask.

## Implementation versus theory diagnosis

### What passed as engineering

- controlled crop/D4 source-coordinate tests, including parser-background and
  generated-background cases;
- content-addressed identifier and repeated-execution tests;
- mixed exact/no-op demonstration semantics;
- explicit unsupported-provenance behavior;
- strict LODO and baseline tests;
- query-boundary and fail-closed gate tests;
- 24/24 directly scoped tests, 115/115 extended ARC compatibility tests, and
  static lint;
- byte-identical cohort, seal, candidate freeze, and failure audit across A/B.

The repository-wide test command is not a clean bounded regression target: the
installed `pytest` entry point has a stale NFS shebang, several legacy `m04a`
tests require `tests/` on `PYTHONPATH`, and the corrected full suite exceeded a
four-minute integration timeout.  These pre-existing harness issues are
reported rather than suppressed; they do not affect the 24 scoped tests.

### What failed as a method

The frozen candidate assumes that a natural parent failure can usually be
factorized as

```text
one persistent source anchor role
  x one shared color delta
  x one unary geometric effect
```

The reserve result rejects that factorization.  The dominant 67% of tasks fail
before effect selection because the transformation needs multiple deltas,
conditional effects, or a role that is stable relationally but not by raw
color.  A further 24% has a coherent delta but lies outside the twelve-effect
support algebra.  Development performance was optimistic because the 12-task
set had already been selected for `anchor_mask_unreachable` near misses; the
reserve cohort was unconditioned.

## Relation to prior work

ARGA demonstrates that object graphs, graph transformations, and constrained
DSL search can provide strong ARC inductive bias.  Therefore adding an object
graph or a DSL is not by itself a contribution.  The intended distinction here
is a causal, content-addressed bridge from an executable failure certificate to
a frontier-changing typed action; v0.4 implements the audit substrate but does
not yet demonstrate natural utility.

PROSE-style witness functions and version-space algebra provide the closer
formal analogy for the missing step: output constraints should be propagated
backward into typed role/effect obligations, then compactly intersected across
demonstrations.  Enumerating a dozen raster masks after identifying one color
delta is not a sufficiently expressive inverse semantics.

Recent reflective multi-module ARC systems such as ARCANA motivate persistent
scene state, executors, and shared workspaces.  This project is stricter about
replay, matched native cost, blinded sequential gates, and causal lesions.  Its
current disadvantage is decisive: the candidate distribution and relational
language are much weaker, so stronger auditability has not translated into
solver accuracy.

Strong ARC systems based on test-time adaptation, recursive refinement, and
large synthetic candidate distributions primarily improve the probability that
the correct hypothesis is generated.  The present experiment addresses a
different question--whether a failure can recruit a new representation more
efficiently than restart--and cannot compensate for a weak first-stage parent
or an impoverished effect algebra.

## Retained insight and next bounded candidate

The persistent provenance layer should be retained.  It is exact, compatible,
replayable, and covers nearly all observed residual cells.  The twelve-effect
sensor should be retired for natural-task claims, not widened post hoc on this
reserve cohort.

The next candidate must be developed on a separate source and confirmed on new
family-disjoint data.  It should target the observed hierarchy rather than add
unmotivated masks:

```text
executable trace + persistent source roles
  -> typed multi-delta failure obligations
  -> n-ary relational role-binding version space
  -> bounded conditional EffectGraph
  -> affected-subtree re-execution
  -> exact verification
```

Minimum design constraints:

1. infer a sound lower bound on required effect cardinality before allowing
   more than one node;
2. bind roles by shape, topology, containment, relative position, and source
   lineage rather than raw color alone, with target colors instantiated from
   role bindings separately in each demonstration;
3. propagate target constraints backward with explicit witness functions;
4. represent multi-color and erase/add effects as typed sets, not independent
   pixel edits;
5. keep identity/global/component and equal-cost restart controls;
6. repeat strict LODO on a new family-disjoint cohort before executing queries;
7. keep the controller frozen until unique frontier and natural recovery gates
   both pass.

This is a continuation of the original functional-recruitment hypothesis at a
more appropriate abstraction level.  A function switch must change the
reachable executable explanation space; merely reordering providers still does
not qualify.
