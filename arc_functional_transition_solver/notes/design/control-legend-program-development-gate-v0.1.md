# Control-legend object program v0.1 development gate

## Status and decision boundary

This is a preregistered implementation gate for one opt-in representation
family selected from the frozen relational-delta v0.2 terminal matrix.  The
50-task ARC-AGI-2 public-training cohort, including query answers for the two
selected tasks, is already outcome-exposed.  Therefore the gate can establish
semantic correctness, deterministic reachability, and candidate novelty, but
cannot establish held-out generalization or publication-level solver gain.

No legacy provider, router, score, budget, checkpoint, or artifact may change.
The implementation lives in a new DSL namespace and controller training remains
forbidden.

## Representation hypothesis

The selected family treats a compact isolated lane of ordered color pairs as a
control program:

```text
unique isolated pair lane
  -> ordered rewrite rules [(source_0, target_0), ...]
  -> payload cells or monochrome payload objects
  -> one typed render operation
```

Repeated adjacent copies of the same pair form one rule but all pair cells
remain in the protected control mask.  Rules are applied once in lane order;
the current color after an earlier rule is visible to later rules.  This
sequential semantics is required by the demonstrations and differs from an
unordered palette dictionary.

The bounded v0.1 AST contains:

- parse axis: horizontal-pair/vertical-sequence or
  vertical-pair/horizontal-sequence;
- pair direction: forward or reverse;
- sequence direction: forward or reverse;
- control selector: `unique_isolated_pair_lane` with fixed maximum gap 2;
- render:
  - `ordered_rewrite_payload`: sequentially rewrite non-control cells;
  - `fill_exterior_bbox_background`: for each source-color payload component,
    resolve its final rule color and write it into the exterior background
    region of the one-cell-padded bounding box, preserving enclosed holes and
    all existing non-background cells.

This gives exactly 16 programs per blind task.  Program identity is the
SHA-256 of strict canonical JSON.  Every program carries a node-level trace for
parse, correspondence, payload assignment, mask, and render.

## Parser invariants

For a candidate lane:

1. both cells of every pair are non-background and have different colors;
2. cells immediately outside the pair axis are background or canvas boundary;
3. observations share one pair-axis coordinate and consecutive observations
   are separated by at most two cells along the sequence axis;
4. at least two rules remain after collapsing adjacent duplicate pairs;
5. exactly one qualifying lane exists for the program's axis;
6. every observed pair cell is protected from payload rendering.

The modal color is the input-side background hypothesis, with the smallest
color breaking exact count ties.  Ambiguous or absent lanes invalidate the
program; there is no fallback.

## Semantic fixture gate

The following named tests must pass before candidate construction:

1. `test_ordered_rewrite_applies_rules_sequentially_and_protects_control`;
2. `test_fill_exterior_bbox_preserves_enclosed_holes`;
3. `test_repeated_pairs_collapse_without_losing_control_cells`;
4. `test_vertical_lane_is_supported_by_the_same_typed_program`;
5. `test_ambiguous_control_lanes_are_rejected`;
6. `test_program_round_trip_is_strict_and_content_addressed`;
7. `test_enumeration_is_bounded_and_query_output_free`;
8. `test_demo_exact_synthesis_rejects_near_miss_render`.

The test receipt, names, source hashes, runtime, and deterministic environment
must be bound into the candidate freeze.

## Frozen construction protocol

- cohort ID:
  `dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac`;
- existing v0.2 candidate freeze ID:
  `85811f28942fc447d529be60e217d4cd4b5326c485ea901dda3d19b983fe32e5`;
- existing combined v0.2 result ID:
  `14d4ad29bec0da59fc6e405b72677cc917cf1f42d41e25af776003d3ccb3b884`;
- task count: 50;
- program cap: 16 per task;
- selected candidate cap: 2 per task;
- candidate construction reads demonstrations and blind query inputs only;
- query outputs and public evaluation are unavailable during freeze;
- the complete freeze is independently reconstructed twice and must be
  byte-identical;
- a new frontier is counted only when a demo-exact query-output bundle is not
  present in the frozen v0.2 full candidate pool.

Scoring occurs in a separate command after the candidate freeze.  The scorer
validates every source hash and immutable ID before reading public-training
query outputs.

## Development decision map

The representation probe passes its bounded implementation gate only if:

1. all eight semantic fixtures pass;
2. both `5adee1b2` and `e4888269` have at least one demonstration-exact,
   query-valid program;
3. both tasks have a content-novel candidate bundle relative to v0.2;
4. both tasks are query-exact under pass@2 after the freeze;
5. neither task was already solved by the frozen legacy plus v0.2 result;
6. replay, source contracts, and cost counts close exactly.

Because query outcomes were exposed during method diagnosis, passing means
only that the selected representation is implemented correctly and reaches a
previously absent candidate region.  It does **not** authorize the future
claim of at least 5/100 unique recovery, visual-posterior guidance, typed
repair superiority, controller training, or causal functional switching.

After this gate, the family must be frozen before any fresh, family-disjoint
confirmation set is scored.  On that fresh set, the representation remains
eligible only if it contributes at least 3--5 unique tasks per 100 under the
same native-cost ledger.  Visual-posterior structured proposals and equal-cost
cold restart are separate later gates.
