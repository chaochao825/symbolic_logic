# M02c/M05f axis-ray rectangle-contact contract

## Registration status

This is the pre-implementation contract for the next failure-driven symbolic
development slice. It was frozen after the M05e evidence bundle passed independent
audit and before M02c/M05f code, synthetic v0.6 cases, or a successor public pool
were created.

The direct post-hoc development target is `1f642eb9`. This target was selected by
comparing two previously diagnosed failures on the same fixed 20-task ARC-AGI-2
public-training smoke:

- `1f642eb9` needs a copy-only axis ray from external singleton markers to one
  solid rectangle boundary;
- `1b8318e3` needs multiple-anchor assignment, move-and-erase behavior, contact-slot
  competition, and a still-ambiguous discrete trajectory policy.

The second task is explicitly out of scope. A blind structural scan using only
training inputs and the existing M02a `background + 4-connectivity + single-color`
objects finds exactly one matching task in the fixed smoke: `1f642eb9`, at
`background=0`. Its three demonstrations contain 5, 3, and 6 admissible rays. This
is a development-set design diagnostic, not holdout evidence or a guarantee about
the query.

## Scientific decomposition

The checkpoint is named **M02c/M05f**, not M05f alone:

- M02c adds an immutable, content-addressed relation sidecar without changing M02a
  or M02b records;
- M05f consumes the relation contract and performs one local paint action;
- M06 proposes and evaluates the typed instruction under the unchanged depth, beam,
  instruction-option, and exact-program caps.

The intended executable names are:

```text
relation parser: afts-axis-aligned-bbox-contact/v0.1
relation stage:  M02c_axis_aligned_singleton_bbox_contact
DSL action:      paint_bbox_contacts(background)
action stage:    M05f_bbox_contact_renderer
proposer:        afts-bbox-contact-proposer/v0.1
```

The only DSL argument is `background`. Anchor color, marker colors, object IDs,
bounding box, directions, and destination cells are dynamically derived from each
input and may not appear as instruction parameters.

## Blind observable relation contract

For a declared ARC color `background`, construct exactly the existing M02a
segmentation with 4-connectivity and `single_color` components. Coordinates are
zero-based `(row, column)` pairs. A complete contact hypothesis has the following
properties:

1. An anchor pre-candidate is a monochrome connected object whose bounding-box
   height and width are both at least two. A filled anchor candidate additionally
   has `area = bbox_height * bbox_width`. A pre-candidate that is not filled remains
   in the sidecar as an incompatible diagnostic; it is never complete.
2. All filled anchor candidates form the competing-anchor set. Relative to one
   filled candidate, the other filled candidates are not markers. Every remaining
   non-background object is a one-cell marker whose color differs from that
   candidate's anchor color.
3. Each marker lies strictly outside the anchor bounding box and is aligned in
   exactly one way:
   - its row lies in the anchor row interval and it is strictly left or right; or
   - its column lies in the anchor column interval and it is strictly above or
     below.
4. The projected cell is the nearest bounding-box boundary cell on the aligned
   axis. `axis=row` denotes a horizontal ray whose endpoints share a row;
   `axis=column` denotes a vertical ray whose endpoints share a column. `direction`
   is the marker-to-anchor travel direction: `right` for a marker left of the box,
   `left` for one right of it, `down` for one above it, and `up` for one below it.
5. The open ray strictly excludes the marker and projected cell. `gap` is their
   Manhattan distance minus one, hence exactly the number of open-ray cells.
   `intervening_background_count` counts cells in that interval equal to the
   declared background, and `ray_clear` is true exactly when that count equals
   `gap`. Every required marker relation must have `ray_clear=true` for the
   hypothesis to be complete.
6. Every projected boundary coordinate is occupied by the anchor in the input.
7. No two clear marker rays project to the same boundary coordinate.
8. At least one marker relation exists. A hypothesis is complete if and only if its
   anchor is filled, every non-anchor object required by rule 2 is a different-color
   singleton with exactly one aligned relation, every such relation is clear, and
   the clear destinations are collision-free. Execution later requires exactly one anchor
   candidate to yield a complete hypothesis; two complete candidates are a
   reachable `non_unique_selection` case.

The M02c sidecar is planned as `relation_parses.jsonl`. Each observable training
input, training output, and query input receives one replayable row. Query outputs
remain absent. For each grid, its background domain is exactly
`background_hypotheses(grid, include_none=True, max_backgrounds=3)` with `None`
removed while preserving returned order. The row contains one background-specific
bundle per remaining color and binds the ordered collection to the blind task hash,
pair role/index, grid hash, parser semantics, and stage. It is neither the
cross-demonstration common-background domain nor an enumeration of all ten colors.

The minimum relation and hypothesis records are:

```text
BBoxContactRelation
  relation_id
  marker_object_id
  anchor_object_id
  source_m02a_parse_id
  marker_coordinate
  projected_boundary_coordinate
  axis                         # row | column
  direction                    # left | right | up | down
  gap
  intervening_background_count
  ray_clear

BBoxContactHypothesis
  relation_parse_id
  source_m02a_parse_id
  background
  anchor_object_id
  anchor_color
  anchor_bbox
  anchor_fills_bbox
  relations
  unmatched_object_ids
  non_singleton_object_ids
  destination_collision_groups
  structural_status
```

All IDs cover the semantics version, source M02a parse ID, and complete canonical
input-derived content. The formal validator must reconstruct the sidecar from its
blind grid and require exact row equality. `parse.py` and its existing
`ObjectRelation` are not modified; otherwise the already frozen M02a bytes would be
invalidated.

## M05f renderer and invalid precedence

`paint_bbox_contacts(background)` copies the input grid, paints each projected
anchor-boundary cell with its marker's input color, and changes nothing else. In
particular, it preserves the external marker, the clear ray, canvas shape,
background, and every non-contact anchor cell. The action is neither a move, a
stamp, a line renderer, nor a destination-list interpreter.

If exactly one complete hypothesis exists, it executes even when other anchor
candidates are invalid. Unless exactly one complete hypothesis exists, the action
fails closed in this precedence order; the first case handles more than one
complete hypothesis, and the remaining cases apply when the complete count is zero:

1. `non_unique_selection`: more than one complete anchor interpretation;
2. `target_collision`: at least one filled-anchor hypothesis has two or more clear,
   otherwise geometrically admissible rays sharing a destination cell;
3. `occluded_ray`: absent a collision, at least one geometrically admissible ray is
   not clear; blocked rays do not participate in collision groups;
4. `incompatible_relation_geometry`: absent the above, an anchor pre-candidate
   exists but is non-solid, or a remaining object is non-singleton, unaligned, or
   same-colored relative to a filled anchor;
5. `empty_selection`: no anchor pre-candidate exists or no aligned relation exists.

Every filled-anchor hypothesis first records all geometrically aligned singleton
relations, including same-color diagnostics. For collision and occlusion precedence,
"geometrically admissible" means an aligned singleton whose color differs from the
anchor; same-color relations participate only in incompatible geometry. The parser
then derives clear-ray collision groups, occlusion flags, incompatible
object diagnostics, and its local status. The action applies the above precedence
over the complete set of hypotheses, independent of their serialization order.

Argument type errors are rejected before execution. Specialized action codes map to
versioned DSL invalid codes; internal exceptions may not be collapsed into a normal
selection failure.

## Blind proposer and cost closure

The proposer calls
`background_hypotheses(train_input, include_none=True, max_backgrounds=3)`, removes
`None`, and intersects those colors across all demonstrations. It builds a relation
bound for every common background and
examines every demonstration without early stopping. Only a background with one
complete relation hypothesis in every training input becomes an admissible binding.
Each admissible binding is executed on every demonstration. A proposal survives
only when all executions are valid and the produced shape agrees with the observed
training-output shape, directly or transposed.

The proposer may not compare any produced pixel with a training-output pixel. It
may not inspect query outputs, task IDs, oracle output hashes, residual locations,
or destination coordinates inferred from targets. Ordinary M06 demo-exact program
evaluation, after proposal, remains the only pixel-exact selector.

The v6 search row must record at least:

```text
bbox_contact_bounds
  background
  max_object_count
  max_anchor_candidate_count
  max_relation_count
bbox_contact_structural_check_count
bbox_contact_anchor_candidate_count
bbox_contact_relation_check_count
bbox_contact_admissible_binding_count
bbox_contact_instruction_proposals
bbox_contact_action_trial_count
bbox_contact_demo_execution_count
bbox_contact_options_after_cap          # derived summary/funnel field
```

Required closure checks include:

```text
I = all (common background, demonstration input) pairs
O_i = number of fixed-view M02a objects for i
A_i = number of filled anchor candidates for i
structural_check_count = |I|
anchor_candidate_count = sum_i A_i
relation_check_count = sum_i A_i * (O_i - A_i)
for each background bound:
  max_object_count = max_demo O_i
  max_anchor_candidate_count = max_demo A_i
  max_relation_count = max({0} union recorded relation counts of its filled anchors)
action_trial_count = admissible_binding_count
demo_execution_count = action_trial_count * demonstration_count
proposal_count <= action_trial_count
options_after_cap <= proposal_count
```

For `1f642eb9`, the pre-registered blind-input expectation is two common background
bindings (`0` and `8`) over three demonstrations, hence six structural checks.
Only background `0` is admissible: object counts are 6/4/7, there is one anchor in
each input, relation checks total 5+3+6=14, and the expected action ledger is one
trial, three complete demonstration pre-executions, one proposal, and one retained
option after the cap. These are proposer expectations, not query-coverage claims.

The new proposal is ordered after M05e proposals and before generic object/crop/
recolor enumeration. This placement is part of the contract because the target has
more than 64 pre-cap options and a tail insertion would silently remove M05f.

## Synthetic v0.6 controls

Append two three-task families without changing the random-number consumption or
content of the old 45 controls:

- `paint_bbox_contacts`;
- `paint_bbox_contacts_rotate180`.

Positive cases must vary background and anchor colors, anchor position and
rectangular dimensions, gaps, all four directions, multiple markers per side,
repeated marker colors, and marker counts. Composition outputs must differ from the
input, the one-step prefix, and a suffix-only execution.

Unit-level invalid controls must cover:

- two complete anchors (`non_unique_selection`);
- two rays landing on one corner (`target_collision`);
- an intervening blocker (`occluded_ray`);
- an unaligned singleton;
- a non-singleton marker;
- a non-solid anchor;
- an empty relation set;
- a wrong background;
- canonical-ID, coordinate, semantics-version, sidecar, proposer-ledger, cap, and
  config tampering.

The proposer must be invariant to query-input changes and to any same-shape change
of training-output pixels. Synthetic generator replay, exact generator-program
reachability, semantic/output coverage, and old-family retention are all reported
separately.

## Evidence gate

The same 20-task/21-pair development smoke and caps remain fixed. Before any claim
upgrade, the successor must demonstrate:

1. public blind tasks, oracle rows, M02a parses, and M02b panel parses are bytewise
   identical to the M05e checkpoint;
2. the old 45 synthetic task identities are retained and receive no M05f proposal;
3. all seven previously solved public tasks remain covered;
4. at most the pre-registered direct task `1f642eb9` is added by this slice;
5. the relation sidecar, bounds, costs, proposal ordering, cap survival, program
   execution, candidate emission, and parent manifests replay exactly;
6. both synthetic and public evaluation bundles pass full parent replay;
7. an independent code review and an independent formal-evidence audit report no
   unresolved P0, P1, or P2 findings.

The expected direct output key
`f23c4677b41d66ff740e6fa4136e9038fb507cbe6d03b751d9a1ea684d67818c`
is recorded only as a post-hoc diagnostic target. Coverage must not be described as
improved until the blind pool is frozen and the oracle-stage evaluator verifies it.

This checkpoint cannot support claims about nearest-anchor motion, general line
drawing, arbitrary object relations, heterogeneous-source complementarity, masked
diffusion, learned ranking, functional switching, holdout transfer, or public
evaluation.
