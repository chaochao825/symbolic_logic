# E01a public-training smoke failure taxonomy

## Scope and evidence status

This is a post-hoc, human-interpreted diagnosis of the 18 unsolved tasks in the
fixed 20-task ARC-AGI-2 public-training development smoke. It is not a new solver
result. The pool was frozen before oracle outputs were read, so this diagnostic
analysis cannot change the reported E01a coverage.

Subsequent status: the registered M03a/M05b slice verifies `c59eb873` with
`scale_pixels(2,2)` and `a416b8f3` with `tile_grid(1,2)`. The following independent
M02b/M05c slice verifies `a68b268e` with `overlay_panel_grid(background=0)` and
records the intended ragged lattice for `92e50de0` without solving it. The following
M05d slice verifies `8e5a5113` with
`broadcast_panel_sequence_d4(background=0, step=rotate90)`. The subsequent M05e
slice verifies `92e50de0` with
`broadcast_panel_lattice_periodic(background=0, row_period=2, column_period=2)`.
The subsequent additive M02c/M05f slice verifies `1f642eb9` with
`paint_bbox_contacts(background=0)`. All seven earlier solved tasks remain covered.
The table below remains the pre-implementation diagnosis; 12 tasks now remain
without a correct candidate.

The machine-verifiable observation is common to all 18 tasks: the formal search
ledger records `first_exact_expansion = null`, zero retained demo-exact programs,
and zero emitted candidates under maximum depth 2, beam width 64, and at most 64
demo-derived instruction options. The capability labels and proposed rules below
are manual interpretations of the demonstrations and therefore remain diagnostic,
not ground-truth causal annotations.

Evidence inputs:

- frozen search ledger:
  `results/e01a_symbolic_v1/public_train_dev_pool/search.jsonl`;
- frozen blind task records:
  `results/e01a_symbolic_v1/public_train_dev_pool/tasks.jsonl`;
- post-freeze oracle task records:
  `results/e01a_symbolic_v1/public_train_dev_oracle/oracle.jsonl`;
- verified aggregate result:
  `results/e01a_symbolic_v1/public_train_dev_eval/summary.json`.

## Label definition

- `parse`: the required scene decomposition or relation is absent from M02a.
- `shape`: the output canvas cannot be proposed or constructed by the current DSL.
- `selector`: useful objects exist, but the rule selecting or matching them is absent.
- `primitive`: the local transformation or rendering action is absent.
- `binding`: a primitive or value must be chosen from task context rather than fixed
  as one program-wide constant.
- `composition-depth`: the rule is expressible with current primitives but lies
  outside the measured depth bound.
- `search-pruning`: an expressible in-budget rule is lost by beam or semantic
  deduplication.

Each task receives one primary label even when a secondary capability is also
needed. This makes the aggregate counts mutually exclusive.

## Task-level diagnosis

| Task | Expansions | Interpreted rule | Primary | Secondary | Smallest useful capability | Confidence |
|---|---:|---|---|---|---|---:|
| `1b8318e3` | 3,214 | Move each colored singleton toward its nearest gray 2x2 anchor until it becomes 8-neighbor adjacent. | primitive | parse | Nearest-object relation plus deterministic move-until-touching. | 0.94 |
| `ce602527` | 2,836 | Select the small motif whose pixelwise 2x expansion matches a larger exemplar clipped by the grid boundary, then crop the small motif. | selector | parse, shape | Same-color macro grouping plus scale-and-visible-clip matching. | 0.995 |
| `0becf7df` | 2,206 | Read two color-swap pairs from the top-left 2x2 legend and apply them outside the legend only. | binding | parse | Region-scoped color map whose arguments are induced from a legend. | 1.00 |
| `fcc82909` | 2,143 | Beneath each colored 2x2 block, draw a green bar whose height is the block's number of distinct colors. | primitive | binding | For-each relative rectangle rendering with feature-to-size binding. | 1.00 |
| `8e5a5113` | 3,340 | Split three 3x3 panels on full-height gray separators; place 90-degree and 180-degree rotations of the first panel in the empty panels. | primitive | parse, composition | Separator split, panel-local D4, and panel reassembly. | 1.00 |
| `1f642eb9` | 3,151 | Project external colored markers along aligned rows or columns onto the first boundary cell of a filled cyan rectangle. | primitive | parse | Row/column line-of-sight relation plus ray-to-bbox overwrite. | 1.00 |
| `d2acf2cb` | 2,696 | Among row/column segments delimited by color 4, choose the greatest span and apply `0<->8, 6<->7` only inside it. | selector | parse, binding | Aligned-marker segment relation, max-span selector, and masked color map. | 0.97 |
| `f5c89df1` | 1,473 | Extract the blue template at source anchor 3 and stamp registered copies at anchors 2, merging overlaps and erasing source markers. | primitive | selector | Anchor-aware template extraction, optional D4 registration, and multi-target stamp. | 0.82 |
| `92e50de0` | 1,387 | Parse a 3x3 panel lattice and copy the residual motif to panels with matching row/column parity, clipping partial edge panels. | parse | primitive | Periodic separator/panel lattice with indices and clipped motif stamping. | 0.995 |
| `a68b268e` | 2,821 | Split a 9x9 grid at the central separator cross and overlay the four 4x4 panels with zero transparent and row-major priority. | parse | primitive, shape | Separator-cross panel view plus ordered transparent overlay. | 1.00 |
| `2685904e` | 3,340 | Bind integer `n` from the top run of color 8; keep colors occurring exactly `n` times in the code row and repeat that row `n` times above the divider. | binding | parse | Integer feature binding reused by a count predicate and repeat renderer. | 1.00 |
| `36d67576` | 2,476 | Find the fully annotated anchored motif and copy its missing color-1/3 labels into D4-congruent motif instances. | selector | parse, primitive | Anchor-aware motif canonicalization, congruence matching, and fill-zero-only label transfer. | 0.98 |
| `e39e9282` | 1,561 | Rewrite adjacent color-9 markers differently around 3x3 blocks of color 5 versus color 6, using local copy, move, and erase actions. | primitive | binding | Relation-scoped marker rewrite with policy selected by block color. | 0.98 |
| `c59eb873` | 2,269 | Replace every input cell by a 2x2 block, mapping `H x W` to `2H x 2W`. | shape | none | Typed pixel scaling with row and column factors. | 1.00 |
| `2697da3f` | 497 | Crop foreground motif `P` and place oriented copies around an empty center to form an expanded square cross. | shape | composition | Explicit canvas construction and spatial composition of D4 copies. | 0.99 |
| `f3e62deb` | 1,513 | Move a hollow 3x3 frame to an edge selected by its color while preserving the orthogonal offset; infer the unseen color-edge pair by bijection completion. | primitive | binding | Snap-object-to-edge plus contextual color-to-edge binding. | 0.99 |
| `94133066` | 3,340 | Crop the main panel and choose the D4 transform that aligns its internal keypoint layout with same-color external keypoints. | binding | parse, selector | Scene-context-preserving D4 selection by keypoint-layout alignment. | 0.99 |
| `a416b8f3` | 2,710 | Concatenate the input with itself horizontally, mapping `H x W` to `H x 2W`. | shape | none | Typed horizontal tile or concatenation. | 1.00 |

`94133066` provides a useful counterexample to a pure grammar diagnosis. Each
demonstration is individually solved at depth 2 by cropping the largest panel and
applying a different D4 transform. A single fixed instruction sequence cannot solve
all demonstrations because the transform must be selected from each input's
external keypoint layout. The missing mechanism is contextual binding, not D4 itself.

## Aggregate bottleneck

| Primary class | Tasks | Count |
|---|---|---:|
| primitive | `1b8318e3`, `fcc82909`, `8e5a5113`, `1f642eb9`, `f5c89df1`, `e39e9282`, `f3e62deb` | 7 |
| selector | `ce602527`, `d2acf2cb`, `36d67576` | 3 |
| binding | `0becf7df`, `2685904e`, `94133066` | 3 |
| shape | `c59eb873`, `2697da3f`, `a416b8f3` | 3 |
| parse | `92e50de0`, `a68b268e` | 2 |
| composition-depth | none | 0 |
| search-pruning | none identified | 0 |

This taxonomy does not prove that wider search has zero value elsewhere. It shows
that every manually reconstructed rule for these 18 tasks requires at least one
semantic operation or representation outside the current program space. Therefore,
a width-only rerun is not the next informative experiment.

## Historical failure-driven symbolic backlog

1. Completed: add explicit output-shape algebra and typed `scale_pixels` and
   `tile_grid` operations.
   This is a small, falsifiable slice with direct expected coverage on
   `c59eb873` and `a416b8f3`; it also establishes shape guards needed by later
   composition.
2. Completed: add separator-band and panel hypotheses as independent M02b views
   without modifying M02a object hypotheses, then pair them with ordered transparent
   panel overlay. This solves `a68b268e` and supplies the indexed representation
   needed by `92e50de0` and `8e5a5113`.
3. Completed: add the parameterized M05d indexed panel-sequence D4 action and
   reassembly. This solves `8e5a5113`; periodic two-axis broadcast with ragged
   clipping remains a separate operation for `92e50de0`.
4. Completed: add the parameterized M05e two-axis periodic panel-lattice action
   with suffix-only top-left clipping. This solves `92e50de0`, preserves the first
   six solved tasks, and leaves leading/interior ragged lattices and multiple seeds
   explicitly invalid.
5. Completed narrow subset: add the M02c singleton-to-solid-rectangle axis-ray
   sidecar and M05f copy-to-boundary renderer. This solves `1f642eb9` while leaving
   aligned max-span segments, nearest anchors, movement, and scoped color maps as
   separately falsifiable future capabilities.

The current plan pauses this task-by-task backlog and pre-registers M04a global
masked-grid generation. Items 6-8 remain possible only if systematic symbolic work
resumes; they are not the current implementation order. See
`notes/results/e01a-next-source-diagnostic.md`.

6. Add anchor-aware motif canonicalization and spatial stamping for
   `f5c89df1` and `36d67576`.
7. Add contextual value/operation binding, including integer reuse, legend-derived
   mappings, and input-dependent D4 choice. This should be represented explicitly
   rather than hidden inside task-specific monolithic primitives.
8. Only after those capabilities enter the bounded grammar, compare exhaustive
   low-cost enumeration against the beam to measure false pruning.

For every slice, use generator-known positive controls and rerun the identical
20-task development smoke. Report both newly solved tasks and regressions. Do not
promote these post-hoc task diagnoses to holdout claims, and do not tune a learned
ranker to compensate for an empty candidate set.
