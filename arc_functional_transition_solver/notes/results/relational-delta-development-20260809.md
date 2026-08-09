# Relational delta-program v0.2 development result

## Decision

The outcome-exposed v0.2 development gate has a split result:

- **solver representation: pass on the bounded development criterion**;
- **natural residual-control mechanism: fail**.

The opt-in grammar adds two query-exact, pass@2 task recoveries on the frozen
50-task ARC-AGI-2 public-training development cohort.  Both are absent from the
frozen legacy plus relational-mask v0.1 result.  The same run finds two natural
parents that satisfy the strengthened near-miss definition, but it finds no
content-novel legal child outside the initial prefix.  Typed repair and
equal-cost restart therefore both recover zero tasks, and no action is declared
frontier-changing.

This is implementation and reachability evidence only.  The cohort and the
three diagnostic query outputs were already exposed during failure analysis;
the result is not held-out generalization evidence and does not authorize a
controller, visual LODO routing, or a brain-switching claim.

## Frozen result

| Endpoint | Result |
|---|---:|
| Tasks | 50 |
| Semantic fixtures | 9/9 passed |
| Tasks with a demo-exact v0.2 program | 2/50 |
| Raw query oracle | 2/50 |
| Pass@2 | 2/50 |
| Unique pass@2 over frozen legacy + v0.1 | 2/50 |
| Diagnostic tasks solved | 2/3 |
| Identity-normalized quality parents | 2 |
| Actions with `novel_frontier_count > 0` | 0 |
| Typed-repair recoveries | 0 |
| Equal-cost restart recoveries | 0 |

All candidate construction was performed from the existing blind payloads.
The candidate artifact records `query_gold_read=false`, was reconstructed twice
byte-identically, binds the exact nine semantic-test names and JUnit hash, and
binds the DSL, gate script, semantic tests, and preregistration source hashes.
Public evaluation was not read and no training was started.

## What the two recoveries mean

### `7acdf6d3`: relation region plus erase/add

The exact program assigns the minority foreground color as the actor and the
majority foreground color as support.  Eight-connected support components
define row-interior regions; the unique region whose area equals the actor
cardinality becomes the target.  The complete actor mask is erased and its
color is written into that region.

- exact program ID:
  `b5dcb86bda70293d94ecc9f390616f2a4a529fd42810b5e59516f3c2a8138693`;
- relation: `actor_area_to_row_interiors`;
- target: `region`;
- operation: `erase_add`;
- query exact: yes.

Its natural add-only parent is genuinely closer than identity:

- identity mismatches: 10;
- parent mismatches: 5;
- delta precision: 1.0;
- delta recall: 0.5;
- every demo is no worse than identity;
- topology-valid: yes.

The certificate correctly diagnoses `delta_mask` and recommends
`edit_delta_mask`.  It still cannot emit a legal exact single-slot child: the
exact program changes both `mask.erase` and `render.operation`, while the
action contract permits one existing slot.  This is an actuator-language
limitation, not a pixel-agreement artifact.

### `320afe60`: topology role plus component relocation/recoloring

The first nearest-edge interpretation was falsified during semantic preflight:
several demonstration components nearer the left edge move right.  The
demo-consistent relation instead assigns a topology role.  Solid components or
components with a closed hole move to the left border and receive color 2;
components with an open indentation but no hole move right and receive color 3.
The component shape is preserved exactly.

- representative exact program ID:
  `3af197d3a0ab4fe658dd0df86e565ed4a4944d0c757639d7c5741544eed53be8`;
- relation: `component_to_horizontal_border`;
- target: `component`;
- operation: `recolor_component`;
- palette: left 2, right 3;
- both query outputs exact: yes.

The swapped-palette parent also passes the stricter quality gate:

- identity mismatches: 267;
- parent mismatches: 139;
- delta precision: 1.0;
- delta recall: 1.0;
- every demo is no worse than identity;
- topology-valid: yes.

The certificate correctly diagnoses `render` and includes `render.palette`.
The exact palette program, however, is already in the initial content-addressed
prefix.  Re-emitting it would not change the frontier, so the novelty guard
correctly suppresses the repair claim.

## Why `3ad05f52` still fails

All four render operations execute correctly on controlled semantic fixtures,
including complete topological-hole filling.  Nevertheless, `3ad05f52` has no
demo-exact program.  Identity makes 406 aggregate demonstration errors; the
current `support_enclosed_regions` fill makes 528.  The desired region is not a
set of independent pixel holes.  It is a cell complex induced by repeated cyan
walls and openings, with colored seed regions propagating through a relational
adjacency graph.

The missing representation is therefore approximately:

```text
infer repeated wall coordinates and cells
  -> build room/corridor adjacency through wall gaps
  -> assign payload seed cells
  -> select the seed-connected relational subgraph
  -> rasterize complete cell interiors and connecting corridors
```

This is a representation null for the current relation vocabulary, not an
execution, replay, hashing, or training failure.

## Mechanism interpretation

The small v0.2 grammar contains only 10--34 programs per task, below the frozen
64-program initial prefix.  Consequently, every available exact program and
every quality parent are already in the initial search region.  Across all 50
tasks:

- typed and restart arms each reserve 16 trials;
- every trial executes all demonstrations and query inputs, using parent replay
  as padding when no new program exists;
- total execution counts match on every task;
- `frontier_changed == (novel_frontier_count > 0)` on every task.

This explains the apparent tension between two good near-misses and zero repair:
the new provider solves the tasks directly, but the current factorization does
not create an unseen legal frontier for residual control.  The result supports
a stronger static provider, not functional switching.

## Failure classification

| Observation | Classification | Consequence |
|---|---|---|
| Same-canvas near-miss code initially raised on shape-changing demonstrations | Implementation boundary defect, caught before artifact creation | Added explicit `canvas_incompatible` status and regression coverage; failed run retained under ignored `trash/`. |
| Nearest-edge interpretation fails `320afe60` demos | Theory/representation hypothesis failure | Replaced before freeze by a topology-role relation and recorded the amendment. |
| Two exact new task candidates | Positive representation evidence on exposed development data | Advance the solver representation candidate to a genuinely fresh confirmation source. |
| Two quality parents but zero novel child | Search-frontier/action-language null | Do not claim repair or residual causality. |
| `3ad05f52` remains unsolved | Missing lattice/cell-complex relation | Add an explicit relational region graph; do not tune pixel-hole weights. |

## Next bounded gate

1. Add a versioned `lattice_cell_complex` relation with explicit wall, cell,
   gap, seed, adjacency, and rasterization traces.  Keep v0.2 immutable.
2. Test it first on controlled irregular-lattice fixtures, then on the already
   exposed `3ad05f52` as development evidence.
3. Separate base generation from repair only when the larger grammar creates a
   genuine unseen frontier.  Do not hide an already enumerated exact program to
   manufacture a repair gain.
4. Confirm unique coverage and natural typed repair on a newly frozen semantic
   family source.  The current 50 tasks cannot be reused for that claim.
5. Keep controller and visual LODO training frozen until at least two natural
   actions have positive novelty and typed recovery beats equal-cost restart.

## Immutable identifiers

- baseline v0.1 result ID:
  `916221630a9a3472e43e05a33932486f1d57f6a49d274bc75677c8b4ae678f6e`;
- v0.2 candidate freeze ID:
  `85811f28942fc447d529be60e217d4cd4b5326c485ea901dda3d19b983fe32e5`;
- candidate freeze file SHA-256:
  `c406cb999ec452c0a5a0a76c32c5a9c60754bb7319caf9846e9a49edb004809e`;
- v0.2 result ID:
  `14d4ad29bec0da59fc6e405b72677cc917cf1f42d41e25af776003d3ccb3b884`;
- result file SHA-256:
  `29c244473a07ed1794bec89df1ab1d9ec16004fb16d1840c1d7883400e0b4d99`;
- semantic JUnit SHA-256:
  `c6cad61a4193efe39ee9cdec96009f5f79643f08d32b10ca054b901203955234`.

Machine-readable artifacts are under
`results/relational_delta_dev_20260809/`.
