# Control-legend object program v0.1 development result

## Decision

The bounded, outcome-exposed representation gate **passes**, while every
solver-scale and mechanism-scale gate remains closed.

The new opt-in DSL contributes two demonstration-exact, query-exact, and
content-novel candidates on the frozen 50-task development cohort:
`5adee1b2` and `e4888269`.  Neither task was solved by the frozen legacy plus
relational-delta v0.2 system.  The selectable union therefore rises from 2/50
to 4/50 on this exposed cohort.

This is the requested proof that one coherent representation family can enter
a previously absent candidate region.  It is not a strong candidate-generation
anchor: the new family solves exactly its two selected development tasks and no
other task in the cohort.  It also does not implement or validate typed repair,
visual-posterior proposal, equal-cost cold restart, router learning, or causal
functional switching.

## Frozen results

| Endpoint | Result |
|---|---:|
| Tasks | 50 |
| Semantic fixtures | 8/8 passed |
| Grammar size | 16 programs/task |
| Program trials | 800 |
| Demonstration program executions | 2,288 |
| Query program executions after demo verification | 4 |
| Tasks with a demo-exact program | 2/50 |
| Standalone raw oracle | 2/50 |
| Standalone pass@2 | 2/50 |
| Unique pass@2 over frozen legacy + v0.2 | 2/50 |
| Combined selectable union | 4/50 |
| Fresh-set 5/100 gate | not run / closed |
| Typed-repair gate | not run / closed |

Candidate construction used demonstrations and blind query inputs only.  It
was reconstructed twice byte-identically before the separate scoring command
read public-training query outputs.  The scorer revalidated the cohort, every
blind/source hash, the old freeze and result IDs, the semantic-test receipt,
and the complete source contract.  Public evaluation was not read and no
controller was trained.

## What was actually added

The representation is a small typed control program, not a task-ID rule:

```text
unique isolated pair lane
  -> ordered color correspondences
  -> protected control mask
  -> payload cells or source-color components
  -> typed render
```

It supports horizontal and vertical pair lanes, both pair directions, both
sequence directions, strict ambiguity rejection, repeated-pair collapse,
content-addressed AST round trips, and node-level execution traces.  The two
render nodes are:

- `ordered_rewrite_payload`: apply the ordered rules once, allowing a color
  produced by an earlier rule to be consumed by a later rule;
- `fill_exterior_bbox_background`: map each source-color payload component,
  flood only the exterior background of its one-cell-padded bounding box, and
  preserve enclosed holes and every existing object cell.

The legacy relational-delta v0.2 enumeration and all prior numerical artifacts
remain unchanged.

## Task-level interpretation

### `e4888269`: ordered rewrite program

The demonstrations are not explained by an unordered palette dictionary.  For
example, a sequence can contain `1 -> 4`, then later `4 -> 6`, then `6 -> 7`;
a payload cell starting at 1 must finish at 7.  Rule order changes across
demonstrations and query inputs, so the program parses the control lane anew on
every grid and excludes those cells from rewriting.

- exact program ID:
  `94a42d14dff6236312946cee9af3a87922ddc4f94797b920884daad2673447dd`;
- parse: horizontal pairs, forward pair direction, forward sequence;
- render: `ordered_rewrite_payload`;
- query outputs: 2/2 exact;
- v0.2 candidate count for this task: 0;
- content-novel frontier: 1.

### `5adee1b2`: exterior object-region rendering

Repeated color pairs at the canvas edge collapse into two mapping rules.  The
source-color payload objects remain unchanged.  Their mapped color fills the
background reachable from the boundary of each one-cell-padded bounding box;
background holes fully enclosed by the object stay unfilled.

- representative deduplicated program ID:
  `c353e6d94ef7fb74ced81da7c623be2309ee586e610b479747ea1ffe20a5d9f8`;
- two demo-exact programs differ only by sequence direction; because the
  demonstrated rules are independent, they produce one exact query bundle;
- render: `fill_exterior_bbox_background`;
- query outputs: 1/1 exact;
- v0.2 candidate count for this task: 0;
- content-novel frontier: 1.

## Why this is still far from the original motivation

The positive result repairs one localized candidate-language omission.  It
does not overturn the failure matrix:

- 46/48 v0.2 failures remain outside this family;
- the exposed 50-task selectable union is only 8%, far below the 25%--30%
  precondition for controller work;
- no visual posterior selected the program or filled an AST hole;
- no natural near-miss was repaired;
- no matched-cost restart comparison exists;
- no residual intervention changed an action.

The correct conclusion is therefore “frontier expansion demonstrated on one
bounded family,” not “strong object/code provider” and not “dynamic brain-area
switching.”

## Failure/validity classification

| Observation | Classification | Consequence |
|---|---|---|
| All semantic and replay guards pass | implementation evidence | The AST, parser, protected mask, render topology, hashing, and split boundary behave as specified. |
| Two selected tasks gain exact novel candidates | positive, outcome-exposed representation evidence | Keep the family frozen as a narrow optional provider. |
| No additional task among the other 48 is solved | candidate breadth limitation | Do not treat this as the requested reference-scale anchor. |
| Fresh 5/100 and typed-repair gates remain false | missing evidence, not a negative run | Do not train a controller or claim visual guidance. |

## Next bounded action

Freeze this family and stop adding task-level relation clauses.  The next
mainline action is to reproduce a published, reference-scale candidate
generator with released code/checkpoints and an auditable native-cost ledger.
Only after that anchor is fixed should a fresh, task-disjoint protocol compare:

```text
anchor cold restart
versus
visual posterior -> typed object/AST proposal -> anchor-conditioned search
```

The required confirmatory endpoint remains at least 5/100 unique recoveries at
equal native cost.  Failure at that gate must trigger a theory-level redesign
of dynamic functional switching rather than more router training.

## Immutable identities

- v0.1 candidate freeze ID:
  `7f607cb446f742cdb8f64d278c817955bb939b3878b8cb87b94d2f945b561a6c`;
- candidate freeze SHA-256:
  `47f51b6673434c8463add8430b42277547fc1208c7bf477be0af1b805d3e96f0`;
- v0.1 result ID:
  `77ff5b4d5e0ab66c1a3ef2f88217efda7ca07b5c34fc63b96974c70a00a2a4b5`;
- result SHA-256:
  `90df8276b9ab2c1eb32576f21265b4c5b1d673ef837bfc70d5bad812151c01a0`;
- existing v0.2 freeze ID:
  `85811f28942fc447d529be60e217d4cd4b5326c485ea901dda3d19b983fe32e5`;
- existing combined v0.2 result ID:
  `14d4ad29bec0da59fc6e405b72677cc917cf1f42d41e25af776003d3ccb3b884`.

Machine-readable artifacts are under `results/control_legend_dev_20260810/`.
