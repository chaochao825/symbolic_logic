# E01a M02b/M05c panel-expansion checkpoint

## Status and scope

This checkpoint is a verified, failure-driven symbolic development-smoke result. It
adds an independent full-span separator/panel parser and one typed ordered-overlay
operation. It does not add masked diffusion, a code model, a learned ranker, a
controller, panel-local D4 broadcasting, or periodic clipped stamping.

The parser and operation were chosen after inspecting failures on the same fixed
20-task ARC-AGI-2 public-training development smoke. The measured uplift is therefore
a post-hoc implementation diagnostic, not holdout generalization. The case set,
oracle set, search depth, beam width, instruction cap, and retained-program cap are
unchanged from the shape checkpoint.

- Evidence root: `results/e01a_panel_v1`.
- Runtime source fingerprint:
  `98b72451934a253ca50fcabb2e354a81f52c8d1bb3bfe9a37829b7f4a894c627`.
- Test-source fingerprint:
  `0f5246f63ec5729f539f2850ae1baca3ca5dee8fd6195ae9218d1a04506a831d`.
- Shared source-snapshot ZIP SHA-256:
  `b4bd5fa30f27dd47cad638a57f74b03d6cd6afd53844e71e24e036c317571463`.
- Full unit suite at freeze: 105/105 passing with warnings treated as errors.
- Independent code review: PASS with no P0, P1, or P2 findings.
- Both evaluation bundles pass `full_parent_replay_pass` through the verifier entry
  point.

## Frozen semantics and evidence boundary

`afts-full-span-panels/v0.1` detects same-color rows and columns spanning the full
grid, merges adjacent indices into separator bands, and takes complement intervals as
ordered panel rows and columns. It emits single-axis sequences and two-axis lattices,
including ragged edge panels, only when the indexed panels reconstruct the original
grid exactly. Outer-frame-only interpretations and views with fewer than two panels
are rejected. This is a deliberately narrow M02b view, not general line, motif,
frame, or arbitrary-region parsing.

M02b is stored independently in `panel_parses.jsonl`. Every observable train input,
train output, and query input has one content-addressed row, including rows with no
hypotheses. The public sidecar contains 161 rows and 115 hypotheses; its SHA-256 is
`77b068ff62254344ac76ce9bf48d289ad2ca73e11bc4044bb83b929e83ad4b28`.
The raw sidecar bytes enter the v3 pool behavior hash, and the verifier re-executes
the parser and checks canonical coverage and ordering rather than trusting hashes
alone.

M02a was not modified. On the identical public blind bundle, its 161-record
`parses.jsonl` is byte-identical to the shape checkpoint: 15,087,233 bytes with
SHA-256
`3b1cf3cebaf6ab2c21841920220308c0e79f6551bea1ef610f2e2eb461dbd9d5`.
The fixed M02a parse/object golden IDs remain
`99556c7f8f90a537f485` and `8c30f36fe2a5be88b579`.

DSL v0.3 adds only:

```text
overlay_panel_grid(background)
```

The operation requires one eligible two-axis lattice whose separator color differs
from the declared background and whose panels have equal shape.
It overlays panels in row-major order, taking the first value not equal to
`background` at each cell. Empty selection, non-unique selection, and incompatible
panel shapes are explicit semantic invalid results. It does not broadcast motifs,
transform individual panels, fill a larger canvas, or accept ragged panels.

The blind proposal stage intersects demonstration-derived background hypotheses,
requires valid panel execution on every training input, and uses observable
demonstration output shapes only as a direct/swapped shape gate. It never reads a
query output. Panel proposals are placed before the large object/background/color
Cartesian block so that a valid proposal remains reachable under the unchanged
64-instruction cap.

## Generator-known synthetic control v0.3

Version 0.3 contains 11 families and 33 tasks: the 27 prior controls plus three
atomic panel-overlay tasks and three overlay-then-rotate90 composition tasks. Panel
separator color varies between examples, and forced conflicting cells distinguish
row-major priority from fixed color priority.

- Case-set ID:
  `e97afdf6a405e837e29206620df4fdab48775eb09d291a95dacdad0e5f90ceff`.
- Oracle-set ID:
  `5ce261f3d443fdc871bb315ce6f7abf093814eaf2e49a1b80aef8473daa1a34d`.
- Pool-spec ID:
  `6d2c352bd2d51526937c14511e7f1d5c51d08a6c27ab2ceb9708950cfc008e5a`.
- Pool-content ID:
  `0ed1a42e5944dc260799951486fe853264e543a004aba32f5d72ec934aedff32`.
- Eval-spec ID:
  `43a8510f111158789d800508902cf949f078576d80268fb192b2de18c4223b07`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `65d0fd0e4caa1bbbde2d1cdf3bbb0a22ecfcbf1b4650ce766cabb483c7877613`,
  `504ff24f960771991dec8b02dd0fa9f3e32eda83289ed65fd658ece03c00e564`,
  `0b8dc0f34e60dc7fbd668ad4b65da269a9d2f3ba178254ff448f46d1f351a155`,
  and `08fb01a544e0c31beef48fc5d65ec33665bdf58cdfdfd7a3d0aa393231af3654`.
- 33 tasks, 66 query pairs, 264 panel-parse rows, 148 panel hypotheses,
  65,506 expansions, 327,530 executions, 467 retained demo-exact programs, 921
  candidate records, and 84 unique query outputs.
- Generator replay, semantic program coverage, task-first output coverage, micro
  pair coverage, and strict task coverage are all 1.0 at output/semantic rank 1.
- Both new families have exact generator-AST evaluated, retained, and rank-1 rates
  of 1.0.
- Across all 33 tasks, exact generator-AST evaluated/retained recall is 30/33 =
  0.909091. The three misses are the unchanged `crop_rotate90` family: after the DSL
  version changes program-ID tie-breaking, the depth-1 semantic-deduplicated frontier
  selects equivalent `crop_largest_object` prefixes, so the exact generator
  compositions are not evaluated at depth 2. Those three tasks still have generator
  replay, semantic coverage, and output coverage of 1.0; this structural identity
  regression is reported rather than relabeled as exact-AST success.
- Summed measured search wall time is 567.22 seconds; maximum per-task traced peak
  memory is 12,252,013 bytes.

The suite remains a grammar-aligned execution/search control. It does not estimate
real ARC coverage or compositional OOD generalization.

## Fixed ARC-AGI-2 public-training development smoke

- Dataset commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Case-set ID, byte-identical blind tasks, and oracle-set ID are unchanged:
  `93fe4870982313fb840d27bc4082e079f41d8b58d4f679cea15b5e5b1381b177`
  and
  `1b6e1c3ad153e7f090234829c23d5df1f3778bb14ea374a58d30060edb8f758e`.
- Pool-spec ID:
  `56da61f1d24ba2cedc3836f80c0d84b3c22456a4d0315c831c3bcb3ec248664e`.
- Pool-content ID:
  `c2a4677b41d2ee230792e4c31544a2a624e9c066f36e6c251ae0f160c247b1cf`.
- Eval-spec ID:
  `495a351fe263266e15eb4622fa7019c304647ea75086ff50e7407b68e216a8df`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `0774ed3c40c381343b430254e78ae2f518509ee41482c9599cea95d37cd32176`,
  `0de23635ccb879be376dd023bdb29fdbaea40b164f7d688d5c3e5a53aaecf09a`,
  `00dc886872389167cdd22f8850083d4d00ae3fc045e018f7e3a2b43fdc4d7629`,
  and `144032ad96fe58c17ec2a6dae03241f651419591cc08b76125483ec00114e759`.
- 20 tasks and 21 query pairs; 15 shape proposals; 161 panel-parse rows and 115
  panel hypotheses; 46,161 expansions; 210,573 executions; 193 retained demo-exact
  programs; 145 candidate records; and 19 unique outputs.
- Task-first output coverage and strict task coverage are 5/20 = 0.25 at output
  rank 1 and remain 0.25 through rank 128.
- Micro pair coverage is 5/21 = 0.238095 at output rank 1 and remains unchanged
  through rank 128.
- Semantic solving-program coverage is 4/20 = 0.20 at program rank 1 and 5/20 =
  0.25 by program rank 8. Output deduplication still places every solved task's
  correct grid at output rank 1.
- Summed measured search wall time is 1,123.49 seconds; maximum per-task traced peak
  memory is 25,431,515 bytes.

### Matched-budget before/after

| Measure | Shape v1 | Panel v1 | Change |
|---|---:|---:|---:|
| task-first/strict coverage@1 | 4/20 = 0.20 | 5/20 = 0.25 | +1 task |
| micro pair coverage@1 | 4/21 = 0.190476 | 5/21 = 0.238095 | +1 pair |
| expansions | 46,053 | 46,161 | +108 |
| program executions | 209,817 | 210,573 | +756 |
| retained demo-exact programs | 172 | 193 | +21 |
| candidate records | 124 | 145 | +21 |
| unique query outputs | 18 | 19 | +1 |

The prior solved set `1f85a75f`, `b1948b0a`, `c59eb873`, and `a416b8f3`
remains covered. Only `a68b268e` receives a panel instruction proposal:
`overlay_panel_grid(background=0)`. It is first demo-exact at expansion 9, retains
21 exact programs, emits 21 candidate records collapsing to one unique query output,
and solves the query at output rank 1.

`92e50de0` is a deliberate negative boundary. Its four input grids all receive one
panel lattice, with 36, 49, 49, and 64 indexed panels, so the M02b representation is
present. The ragged examples are invalid for ordered overlay, no panel instruction is
proposed, and the task remains without an exact program or candidate. Solving it
requires a separate periodic row/column broadcast and clipped-stamping operation.

Fifteen tasks now remain without a correct candidate. This checkpoint therefore
still does not justify ranker or controller training.

## Decision and next action

The slice passes its narrow gate: the new parser has executable, content-addressed
replay; both new generator-known families pass; the prediagnosed direct task is added
at small matched-budget cost; all four prior tasks remain solved; and the ragged
periodic task stays outside the claimed operation boundary.

The next registered symbolic slice was M05d, an indexed single-axis panel-sequence
D4 group action. It has now been executed under `results/e01a_panel_d4_v1`: the
parameterized action solves `8e5a5113` while preserving this checkpoint's five-task
solved set. Two-dimensional periodic broadcast with ragged clipping remains a
separate later slice for `92e50de0` so that the two inductive biases can be measured
independently. See `notes/results/e01a-panel-d4-expansion.md`.
