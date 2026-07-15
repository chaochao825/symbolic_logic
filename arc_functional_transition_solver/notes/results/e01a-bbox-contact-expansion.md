# E01a M02c/M05f axis-ray bbox-contact checkpoint

## Status and scope

This checkpoint is a verified-by-replay, failure-driven symbolic development-smoke
result. It adds one immutable relation sidecar, one typed local renderer, and one
blind structural proposer. It does not add masked diffusion, a code model, a learned
ranker, a controller, arbitrary line drawing, object motion, or multi-source merging.

The subsystem was pre-registered in
`notes/design/m02c-axis-ray-contract.md` after inspecting failures on the same fixed
20-task ARC-AGI-2 public-training development smoke. Its direct target was at most
`1f642eb9`; `1b8318e3` and its multiple-anchor move-and-erase behavior were explicitly
excluded. The measured uplift is therefore a post-hoc implementation diagnostic,
not holdout generalization.

- Evidence root: `results/e01a_bbox_contact_v1`.
- Runtime source fingerprint:
  `3955ed3dedf1a34176e55dd38e85990337fed98512e5e626cd5edf4f45349c67`.
- Test-source fingerprint:
  `3372e76503d01a8e66d4c37d9fe8205605d0e104b94b9d66f452a311f5e025dd`.
- Shared source-snapshot ZIP: 93,895 bytes, SHA-256
  `d115eb1d5b95ede00a7fd6bde2ec44bda2caf5e7eb0a9244226a25fa71b4672a`.
- Full unit suite at freeze: 133/133 passing with warnings treated as errors.
- Independent code review: PASS with no remaining P0, P1, or P2 findings.
- Both evaluation bundles and both standalone formal verifier commands return
  `full_parent_replay_pass`.
- Independent formal-evidence audit: PASS with no remaining P0, P1, or P2
  findings after rebuilding all 161 public and 408 synthetic M02c rows, replaying
  every bbox bound/proposal/cost ledger, and recomputing all eight closed-world
  artifact manifests and parent bindings.

## Frozen relation and action semantics

M02c records `afts-axis-aligned-bbox-contact/v0.1` hypotheses in
`relation_parses.jsonl`. For each declared background, it reuses exactly the M02a
4-connected, single-color component view. A filled monochrome object whose bounding
box is at least 2 by 2 is an anchor candidate. Every other required object must be a
different-color singleton strictly outside the anchor and aligned to one of its row
or column spans. The relation points from the singleton to the nearest anchor
boundary cell and records marker/anchor identities, the source M02a parse identity,
axis, direction, gap, open-ray background count, ray clarity, and projected cell.

A complete hypothesis requires a filled anchor, at least one relation, a clear ray
for every required marker, no unmatched or non-singleton objects, and no projected
destination collision. Two complete anchors are non-unique. When no complete
hypothesis exists, the renderer fails closed in this order:

1. `target_collision`;
2. `occluded_ray`;
3. `incompatible_relation_geometry`;
4. `empty_selection`.

`non_unique_selection` precedes those cases when more than one complete anchor
exists. These specialized outcomes remain distinct versioned DSL invalid codes.

DSL v0.6 adds only:

```text
paint_bbox_contacts(background)
```

The `afts-bbox-contact-render/v0.1` action copies each marker color to its projected
anchor-boundary cell. It preserves the marker, open ray, canvas, background, and all
other anchor cells. Anchor color, bounding box, marker list, directions, and
destinations are derived from the input and cannot be instruction parameters. The
action is copy-only: it is not a move, erase, line, stamp, or destination-list
primitive.

The sidecar is closed-world and replayable. Every observable training input,
training output, and test input has one row. Each row evaluates exactly the ordered
non-`None` backgrounds returned by
`background_hypotheses(grid, include_none=True, max_backgrounds=3)`. Query outputs
are absent, and the formal validator reconstructs every row from the blind grid.
M02a and M02b are not modified.

## Blind proposal and cost ledger

`afts-bbox-contact-proposer/v0.1` intersects the training-input background domains,
constructs one structural bound for every common background, and visits every
demonstration without early stopping. A background becomes an admissible binding
only when every training input has exactly one complete relation hypothesis. Each
binding is executed on every demonstration. A proposal survives only when all
executions are valid and their shapes agree with the observed output shapes,
directly or transposed.

The proposer never compares produced pixels with training-output pixels and never
reads query outputs, task IDs, oracle hashes, residual coordinates, or
target-derived destinations. Ordinary M06 demo-exact execution is the pixel-exact
selector. The v6 search row separately records relation bounds, structural checks,
anchor candidates, relation checks, admissible bindings, action trials,
demonstration pre-executions, proposals, cap survival, and the ordinary program
execution count. The formal replay verifies the registered closure
`relation_checks = sum A_i(O_i-A_i)` and all associated bound and cost identities.

The proposal is ordered after M05e and before generic object/crop/recolor options.
This ordering is behaviorally relevant because `1f642eb9` has more than 64 pre-cap
options.

## Generator-known synthetic control v0.6

Version 0.6 contains 17 families and 51 tasks: all 45 v0.5 controls plus three
atomic `paint_bbox_contacts` tasks and three
`paint_bbox_contacts_rotate180` compositions. The new cases cover backgrounds 0,
2, and 3; different anchor colors, shapes, and positions; all four ray directions;
gaps 0, 1, and 2; marker counts 3 through 6; repeated marker colors; and multiple
markers on one side. Unit controls separately cover ambiguity, collisions,
occlusion, unaligned and non-singleton markers, non-solid anchors, empty relations,
wrong backgrounds, and identity/sidecar/ledger/config tampering.

- Case-set ID:
  `79ef5b6f4439d1cf71439ff9d7ee7e172147225bbb8dbb9e5dfd968571e6b39f`.
- Oracle-set ID:
  `9d180c75df5e18f8a76ea3ffeda65ee260ae80257a0db975e7cb2b9e59536da9`.
- Pool-spec ID:
  `b1b76eccce5b500d9e755842dfa471d83fd7fcdc8ded2dbf9e4194c3b982d207`.
- Pool-content ID:
  `fd36d3c0986517d89c18d38bc59311c4cf79deab9a4d610102b5b96c9b70aca2`.
- Eval-spec ID:
  `fa03b465bc345ff4c8b2742323893b54271bbb568dcc700a1394f8c0e3aa98b7`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `cfe3b929613d3b10ada3c7ad75ddd5801fb7fa3583a5f485b61694466e3ee380`,
  `72356ab3569dc99a0c0d9ae992f146476757e2e8974be500c2725c12ed81096a`,
  `75e139d2da97c68af13ba322f50ed5054e7203845c9766c75e8898092db3395e`,
  and `a23a375382b8bc630dee77976cac3651d9bf5e23d82fed9860f46fee131694fe`.
- 51 tasks and 102 query pairs; 408 rows in each M02a, M02b, and M02c sidecar;
  816 background-specific relation bundles and 1,558 relation hypotheses.
- 111,199 expansions, 555,995 program executions, 592 retained demo-exact
  programs, 1,166 candidate records, and 123 unique query outputs.
- The M02c/M05f ledger records 60 bounds, 180 structural checks, 55 anchor
  candidates, 289 relation checks, six admissible bindings, six action trials,
  18 demonstration pre-executions, six proposals, and six cap-surviving options.
- Generator replay, generator evaluated/retained rates, semantic/output coverage,
  task-first coverage, micro pair coverage, and strict task coverage are all 1.0.
- Exact generator-AST recall is 38/51 at rank 1 and 51/51 by rank 8. The atomic new
  family is 3/3 at rank 1. Each new composition's exact generator AST is rank 2,
  hence 0/3 at rank 1 and 3/3 by rank 8, while an output-equivalent program is
  already present at rank 1.
- The old 45 controls now place 35 exact generator ASTs at rank 1, four at rank 2,
  three at rank 3, and three at rank 4, while all 45 remain present by rank 8.
  Relative to DSL v0.5, the three old `crop_rotate90` controls move from rank 1 to
  rank 3, the three `rotate90_recolor` controls from rank 1 to rank 2, and the three
  `keep_largest_object` controls from rank 1 to rank 4. One earlier panel-sequence
  composition accounts for the fourth old rank-2 case. The versioned DSL program
  IDs change deterministic hash tie-breaking under the retained exact-program cap;
  old search totals and candidate-output sets are unchanged, all old tasks retain
  semantic/output coverage at rank 1, and none receives an M05f proposal. This is
  exact-AST identity churn, not a capability regression.
- The ordered old-45 blind identity digest is
  `d68083185d87ea33505e3c13cc15827066952272e27e24520c69ffe6000fb988`.
  The old subset remains exactly 98,467 expansions, 492,335 executions, 534 exact
  programs, 1,050 candidates, and 111 unique outputs, with zero M05f proposals.
  These are the complete v0.5 totals. The six new tasks account for the remaining
  12,732 expansions, 63,660 executions, 58 exact programs, 116 candidates, and 12
  unique outputs.
- Summed measured search wall and process time are 2,007.4648222 and
  2,002.859375 seconds; maximum per-task traced peak memory is 17,000,058 bytes.

The suite is a grammar-aligned execution/search control, not an estimate of real
ARC coverage or compositional OOD generalization.

## Fixed ARC-AGI-2 public-training development smoke

- Dataset commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Case-set ID and oracle-set ID are unchanged:
  `93fe4870982313fb840d27bc4082e079f41d8b58d4f679cea15b5e5b1381b177`
  and
  `1b6e1c3ad153e7f090234829c23d5df1f3778bb14ea374a58d30060edb8f758e`.
- Pool-spec ID:
  `6772be7c69a768b15d8e9c8f7b788400acf9feba311de97a613c4968fc08f232`.
- Pool-content ID:
  `e2b5ea2084fdf1ac072a6e15c7d13024dd8c63a019961b4c1c4b9fb0386ad49d`.
- Eval-spec ID:
  `a93c3da669fa59fe5483634095c94aaf056f3608ccd4e07e542bb9371cd41c34`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `7df2afbe088ad866f329ab654d90dd1678a0e985adee7af138fee692e66aab3b`,
  `8cdcbe50defe98de577b21aaf4e94ac385dbbfd454c1aff14212cc95a43c64a0`,
  `3a39d4703955a5359166b4a0ffdcf73689d62e7c2468b8b7525d2428c570ca7e`,
  and `3f4ce2b734ad281b22a485a4e51edf7aa10e435137deda4b95c2bdc432f531d0`.
- Blind tasks remain 82,753 bytes with SHA-256
  `e80dacf901852e878d49bc2f2fe124792460a2bbfe4553adc10c2e36aea7f290`;
  oracle rows remain 16,711 bytes with SHA-256
  `f3ec853c33f7b78482769ec260a5215b7c266f0803a942419e7d8072341afafa`.
- M02a remains 15,087,233 bytes with SHA-256
  `3b1cf3cebaf6ab2c21841920220308c0e79f6551bea1ef610f2e2eb461dbd9d5`;
  M02b remains 336,259 bytes with SHA-256
  `77b068ff62254344ac76ce9bf48d289ad2ca73e11bc4044bb83b929e83ad4b28`.
- The additive M02c sidecar has 161 rows, 322 background bundles, 892
  hypotheses, 1,763,915 bytes, and SHA-256
  `f4101d804770242d705af168563a607a80452348c9356dc355b100d5fc963f46`.
- 20 tasks and 21 query pairs; 48,177 expansions; 218,637 program executions;
  229 retained demo-exact programs; 181 candidate records; and 22 unique outputs.
- The aggregate M02c/M05f ledger records 28 bounds, 100 structural checks, 34
  anchor candidates, 216 relation checks, one admissible binding, one action trial,
  three demonstration pre-executions, one proposal, and one cap-surviving option.
- Task-first, strict, and semantic-program coverage are 8/20 = 0.40 at rank 1 and
  every reported cutoff. Micro pair coverage is 8/21 = 0.380952 at rank 1 and
  remains unchanged through rank 128.
- Summed measured search wall and process time are 1,559.3622582 and
  1,553.828125 seconds; maximum per-task traced peak memory is 32,412,259 bytes.

### Matched-cap before/after

| Measure | M05e periodic v1 | M02c/M05f v1 | Change |
|---|---:|---:|---:|
| task-first/strict coverage@1 | 7/20 = 0.35 | 8/20 = 0.40 | +1 task |
| micro pair coverage@1 | 7/21 = 0.333333 | 8/21 = 0.380952 | +1 pair |
| semantic program coverage@1 | 7/20 = 0.35 | 8/20 = 0.40 | +1 task |
| instruction options pre-cap | 1,764 | 1,765 | +1 |
| instruction options post-cap | 1,167 | 1,167 | 0 |
| instruction options truncated | 597 | 598 | +1 |
| expansions | 48,177 | 48,177 | 0 |
| program executions | 218,637 | 218,637 | 0 |
| retained demo-exact programs | 220 | 229 | +9 |
| candidate records | 172 | 181 | +9 |
| unique query outputs | 21 | 22 | +1 |

The one new proposal displaces one generic tail option under the same 64-option cap;
the aggregate post-cap option count, expansion count, and ordinary execution count
therefore remain fixed. Relation construction and its three proposal-stage demo
executions are separately accounted rather than hidden inside
`program_execution_count`.

All seven prior solved tasks remain covered:
`1f85a75f`, `b1948b0a`, `c59eb873`, `a416b8f3`, `a68b268e`, `8e5a5113`, and
`92e50de0`. Only `1f642eb9` changes its task metrics. Its blind ID is
`blind_31f6a1f20461912a9616b2970ebd6a41965cef7212780c05b8bf496d15447464`.
The target retains 3,151 expansions and 12,604 ordinary program executions, while
its pre/post/truncated option ledger changes from 116/64/52 to 117/64/53.

For background 0, its bound is seven objects, one filled anchor candidate, and at
most six relations; background 8 has seven objects but no anchor or relation. The
three demonstrations account for six structural checks, three anchor candidates,
14 relation checks, one binding, one trial, three pre-executions, and one proposal.
The blind structural expectations registered before implementation are therefore
met exactly.

Normal demo-exact evaluation first succeeds at expansion 9. It retains nine exact
programs and emits nine candidates, all collapsing to one query output. The atomic
rank-1 program is

```text
paint_bbox_contacts(background=0)
```

with program ID
`a0d9bff7efbc3f4574a25b3ba852701e43e2c9a37a33f65e2ed9eaa19f02669a`
and output key
`f23c4677b41d66ff740e6fa4136e9038fb507cbe6d03b751d9a1ea684d67818c`.
All +9 exact programs, +9 candidates, and +1 unique public output occur on this task
alone.

Twelve public-smoke tasks remain without a correct candidate. This checkpoint does
not justify ranker or controller training.

## Decision and next action

The narrow M02c/M05f gate passes at the formal replay level: the registered direct
task is added, no other public task changes coverage, all previous solutions are
preserved, the old synthetic blind-task identities and aggregate behavior totals
are retained, M02a/M02b
bytes remain frozen, and relation/proposal/cap/cost closure replays exactly.

The evidence supports only singleton-to-solid-rectangle axis relations and a
copy-to-first-boundary-cell renderer. It does not support nearest-anchor assignment,
move-and-erase behavior, arbitrary rays, general scene graphs, template stamping,
learned ranking, functional switching, holdout transfer, or public evaluation.

A subsequent input-only structural scan over the fixed smoke and all 1,000
ARC-AGI-2 public-training tasks found that conservative signatures for the cleanest
remaining legend-map (`0becf7df`) and feature-to-bar (`fcc82909`) diagnoses each
identify only that diagnosed task. Max-span scoped rewriting remains ambiguous, and
nearest-anchor motion requires a substantially larger assignment/action contract.
The next planned direction is therefore M04a global masked-grid generation rather
than a fifth consecutive single-task symbolic macro. Its immediate action is to
pre-register and review a source contract; no M04a checkpoint or pool is frozen yet.
The first slice is generation-only, while local residual-directed repair remains a
separate M11 experiment. E01b remains the first gate at which cross-source
complementarity may be claimed.
