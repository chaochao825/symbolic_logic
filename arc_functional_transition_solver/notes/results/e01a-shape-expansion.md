# E01a M03a/M05b shape-expansion checkpoint

## Status and scope

This checkpoint is verified as a failure-driven symbolic development-smoke result.
It adds a blind axis-integer output-shape proposer and two typed DSL primitives:
pixel scaling and whole-grid tiling. It does not add M02b parsing, diffusion, a code
model, a learned ranker, or a functional controller.

The two target rules were selected after inspecting failures on the same development
smoke. Consequently, the measured uplift is a post-hoc development diagnostic, not
holdout generalization. The fixed 20-task case set, search depth, beam width,
instruction cap, and retained-program cap are unchanged from E01a v1.

- Evidence root: `results/e01a_shape_v1`.
- Runtime source fingerprint:
  `b1abacbec9e4c9f5175f55fb58844c280c4a6800ef8896d8afa884666c59ac2a`.
- Test-source fingerprint:
  `dcecf7ce3112f91b7208c21c1f787a1f67b985f53b89653cbd2539dc8d4760c1`.
- Shared source-snapshot ZIP SHA-256:
  `46b3492d15804c387c3c1cf6836df6c91170b0f2daf8cf6e5a3f776bcfc8ce80`.
- Full unit suite at freeze: 94/94 passing with warnings treated as errors.
- Design CSV validator: passing.
- Independent code review: PASS with no P0, P1, or P2 findings.
- Both synthetic and public evaluation bundles passed
  `full_parent_replay_pass` through the independent verifier entry point.

## Frozen semantics

`afts-output-shape/v0.1` accepts a proposal only when every demonstration has the
same positive integer row and column ratio between output and input dimensions. It
projects that ratio onto query-input dimensions without reading query outputs. A
projection exceeding 30 by 30 is rejected. The `(1,1)` rule is recorded as an
identity shape hypothesis but emits no redundant DSL instructions.

For each non-identity ratio, search proposes both:

- `scale_pixels(row_factor, column_factor)`, which repeats individual cells; and
- `tile_grid(row_repeats, column_repeats)`, which repeats the whole grid.

All demonstrations must execute exactly, so dimensions alone do not choose between
these semantics. The two instructions appear after D4 and an optional inferred
color map but before the background/object/color Cartesian products. They therefore
remain reachable under the existing 64-instruction cap. Every intermediate output
is checked against the ARC size limit before allocation; overflow is a semantic
invalid result rather than an executor internal error.

This is M03a plus M05b, not full M03. It excludes output-size constants, reductions,
axis-swapped ratios, crop-derived sizes, feature-bound formulas, arbitrary canvases,
and multi-object spatial composition.

## Generator-known synthetic control v0.2

The v0.1 seven-family suite is retained as historical evidence. Version 0.2 appends
three `scale_pixels_2x2` tasks and three `tile_horizontal_2` tasks without changing
the old family seeds.

- Case-set ID:
  `1d2be4cf3d0d7e3a14191f59175a2a063a7db171edf770b77a783994563bbb4f`.
- Oracle-set ID:
  `f8c77054240163d21c0db07eebfcbca46901af316d76ca1c6005043d4410e98b`.
- Pool-spec ID:
  `8c7eb7010e0d763a8c4fcacbaf1bb4635be05ed7b4302de3fa7ce93ce02a4956`.
- Pool-content ID:
  `1b126f5615c7ef142d4c0a7c1f75bf3e2cd65da4be919a99517f923bd3d2cb10`.
- Eval-spec ID:
  `a02eb5d606c0c396c44e11a4878897a783ef3355971827c968a0adb53c935035`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `0f0944ab5c832a4a19f792134179f740474dbcdabb19ab8fe841f523e1e0e2eb`,
  `7baa4d634156a234b3de7cacd0cac722ecec36698a75ed2aa26abd8e3779aa8c`,
  `f30e76ea8176f72b0f11daa3ef822ffdd8c229c96d6cc1960b34e29c52885c33`,
  and `f3dcfefb7ce0ef8f0d1e76e2a0d89687931e82a962401bc0e5a6d933a5e77005`.
- 27 tasks, 54 query pairs, 15 shape proposals, 54,412 expansions, 272,060
  program executions, 413 retained demo-exact programs, 813 candidate records,
  and 72 unique query outputs.
- Generator replay, generator evaluated, and generator retained rates are 1.0.
- Semantic program coverage and all output coverage measures are 1.0 at rank 1.
- Exact generator-AST recall is 15/27 = 0.555556 at program rank 1 and 1.0 by
  rank 8. Equivalent exact programs explain the rank-1 structural gap.
- Summed measured search wall time is 346.88 seconds; maximum per-task traced peak
  memory is 13,716,645 bytes.

This suite remains a grammar-aligned execution/search control. It is not real ARC
coverage or compositional OOD evidence.

## Fixed ARC-AGI-2 public-training development smoke

- Dataset commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Case-set ID, unchanged from v1:
  `93fe4870982313fb840d27bc4082e079f41d8b58d4f679cea15b5e5b1381b177`.
- Oracle-set ID, unchanged from v1:
  `1b6e1c3ad153e7f090234829c23d5df1f3778bb14ea374a58d30060edb8f758e`.
- Pool-spec ID:
  `ce41da03c0504675f03c4ff000c9fd603bf21e071273f05e4dec9d9c923013af`.
- Pool-content ID:
  `e9b88abef2cb7a0f0e58f1a4a4dffe5eb6fca2bf19cb7846fa32303a1f53dd2e`.
- Eval-spec ID:
  `ad50f0ce0c3e986ea1e2532bcde3efc43ff129127ea42d448340c5b3a46a2d13`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `f134d7f61ab2b39934d02a347a00efc31c15af514e93895834ff9915951bcad8`,
  `efcd163148c764efb509f26e2ebccbd3588f2e3d13122bcd46fccbd92c154a2a`,
  `c33434b1d439213bc1fc843904b20a396a886f146ce28d0d7ecdc1b3828dbe9b`,
  and `a856dece6bdc8ae6e52fb71dbe0e48512eee228b793b946e8c90d0f3101ae5e1`.
- 20 tasks and 21 query pairs; 15 shape proposals, of which 13 are identity and
  two are non-identity; 46,053 expansions; 209,817 program executions; 172
  retained demo-exact programs; 124 candidate records; and 18 unique outputs.
- Task-first output coverage and strict task coverage are 4/20 = 0.20 at output
  rank 1 and remain 0.20 through rank 128.
- Micro pair coverage is 4/21 = 0.190476 at output rank 1 and remains unchanged
  through rank 128.
- Semantic solving-program coverage is 3/20 = 0.15 at program rank 1 and 4/20 =
  0.20 by program rank 8. This program ordering does not reduce output-rank-1
  coverage.
- Summed measured search wall time is 1,117.95 seconds; maximum per-task traced
  peak memory is 25,440,452 bytes.

### Matched-budget before/after

| Measure | E01a v1 | Shape v1 | Change |
|---|---:|---:|---:|
| task-first/strict coverage@1 | 2/20 = 0.10 | 4/20 = 0.20 | +2 tasks |
| micro pair coverage@1 | 2/21 = 0.095238 | 4/21 = 0.190476 | +2 pairs |
| expansions | 45,864 | 46,053 | +189 |
| program executions | 209,061 | 209,817 | +756 |
| retained demo-exact programs | 140 | 172 | +32 |
| candidate records | 93 | 124 | +31 |
| unique query outputs | 13 | 18 | +5 |

The prior solved tasks `1f85a75f` and `b1948b0a` remain covered. The two new
covered tasks are:

- `c59eb873`: `scale_pixels(2,2)` is first exact at expansion 9; the task retains
  17 exact programs and produces three unique outputs.
- `a416b8f3`: `tile_grid(1,2)` is first exact at expansion 10; the task retains
  15 exact programs and produces two unique outputs.

All other 16 tasks still have zero retained demo-exact programs and zero candidates.
The remaining primary failure labels are therefore seven missing primitives, three
selectors, three contextual bindings, two parser views, and one output-shape or
spatial-composition rule. The result supports the failure taxonomy but does not show
that these counts generalize beyond this post-hoc development set.

## Decision and next action

The slice passes its narrow gate: the two new primitives pass generator-known
controls, add the two prediagnosed correct outputs at almost unchanged executor
budget, preserve both old solved tasks, and survive full replay. It does not justify
ranker or controller training because 16/20 tasks still have empty candidate sets.

That registered successor has now been executed under `results/e01a_panel_v1`.
The independent M02b sidecar plus `overlay_panel_grid(background)` preserves M02a
bytes, adds `a68b268e`, and raises the same post-hoc smoke to 25% strict/task-first
coverage. `92e50de0` receives the intended ragged panel representation but remains
unsolved, preserving the boundary to a separate parity-broadcast and clipped-stamp
primitive. See `notes/results/e01a-panel-expansion.md`.
