# E01a M05e periodic panel-lattice checkpoint

## Status and scope

This checkpoint is a verified, failure-driven symbolic development-smoke result. It
keeps the frozen M02a and M02b representations and adds one typed action that
broadcasts an inferred seed over a two-axis panel congruence class, with clipping
restricted to shorter final panel rows and columns. It does not add masked
diffusion, a code model, a learned ranker, a controller, arbitrary stamping,
multi-source merging, or relational object selection.

The action was chosen after inspecting failures on the same fixed 20-task ARC-AGI-2
public-training development smoke. The measured uplift is therefore a post-hoc
implementation diagnostic, not holdout generalization. The case set, oracle set,
search depth, beam width, instruction cap, and retained-program cap are unchanged
from the indexed panel-sequence D4 checkpoint.

- Evidence root: `results/e01a_panel_periodic_v1`.
- Runtime source fingerprint:
  `d34264e84123b503fff917664e09a8fd4b971a716a91800d1d2d67dc96ab2095`.
- Test-source fingerprint:
  `c055dfa088ebad015d302132494a9737a0f046d0a9038633ddeb198bf48f7b0c`.
- Shared source-snapshot ZIP SHA-256:
  `cc69e994f9e555f0e71a5253fa26488df5a2ac05fb4b212bf3011d221c405f1b`.
- Full unit suite at freeze: 120/120 passing with warnings treated as errors.
- Independent code review: PASS with no remaining P0, P1, or P2 findings.
- Both evaluation bundles pass `full_parent_replay_pass` through the verifier entry
  point.

## Frozen action semantics

DSL v0.5 adds:

```text
broadcast_panel_lattice_periodic(background, row_period, column_period)
```

`afts-panel-lattice-periodic/v0.1` considers only M02b hypotheses with both row and
column separator bands and `separator_color != background`. Relative to the declared
background, exactly one panel must contain non-background pixels. That dynamic seed
must have the nominal full panel shape. Panel heights must have the form
`H, ..., H, h_last`, and widths must have the form `W, ..., W, w_last`, where only
the final values may be shorter. Leading or interior raggedness and a seed in a
partial panel are incompatible.

If the seed is at panel index `(s_r, s_c)`, destination `(r, c)` is selected exactly
when

```text
(r - s_r) mod row_period = 0
(c - s_c) mod column_period = 0.
```

A full destination receives the whole seed. A shorter bottom/right destination
receives the visible top-left prefix `seed[:height, :width]`. Unselected panels,
separator bands, and other pixels are preserved. Periods are integers in `[1, 30]`;
a period larger than the finite lattice count is valid and simply makes no extra
copy on that axis.

The instruction has no phase, seed index, separator color, panel count, panel size,
parity label, destination list, or overwrite policy. Multiple complete hypotheses
or multi-seed evidence are non-unique; incompatible suffix geometry follows; an
otherwise absent/empty selection is last. One complete valid hypothesis executes
even when other hypotheses are invalid.

M02a and M02b were not changed. On the public bundle, all of the following are
byte-identical to `results/e01a_panel_d4_v1`:

- blind tasks: 82,753 bytes, SHA-256
  `e80dacf901852e878d49bc2f2fe124792460a2bbfe4553adc10c2e36aea7f290`;
- oracle rows: 16,711 bytes, SHA-256
  `f3ec853c33f7b78482769ec260a5215b7c266f0803a942419e7d8072341afafa`;
- M02a parses: 15,087,233 bytes, SHA-256
  `3b1cf3cebaf6ab2c21841920220308c0e79f6551bea1ef610f2e2eb461dbd9d5`;
- M02b panel parses: 336,259 bytes, SHA-256
  `77b068ff62254344ac76ce9bf48d289ad2ca73e11bc4044bb83b929e83ad4b28`.

## Blind proposal and cost ledger

`afts-panel-lattice-periodic-proposer/v0.1` uses only training inputs to find a
structurally unique eligible lattice for each common background. Its maximum row and
column periods are the largest observed training lattice counts minus one. It then
executes every period pair on every demonstration without early stopping. A proposal
survives when every execution is valid and its shape agrees with the observed
demonstration-output shape, directly or swapped. It does not compare output pixels
and never reads a query output. This intentionally broad structural/shape gate keeps
the action usable as a depth-2 composition prefix; ordinary demo-exact program
evaluation selects the fitting period.

The v5 search row records bounds, proposals, structural checks, period trials,
demonstration pre-executions, pre/post-cap option counts, and truncation. Canonical
validation binds every proposal to one bound, checks ordered uniqueness and the
finite domain, and closes positive trial cost to positive demo-execution cost. These
fields enter the behavior hash and full-parent replay. Pre-executions remain separate
from `program_execution_count`.

## Generator-known synthetic control v0.5

Version 0.5 contains 15 families and 45 tasks: all 39 prior controls plus three
atomic periodic-lattice tasks and three periodic-then-rotate180 composition tasks.
The new controls cover `(row_period, column_period)` values `(2,2)`, `(2,3)`, and
`(3,2)`; backgrounds `0`, `2`, and `3`; moving source phase; one- and two-cell
separator bands; uniform, bottom-ragged, right-ragged, and doubly ragged suffixes;
and selected versus unselected partial destinations. Composition outputs are
non-degenerate relative to either one-step prefix or suffix.

- Case-set ID:
  `99e2678ef72e4454711315898b028fa51e568479741f2e25be62a4e2e4024c0e`.
- Oracle-set ID:
  `06d0ee6c7aa0462f8c69928218696db2804c17d8620e0a9f139c5964c0aa3160`.
- Pool-spec ID:
  `24ed398419057a13ca082862f1c8cf817c1749aa7d7dcb9dd021eb5eae912df8`.
- Pool-content ID:
  `7c51f06962d7b089edc3ab31a65299d33f0592f98b7a8bb0105255adabc3bc85`.
- Eval-spec ID:
  `2ea5974038dc72333e3fa4dfda0abde52f850c98a5fc7b6a9ce807ccefaaf6cf`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `0892042bf9283b4fc0ecc8cac39b18aba5f02a7150f5e0429340c0918f0a7830`,
  `d129d959af36f744ff767e6ee68d16fcb3c108a2c5ab9e3e9211b1ca4fa380d9`,
  `213e73af0c153a5a01598246284d5b38926af68c99a48a08b4c32b1f5d40cbfc`,
  and `2848d02aa8775e3de7b42adfd1380c0cbb5745a872ebba03f9d093c6b0f675e4`.
- 45 tasks, 90 query pairs, 360 M02a rows, 360 M02b rows, 256 panel
  hypotheses, 98,467 expansions, 492,335 program executions, 534 retained
  demo-exact programs, 1,050 candidate records, and 111 unique query outputs.
- The M05e ledger records six bounds, 162 structural checks, 78 period trials,
  234 demonstration pre-executions, and 78 proposals; all 78 survive the cap.
- Generator replay, evaluated rate, retained rate, semantic-program coverage,
  task-first coverage, micro pair coverage, and strict task coverage are all 1.0.
- Exact generator-AST recall is 44/45 at rank 1 and 45/45 by rank 8. Both new
  families are 3/3 at rank 1. The one rank-1 miss is an old
  `broadcast_panel_sequence_d4_flip_horizontal` control and is present by rank 8.
- Every old 39-task blind row, all 312 old rows in each parse sidecar, and every old
  oracle task semantic projection are retained. The old-subset pool counts remain
  exactly 84,160 expansions, 420,800 executions, 493 exact programs, 973 candidates,
  and 96 unique outputs, with zero M05e bounds or proposals.
- DSL v0.5 changes versioned program IDs and deterministic tie-breaking. The exact
  old `crop_rotate90` generator programs now enter the retained frontier, so the
  old-subset generator identity metric improves without changing old task semantics
  or pool-count totals. This is a search-order identity effect, not new real-task
  capability.
- Summed measured search wall time is 1,771.58 seconds; maximum per-task traced peak
  memory is 16,450,847 bytes.

The suite remains a grammar-aligned execution/search control. It is not an estimate
of real ARC coverage or compositional OOD generalization.

## Fixed ARC-AGI-2 public-training development smoke

- Dataset commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Case-set ID and oracle-set ID are unchanged:
  `93fe4870982313fb840d27bc4082e079f41d8b58d4f679cea15b5e5b1381b177`
  and
  `1b6e1c3ad153e7f090234829c23d5df1f3778bb14ea374a58d30060edb8f758e`.
- Pool-spec ID:
  `35f8d8a22d721d847a8f99f91b037c5d4839c798680624b691b8371c509ccfc1`.
- Pool-content ID:
  `c34da41d0634b882669e0e05fd7732b23746407229fc91749e356247ab29c088`.
- Eval-spec ID:
  `f78a581b44b47b5291b20c908d134c524dd5e484468897d63bbfb6e9a82aa3c4`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `7ce5a523449e3493522070952c871c523793571720c6cce22b61100f8d9d6920`,
  `e0e97c01b3e0f8202ddc459028f9dd1f0bab34fbca1e72a0273165e013f83241`,
  `f7a6a0d5745aa6001af935ea3fddf100b34481ae11ad99aec431259ebf29be4b`,
  and `c4df301982a08fc407bc82e73156193e30dcf0c12914bd5c50ee5e88152d6756`.
- 20 tasks and 21 query pairs; 48,177 expansions; 218,637 program executions;
  220 retained demo-exact programs; 172 candidate records; and 21 unique outputs.
- The M05e ledger records one bound, 100 structural checks, 36 period trials, 108
  demonstration pre-executions, 36 proposals, and 36 M05e options surviving the
  cap. Aggregate options are 1,764 before cap, 1,167 after cap, and 597 truncated.
- Task-first, strict, and semantic-program coverage are 7/20 = 0.35 at rank 1 and
  remain 0.35 through every reported cutoff.
- Micro pair coverage is 7/21 = 0.333333 at rank 1 and remains unchanged through
  rank 128.
- Summed measured search wall time is 1,378.46 seconds; maximum per-task traced peak
  memory is 32,450,027 bytes.

### Matched-cap before/after

| Measure | Panel-D4 v1 | Periodic v1 | Change |
|---|---:|---:|---:|
| task-first/strict coverage@1 | 6/20 = 0.30 | 7/20 = 0.35 | +1 task |
| micro pair coverage@1 | 6/21 = 0.285714 | 7/21 = 0.333333 | +1 pair |
| semantic program coverage@1 | 6/20 = 0.30 | 7/20 = 0.35 | +1 task |
| instruction options pre-cap | 1,728 | 1,764 | +36 |
| instruction options post-cap | 1,167 | 1,167 | 0 |
| instruction options truncated | 561 | 597 | +36 |
| expansions | 46,161 | 48,177 | +2,016 |
| program executions | 210,573 | 218,637 | +8,064 |
| retained demo-exact programs | 211 | 220 | +9 |
| candidate records | 163 | 172 | +9 |
| unique query outputs | 20 | 21 | +1 |

This is a matched-cap comparison, not an equal-compute comparison. Although the
post-cap option count is unchanged, replacing tail bulk options with 36 periodic
actions increases the number of distinct depth-1 semantic representatives for
`92e50de0`; its depth-2 frontier therefore adds 2,016 expansions and 8,064 program
executions. The 100 structural checks and 108 periodic demonstration pre-executions
are separately reported rather than folded into an incomplete total.

The prior solved set `1f85a75f`, `b1948b0a`, `c59eb873`, `a416b8f3`,
`a68b268e`, and `8e5a5113` remains covered. Only `92e50de0` receives an M05e bound
or proposals. Its bound is `background=0`, `max_row_period=6`, and
`max_column_period=6`; all 36 canonical period pairs survive its 134/64/70
pre/post/truncated cap ledger. The proposer does not identify `(2,2)` as correct.
Normal program execution finds that the one-instruction program

```text
broadcast_panel_lattice_periodic(
    background=0,
    row_period=2,
    column_period=2
)
```

is demo-exact and produces the correct query at semantic/output rank 1. Its program
ID is `2706cd8e3be345cc75c4dec5053abf8aa5ca427e24b46228e04bd092da5789d4`,
and its query output key is
`7b71a179631aaf7db1aeeffa6cfe5967f12413dc70c79790761f92fa9ea4bf2d`.
The task is first demo-exact at expansion 16, retains nine exact programs, emits nine
candidate records, and collapses to one unique query output. All +2,016 expansions,
+8,064 executions, +9 exact programs, +9 candidates, and +1 unique output in the
public matched-cap delta occur on this task alone.

Thirteen public-smoke tasks remain without a correct candidate. This checkpoint
therefore still does not justify ranker or controller training.

## Decision and next action

The M05e slice passes its narrow gate: both new generator-known families retain their
exact programs at rank 1, the prediagnosed direct task is added, every prior solved
task is preserved, parser bytes stay frozen, and structural/proposal/cap costs are
fully replayable. Its evidence supports only a suffix-ragged two-axis periodic
renderer with one seed; it does not support arbitrary template stamping, multiple
seeds, overlap policies, or a learned functional controller.

The next registered subsystem was subsequently completed as the additive M02c/M05f
axis-ray bbox-contact checkpoint. It freezes a relation sidecar, adds only
`1f642eb9`, preserves this checkpoint's seven solved tasks, and raises the same
post-hoc smoke to 8/20. See
`notes/results/e01a-bbox-contact-expansion.md`. Nearest-object motion and max-span
scoped rewriting remain outside both contracts. Masked diffusion and code
hypotheses may proceed only as separately frozen candidate sources; E01b remains
the gate for heterogeneous complementarity.
