# E01a M05d indexed panel-sequence D4 checkpoint

## Status and scope

This checkpoint is a verified, failure-driven symbolic development-smoke result. It
keeps the frozen M02a and M02b representations and adds one typed action that applies
an indexed D4 group step across a single-axis panel sequence. It does not add masked
diffusion, a code model, a learned ranker, a controller, two-axis periodic panel
broadcast, or ragged clipped stamping.

The action was chosen after inspecting failures on the same fixed 20-task ARC-AGI-2
public-training development smoke. The measured uplift is therefore a post-hoc
implementation diagnostic, not holdout generalization. The case set, oracle set,
search depth, beam width, instruction cap, and retained-program cap are unchanged
from the panel-overlay checkpoint.

- Evidence root: `results/e01a_panel_d4_v1`.
- Runtime source fingerprint:
  `5cc8c9a5a2cc2a13d34c40b96606382f6d98f49fb5d8cbaa264278a4b4a70530`.
- Test-source fingerprint:
  `bb670fb6b8f47b805af6211c000f31c43f14850ac5d890b33dbb3ef8d71736be`.
- Shared source-snapshot ZIP SHA-256:
  `69e6f1aae8a5198c071112acb2318df519ec2d1d1ac822190bc65dc48e9240f5`.
- Full unit suite at freeze: 113/113 passing with warnings treated as errors.
- Independent code review: PASS with no remaining P0, P1, or P2 findings.
- Both evaluation bundles pass `full_parent_replay_pass` through the verifier entry
  point.

## Frozen action semantics

DSL v0.4 adds:

```text
broadcast_panel_sequence_d4(background, step)
```

`afts-panel-sequence-d4/v0.1` consumes the unchanged M02b hypotheses and considers
only a horizontal or vertical single-axis panel sequence whose separator color is
different from `background`. All panels must have equal shape. Relative to the
declared background, exactly one panel must contain a non-background seed. If its
sequence index is `s`, destination panel `i` receives

```text
step^(i-s)(seed)
```

where `step` is one of the eight D4 elements. Exponents are reduced modulo the group
order, so panels before an interior seed use true negative powers. Each transformed
panel must fit its destination bounding box exactly; this makes shape-swapping D4
steps invalid on incompatible rectangular destinations. Reassembly changes only
panel cells and preserves separators and every non-panel cell.

The action has no axis, seed index, panel count, separator color, motif size, or
explicit transform-list parameter. The seed is inferred independently on every
input, so a fixed program supports horizontal and vertical sequences, different
lengths, and moving seed positions.

Invalid selection is deterministic across the whole hypothesis set. One complete
valid hypothesis executes; multiple complete hypotheses are non-unique. With no
complete hypothesis, multi-seed evidence takes priority over incompatible-shape
evidence, which in turn takes priority over an empty selection. The DSL maps these
outcomes onto the existing explicit selection/shape invalid codes.

M02a and M02b were not changed. On the public bundle, all of the following are
byte-identical to `results/e01a_panel_v1`:

- blind tasks: 82,753 bytes, SHA-256
  `e80dacf901852e878d49bc2f2fe124792460a2bbfe4553adc10c2e36aea7f290`;
- oracle rows: 16,711 bytes, SHA-256
  `f3ec853c33f7b78482769ec260a5215b7c266f0803a942419e7d8072341afafa`;
- M02a parses: 15,087,233 bytes, SHA-256
  `3b1cf3cebaf6ab2c21841920220308c0e79f6551bea1ef610f2e2eb461dbd9d5`;
- M02b panel parses: 336,259 bytes, SHA-256
  `77b068ff62254344ac76ce9bf48d289ad2ca73e11bc4044bb83b929e83ad4b28`.

The fixed M02a parse/object and M02b panel golden identities also remain unchanged.

## Blind proposal and cost ledger

`afts-panel-sequence-d4-proposer/v0.1` forms the Cartesian product of common
demonstration background hypotheses and all eight canonical D4 steps. Every trial is
executed on every training input without early stopping. A proposal is retained when
execution is structurally valid on every demonstration and its output shape agrees
with the observed demonstration output shape, directly or swapped. Demonstration
output pixels are deliberately not used as a proposal gate; normal typed search
evaluates all surviving actions, which preserves their use as prefixes in
compositions.

The v4 search row separately records:

- proposed M05d instructions;
- D4 trial count and demonstration pre-execution count;
- instruction counts before and after the fixed 64-option cap;
- explicit truncation count; and
- how many M05d options survive that cap.

These fields enter the pool behavior hash and are recomputed by the evaluator and
verifier. They are not folded into `program_execution_count`, because that would
mislabel a partial cost sum as a complete execution total.

## Generator-known synthetic control v0.4

Version 0.4 contains 13 families and 39 tasks: all 33 prior controls plus three
atomic `broadcast_panel_sequence_d4` tasks and three non-degenerate
`broadcast_panel_sequence_d4`-then-`flip_horizontal` composition tasks. The controls
vary horizontal/vertical direction, two through five panels, first/interior/last
seed positions, separator color and thickness, D4 step, and square/rectangular
behavior. Every old blind-task row and all 264 old rows in each of the M02a and M02b
parse sidecars are retained byte-for-byte; old oracle task semantics are unchanged.

- Case-set ID:
  `1b4c5befb9b488b54964020e5e436f46a0b7f302e57abde0ee404b0fc50dfc71`.
- Oracle-set ID:
  `8c3710f14ad753063ee95c5d7022b325369c0d9f91abcb42bbdb4af19cb57c2c`.
- Pool-spec ID:
  `e5cb12eaa07bb1f213379991ae2b2ecc1a4cd0b7e43905beccf1634db2150fb1`.
- Pool-content ID:
  `fcfbc8195c0c027694b8bf4f207481dbde15ebf156a67c440742168ef99351da`.
- Eval-spec ID:
  `f53f98b660569c125c696d698993b85ff961e5e35cad2c50d4a7cf0f2d22d1c8`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `f02ab26845cd0b360a484d2cbbcf9111faa6e129d3e5c10f4fa12ec7916ceca6`,
  `039ee08e9a18b7bd338bff192ac12e08c6321fb25d9061fddb250869119a9884`,
  `532715569474f37f9d1e785b4a81786b0ec23c6265c827aea73ec0319f1b3a71`,
  and `b25a3f98716e7be24bee6fb3170f199e99b6458f0cae755ff5358df07e80327b`.
- 39 tasks, 78 query pairs, 312 M02a parse rows, 312 M02b parse rows, 208
  panel hypotheses, 84,160 expansions, 420,800 program executions, 493 retained
  demo-exact programs, 973 candidate records, and 96 unique query outputs.
- The proposal ledger records 384 D4 trials, 1,152 demonstration pre-executions,
  and 44 M05d proposals; all 44 survive the instruction cap.
- Generator replay, semantic-program coverage, task-first coverage, micro pair
  coverage, and strict task coverage are all 1.0 at output/semantic rank 1.
- The atomic family has exact generator-AST evaluated, retained, and rank-1 rates of
  1.0. The composition family has evaluated and retained rates of 1.0, exact
  generator-AST recall 2/3 at rank 1 and 3/3 by rank 8, and semantic/output coverage
  3/3 at rank 1.
- Across all 39 tasks, exact generator-AST evaluated/retained recall is 36/39 =
  0.923077. The three misses remain the unchanged `crop_rotate90` family: the
  depth-1 semantic-deduplicated frontier selects equivalent prefixes, so those exact
  generator compositions are not evaluated at depth 2. Their semantic/output
  coverage remains 1.0.
- Summed measured search wall time is 788.71 seconds; maximum per-task traced peak
  memory is 12,424,056 bytes.

The suite remains a grammar-aligned execution/search control. It is not an estimate
of real ARC coverage or compositional OOD generalization.

## Fixed ARC-AGI-2 public-training development smoke

- Dataset commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Case-set ID and oracle-set ID are unchanged:
  `93fe4870982313fb840d27bc4082e079f41d8b58d4f679cea15b5e5b1381b177`
  and
  `1b6e1c3ad153e7f090234829c23d5df1f3778bb14ea374a58d30060edb8f758e`.
- Pool-spec ID:
  `a02f946f7bfcc3a6a174feb831b6262826d7f7a8bd3f9b603e2f22d48148b586`.
- Pool-content ID:
  `a467e18fc3734bc2e4013c78bbd07bb2a17b20b7a40673198d59048face3fc48`.
- Eval-spec ID:
  `5a68fd6d716bf13a56198fc7496001c7b020800b2a465b674e74bef38dbb1ff1`.
- Blind, oracle, pool, and evaluation artifact-manifest SHA-256 values:
  `933b5bbb6302d7165246847b9bab70e8f3d6a961e197848915c6bea226d027d3`,
  `7c88ec96e6f4f5eb2561ea60af3ba58ce6318feca4efe4d3db1600af7883fba1`,
  `b0212ba73dee4f462c4651dae962f5fff1f4c8e986dc081a1e34abda33485c2f`,
  and `43858f116f464b64f54be0a0417ddb991c3e2bc3453200a28020f96067d7a499`.
- 20 tasks and 21 query pairs; 46,161 expansions; 210,573 program executions;
  211 retained demo-exact programs; 163 candidate records; and 20 unique outputs.
- The M05d ledger records 224 D4 trials, 800 demonstration pre-executions, eight
  proposals, and eight M05d options surviving the cap. Total instruction options are
  1,728 before cap, 1,167 after cap, and 561 truncated across tasks.
- Task-first, strict, and semantic-program coverage are 6/20 = 0.30 at rank 1 and
  remain 0.30 through every reported cutoff.
- Micro pair coverage is 6/21 = 0.285714 at rank 1 and remains unchanged through
  rank 128.
- Summed measured search wall time is 1,158.90 seconds; maximum per-task traced peak
  memory is 25,613,580 bytes.

### Matched-cap before/after

| Measure | Panel v1 | Panel-D4 v1 | Change |
|---|---:|---:|---:|
| task-first/strict coverage@1 | 5/20 = 0.25 | 6/20 = 0.30 | +1 task |
| micro pair coverage@1 | 5/21 = 0.238095 | 6/21 = 0.285714 | +1 pair |
| semantic program coverage@1 | 4/20 = 0.20 | 6/20 = 0.30 | +2 rank-1 tasks |
| expansions | 46,161 | 46,161 | 0 |
| program executions | 210,573 | 210,573 | 0 |
| retained demo-exact programs | 193 | 211 | +18 |
| candidate records | 145 | 163 | +18 |
| unique query outputs | 19 | 20 | +1 |

The unchanged expansions and program executions result from the same per-task
instruction cap, not from equal actual work: M05d additionally incurs the separately
reported 224 trials and 800 demonstration pre-executions. The public action options
also change which tail options survive truncation. This is therefore a matched-cap
comparison, not a claim of identical wall time or complete execution count.

The semantic-program rank-1 change is not two newly solved tasks. `8e5a5113` is the
one new solution; the already solved `1f85a75f` moves from semantic program rank 2
to rank 1 because DSL v0.4 changes versioned program IDs and deterministic tie-break
ordering. Output-level coverage for `1f85a75f` was already rank 1 in panel v1.

The prior solved set `1f85a75f`, `b1948b0a`, `c59eb873`, `a416b8f3`, and
`a68b268e` remains covered. Only `8e5a5113` receives M05d proposals: the eight
canonical D4 steps at `background=0`. Its cap ledger is 114/64/50 before/after/
truncated, with 16 trials and 48 demonstration pre-executions. It is first
demo-exact at expansion 10, retains 18 exact programs, emits 18 candidate records
collapsing to one unique output, and solves the query at rank 1. The one-instruction
program

```text
broadcast_panel_sequence_d4(background=0, step=rotate90)
```

is demo-exact and produces the correct query output.

`92e50de0` remains the negative boundary: it receives no M05d proposal, exact
program, or candidate. Its requirement is a distinct two-axis periodic selection and
ragged clipped-stamping rule, not the equal-shape single-axis group action claimed
here.

At this historical checkpoint, fourteen public-smoke tasks remained without a
correct candidate. The subsequent M05e checkpoint reduces that count to thirteen;
neither checkpoint justifies ranker or controller training.

## Decision and next action

The M05d slice passes its narrow gate: both new generator-known families are solved
semantically at rank 1, the exact generator composition remains reachable, the
prediagnosed direct task is added, every prior solved task is preserved, representation
bytes stay frozen, and proposal/cap costs are replayable.

The next registered symbolic slice was M05e, a two-axis periodic panel broadcast
with bottom/right suffix clipping. It has now passed its registered gate: the blind
36-period-pair domain for `92e50de0` contains the demo-exact `(2,2)` program, the
atomic/composition/invalid controls pass, the structural/trial/demo/cap ledger is
replayable, and all six prior solved tasks are preserved. See
`notes/results/e01a-panel-periodic-expansion.md`. Relational selectors and local
renderers are now the next separately registered subsystem.
Learned ranking, masked diffusion, code hypotheses, and functional switching remain
separate source/controller milestones rather than retroactive explanations of this
symbolic uplift.
