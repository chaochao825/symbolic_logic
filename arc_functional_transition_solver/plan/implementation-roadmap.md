# Implementation Roadmap

The roadmap is ordered by information value. A later phase is not used to hide a
failure in an earlier bottleneck.

## Phase 0: benchmark and evidence spine

Build the smallest trustworthy executable core:

- pinned ARC-AGI-1/2 loaders;
- official pair-fraction pass@1/pass@2 plus strict all-pairs task diagnostics;
- immutable candidate record and canonical output deduplication;
- run manifest and per-task provenance;
- identity and simple geometric/color baselines;
- unit tests for wrong shape, wrong color, one-cell mismatch, two attempts, and
  multiple test inputs.

**Exit gate:** E00 passes and every score can be traced to a candidate and data commit.

## Phase 1: E01a symbolic coverage controls before learned routing

- M02a connected-component parse bundles over alternative backgrounds, 4/8
  connectivity, and single/multicolor objects; M02a does not complete M02, whose
  line, corner, frame, motif, panel, and symmetry views remain future work;
- compact typed DSL with deterministic execution;
- deterministic bounded enumeration or beam search for short programs;
- residual extraction across demonstrations;
- generator-known grammar-aligned synthetic controls v0.1, the failure-driven
  scale/tile v0.2 extension, the independent panel-overlay v0.3 extension, and the
  indexed panel-sequence D4 v0.4, periodic panel-lattice v0.5, and axis-ray
  bbox-contact v0.6 extensions,
  followed by an ARC-AGI-2
  public-training development smoke;
- separate generator-program recall from semantic/output coverage, with oracle
  query outputs available only after the candidate pool is frozen.

**Exit gate:** E01a produces reproducible blind candidate pools, recovers known paths
on the grammar-aligned controls, and reports the bounded symbolic ceiling on the
public-training development smoke. Synthetic v0.1 validates generator, executor, and
search plumbing only; v0.2 adds shape/scale/tile controls, v0.3 adds panel
parse/overlay controls, and v0.4 adds panel-sequence D4 controls under the same
boundary; v0.5 adds two-axis periodic broadcast and suffix-clipping controls;
v0.6 adds M02c relation-sidecar and M05f contact-render controls. None is evidence
of real ARC expression coverage,
compositional OOD generalization, or heterogeneous source complementarity. If the
public-training smoke is weak, expand M02 and the DSL rather than training a
controller.

**2026-07-11 checkpoint:** the E01a gate is complete as a bounded symbolic-control
measurement. The 21-task synthetic control reaches 100% semantic/output coverage,
while the fixed 20-task ARC-AGI-2 public-training development smoke reaches only
10% task-first and strict task coverage (2/21 = 9.52% micro pair coverage), with no
gain through output rank 128. This is evidence to expand M02/DSL/search coverage,
not evidence to begin controller training. See
`notes/results/e01a-symbolic-coverage.md` and `results/e01a_symbolic_v1`.

**2026-07-11 shape checkpoint:** the first failure-driven M03a/M05b slice is also
complete. A blind axis-integer shape proposer plus typed pixel scaling and whole-grid
tiling raises the identical 20-task development smoke from 10% to 20% task-first and
strict coverage and from 2/21 to 4/21 micro pair coverage. It uses 46,053 expansions
and 209,817 executions versus 45,864 and 209,061 in v1; both old solved tasks remain
covered. The 27-task synthetic v0.2 control retains 100% semantic/output coverage at
rank 1. Both evaluation bundles pass full parent replay. Because the two rules were
chosen after inspecting this same smoke, the uplift is a post-hoc implementation
diagnostic, not holdout generalization. See
`notes/results/e01a-shape-expansion.md` and `results/e01a_shape_v1`.

**2026-07-11 panel checkpoint:** the independent M02b/M05c slice is complete. A
content-addressed full-span separator/panel sidecar plus typed ordered transparent
overlay raises the identical development smoke from 20% to 25% task-first/strict
coverage and from 4/21 to 5/21 micro pair coverage. It uses 46,161 expansions and
210,573 executions, versus 46,053 and 209,817 in the shape run, and preserves all
four earlier solved tasks. All 33 synthetic v0.3 controls retain 100% semantic and
output coverage at rank 1; the two new panel families also retain exact generator
ASTs at rank 1. Both evaluation bundles pass full parent replay. `92e50de0` has the
intended ragged panel representation but remains unsolved, so this is not periodic
broadcast evidence. The uplift is again post-hoc development evidence, not holdout
generalization. See `notes/results/e01a-panel-expansion.md` and
`results/e01a_panel_v1`.

**2026-07-11 panel-D4 checkpoint:** the M05d indexed single-axis panel-sequence
action is complete. It raises the identical development smoke from 25% to 30%
task-first/strict coverage and from 5/21 to 6/21 micro pair coverage while keeping
46,161 expansions and 210,573 program executions under the same caps. Its separate
ledger records 224 D4 trials and 800 demonstration pre-executions. All five earlier
solved tasks remain covered, and only `8e5a5113` receives the eight public M05d
proposals. All 39 synthetic v0.4 controls retain semantic/output coverage 1.0 at rank
1, and both new families retain their exact generator programs. Both evaluation
bundles pass full parent replay. `92e50de0` remains outside the equal-shape
single-axis contract, so this is not evidence for two-axis periodic broadcast or
ragged clipping. See `notes/results/e01a-panel-d4-expansion.md` and
`results/e01a_panel_d4_v1`.

**2026-07-11 periodic-panel checkpoint:** the M05e two-axis periodic panel-lattice
action is complete. It raises the same development smoke from 30% to 35%
task-first/strict coverage and from 6/21 to 7/21 micro pair coverage. Only
`92e50de0` receives a periodic bound and proposals; normal demo-exact evaluation
selects `(2,2)` at rank 1, while all six prior solved tasks remain covered. The v0.5 synthetic control
has 45 tasks: semantic/output coverage is 45/45 at rank 1, both new families are
3/3 at rank 1, and the exact generator AST is 44/45 at rank 1 and 45/45 by rank 8.
The public pool expands from 46,161/210,573 to 48,177/218,637 expansions/executions.
Across the full bundle, the separate ledger records 100 structural checks; only
`92e50de0` produces a bound and therefore accounts for all 36 period trials and 108
complete demonstration executions. Both bundles pass full parent replay.
This remains post-hoc development evidence, not holdout or public-evaluation
evidence. See `notes/results/e01a-panel-periodic-expansion.md` and
`results/e01a_panel_periodic_v1`.

**2026-07-11 bbox-contact checkpoint:** the additive M02c/M05f slice is complete.
It raises the identical development smoke from 35% to 40% task-first/strict
coverage and from 7/21 to 8/21 micro pair coverage. Only `1f642eb9` receives an
admissible binding and proposal; `paint_bbox_contacts(background=0)` is exact at
rank 1 while all seven earlier solved tasks remain covered. The v0.6 synthetic
control has 51 tasks with semantic/output coverage 51/51 at rank 1 and exact
generator AST 38/51 at rank 1, 51/51 by rank 8. The old 45 controls retain their
blind identities, search totals, candidate-output sets, and rank-1 semantic/output
coverage; versioned program-ID ordering moves nine old exact ASTs behind rank 1.
The public pool retains 48,177 expansions, 218,637 ordinary executions, and 1,167
post-cap options. M02c/M05f separately records 100 structural checks, 216 relation
checks, one action trial, and three demonstration pre-executions. Both bundles pass
full parent replay. This remains post-hoc development evidence, not holdout or
public-evaluation evidence. See
`notes/results/e01a-bbox-contact-expansion.md` and
`results/e01a_bbox_contact_v1`.

## Phase 2: frozen candidate selection

- generate and freeze candidate pools;
- verifier ladder with necessary versus preference checks separated;
- task-grouped ranker dataset;
- hand MDL, XGBoost, MLP, tree, and DLGN comparisons;
- false-prune and anti-overfit audits.

**Exit gate:** improve top-k recall or pass@2 on a frozen pool without task leakage.

## Phase 3: masked grid generation and repair

- begin with a small, reproducible masked grid model or an audited open baseline;
- enumerate or learn output shapes separately and expose the bottleneck;
- compare one-shot, cold restart, global soft-mask recursion, random local masks, and
  residual-directed masks;
- record unique coverage and cost rather than only sample count.

**Exit gate:** masked generation adds complementary oracle coverage; local repair beats
cold restart on exact recovery per model call.

## Phase 4: open-ended code and macro induction

- sandboxed Python or DSL proposal interface;
- exact execution on every demonstration;
- convert verified reusable fragments into typed macros;
- retain prompt, model, token, API, and cost provenance.

**Exit gate:** code proposals solve tasks outside the fixed DSL often enough to justify
their cost and do not compromise reproducibility of the non-API track.

After the masked grid, code, rule, and DSL sources are independently frozen, run
E01b_heterogeneous_marginal_coverage at matched budgets. E01b, not E01a, tests whether
the source union adds non-duplicate correct-output coverage after canonical
deduplication.

## Phase 5: functional controller

- define the action space and four-valued dynamic transition mask;
- implement round robin, fixed weights, bandit, hand policy, XGBoost, MLP, and DLGN;
- train on action outcomes or offline oracle value with task-grouped splits;
- compare matched-budget accuracy-cost frontiers and switch traces.

**Exit gate:** C1 is supported or rejected. DLGN is promoted only if E08 shows a
non-dominated practical frontier.

## Phase 6: program/trace denoising

Add syntax-aware trace inpainting only after beam/mutation baselines are strong.
Prior work already applies diffusion to syntax trees, so novelty must come from ARC
execution residuals, cross-space repair, or controller coupling.

**Exit gate:** exact recovery and end-to-end gains over typed mutation and resynthesis.

## Phase 7: sealed release audit and paper

- freeze code, model weights, prompts, features, and primary budgets;
- run the public evaluation release gate once;
- archive task-level results and failures;
- backfill only verified numbers into the paper;
- run independent reproducibility and claim reviews.

**Exit gate:** all result claims have backing files and protocol labels.

## Immediate next implementation slice

The M03a scale/tile, M02b/M05c panel-overlay, M05d indexed panel-sequence D4, M05e
periodic panel-lattice, and M02c/M05f bbox-contact slices are complete. The last four
symbolic slices each added exactly one task; 12 of the fixed 20 tasks still have zero
raw candidates. A limited input-only diagnostic over all 1,000 ARC-AGI-2
public-training tasks found that strict signatures for the two cleanest remaining
legend-map and feature-to-bar diagnoses each identify only their already diagnosed
task. This does not exclude broader symbolic opportunities, but those two exact
contracts would add another single-task post-hoc result; the max-span and
nearest-anchor alternatives still require ambiguous or substantially larger
contracts. Details and limitations are recorded in
`notes/results/e01a-next-source-diagnostic.md`.

The next planned slice is M04a global masked-grid generation. Its pre-implementation
contract at `notes/design/m04a-global-masked-grid-contract.md` is frozen after
independent data-leakage, model/sampler, and evidence/remote-safety reviews. The
model/sampler/training primitives and production preflight/evidence closure are now
implemented, including a successful local 50,525-row full training-bundle roundtrip.
No production checkpoint, pool, or result exists yet; only a later passing run may
be called a frozen candidate source. The frozen contract requires:

1. condition only on demonstrations, query inputs, and a deployable non-oracle shape
   hypothesis; prohibit query outputs, oracle residuals, oracle-selected seeds,
   oracle early stopping, and training on the fixed 20 tasks or their augmentations;
2. bind the training-data identity, model architecture and checkpoint, tokenizer,
   seed list, initial mask, mask schedule, denoising-step count, temperature, and
   candidate budget before reading successor query outputs;
3. record the shape source, masked-token counts, forward calls, CPU/GPU time, peak
   memory, raw/format-valid/unique candidates, duplicate classes, and failures for
   every task and seed;
4. use generator-known held-out families and compositions plus D4/color
   counterfactuals, with both identity-shape and nonidentity-shape cases, to test
   the source without conflating grammar-aligned training recall with real ARC;
5. freeze the M04a pool independently, then deduplicate its outputs against the
   frozen DSL v0.6 pool before oracle evaluation; the first real-data success gate
   is at least one new correct output among the 12 zero-candidate tasks, but a
   verified zero gain must be reported rather than tuned away;
6. evaluate global generation and cold restarts only. Residual-directed local
   remasking, random local masks, and global-soft-mask recursion belong to the later
   M11 repair experiment and may not be claimed by this slice;
7. pass closed-world artifact, checkpoint/data, parent, cost-ledger, candidate, and
   full replay audits before it enters E01b.

Do not train the controller or use a ranker to conceal missing candidates. A code
model and independent rule source likewise need their own frozen contracts and
pools. E01b begins only after at least two sources can be compared under matched
budgets and canonical output deduplication; E01a/M04a development results must not
be described as heterogeneous complementarity or public evaluation.
