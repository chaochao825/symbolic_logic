# Evaluation Protocol

## Dataset authority and sealing

At execution time, pin a commit of the official ARC-AGI-1 and ARC-AGI-2
repositories. The current ARC-AGI-2 repository reports 1,000 public training tasks
and 120 public evaluation tasks, while the historical 2025 technical report describes
the then-used split differently. The pinned data revision, not an old paper count,
governs each experiment.

Routine method development must not inspect or repeatedly score on public evaluation
outputs. Use:

1. a fixed development split from public training tasks;
2. a fixed held-out training-task audit split;
3. controlled synthetic compositional OOD suites with known generating programs;
4. the public evaluation set only at predeclared release gates;
5. private or semi-private scores only through their official protocol.

Sealing is benchmark-version specific. The pinned repositories contain substantial
cross-version reuse: ARC-AGI-2 public training overlaps heavily with ARC-AGI-1 public
evaluation. A method trained or selected with ARC-AGI-2 public training therefore may
not call ARC-AGI-1 public evaluation held out. Before every split, compute an
order-insensitive semantic task fingerprint over train/test pair multisets and record
all intersections. The primary claim uses only an ARC-AGI-2 public-training split and
makes no ARC-AGI-1 held-out claim.

## Primary pre-registration

- Dataset: a fixed 150-task ARC-AGI-2 public-training holdout, selected once by a
  recorded semantic-fingerprint split after near-duplicate auditing.
- Development: the remaining ARC-AGI-2 public-training tasks, with a separate fixed
  validation subset for the static schedule and controller.
- Method: one preselected XGBoost execution-aware next-action controller.
- Comparator: the strongest validation-tuned static weighted schedule.
- Budget: 180 wall-clock seconds per task, non-API, on one recorded hardware stack.
- Resource equality: identical generator checkpoints, actions, seed policy, attempt
  limit, and end-to-end cost ledger.
- Primary effect: official two-attempt pair-fraction score delta at least 0.02 and a
  paired task-bootstrap 95% confidence interval whose lower bound is above zero.
- Primary fixed generation seeds: 0, 1, and 2. The controller-training seed and
  selected checkpoint are frozen from validation before holdout execution.

Failure of this test rejects C1. Results on other budgets, datasets, controllers, or
Pareto curves are secondary and cannot rescue it.

## Primary metrics

- official pair-fraction `pass@1`, averaged within task and then across tasks;
- official pair-fraction `pass@2` under two attempts per test input;
- strict all-test-pairs-solved rate at one and two attempts;
- per-task wall time, model calls, CPU/GPU time, and timeout rate;
- cost-normalized accuracy and Pareto frontier.

For each test pair, an answer receives credit only when the entire output grid is
exact. The current official scorer gives a multi-test task partial task credit when
only some test pairs are solved; the strict all-pairs rate remains a separate
diagnostic and must not be mislabeled as the official score.

## Diagnostic metrics

- output-shape recall@k;
- oracle output coverage@k per source and for the union;
- unique candidate count after canonical deduplication;
- true candidate recall after each hard-filter stage;
- ranker recall@k, mean reciprocal rank, calibration, and top-2 diversity;
- program and parse coverage where ground truth is available;
- expanded nodes and peak beam size;
- repair exact-recovery rate per model call;
- source complementarity and marginal solved tasks;
- executor errors and sandbox failures.

All denominators are explicit. `coverage@k` is computed per test pair over the first
`k` unique candidate outputs unless marked program-level. `ranker recall@k` is the
fraction of test pairs whose exact output is in the ranked first `k`, conditional on
the exact output existing in the frozen pool. `official pass@2` is the mean, across
tasks, of the fraction of test pairs solved in two attempts. Strict task rate requires
all test pairs in a task to be solved.

## Residual and oracle policy

- `R_demo` uses known demonstration targets.
- `R_query_internal` uses only uncertainty, cross-view/source disagreement, inferred
  invariant violations, and execution/type evidence available without the answer.
- `R_oracle` compares with the held-out query answer and is never a policy feature.

Repair masks, rankers, controllers, stopping, and hyperparameters may use only the
first two. Oracle error may be used after predictions are frozen for a ceiling or
stratified analysis. Each mask stores evidence provenance.

## Budget matching

Every main comparison fixes:

- hardware and accelerator model;
- maximum wall time and memory;
- maximum neural model calls or token budget;
- maximum program expansions;
- candidate canonicalization and verification cost accounting;
- number of allowed answers;
- task order and timeout semantics.

Report both wall-time matching and a hardware-independent operation ledger when
possible. A method that uses extra candidates or an extra foundation model is not a
matched comparison.

## Leakage controls

- Split by task, never by demonstration pair.
- Do not include task ID, file order, or stable hash features in learned policies.
- Fit rankers using task-grouped folds so candidates from one task never cross folds.
- Record all synthetic generators and reject exact or near duplicate train/test tasks.
- Record the pretraining and ARC exposure policy for each model.
- Do not call a public-evaluation score held-out if it influenced hyperparameters.
- Separate `contest_private`, `semi_private_verified`, `public_eval`,
  `public_train_holdout`, and `synthetic_ood` in every table.

## Statistical reporting

- The task is the primary analysis unit; within-task test-pair fractions remain
  grouped during resampling.
- Use the fixed generation seeds 0, 1, and 2 for the primary experiment. Aggregate
  seed outcomes within task, then apply a paired task bootstrap with 10,000 resamples
  and a two-sided 95% percentile interval.
- Report the point delta, interval, and the predeclared 0.02 minimum practical effect.
- E06 is the sole primary comparison. Secondary controller/budget comparisons use
  Holm correction within their declared family.
- The controller checkpoint and training seed are selected on validation and frozen
  before the holdout run.
- Preserve negative and contradictory results.

## Feature governance

Candidate IDs, task IDs, file names, hashes, repository paths, and task order are
provenance only. Learned interfaces receive an explicit feature whitelist. A run fails
validation if any provenance field enters a feature tensor or table.

## Frozen-pool versus online protocols

- Frozen candidate pools evaluate filters, rankers, calibration, and the top-2
  selector under identical candidates.
- Online matched-budget runs evaluate the next-action controller. Each controller
  generates its own trajectory from identical checkpoints, actions, seeds, and
  resource limits.

Offline policy training must state whether all actions were enumerated. Otherwise it
must record the logging policy and a justified off-policy estimator.

## E01 coverage staging

E01a_symbolic_coverage and E01b_heterogeneous_marginal_coverage answer different
questions and must not be pooled into one claim.

E01a uses three separately materialized stages. The dataset stage writes blind tasks
containing demonstrations and query inputs plus a sealed oracle sidecar. The pool
stage reads only the blind bundle and freezes parse, program, candidate, execution,
and cost records. The evaluation stage may then read the oracle sidecar to compute
generator-program recall, semantic/output coverage, pair-level coverage, and strict
task coverage. Oracle query outputs, generator family labels, and generating programs
must not affect search, stopping, deduplication, or hyperparameters.

The synthetic v0.1 suite is deliberately aligned with the compact DSL grammar; v0.2
appends scale and horizontal-tile controls, v0.3 appends panel-overlay controls, and
v0.4 appends indexed panel-sequence D4 atomic and composition controls under the
same claim boundary; v0.5 appends suffix-ragged two-axis periodic panel controls.
Version 0.6 appends axis-ray rectangle-contact atomic and composition controls plus
an independently replayed M02c relation sidecar. These are controls for deterministic generation, typed
execution, bounded search, and evidence plumbing. Their recall is not an estimate
of real ARC expression coverage or compositional OOD generalization. M02a similarly
covers connected-component views only, and M02b covers only full-span separator
panels; neither may be reported as the complete M02 parser. The subsequent
ARC-AGI-2 public-training run is a development smoke: query outputs remain hidden
until pool freeze, but the result is neither the primary holdout nor a release claim.

M04a is separately registered by the frozen pre-implementation contract in
`notes/design/m04a-global-masked-grid-contract.md`. Its blind generation population
is every test pair of the fixed 20-task smoke, never an oracle-selected subset; the
12 DSL-zero-candidate tasks are a blind reporting subgroup. Training is permitted
only from the physically sanitized 662 ARC-AGI-2/262 ReARC training parents, while
checkpoint selection uses the clean 145/51 validation parents. The model, 64-lane
sampler, full trace, and training/inference cost ledgers must pass their evidence
gate before oracle evaluation. This registration is not implementation evidence:
no M04a checkpoint, pool, or result exists yet, and local repair remains M11.

E01b runs only after the DSL, masked grid diffusion, code-model, and rule-source pools
are independently frozen. It uses matched source budgets and canonical output
deduplication to report per-source coverage, union coverage, and each source's
marginal solved pairs or tasks. Passing E01a does not satisfy E01b or Gate 1's
complementary-source requirement.

### E01a verified checkpoint (2026-07-11)

The source-bound evidence root is `results/e01a_symbolic_v1`, with runtime source
fingerprint
`3eb556bcfed1ea8a865a7bf154a465e449f53aeac0455e1bf019cc83629d82ad`.
Both synthetic and public-training evaluation bundles passed a full replay that
reloaded their blind/oracle/pool parents, reran parsing and search, and recomputed all
pair, task, slice, funnel, integrity, and summary records.

- The 21-task, 42-query grammar-aligned synthetic control obtains 1.0 semantic and
  output coverage at rank 1; generator replay/evaluated/retained rates are 1.0.
- The fixed 20-task, 21-query ARC-AGI-2 public-training development smoke obtains
  0.10 task-first pair coverage, 2/21 = 0.095238 micro pair coverage, 0.10 strict
  task coverage, and 0.10 semantic solving-program coverage. Coverage is unchanged
  through output rank 128.
- The public smoke used 45,864 expansions and 209,061 program executions to produce
  only 13 unique query outputs, confirming that candidate support is the bottleneck.

These numbers are diagnostics, not a primary holdout, public-evaluation, pass@2,
release, compositional-OOD, or source-complementarity result. At that checkpoint,
the next registered action was failure classification plus M02, shape, DSL, and
search expansion on the same development smoke; ranker/controller training remained
premature.

### E01a M03a/M05b shape checkpoint (2026-07-11)

The source-bound successor root is `results/e01a_shape_v1`, with runtime source
fingerprint
`b1abacbec9e4c9f5175f55fb58844c280c4a6800ef8896d8afa884666c59ac2a`.
It keeps the same 20-task case-set ID and the same depth, beam, instruction, and
retention caps. Both evaluation bundles again pass full parent replay.

- Synthetic v0.2 has 27 tasks and 54 query pairs. Semantic/output coverage is 1.0
  at rank 1; exact generator-AST recall is 15/27 at rank 1 and 27/27 by rank 8.
- The fixed public-training development smoke reaches 0.20 task-first and strict
  task coverage and 4/21 = 0.190476 micro pair coverage at output rank 1, with no
  further gain through rank 128.
- The matched search uses 46,053 expansions and 209,817 executions, versus 45,864
  and 209,061 in v1. Unique query outputs increase from 13 to 18.
- The old solved tasks remain covered; `c59eb873` is added by pixel scaling and
  `a416b8f3` by horizontal whole-grid tiling. The other 16 tasks still emit no
  candidates.

This uplift was obtained after diagnosing failures on the same development tasks.
It is therefore post-hoc error-driven evidence that the implementation behaves as
intended, not an estimate of holdout improvement. M03a covers direct positive
integer axis ratios only, and the parser remains M02a. See
`notes/results/e01a-shape-expansion.md` for identities, costs, and limitations.

### E01a M02b/M05c panel checkpoint (2026-07-11)

The source-bound successor root is `results/e01a_panel_v1`, with runtime source
fingerprint
`98b72451934a253ca50fcabb2e354a81f52c8d1bb3bfe9a37829b7f4a894c627`.
The public blind tasks and oracle labels are byte-identical to the shape checkpoint,
and depth, beam, instruction, and retention caps remain fixed. M02b is an independent
sidecar; the public M02a parse bytes remain unchanged. Both evaluation bundles pass
full parent replay, including executable panel-parser replay.

- Synthetic v0.3 has 33 tasks and 66 query pairs. Semantic/output coverage is 1.0
  at rank 1. The two new panel families have exact generator-AST recall 1.0 at rank
  1; all-suite exact-AST evaluated/retained recall is 30/33 because semantic
  deduplication selects equivalent programs for the old `crop_rotate90` family.
- The fixed public-training development smoke reaches 0.25 task-first and strict
  coverage and 5/21 = 0.238095 micro pair coverage at output rank 1, unchanged
  through rank 128.
- The matched search uses 46,161 expansions and 210,573 executions, versus 46,053
  and 209,817 in the shape run. Unique query outputs increase from 18 to 19.
- All four prior solved tasks remain covered. `a68b268e` is the only public task to
  receive an overlay proposal and is added at output rank 1. `92e50de0` receives a
  ragged indexed panel lattice but no eligible overlay and remains unsolved.

This is post-hoc error-driven evidence for a narrow panel representation and ordered
overlay, not an estimate of holdout improvement or evidence for periodic broadcast,
full M02, heterogeneous sources, or functional switching. See
`notes/results/e01a-panel-expansion.md` for identities, costs, and limitations.

### E01a M05d indexed panel-sequence D4 checkpoint (2026-07-11)

The source-bound successor root is `results/e01a_panel_d4_v1`, with runtime
source fingerprint
`5cc8c9a5a2cc2a13d34c40b96606382f6d98f49fb5d8cbaa264278a4b4a70530`.
The public blind tasks, oracle labels, M02a parses, and M02b panel parses are
byte-identical to the panel checkpoint. Depth, beam, instruction, and retention caps
remain fixed. Both evaluation bundles pass full parent replay, including proposal,
D4 trial/demo cost, instruction-cap, parser, program, and candidate replay.

- Synthetic v0.4 has 39 tasks and 78 query pairs. Semantic/output coverage remains
  1.0 at rank 1. Both new M05d families have generator evaluated/retained rates of
  1.0; the atomic family has exact generator-AST 3/3 at rank 1, and the composition
  family reaches 3/3 by rank 8. All-suite exact generator-AST evaluated/retained
  recall is 36/39 because the three unchanged `crop_rotate90` controls retain the
  previously reported semantic-frontier identity issue.
- The fixed public-training development smoke reaches 0.30 task-first, strict, and
  semantic-program coverage and 6/21 = 0.285714 micro pair coverage at rank 1, with
  no further gain through the reported cutoffs.
- The matched-cap search still uses 46,161 expansions and 210,573 program
  executions. M05d additionally records 224 D4 trials and 800 demonstration
  pre-executions; those costs are reported separately rather than mislabeled as a
  complete combined execution total. Unique query outputs increase from 19 to 20.
- All five prior solved tasks remain covered. Only `8e5a5113` receives M05d
  proposals; its atomic `background=0, step=rotate90` program solves at rank 1.
  `92e50de0` receives no proposal or candidate and remains a negative boundary for
  two-axis periodic broadcast with ragged clipping.

This is post-hoc error-driven evidence for one narrow single-axis group action, not
an estimate of holdout improvement or evidence for periodic two-axis broadcast,
full M02, heterogeneous sources, masked diffusion, or functional switching. See
`notes/results/e01a-panel-d4-expansion.md` for identities, costs, and limitations.

### E01a M05e periodic panel-lattice checkpoint (2026-07-11)

The source-bound successor root is `results/e01a_panel_periodic_v1`, with
runtime source fingerprint
`d34264e84123b503fff917664e09a8fd4b971a716a91800d1d2d67dc96ab2095`.
The public blind tasks, oracle labels, M02a parses, and M02b panel parses are
byte-identical to the M05d checkpoint. Depth, beam, instruction, and retention caps
remain fixed. Both evaluation bundles pass full parent replay, including periodic
bounds, structural/trial/demo cost, cap, parser, program, and candidate replay.

- Synthetic v0.5 has 45 tasks and 90 query pairs. Generator replay,
  evaluated/retained rate, and semantic/output coverage are all 1.0. Exact
  generator-AST recall is 44/45 at rank 1 and 45/45 by rank 8; both new periodic
  families are 3/3 at rank 1. Every old blind/parse row is retained.
- The fixed public-training development smoke reaches 0.35 task-first, strict, and
  semantic-program coverage and 7/21 = 0.333333 micro pair coverage at rank 1,
  unchanged through the reported cutoffs.
- Under the same caps, expansions increase from 46,161 to 48,177 and program
  executions from 210,573 to 218,637. The 36 periodic options enlarge only the
  `92e50de0` semantic frontier. Periodic proposal work is separately recorded as
  100 structural checks, 36 trials, and 108 demonstration pre-executions.
- All six prior solved tasks remain covered. Only `92e50de0` receives an M05e
  bound/proposals; all 36 period pairs survive the cap, while normal demo-exact
  execution selects `background=0, row_period=2, column_period=2` at rank 1.

This is post-hoc error-driven evidence for one narrow suffix-ragged periodic
renderer, not an estimate of holdout improvement or evidence for arbitrary
stamping, full M02, heterogeneous sources, masked diffusion, or functional
switching. See `notes/results/e01a-panel-periodic-expansion.md` for identities,
costs, and limitations.

### E01a M02c/M05f bbox-contact checkpoint (2026-07-11)

The source-bound successor root is `results/e01a_bbox_contact_v1`, with runtime
source fingerprint
`3955ed3dedf1a34176e55dd38e85990337fed98512e5e626cd5edf4f45349c67`.
The public blind tasks, oracle labels, M02a parses, and M02b panel parses are
byte-identical to the M05e checkpoint. M02c is an additive closed-world relation
sidecar; depth, beam, instruction, and retention caps remain fixed. Both evaluation
bundles pass full parent replay, including relation-sidecar, bound, proposal, cost,
cap, parser, program, candidate, and parent-artifact replay.

- Synthetic v0.6 has 51 tasks and 102 query pairs. Generator replay,
  evaluated/retained rate, and semantic/output coverage are all 1.0. Exact
  generator-AST recall is 38/51 at rank 1 and 51/51 by rank 8. The new atomic
  family is 3/3 at rank 1; each new composition's exact generator AST is rank 2.
  Versioned program-ID ordering also moves nine old exact ASTs behind rank 1, while
  all old task identities, search totals, output sets, and semantic/output rank-1
  coverage remain unchanged.
- The fixed public-training development smoke reaches 0.40 task-first, strict, and
  semantic-program coverage and 8/21 = 0.380952 micro pair coverage at rank 1,
  unchanged through the reported cutoffs.
- The 64-option cap retains the same aggregate 1,167 post-cap options, 48,177
  expansions, and 218,637 ordinary program executions as M05e. The one M05f option
  replaces one generic tail option. Its separate aggregate ledger records 100
  structural checks, 34 anchor candidates, 216 relation checks, one admissible
  binding/action trial, and three demonstration pre-executions.
- All seven prior solved tasks remain covered. Only `1f642eb9` receives an
  admissible binding and proposal; `paint_bbox_contacts(background=0)` is exact at
  semantic/output rank 1. The other 12 tasks remain without a correct candidate.

This is post-hoc error-driven evidence for one narrow singleton-to-solid-rectangle
axis relation and copy-to-boundary renderer. It is not an estimate of holdout
improvement or evidence for nearest-anchor motion, arbitrary line drawing, full
M02, heterogeneous sources, masked diffusion, learned ranking, or functional
switching. See `notes/results/e01a-bbox-contact-expansion.md` for identities, costs,
rank-order caveats, and limitations.

## Release gates

### Gate 0: harness integrity

Dataset loading, exact scoring, two-attempt handling, multi-test tasks, provenance,
and deterministic baselines pass unit tests.

### Gate 1: candidate coverage

At least two genuinely complementary sources produce reproducible candidates, and
oracle coverage is measured before ranker work.

### Gate 2: verifier safety

Each hard filter demonstrates a predeclared true-candidate recall floor on synthetic
ground truth and frozen real candidates.

### Gate 3: ranking and repair

Ranker and repair experiments use frozen pools. The public evaluation set remains
sealed.

### Gate 4: controller

Controller comparisons are matched on candidate generators, features, and budgets.

### Gate 5: public evaluation audit

Run once for a tagged method release and archive the exact configuration. A new run
requires a material method revision and a written reason.
