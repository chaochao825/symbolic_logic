# Residual-Compiled Metareasoning P1: Cost-Carrying DAG and Switching Audit

Date: 2026-07-26
Source branch before this change: `agent/arc-online-controller-20260724` at
`ab9442facfc92bb04ebfd3c3289f9d6396803f8a`
Training launched: **no**
ARC evaluation opened: **no**

## Verdict

The project now has a sharper theory and a stronger cost-audit primitive, but
the new experiment rejects the current learned functional-switching premise:

> On the present three-source pool, observed residual tokens add no detectable
> information or held-out coverage benefit to the second-option decision.

The direct residual interventions are behaviorally inert.  A GRU, MLP, or
diffusion action planner is therefore not justified on this candidate pool.
Candidate language and repair capability must improve first.

This is a useful negative result.  It prevents a larger controller from being
mistaken for progress and turns the brain-region analogy into measurable gates.

## Theory change

The method is now defined as residual-compiled metareasoning rather than a
collection of brain-inspired modules.  The controller state is

\[
b_t=(\phi(T), C_t, R_t, F_t, H_t, B_t),
\]

and each provider is a typed, temporally extended option.  The intended
controller estimates the value of computation of an option and stops when all
legal options have non-positive value after native cost and switch cost.

The functional brain analogy has five falsifiable requirements:

1. specialists have stable and complementary task selectivity;
2. residual interventions causally change coupling to the next specialist;
3. option sequences transfer compositionally;
4. module and bridge lesions produce selective deficits;
5. switching beats the strongest static schedule under physical budgets.

Only the first property has preliminary support.  The current experiment does
not support the second or fifth.  The full pre-registration is in
`notes/design/residual-compiled-metareasoning-plan.md`.

## What was implemented

### Cost-carrying action DAG

`NativeCostVector` and `NativeBudgetLedger` preserve named physical units rather
than silently equating them with NCU.  A frozen action batch now hashes:

- action operator and parent;
- content-addressed candidates;
- realized provider-native cost vector.

The pool schema is bumped from `afts.frozen-action-pool/v2` to `/v3`.  Frozen
replay returns the cost vector from the action batch, so policy cost summaries
no longer require a side lookup into the live recorder cache.  The provenance is
explicitly `realized_discovery_outcome`; this is audit evidence, not yet a legal
inference-time reservation estimate.

A one-task rerun produced:

- result ID `aa50e1d9847a8bdd61eb4196fb37682c0565508804e7b7d81d3907292e0369d3`;
- pool schema `/v3` and pool ID
  `4de29bd5c9ad7c186523fd214953c43e74e10cebbfe563f6c7c822b10cd93145`;
- zero behavioral differences from the corresponding previous task across all
  policies, metrics, actions, selections, and cost summaries.

The schema-v3 change alone did **not** establish matched physical compute:
using the realized cost of a frozen outcome to decide whether to run it would
be hindsight lookahead.  The follow-up implementation below adds a reservation
that is calibrated before the analysis task is opened.

### Fit-calibrated native token bucket

The offset 0-99 fit block is used to calibrate a conservative, dimension-wise
maximum reservation (`q=1.0`) for each typed operator.  The controller now:

- masks an action before selection if its reservation does not fit;
- charges the reservation even when the frozen provider abstains;
- records realized work separately from reserved work;
- invalidates the native-budget claim if realized work exceeds a reservation;
- rejects non-numeric diagnostic leaves instead of silently treating booleans
  as costs.

The profile was created without reading query labels or oracle-success fields:

- profile ID
  `8f54b70c3f3c1b926fc873c58a7d7e6e4bbac3e7b4da9e943aef11c1320b95b8`;
- contract ID
  `168b711f134e84d4f8e0aac41304718875fa1057661ea7b8fa351b401d9eb98a`;
- per-call maxima include 11,690 scene-rule trials, 35,070 demo scene
  executions, 931 typed-DSL expansions, two CA policy fits, and one repair
  attempt.

This establishes pre-action native-budget legality for the frozen replay
experiment.  It does not make heterogeneous dimensions interchangeable, and it
does not claim that reserved maxima equal realized expenditure.

### Option-switching audit

The new audit treats `dsl_only`, `ca_only`, and `scene_only` as three
temporally extended options.  It fits on the previously opened offset 0-99 block
and tests on the previously opened offset 300-399 block.  It uses exactly two
option slots and no query-output feature.

Compared models:

- fit-block best fixed pair;
- shallow static decision tree;
- static gradient boosting;
- dynamic gradient boosting with option history but no residual tokens;
- dynamic gradient boosting with residual tokens;
- residual-empty and residual-shuffled interventions;
- deterministic random pair;
- post-hoc oracle source scheduler as a diagnostic ceiling.

The source-success label is post-hoc and is never an inference feature.  Native
cost is reported as a raw vector.  A median-normalized scalar is descriptive
only and is not used to claim physical budget matching.

## Evidence baseline over 200 comparable development tasks

| Quantity | Result |
|---|---:|
| heterogeneous raw pool coverage | 29/200 |
| selectable pool coverage | 27/200 |
| structured pass@2 | 27/200 |
| fixed CA-first pass@2 | 23/200 |
| structured vs grounding-only action differences | 0/200 |
| structured vs fixed CA-first paired wins/losses | 4/0 |
| repair actions | 76 |
| uniquely recovered tasks from repair | 0 |

The 200-task error decomposition is therefore approximately

\[
173 = 171\;\text{candidate-language gap}
    +2\;\text{selectability gap}
    +0\;\text{controller discovery gap}
    +0\;\text{final selection gap}.
\]

The controller still cannot optimize around a correct candidate that does not
exist.

### Existing realized cost mismatch

| Policy | pass@2 | scene-rule trials | typed-DSL expansions |
|---|---:|---:|---:|
| structured/grounding-only | 27 | 371,439 | 101,367 |
| fixed CA-first | 23 | 9,002 | 126,359 |
| coverage-aware expanded | 20 | 48,980 | 129,588 |

Structured uses about 41.3 times the scene-rule trials of fixed CA-first for
four extra development tasks.  This remains an unmatched-cost observation, not
an efficiency gain.

## New option-switching results

### Source selectivity

| Block | DSL | CA | scene | three-source union | canonical pass |
|---|---:|---:|---:|---:|---:|
| fit 0-99 | 7 | 5 | 3 | 11 | 11 |
| analysis test 300-399 | 6 | 8 | 5 | 16 | 16 |

The sources remain complementary, and the fit-selected best fixed pair is
DSL+CA.  However, there are only 11 positive fit tasks, which sharply limits any
controller learner.

### Two-option coverage on the analysis test block

| Policy | solved | normalized native work | solved/work |
|---|---:|---:|---:|
| best fixed pair, DSL+CA | 13/100 | 561.678 | 0.02315 |
| static shallow tree | 13/100 | 561.678 | 0.02315 |
| static boosting | 13/100 | 627.781 | 0.02071 |
| dynamic history, no residual | 13/100 | 574.444 | 0.02263 |
| dynamic residual | 13/100 | 568.588 | 0.02286 |
| residual ablated | 13/100 | 568.588 | 0.02286 |
| residual shuffled | 13/100 | 568.588 | 0.02286 |
| deterministic random pair | 9/100 | 581.215 | 0.01549 |
| oracle source scheduler | 16/100 | 353.912 | 0.04521 |

The normalized work column divides each named dimension by its positive median
on the fit block and sums the ratios.  It is useful only for within-audit
description; raw vectors in the JSON are authoritative.

### Residual intervention and conditional information

- dynamic residual versus separately trained history-only model: four second
  choices differ, but coverage is identical, with 0 wins and 0 losses;
- replacing all residual tokens by the empty set changes **0/100** choices;
- shuffling residuals within the selected first source changes **0/100** choices;
- empirical
  \(I(m^*_2;R_1\mid\phi(T),m_1)=0.01070\) bit;
- stratified permutation mean is 0.02202 bit, the 95th percentile is 0.06270
  bit, and permutation `p=1.0`.

The four differences between two independently fitted model variants are not a
residual effect: direct ablation and shuffle of the residual input leave the
residual model unchanged.  The residual feature has been ignored.

The automatic gate is therefore:

`residual_behaviorally_inert`.

Raw audit result ID:
`3e6df17d9e2127df7f40a095272207f43d9494bbab65788dfb88a6650057cbd8`.

## Native-budget phase ablation

A deterministic 20-task pilot from the already opened offset 300 analysis
block was rerun under the fit-calibrated reservation contract.  This is a
development ablation, not a final accuracy estimate.

| Policy | pass@2 | native-comparable tasks | reservation violations | masked actions |
|---|---:|---:|---:|---:|
| structured deliberation | 4/20 | 20/20 | 0 | 17 |
| summary grounded | 4/20 | 20/20 | 0 | 17 |
| static grounding only, phase disabled | 4/20 | 20/20 | 0 | 17 |
| static task router | 4/20 | 20/20 | 0 | 19 |
| fixed CA-first | 2/20 | 20/20 | 0 | 42 |

The decisive mechanistic comparison is negative:

- structured, summary-grounded, and true phase-disabled grounding have
  identical action traces on **20/20**, identical selections on **20/20**, and
  identical realized and reserved native-cost vectors;
- structured and static task routing differ on 18/20 action traces, but have
  identical selected hypotheses and 0 paired pass wins/losses;
- static task routing spends 52,666 scene trials and 163,150 demo scene
  executions, versus 36,573 and 114,871 for structured, with the same 4/20
  pass rate;
- structured beats fixed CA-first on 2 tasks and loses on 0, but fixed CA-first
  avoids scene search entirely, so this pilot does not establish superior
  compute efficiency;
- all three grounding variants execute 14 repair actions and recover zero new
  tasks.

Adding the true ablation policy does not perturb the pre-existing policies:
their per-task actions, metrics, selections, and native costs are identical to
the preceding pilot on all 20 tasks.  Pool IDs change because the manifest
content-addresses the enlarged replay-policy/provenance set.

Raw result ID:
`fbefd0b0b17131eb21980943a746ad2efaf342a72729852fed10aaaa83f6bb98`.

The audit, calibration profile, and final phase ablation were replayed from
source commit `f62a0fe5ea326a0b20bcccd6543575e4f6230925`.  Relative to the pre-commit
pilot, all 20 pool IDs, every policy/task action and metric record, aggregate
accuracy, realized native cost, and reserved native cost are unchanged.

The correct conclusion is therefore stronger than “no significant gain”: the
current phase/residual terms have no observable causal effect on controller
behavior in this sample.  The static grounding model, not stateful switching,
explains the 4/20 result.

## Interpretation against the original motivation

### What survives

- ARC benefits from specialized, heterogeneous inductive biases.
- Content-addressed candidates, typed actions, provenance, oracle isolation,
  and deterministic replay are a credible foundation.
- A residual loop remains theoretically sensible and is aligned with refinement
  and rational-metareasoning work.

### What is currently falsified or absent

- Current residuals do not produce useful learned module switching.
- Current repair does not recover a natural ARC task outside the root pool.
- Current positive accuracy comes from candidate-language expansion and static
  representation priority.
- Masked diffusion and code/LLM options are not present in the formal pool.
- A conservative native token bucket is implemented and violation-free in the
  20-task pilot, but learned value-of-computation stopping is not implemented.

Thus, the system is still an auditable heterogeneous solver platform, not yet a
validated brain-like adaptive algorithm.

## Decision for the next implementation phase

Do not train a larger controller on the current data.  Proceed in this order:

1. **Candidate coverage:** add one credible masked-grid provider and one
   code/program provider.  Each must add at least 3-5 unique selectable tasks per
   100 development tasks at bounded native cost.
2. **Near-miss repair benchmark:** create parent/family-disjoint tasks with
   known fault types.  Compare residual-conditioned/blind and
   same-/cross-representation repair against equal-cost cold restart.  Require
   at least 5/100 unique recoveries before returning to natural-ARC repair
   claims.
3. **Repeat the residual audit:** only after the richer options create enough
   positive and heterogeneous outcomes.  Start with static boosting and a
   contextual bandit; use MLP/GRU only if residual ablation changes decisions and
   improves coverage.
4. **Scale native-budget evaluation:** reuse the fit-only reservation contract,
   add wall-clock/GPU/model-token dimensions for new providers, and run at least
   100 development tasks before any efficiency claim.
5. **Causal brain-like tests:** residual injection, representation-bridge
   lesion, provider-cost intervention, and unseen-option insertion are required
   before using functional-switching language as a contribution.

The final ARC evaluation remains sealed.  All ARC-AGI-1 training blocks used so
far are development data.

## Verification

- 32 relevant controller, deliberation, and metareasoning tests pass with the
  repository CA backend on `PYTHONPATH`;
- Ruff lint and format checks pass for all changed Python files;
- `git diff --check` passes;
- the one-task schema-v3 replay has zero behavior/metric regression;
- the 20-task native-budget ablation completes without failures or reservation
  violations, and its common-policy outputs exactly reproduce the prior pilot;
- audit input task sets are disjoint and their ID lists are content hashed.
