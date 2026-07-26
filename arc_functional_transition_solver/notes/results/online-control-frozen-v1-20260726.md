# Frozen-action online controller: ARC-AGI-1 training audit

## Decision

Keep the heterogeneous, content-addressed evaluation infrastructure, but do not
train or tune the current residual-first controller.  The 100-task run shows
real DSL/CA complementarity, yet the residual-first policy uses less than half
of the selectable union coverage and produces no uniquely recovered repair.
The next milestone should improve action-causal routing and repair coverage on a
separate development partition before adding learned routers or neural candidate
generators.

This is a bounded result on a deterministic sample of the public ARC-AGI-1
training split.  It is not an ARC leaderboard result and it does not establish
the full diffusion/code/DSL/DiffLogic architecture.

## Frozen protocol

- Code under test: `27112ffb7c97433c682a9b5ab2277d3e2c114991`
- Dataset: ARC-AGI-1 `training`
- Selection: SHA-256 ordering of `20260726:<task_id>`
- Main run: first 100 tasks in that ordering
- Prefix run: first 32 tasks in the same ordering
- Training/model downloads/remote model calls: none
- Oracle access: post-hoc, after each policy replay
- Candidate discovery: DSL and sparse-CA typed actions, then immutable
  `(provider, operator, parent_hypothesis_id) -> batch` freezing
- Per-task normalized budget cap: 11 compute units, 4 controller steps,
  3 provider calls, 1 repair attempt, 7 candidate slots
- Output: pass@2

The comparison is strict with respect to frozen action identity and the
normalized-compute ledger.  Native work (for example DSL program executions,
grid sizes, or CA fits) is recorded only descriptively, and policies can stop
below the common cap.  Consequently this run supports a cap-matched NCU claim,
not matched FLOPs or matched wall-clock time.

## Main result (n=100)

The heterogeneous pool has selectable oracle coverage on 10/100 tasks and raw
oracle coverage on 11/100.  The extra raw-only task (`4c4377d9`) has an
oracle-correct query output but no candidate satisfying the operational
demo-exact plus hard-verification eligibility gate, so it cannot contribute to
pass@2.

| Policy | Pool scope | Pass@2 | Selectable pool coverage | Exploration recall | Selection utilization | Realized NCU | Unique repair recovery |
|---|---|---:|---:|---:|---:|---:|---:|
| residual-first | heterogeneous | 4/100 | 10/100 | 0.40 | 1.00 | 1014 | 0 |
| static task router | heterogeneous | 4/100 | 10/100 | 0.40 | 1.00 | 1014 | 0 |
| random, seed 0 | heterogeneous | 8/100 | 10/100 | 0.80 | 1.00 | 958 | 0 |
| round-robin | heterogeneous | 7/100 | 10/100 | 0.70 | 1.00 | 972 | 0 |
| fixed DSL-first | heterogeneous | 7/100 | 10/100 | 0.70 | 1.00 | 1016 | 0 |
| fixed CA-first | heterogeneous | 5/100 | 10/100 | 0.50 | 1.00 | 897 | 0 |
| DSL-only | DSL plus generic repair | 7/100 | 7/100 | 1.00 | 1.00 | 932 | 0 |
| CA-only | CA | 5/100 | 5/100 | 1.00 | 1.00 | 423 | 0 |

Conditional selection utilization is 1.00 for every policy: whenever a policy
observes a selectable correct hypothesis, MDL plus hard verification places a
correct answer in the final two.  All observed loss is therefore in exploration
and action scheduling, not final ranking.

The pass-rate Wilson 95% intervals are wide: residual-first 1.6--9.8%, random
4.1--15.0%, DSL-first 3.4--13.8%, and union coverage 5.5--17.4%.  The paired
two-sided exact sign test for random versus residual-first is `p=0.125`
(four random-only successes and zero residual-only successes).  This run is
strong enough for a negative engineering gate, but not for a population-level
superiority claim.

## Complementarity and controller losses

Selectable coverage by source is:

- DSL: 7 tasks
- sparse CA: 5 tasks
- intersection: 2 tasks (`0d3d703e`, `c8f0f002`)
- DSL-only contribution: 5 tasks
- CA-only contribution: 3 tasks
- heterogeneous union: 10 tasks

Thus heterogeneity adds three tasks over the best single source (10 versus 7),
which supports the portfolio part of the original motivation.  The current
residual-first policy, however, reaches only 4/10 of that union.  Its pass set is
identical to the static router's pass set.  Residual feedback changes the stable
trace on 55/100 tasks, but never changes aggregate coverage or pass@2.

The ten operationally covered tasks are:

| Task | DSL pool | CA pool | Residual | Random | Round-robin | DSL-first | CA-first |
|---|---:|---:|---:|---:|---:|---:|---:|
| `3c9b0459` | yes | no | no | no | no | yes | no |
| `a87f7484` | yes | no | yes | yes | yes | yes | no |
| `c0f76784` | no | yes | no | yes | yes | no | yes |
| `a68b268e` | yes | no | yes | yes | yes | yes | no |
| `74dd1130` | yes | no | no | no | no | yes | no |
| `68b16354` | yes | no | no | yes | no | yes | no |
| `0d3d703e` | yes | yes | yes | yes | yes | yes | yes |
| `9edfc990` | no | yes | no | yes | yes | no | yes |
| `913fb3ed` | no | yes | no | yes | yes | no | yes |
| `c8f0f002` | yes | yes | yes | yes | yes | yes | yes |

Useful action-level evidence does exist.  DSL
`suffix_resynthesize` supplies correct hypotheses for `3c9b0459`, `74dd1130`,
`68b16354`, `a68b268e`, and `0d3d703e`; `shape_resynthesize` supplies one for
`a87f7484`.  CA `local_transition_search` supplies the CA-only successes.
Residual-first misses all three CA-unique tasks and three of five DSL-unique
tasks because its bounded action sequence does not expose the relevant frozen
batch.

## Repair result

There are 58 DSL-only, 58 DSL-first, 57 residual-first, and 42 round-robin
repair attempts.  Most abstain with `no_high_support_local_rule`.  Exactly one
correct repair candidate is generated, on `c8f0f002`, but that task already has
a selectable correct initial hypothesis.  Therefore:

- uniquely recovered tasks: 0
- uniquely recovered tasks per NCU: 0 for every policy
- the claim that residual-directed repair improves unit-compute recovery is not
  supported by this run

## Candidate and cost audit

- 534 unique frozen candidates: 379 DSL, 154 CA, and 1 residual-repair
- Per-task heterogeneous pool size: min 2, median 4, mean 5.34, max 12
- Wall time: 1854.44 seconds
- Recorded discovery time: 1651.26 seconds
- Recorded frozen replay time: 68.75 seconds
- Largest recorded task cost: `3631a71a`, 418.64 seconds; it has four 30x30
  demonstrations and a 30x30 test input

This long tail is evidence that NCU is not yet a sufficient physical-cost
normalizer.  A stronger experiment must cap or normalize DSL program
executions multiplied by evaluated cells, CA fits, and replay executions in
addition to the controller-level action ledger.

## Integrity and reproducibility

- Main result ID:
  `93803927361086c99d251c8f8eb5dd5cfbe78e7d9572d6bf238dde82c0e91618`
- Prefix result ID:
  `d17a433e31383eaee7f464170623c1a0cb4ffa066704f6715398ea0afde35644`
- Both summary hashes independently recompute exactly.
- All 132 pool-manifest hashes independently recompute exactly.
- Both runs have zero failures and `training_started=false`.
- The first 32 main-run task IDs equal the prefix run; all 32 pool manifests are
  byte-identical, and all stable actions, selections, budgets, and metrics match.
- Verification before the run: `335 passed, 9 skipped`; online-controller unit
  suite: `13 passed`; Ruff checks passed.

Artifacts:

- `results/online_control_frozen_v1_arc1_train_seed20260726_n100/summary.json`
- `results/online_control_frozen_v1_arc1_train_seed20260726_n100/pools/`
- `results/online_control_frozen_v1_arc1_train_seed20260726_n32/summary.json`
- `results/online_control_frozen_v1_arc1_train_seed20260726_n32/pools/`

Reproduce the main run from the repository root:

```bash
python arc_functional_transition_solver/scripts/afts_arc_online_matched_budget.py \
  /path/to/ARC-AGI-1/data \
  arc_functional_transition_solver/results/online_control_frozen_v1_arc1_train_seed20260726_n100 \
  --split training --limit 100 --sample-seed 20260726
```

## Gap to the original architecture

The current result validates infrastructure, not the complete brain-region-like
solver:

- active candidate regions are only typed DSL and sparse CA;
- masked diffusion, LLM/code generation, and DiffLogic callbacks remain explicit
  abstention boundaries in this run;
- the state transition is a hand-written bounded policy, not a learned or
  calibrated transition model;
- residual repair is a narrow local rule, not a general cross-representation
  CEGIS loop;
- evaluation is a training-split sample, not a frozen held-out ARC evaluation.

The main positive signal is source complementarity.  The main negative signal
is that the proposed residual controller does not exploit it.

## Next falsifiable milestone

1. Add a physical-cost ledger: DSL expansions, demo/query cell executions, CA
   fits, and replay executions; compare both common caps and realized cost.
2. Compile each residual or execution failure into a scored set of legal actions
   with explicit predicted coverage gain and native cost, rather than a fixed
   priority list.
3. Make the controller coverage-aware: preserve one exploration call for the
   alternate representation when the first source has no eligible exact
   hypothesis, then test against random and round-robin.
4. Redesign repair so that success is measured only on tasks with no correct
   initial candidate; require positive unique recovery before learning a repair
   policy.
5. Freeze this 100-task sample as development data.  Tune only there, then run
   once on a disjoint deterministic training partition and finally a held-out
   evaluation partition.
6. Add code/LLM, masked-diffusion, or DiffLogic generators one at a time only
   when each emits a replayable, typed, content-addressed candidate and adds
   unique selectable pool coverage under the physical budget.

Go forward only if a new controller increases exploration recall and pass@2
over random/round-robin without reducing unique repair efficiency under both
NCU and physical-cost accounting.  Otherwise invest in generator coverage, not
controller complexity.
