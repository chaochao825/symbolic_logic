# Frozen-action online control v2/v3: exact-first emission, pool closure, and paired controller ablation

## Decision

Continue the project, but move the next research milestone toward generator
coverage and genuinely coverage-adding repair rather than a more complicated
router.

The engineering changes in this round are real and reproducible. Exact-first
DSL emission and parent-safe candidate identity raise utilization of the
already available heterogeneous pool. Replay closure makes the frozen-pool
claim valid. Action-specific availability masking removes known-empty actions
and cuts normalized and native work substantially. On the untouched final
100-task block, the coverage-aware controller reaches every selectable pool
task (14/14) and uses the fewest normalized compute units among heterogeneous
policies.

The main method claim is nevertheless only partially supported. The final
controller does not beat simple schedules on pass@2: all seven heterogeneous
policies pass the same 14 tasks. Residual repair and action-conditioned DSL
descendants create correct candidates, but add zero unique heterogeneous
coverage. The remaining ceiling is therefore the candidate language, not the
final MDL selector and no longer primarily the action router.

This is a deterministic audit on disjoint blocks of the public ARC-AGI-1
training split. It is not an ARC leaderboard result and does not validate the
full masked-diffusion/code/DSL/DiffLogic architecture.

## Implemented changes

The implementation was advanced through three immutable code commits:

1. `c7f9fdcd7f9bd23427ba78c7b0d68bd3bc44da55`
   (`Improve ARC candidate emission and online control`)
   - DSL `synthesize` emits demo-exact candidates before task-conditioned
     near-miss seeds and preserves semantic diversity.
   - All eight D4 repair seeds are evaluated; demo-exact seeds are promoted.
   - Parent-insensitive CA actions no longer acquire arbitrary parents.
   - D4-shaped tasks expose DSL before CA.
   - `CoverageAwareResidualPolicy` v2 uses demo-only task features, source
     diversity, and semantic novelty.
   - Metrics record emitted versus accepted candidates, semantic duplicates,
     action-conditioned recovery, repair recovery, native operations, and
     timing.
2. `b2ef41590cff97b547bf440cd518e74b36049a53`
   (`Close frozen ARC pools over replay descendants`)
   - Every frozen policy is replayed before scoring.
   - All oracle-free replay descendants are merged into the common
     content-addressed pool.
   - Pool schema v2 records discovery/replay closure and replay-only content.
3. `d475a41ee750eb670006efe82e97de686ca11a5f`
   (`Prioritize certified ARC repair and mask exhausted actions`)
   - A globally certified color map outranks a local transition repair when
     both are type-legal.
   - The compiler masks an exact `(provider, operator, parent)` action when its
     frozen batch contains no unseen candidate.
   - Empty batches produce a zero-cost stop rather than a known abstention.

No neural model was trained, downloaded, or called. Masked diffusion, LLM/code
generation, and DiffLogic remain explicit future provider boundaries.

## Protocol and data partitions

- Dataset: ARC-AGI-1 public `training` split.
- Deterministic order: SHA-256 of `20260726:<task_id>`.
- Development block: positions 1--100.
- First disjoint block: positions 101--200.
- Final untouched block: positions 201--300.
- Pairwise intersections among all three blocks: zero.
- Oracle access: post-hoc only, after discovery and frozen replay.
- Frozen action key: `(provider, operator, parent_hypothesis_id)`.
- Pool closure: provider batches plus every oracle-free discovery/replay
  descendant, content-addressed before oracle scoring.
- Per-task cap: 11 NCU, 4 controller steps, 3 provider calls, 1 repair attempt,
  and 7 candidate slots.
- Final output metric: pass@2 after demo-exact, hard-rule, and MDL selection.

NCU is an action-level ledger, not a FLOP-equivalent measure. Native counts are
therefore reported separately. Policies can stop below the shared cap.

## Result progression

| Run | Code | Role | Raw pool | Selectable pool | Coverage-aware pass@2 | NCU | Failures |
|---|---|---|---:|---:|---:|---:|---:|
| v1, positions 1--100 | `27112ffb` | historical baseline | 11 | 10 | not present; residual-first 4 | residual-first 1014 | 0 |
| v2, positions 1--100 | `c7f9fdc` | development | 11 | 10 | 10 | 898 | 0 |
| v2b, positions 101--200 | `b2ef415` | disjoint diagnosis | 11 | 9 | 8 | 938 | 0 |
| v3, positions 201--300 | `d475a41` | untouched final validation | 15 | 14 | 14 | 630 | 0 |
| v2b, positions 201--300 | `b2ef415` | paired ablation only | 15 | 14 | 14 | 945 | 0 |

On the original development block, v2 does not expand raw oracle coverage
(11 tasks in both v1 and v2). It instead makes every selectable correct
candidate reachable: heterogeneous policies rise from 4--8 passes in v1 to
10/10 in v2. Candidate count falls from 534 to 393 without losing coverage.
This is an emission and controller-utilization improvement, not new conceptual
coverage.

The first disjoint block exposed two distinct defects. An initial run completed
99/100 tasks and failed `a740d043` because replay emitted a repair descendant
that was absent from the declared pool. Pool closure v2 fixed that validity
error. The valid v2b run then had 9 selectable pool tasks but the coverage-aware
controller passed 8: `a740d043` required the globally certified color-map
repair, while the policy selected a competing local repair. That block was used
to design v3 and is therefore diagnosis data, not final untouched evidence.

The final block was selected before viewing its results and run once from
`d475a41`. It completed all 100 tasks with no pool-closure failure.

## Final untouched result (positions 201--300)

| Policy | Pool scope | Raw / selectable pool | Observed selectable | Pass@2 | Realized NCU | Unique repair recovery |
|---|---|---:|---:|---:|---:|---:|
| coverage-aware v2 | heterogeneous | 15 / 14 | 14 | 14 | **630** | 0 |
| residual-first | heterogeneous | 15 / 14 | 14 | 14 | 651 | 0 |
| random, seed 0 | heterogeneous | 15 / 14 | 14 | 14 | 641 | 0 |
| round-robin | heterogeneous | 15 / 14 | 14 | 14 | 639 | 0 |
| fixed CA-first | heterogeneous | 15 / 14 | 14 | 14 | 652 | 0 |
| fixed DSL-first | heterogeneous | 15 / 14 | 14 | 14 | 642 | 0 |
| static task router | heterogeneous | 15 / 14 | 14 | 14 | 645 | 0 |
| DSL-only | DSL plus generic repair | 11 / 10 | 10 | 10 | 561 | 3 within this restricted scope |
| CA-only | CA | 8 / 8 | 8 | 8 | 126 | 0 |

All heterogeneous policies have exactly the same 14-task pass set. Therefore
the final run supports full selectable-pool utilization and a modest NCU
advantage for the coverage-aware policy, but no pass@2 superiority over random,
round-robin, or fixed schedules. The pass-rate Wilson 95% interval is
8.53--22.14%; raw union coverage 15/100 has interval 9.31--23.28%.

The one raw-only task is `662c240a`. A near-miss DSL root and its
shape-conditioned descendants happen to contain the correct query output, but
the candidate is not demo-exact plus hard-verified and is operationally
ineligible. It is correctly excluded from pass@2.

## Paired ablation: what v3 actually changes

The final 100 tasks were rerun from real commit `b2ef415` with identical task
IDs, generator parameters, pool-closure protocol, and budget. Both runs have
valid hashes and zero failures. Candidate-ID sets are identical on 100/100
tasks, and all shared candidate payloads are byte-equivalent under canonical
JSON. Thus paired differences are controller effects, not generator effects.

| Policy | v2b pass / NCU | v3 pass / NCU | NCU saved | Tasks saving NCU |
|---|---:|---:|---:|---:|
| coverage-aware v2 | 14 / 945 | 14 / 630 | 315 (33.3%) | 77 |
| residual-first | 12 / 1023 | 14 / 651 | 372 | 82 |
| random, seed 0 | 13 / 1002 | 14 / 641 | 361 | 81 |
| round-robin | 13 / 1006 | 14 / 639 | 367 | 81 |
| fixed CA-first | 14 / 977 | 14 / 652 | 325 | 71 |
| fixed DSL-first | 12 / 1017 | 14 / 642 | 375 | 83 |
| static task router | 12 / 1017 | 14 / 645 | 372 | 82 |
| DSL-only | 10 / 810 | 10 / 561 | 249 | 83 |
| CA-only | 8 / 369 | 8 / 126 | 243 | 81 |

The old controller executes 1,078 proposal actions that the frozen pool already
knows cannot emit unseen content; every one ends as `frozen_action_exhausted`.
V3 removes all 1,078. The median per-task saving for the coverage-aware policy
is 3 NCU, with no task becoming more expensive. Its pass@2 is unchanged because
it already reached all 14 selectable tasks, while less robust schedules gain
access to `b6afb2da` and/or `25ff71a9` after preserving budget.

Native counts move in the same direction for coverage-aware v2:

- DSL demo executions: 18,840 to 8,798 (-53.3%).
- DSL program trials: 5,692 to 2,622 (-53.9%).
- DSL query executions: 6,074 to 2,876 (-52.7%).
- CA policy fits: 96 to 62 (-35.4%).
- DSL program expansions: 63,495 to 62,595 (-1.4%).

The small expansion change shows that initial broad DSL enumeration remains the
dominant irreducible cost. The large execution/trial reduction comes from
masking fruitless follow-up actions. End-to-end wall time is almost unchanged
(1211.47 versus 1207.23 seconds) because candidate discovery and two long-tail
tasks dominate total runtime.

Correct repair candidates per NCU rise mechanically because the numerator is
unchanged while NCU falls (3/945 to 3/630, +50% for coverage-aware v2). This is
an efficiency improvement, but not evidence of new repair capability: unique
heterogeneous repair recovery remains zero.

## Candidate complementarity and remaining ceiling

The final heterogeneous pool contains 374 candidates:

- 283 typed-DSL candidates;
- 86 CA candidates (83 sparse categorical CA and 3 bounded CA programs);
- 5 residual-repair candidates.

DSL has raw coverage on 11 tasks and selectable coverage on 10. CA covers 8
tasks. Their intersection is 4, so the heterogeneous selectable union is
14 tasks: six selectable DSL-only contributions, four CA-only contributions,
and four shared tasks. The portfolio architecture remains useful.

Among DSL candidates, 191 are task-conditioned near-miss seeds, 8 are emitted
in the demo-exact lane, and 84 are action-conditioned shape/suffix descendants.
There are seven oracle-correct action-conditioned candidates on four tasks, but
each task already has a correct root candidate. Action-conditioned unique
recovery is therefore zero.

Five residual candidates are emitted. Four are post-hoc oracle-correct. The
coverage-aware run observes three correct repairs (`50cb2852`, `b1948b0a`, and
`6f8cd79b`), but all already have a correct initial candidate. DSL-only records
three repair recoveries (`50cb2852`, `bb43febb`, and `6f8cd79b`), yet all three
are covered by CA in the heterogeneous pool. Repair adds redundancy, not union
coverage.

Coverage remains structurally narrow:

| Demo shape relation | Tasks | Raw pool | Selectable / pass |
|---|---:|---:|---:|
| same input/output shape | 65 | 11 | 11 |
| shrinking output | 27 | 4 | 3 |
| expanding output | 8 | 0 | 0 |

Across the three disjoint 100-task blocks, the versioned runs descriptively
contain 37 raw-covered tasks, 33 selectable-covered tasks, and 32
coverage-aware passes. This 300-task total is not a single frozen evaluation:
the first block was development data and the second influenced v3. It should
not be used as an inferential headline. It does show that roughly nine out of
ten tasks still lie outside the current candidate language.

## Failure disclosure and diagnostic runs

All outputs are retained, including invalid and intermediate diagnostics:

- `online_control_frozen_v2_arc1_train_seed20260726_holdout_offset100_n100`
  is invalid as a 100-task aggregate: 99 tasks completed and `a740d043` failed
  with `observed candidate is absent from the declared frozen pool`.
- `online_control_frozen_v2_pool_closure_regression_a740d043` verifies pool
  closure on that task: discovery has 4 candidates, closure has 5, including 1
  replay-only candidate.
- `online_control_frozen_v2b_arc1_train_seed20260726_holdout_offset100_n100`
  is the valid closure-corrected rerun.
- `online_control_frozen_v3_repair_priority_regression_a740d043` shows all
  heterogeneous policies can choose the correct global color map, but still
  spends 11 NCU and contains one exhausted frozen action.
- `online_control_frozen_v3b_action_mask_regression_a740d043` keeps the pass,
  reduces NCU to 8, and has zero exhausted actions.

The two one-task v3 diagnostics and the closure diagnostic were run while the
corresponding changes were uncommitted, so their `source_commit` field names
the preceding base commit. They are regression evidence, not canonical release
runs. The final 100-task v3 summary records the complete immutable commit
`d475a41`.

## Integrity and reproducibility

- Final result ID:
  `be5ed017abe7b1013ca6c7900472b6097ee7ca70208766d311e7cf1d658bca72`.
- Paired v2b ablation result ID:
  `8169f270202ed6be5d3323c5f666a560f278560943a490edee21586ca88b06a1`.
- Development v2 result ID:
  `46ad75f324ea8f8482b07f398a062ea465f49273a918489497fe080bc1bc4c09`.
- Valid first-disjoint v2b result ID:
  `b221ac6cc1a6d94813e04b02ca00698f67106d34b1cef8ff12be43e308750e32`.
- Every official summary hash recomputes exactly.
- Every pool hash in the three official 100-task runs and paired ablation
  recomputes exactly (400/400 manifests).
- Final and paired runs have identical task IDs and identical per-task
  candidate-ID sets.
- All three 100-task blocks are disjoint.
- `oracle_used_during_discovery=false` and
  `oracle_used_during_pool_closure=false` in every final pool.
- Full verification at `d475a41`: `340 passed, 9 skipped` in 445.03 seconds.
- Targeted online-controller verification: `18 passed`; Ruff checks passed.

Canonical artifacts:

- `results/online_control_frozen_v2_arc1_train_seed20260726_n100/`
- `results/online_control_frozen_v2b_arc1_train_seed20260726_holdout_offset100_n100/`
- `results/online_control_frozen_v3_arc1_train_seed20260726_holdout_offset200_n100/`
- `results/online_control_frozen_v2b_arc1_train_seed20260726_holdout_offset200_n100_paired_ablation/`

Additional smoke, failed, and one-task diagnostic artifacts are committed under
their original result-directory names.

## Claim assessment

The proposed claim was that compiling failures and residuals into legal
cross-representation actions, then controlling replayable content-addressed
candidates online under matched compute, would improve oracle-coverage
utilization, final pass@2, and repair rate per compute.

- **Coverage utilization: supported.** On the development block, residual-first
  uses 4/10 selectable tasks in v1, while v2 uses 10/10. On final validation,
  coverage-aware v2 uses 14/14.
- **Final pass@2: partially supported.** V3 raises several simple policies by
  one or two passes in the paired ablation, with identical candidate pools. The
  target coverage-aware policy is already at 14/14 before v3 and gains no pass.
- **Repair per compute: weakly supported only as efficiency.** Correct repairs
  per NCU improve, but unique heterogeneous repair recovery remains zero.
- **Policy novelty over portfolios: not established.** Random, round-robin,
  fixed schedules, static routing, residual-first, and coverage-aware v2 all
  pass the same final 14 tasks.

The strongest defensible contribution is now a valid, replayable experimental
substrate plus an action-availability controller that removes provably wasted
work. The brain-region-like functional switching idea remains promising, but
the evidence says that new regions and better within-region inductive biases
must add unique candidates before a learned switching mechanism can matter.

## Next falsifiable milestone

1. Add a physical-cost token bucket before action execution: DSL expansion,
   demo/query cell execution, CA fits, and replay cost. Cache each typed batch
   once per task so policy comparisons do not inherit first-run timing effects.
2. Expand the DSL in coverage-targeted families, measured one at a time:
   object/component selection, relational movement/recoloring, bounding-box
   crop/insert, canvas expansion and tiling, and object composition. Require
   unique selectable pool coverage on the frozen development block.
3. Redesign repair as typed counterexample-guided synthesis. Compile residuals
   into global color maps, object edits, crop/insert operations, and local CA
   patches; execute a repair only when it can falsifiably improve the parent.
   The gate is at least one task with no correct root candidate that is uniquely
   recovered by repair.
4. Optimize the controller for expected *unique selectable coverage gain per
   native cost*, not total correct candidates. Calibrate only on positions
   1--200, then freeze it before a new disjoint evaluation.
5. Add code/LLM, masked-diffusion, or DiffLogic generators only through the same
   typed, replayable, content-addressed contract. Each provider must justify
   itself by new union coverage under the physical budget.
6. Run a larger untouched public-training audit and then a genuinely held-out
   evaluation split. Do not claim algorithmic superiority until the controller
   beats random/round-robin on paired pass@2 and unique repair recovery, with
   confidence intervals and physical-cost accounting.
