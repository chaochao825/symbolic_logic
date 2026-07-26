# Structured visual deliberation: ARC evidence report

Date: 2026-07-26

## Claim boundary

This report evaluates a no-training, formally replayable slice inspired by five
recent multimodal-reasoning papers.  It does not reproduce their task-specific
models or compare their reported accuracies directly with ARC.  The tested claim
is narrower: a typed object-predicate language can expand selectable ARC
coverage, and a structured factual/residual sketch can improve utilization of a
frozen heterogeneous pool under matched controller reservations.

Normalized compute units (NCU) match controller reservations, not FLOPs or wall
clock.  Provider-native counters and elapsed time are therefore reported as a
separate claim boundary.  Empty frozen actions are replayed once and charged in
the final protocol; the earlier free-empty-action behavior was diagnosed and
removed before the final block was opened.

## Mechanism audit and transfer

| Work | Original mechanism | ARC transfer implemented here | Remaining gap |
|---|---|---|---|
| [Hilbert-Geo](https://arxiv.org/abs/2605.16385) | Parse multimodal conditions into CDL, then search a predicate/theorem bank | Deterministic object parsing, typed selectors/actions, replay and hard verification | Tiny heuristic language; no learned parser or relation/theorem closure |
| [Step-CoT](https://arxiv.org/abs/2603.13878) | Step graph, global memory node, gated teacher/student updates | Content-addressed phase sketch with factual, residual, and provider-call memory | Hand-designed phase transition; no learned graph or step supervision |
| [SDRL](https://arxiv.org/abs/2603.25942) | `Summarize -> Think -> Answer`, factual-summary consistency, accuracy-conditioned diversity | Stable factual tokens separated from diversity scoring based on demo verification | Hand-tuned score; no learned policy or calibrated uncertainty |
| [Visual Thoughts](https://arxiv.org/abs/2505.15510) | Clear, concise textual or image-form visual intermediates carry image information through deeper reasoning | Compact scene/residual sketch instead of verbose prose | Symbolic tokens only; no learned visual cache or image-form thoughts |
| [Agile Deliberation](https://arxiv.org/abs/2512.10821) | Concept scoping, clustered borderline examples, bandit/UCB cluster allocation, and greedy prompt refinement | Semantic-cluster count, residual frontier, and UCB-style representation exploration | Verification replaces human usefulness rewards; no persistent cross-task ambiguity bank or human refinement loop |

The system-level mechanisms that predate this experiment remain unchanged:
content-addressed heterogeneous candidates, typed actions, immutable blackboard
states, demo replay, hard verification, MDL ranking, explicit budget receipts,
and posthoc-only oracle scoring.

## Implemented ablations

Controller variants on the same expanded frozen pool are:

- `coverage_aware_expanded`: old controller, new three-provider pool;
- `summary_grounded`: phase sketch and task-grounding term only;
- `adaptive_diversity`: verification-conditioned representation diversity only;
- `structured_deliberation`: grounding, diversity, and borderline/UCB terms;
- residual, fixed-order, round-robin, static-router, and deterministic-random
  controls.

Provider variants are the previous typed grid DSL, sparse CA, the new scene DSL,
and their union.  `coverage_aware_v2` retains the old DSL+CA scope and is the
capability baseline.

Primary budget is 11 NCU / 4 steps / 3 provider calls / 1 repair / 7 slots.
The frozen tight budget is 6 NCU / 3 steps / 2 calls / 0 repairs / 4 slots.
Both use provider batch size two and pass@2 selection.

## Protocol correction before formal evidence

An audit found that an unattempted frozen action whose discovery batch was
empty was reported as unavailable.  Controllers could therefore avoid the call
without paying its reservation, effectively receiving free provider-availability
lookahead.  This did not expose query outputs, but it broke matched-budget
fidelity.  Commit `0244a3c` changes an empty frozen batch into a legal one-time
abstention replay charged at the full reserved budget, and the regression test
now requires that behavior.  All pre-fix runs were invalidated and moved to
ignored trash; every result ID below is from the corrected protocol.

## Results

### Development block, primary budget

Source commit: `0244a3c5dfda1489383fadeeeba03b8fadc0db13`

Result ID: `ad0cee9b0ff7852f3a693663707c084d51847bb1b4dc1556944e3bdb778dbf40`

Tasks: SHA-256-seeded positions 1--100; 100 completed, zero failed.  Elapsed
time was 1,551.19 seconds.

| Policy / pool | pass@2 | selectable pool | observed selectable | total NCU |
|---|---:|---:|---:|---:|
| `coverage_aware_v2` (old DSL+CA scope) | 10 | 10 | 10 | 898 |
| `coverage_aware_expanded` | 10 | 11 | 10 | 898 |
| `summary_grounded` | 11 | 11 | 11 | 960 |
| `adaptive_diversity` | 10 | 11 | 10 | 984 |
| `structured_deliberation` | 11 | 11 | 11 | 960 |
| `static_task_router` | 8 | 11 | 8 | 1,012 |
| deterministic random | 11 | 11 | 11 | 972 |
| typed DSL only | 7 | 7 | 7 | 815 |
| sparse CA only | 5 | 5 | 5 | 357 |
| scene DSL only | 3 | 3 | 3 | 300 |

The old selectable union contained ten tasks.  The scene DSL added exactly one
new task, `3aa6fb7a`, with the verified rule `all objects -> fill background
cells inside each bounding box with color 1`.  The other two scene successes
overlapped the old DSL/CA pool.  Grounding and the combined policy chose the new
rule; the old expanded controller and diversity-only policy did not.  This is a
single paired win and no losses, which is not statistically persuasive
(two-sided exact paired p-value 1.0), and deterministic random obtained the same
11 passes.

Failure-aware accounting materially changed the cost conclusion.  The old
controller used 898 NCU; the combined policy used 960.  Mean NCU on successful
tasks changed from 7.40 to 8.36.  There were no unique repair recoveries.  Thus
the development result supports a small language-coverage gain but not a
gain-per-compute or controller-superiority claim.

Across all 100 pool manifests, scene synthesis attempted 285,976 rules and
946,508 demonstration executions, produced a candidate on only three tasks,
and produced one old-pool-unique solution.  NCU is consequently insufficient as
a physical-cost equivalence measure.

### Development block, tight budget

Result ID: `21f39379c6847d81510c9c20c0ddd83f849953dda0f8887b5ee55a19755d225d`

The same 100 tasks completed with zero failures in 1,426.90 seconds.

| Policy / pool | pass@2 | selectable pool | observed selectable | total NCU |
|---|---:|---:|---:|---:|
| `coverage_aware_v2` | 10 | 10 | 10 | 588 |
| `coverage_aware_expanded` | 10 | 11 | 10 | 588 |
| `summary_grounded` | 11 | 11 | 11 | 600 |
| `adaptive_diversity` | 10 | 11 | 10 | 600 |
| `structured_deliberation` | 11 | 11 | 11 | 600 |
| `static_task_router` | 8 | 11 | 8 | 600 |
| deterministic random | 9 | 11 | 9 | 597 |

Grounding again recovered `3aa6fb7a` with no loss relative to the old expanded
controller, while diversity alone did not.  The combined method also had two
paired wins and no losses against deterministic random, but two discordances
still give only p=0.5 under a two-sided exact paired test.  It used the full cap
on every successful task (6.0 mean NCU); the expanded controller averaged 5.4.
H1 therefore has a positive development signal at the tight endpoint, H2 is
falsified on this block, and the UCB/diversity terms add nothing beyond the
grounding term.

### Disjoint final block, primary budget

Result ID: `e0004ba94f4475fd0361cfba8509e58f6502a5c182dcd8120cb1fe4057c6a5fe`

Tasks: the pre-registered SHA-256-seeded positions 301--400; 100 completed,
zero failed.  Elapsed time was 2,129.13 seconds.  The source remained commit
`0244a3c5dfda1489383fadeeeba03b8fadc0db13`; no result-dependent code or
weight changes were made after opening this block.

| Policy / pool | pass@2 | selectable pool | observed selectable | total NCU |
|---|---:|---:|---:|---:|
| `coverage_aware_v2` (old DSL+CA scope) | 11 | 13 | 11 | 943 |
| `coverage_aware_expanded` | 10 | 16 | 10 | 946 |
| `summary_grounded` | 16 | 16 | 16 | 962 |
| `adaptive_diversity` | 10 | 16 | 10 | 997 |
| `structured_deliberation` | 16 | 16 | 16 | 962 |
| `fixed_ca_first` | 13 | 16 | 13 | 975 |
| `round_robin` | 10 | 16 | 10 | 1,000 |
| `static_task_router` | 9 | 16 | 9 | 1,003 |
| deterministic random | 12 | 16 | 12 | 992 |
| typed DSL only | 6 | 6 | 6 | 787 |
| sparse CA only | 8 | 8 | 8 | 369 |
| scene DSL only | 5 | 5 | 5 | 300 |

The scene DSL added three selectable tasks not solved by either old specialist:
`00d62c1b`, `810b9b61`, and `b2862040`.  Its other two successes overlapped the
typed DSL.  The combined controller selected all 16 selectable solutions.  It
had six paired wins and no losses against `coverage_aware_expanded`, giving a
two-sided exact paired p-value of 0.03125.  It also had four wins and no losses
against deterministic random (p=0.125), and five wins with no losses against
the old-scope controller (p=0.0625; this last comparison also changes the pool).

The three unique rules are semantically compact rather than accidental long
programs: select a filled rectangle and fill its box with color 4 on background
3 (`00d62c1b`); select objects with holes and outline their boxes in color 3
(`810b9b61`); and select objects with holes and recolor them to 8 on background
9 (`b2862040`).  Each rule is serialized, independently replayed, and exact on
every demonstration before posthoc query scoring.

The gain decomposes cleanly: three wins come from the expanded scene language,
while two additional wins over the old controller are old sparse-CA solutions
that the grounded routing policy reaches.  The sixth win over the expanded
controller is another sparse-CA solution lost by that controller's changed
ordering.  `summary_grounded` and the full policy produced identical action
traces and selections on every one of these 100 tasks (and on both 100-task
development runs), whereas `adaptive_diversity` did not improve the expanded
baseline's endpoint.  The evidence therefore supports grounding/state
representation, not the added diversity or UCB score; in the tested score
geometry those extra terms are behaviorally inert once grounding is enabled.

NCU tells a favorable reservation-level story: pass/100 NCU is 1.66 for the
combined policy versus 1.06 for the expanded baseline.  It does not establish
physical compute efficiency.  The combined policy performed 187,624 scene-rule
trials and 622,741 demonstration scene executions, versus 19,100 and 77,950 for
the expanded baseline.  An exhaustive scene-only sweep performed 304,682 rule
trials and 1,019,879 demo executions to obtain five passes, three unique to the
old pool.  Mean NCU on successful tasks was also higher (8.56 versus 7.90).
There were no repair-unique recoveries.

### Disjoint final block, tight budget

Result ID: `29289e521b42684727b2b7474abfc4444903756f5ca5d9f7383661ce2e663c11`

The same disjoint 100 tasks completed with zero failures in 2,022.18 seconds.
No intermediate correctness was inspected, and the source remained commit
`0244a3c5dfda1489383fadeeeba03b8fadc0db13`.

| Policy / pool | pass@2 | selectable pool | observed selectable | total NCU |
|---|---:|---:|---:|---:|
| `coverage_aware_v2` | 10 | 13 | 10 | 588 |
| `coverage_aware_expanded` | 10 | 16 | 10 | 591 |
| `summary_grounded` | 12 | 16 | 12 | 594 |
| `adaptive_diversity` | 10 | 16 | 10 | 594 |
| `structured_deliberation` | 12 | 16 | 12 | 594 |
| `fixed_ca_first` | 13 | 16 | 13 | 594 |
| `round_robin` | 10 | 16 | 10 | 597 |
| `static_task_router` | 8 | 16 | 8 | 597 |
| deterministic random | 5 | 16 | 5 | 600 |
| typed DSL only | 6 | 6 | 6 | 594 |
| sparse CA only | 8 | 8 | 8 | 369 |
| scene DSL only | 5 | 5 | 5 | 300 |

Grounding again moves in the expected direction: two wins (`00d62c1b` and
`6c434453`) and no losses versus either old controller, but the two-sided exact
paired p-value is 0.5.  It improves selectable-pool utilization from 10/16 to
12/16 and pass/100 NCU from 1.69 to 2.02.  This is not the best tight controller:
`fixed_ca_first` reaches 13/16 at the same 594 NCU, with two wins and one loss
against the combined policy.  The tight result therefore favors a cheap strong
specialist before broader deliberation.

The physical-cost mismatch is larger here.  The combined policy performs
167,648 scene-rule trials and 556,823 demo scene executions; the expanded
baseline performs 2,765 and 8,295 respectively.  That is a 60.6x scene-trial
imbalance under nearly equal NCU.  Mean NCU on successful tasks is 5.50 versus
5.40.  Again there are no repair attempts by construction and no repair
recoveries.

`summary_grounded` and the full policy have identical action traces and final
selections on all 100 tight tasks as well.  Across all four formal task-budget
runs, the diversity/UCB additions never change a single action selected after
grounding is enabled.

## Reproducibility and leakage audit

A post-run assertion audit loaded all four summaries, all 400 referenced pool
manifests, and all 1,505 serialized candidates.  It verified unique task IDs,
source commit, offsets, disjoint development/final task sets, identical
primary/tight task ordering, zero failures, budget upper bounds, manifest/pool
content IDs, candidate counts, posthoc-only oracle mode, and absent explicit
query-output keys.  Proposal and repair action counts exactly match every budget
receipt.  In total, 5,736 empty frozen proposals appear as charged one-time
abstention replays, confirming that the corrected failure accounting is active.

## Interpretation

The candidate-language and controller claims are judged separately.  A higher
expanded-pool ceiling supports the predicate-language hypothesis.  A policy
claim additionally requires a paired gain over `coverage_aware_expanded` on the
same pool; matching a random or static baseline is not evidence of a superior
controller.  Repairs count only when their candidate uniquely recovers a task.

| Frozen hypothesis | Evidence across development and disjoint final runs | Decision |
|---|---|---|
| H1: summary grounding | +1 development and +6 final primary passes over the expanded controller; +1 development and +2 final tight; final-primary p=0.03125 | Supported for pool utilization/pass@2 under NCU, but not for lower native cost or universal tight-budget superiority |
| H2: adaptive diversity | diversity-only never improves the expanded endpoint; full and grounding-only traces are identical in all four runs | Falsified / behaviorally inert |
| H3: object-predicate language | one old-pool-unique development task and three old-pool-unique final tasks | Supported, with very poor exhaustive-search efficiency |
| H4: combined method | final primary 16 vs 11 old-scope and 10 expanded; final tight 12 vs 10, but fixed-CA reaches 13 | Partially supported at matched NCU; not established at matched physical compute |

The original compound claim is therefore only partially supported:

| Claim component | Primary final evidence | Verdict |
|---|---|---|
| Typed, replayable, content-addressed cross-representation control | Every action/candidate has a canonical ID, frozen replay receipt, and hard verification path | Established as infrastructure |
| Better selectable-pool utilization | 10/16 to 16/16 versus the expanded controller | Supported on the disjoint block |
| Better final pass@2 | 10 to 16, six paired wins, zero losses, p=0.03125 | Supported under NCU reservations |
| Better repair recovery per compute | zero unique repair recoveries for every policy | Falsified in this implementation |
| Strictly matched physical compute | 187,624 versus 19,100 scene trials despite similar NCU | Not established; only controller reservations match |

Thus the real positive result is a grounded **representation-routing** loop plus
a useful object-predicate language.  It is not yet evidence that residual-local
repair works, that adaptive diversity helps, or that the method wins under
FLOP/latency-matched compute.

The final-block error budget also changes the priority of future work.  The
expanded union has a correct candidate for 17/100 tasks, only 16 are selectable
under pass@2, and the grounded controller passes all 16.  Therefore 83 tasks are
now candidate-language failures, one is a selection/aliasing failure, and zero
of the selectable tasks are lost by the grounded controller.  More router
heuristics cannot lift the 16% ceiling; provider coverage and cost-efficient
synthesis are the next bottlenecks.

## ARC-specific competitive context

These runs are on SHA-256-selected ARC-AGI-1 **training** tasks and are an
internal mechanism test, not an ARC evaluation or leaderboard submission.  The
16/100 final-block result therefore must not be compared as though it were a
16% public-evaluation score.  Even as a rough capability indicator, the current
solver is not accuracy-competitive with recent dedicated ARC systems:

- [TRM](https://arxiv.org/abs/2510.04871) reports 45% ARC-AGI-1 test accuracy
  and 8% on ARC-AGI-2 using a 7M-parameter recursive network;
- [CompressARC](https://arxiv.org/abs/2512.06104) reports 20% on ARC-AGI-1
  evaluation with a 76K-parameter inference-time MDL learner and no pretraining;
- [Compositional Neuro-Symbolic Reasoning](https://arxiv.org/abs/2604.02434)
  reports 24.4% on ARC-AGI-2 public evaluation, and 30.8% when its object/DSL
  solver is combined with an ARC language solver through a meta-classifier;
- the official [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
  reports 24% on the ARC-AGI-2 private competition set at USD 0.20/task for the
  top open competition entry.  Dataset, leakage controls, and cost definitions
  differ from this experiment.

The present competitive asset is narrower: auditable portfolio control across
heterogeneous, replayable representations.  Recent ARC systems supply exactly
the missing ingredients that could sit inside that substrate: learned recursive
repair (TRM), inference-time compression (CompressARC), and neural object/DSL
proposal plus a learned meta-classifier (Compositional Neuro-Symbolic
Reasoning).  Conversely, none of the current results supports a broad
brain-region-switching or state-of-the-art ARC claim.

### Gap to the original brain-region portfolio motivation

| Intended component | Tested implementation | Evidence / missing piece |
|---|---|---|
| masked diffusion for many candidates | absent from the frozen pool | no learned/global candidate generator or denoising trajectory |
| LLM/code model for open hypotheses | absent | no open-ended rule invention or code-to-typed-action compiler exercised |
| DSL/program synthesis for verifiable rules | old grid DSL plus the new scene DSL | useful: scene language adds three unique final-block tasks, but the union ceiling is only 17/100 |
| sparse CA / D4 / background padding | present | supplies eight specialist solutions and seven old-pool-unique final tasks |
| DiffLogic hard circuits | not exercised as a provider in this experiment | no learned compressible transition circuit in the compared pool |
| LGN/MLP/XGBoost screening and repair policy | hand-written grounding ranks only | grounding helps; diversity/UCB is inert; no learned gain-per-cost calibration |
| brain-area state transition / functional switching | deterministic typed phase sketch and action compiler | auditable but not learned, probabilistic, or diffusion-like |
| residual-directed repair | legal compiled repair actions | zero unique recoveries, so the substantive repair claim remains open |

This is why the result is a credible systems/method increment, not yet a
realization of the initial full architecture.

## Next engineering and research gates

1. Replace exhaustive scene enumeration with demo-difference constraints;
   memoize parses, selected object sets, and transformed grids.  Gate this on at
   least a 10x reduction in scene-rule executions without losing any of the
   four development/final unique scene solutions.
2. Replace uniform NCU with provider-native reservations and caps: grid-cell
   executions, program trials, model tokens/FLOPs, latency, and memory.  Repeat
   the paired comparison with both NCU and native cost matched; do not call the
   present 9.8x scene-trial imbalance a strict physical-compute result.
3. Train an XGBoost/MLP gain-per-native-cost prior over only legal typed actions
   and development receipts.  Keep the hard action compiler and verifier;
   compare it with grounding-only, fixed order, and random on a newly sealed
   block.
4. Make stopping confidence- and representation-aware so two same-provider
   demo-exact guesses cannot terminate exploration when the pool still contains
   an untried specialist with high predicted value.
5. Add replayable neural/code/diffusion specialists: a TRM-like iterative grid
   repairer, a code-model producer compiled into the DSL, and a masked grid
   generator whose samples are content-addressed and demo-verified.  Their raw
   outputs must never bypass typed compilation.
6. Move the next frozen evaluation to untouched ARC-AGI-1 evaluation and then
   ARC-AGI-2 public evaluation.  Report official pass@2, wall time, peak memory,
   tokens/FLOPs, and USD/task alongside candidate ceiling and utilization.
