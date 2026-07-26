# Stateful residual controller

## Scope

This implementation supplies the control mechanism that the v1 functional
router lacked.  It does not claim a new ARC accuracy result.  It turns the
existing providers, replay gate, MDL ranker, and repair operators into an online
process whose next action depends on the observed execution result of the
previous action.

The deployable state is

```text
s_t = (
  blind task features and representation route,
  active content-addressed candidates,
  replay evaluations and residual signatures,
  provider/repair receipts and quarantines,
  exact remaining budget,
  append-only action history
)
```

Every state, action, and action result has a deterministic 24-hex content ID.
An action is bound to the predecessor `state_id`, so a policy cannot replay an
action against a stale blackboard.  All provider inputs remain `BlindTask`; test
outputs are available only to the separate post-hoc metric function after the
controller has stopped.

## Closed loop

```text
heterogeneous replayable candidate sources
  -> immutable blackboard s_t
  -> compile residual/failure signals into a legal action mask
  -> choose one (provider, operator, parent, budget slice)
  -> execute/replay/verify one bounded batch
  -> quarantine content or semantic collisions
  -> update residuals and remaining budget
  -> compile again, or emit explicit STOP
```

`OnlineFunctionalRouterSolver` performs exactly one non-STOP action per state
transition.  Unlike the v1 solver, it does not eagerly invoke every provider and
then repair every parent.  `ResidualFirstPolicy` prioritizes a legal local repair
or a residual-targeted representation switch; `FixedSchedulePolicy` is a
matched-budget run-all ablation that exhausts provider proposals before repair.

## Typed action language

The action type system rejects a route/operator mismatch before execution:

| Route | Legal operators |
|---|---|
| `dsl_program` | `synthesize`, `shape_resynthesize`, `suffix_resynthesize` |
| `code_llm` | `open_hypothesis`, `exception_resynthesize`, `counterfactual_judge` |
| `sparse_ca` | `local_transition_search`, `d4_bgpad_search` |
| `difflogic_hard` | `hard_circuit_search` |
| `masked_diffusion` | `global_sample`, `masked_inpaint` |
| `residual_repair` | `global_color_map`, `local_transition` |

A proposal action can carry a residual parent and evidence signal IDs across
representations.  A context-aware provider receives the full oracle-free
blackboard and this typed action through `act(task, features, decision,
blackboard, action)`.  Existing `propose(...)` providers are adapted without an
API break, but they cannot consume residual context.

The built-in DSL and sparse-CA providers now implement `act(...)` directly.
`shape_resynthesize` searches demo-inferred shape-compatible programs;
`suffix_resynthesize` performs bounded append/tail-replacement edits from a DSL
parent (or a bounded cross-representation one-step search); and sparse-CA
operators select disjoint local versus D4/bgpad policy families.  Every
residual-conditioned candidate records its parent and operator.  The external
code, diffusion, and hard-circuit callbacks remain legacy proposal boundaries
until an integration implements the same contract.

The current compiler implements these deployable mappings:

| Observed evidence | Preferred legal actions |
|---|---|
| execution failure or shape mismatch | DSL shape resynthesis, then code exception resynthesis/global sampling |
| unambiguous same-shape color confusion | composed global-color repair |
| localized same-shape residual | local repair, masked inpainting, sparse CA, hard circuit, DSL suffix resynthesis |
| low CA/query support | switch to DSL/code/masked candidate sources |
| demo-exact grid without replay certificate | compile into DSL/code/DiffLogic executable form |
| two verified, query-distinct hypotheses | explicit `STOP(pass_at_k_filled)` |
| no untried legal action or insufficient budget | explicit budget/no-action STOP |

Global color repair is offered only when the parent is replayable, artifact-
verified, non-exact, same-shape, above the agreement threshold, and the complete
demonstration mapping is unambiguous.  Local repair additionally requires at
least two demonstrations and a bounded mismatch fraction.  The repair itself
still reconstructs and verifies its serialized table, so the compiler is a
legality mask rather than a correctness oracle.

## Strict budget semantics

Wall time is noisy and existing providers expose incomparable native work
(DSL expansions, CA fitting/search, external model calls).  The controller
therefore uses an explicit normalized compute unit (NCU) ledger with five
dimensions:

- normalized compute units;
- controller steps;
- provider calls;
- repair attempts;
- candidate slots.

A proposal with `n` candidate slots reserves `1+n` NCU, one step, one provider
call, and `n` slots.  A repair reserves 2 NCU, one step, one repair attempt, and
one slot.  STOP is free.  The entire reservation is charged even when a source
abstains or returns fewer candidates.  Consequently two policies cannot recycle
unused batch capacity differently.  The compiler emits an action only if every
dimension fits the remaining ledger, and a strict provider that returns more
than its slots is rejected as `budget_contract_violation`.

This is a strict controller-level budget, not a claim that one DSL search and one
GPU sample have equal FLOPs.  Reports distinguish:

- `strict_replayable`: a frozen pool or provider explicitly guarantees the
  action-level budget contract;
- `controller_only`: a legacy provider is centrally capped, but its internal
  computation is not matched.

Only runs for which every invoked candidate provider is `strict_replayable` set
`strict_budget_comparable=true`.  Accuracy/efficiency claims should initially
use frozen candidate pools; native compute comparisons require provider-specific
adapters that bind and report their own search/FLOP limits.

## Frozen heterogeneous pools

`FrozenCandidatePoolProvider` is the controlled experiment boundary.  Its pool
ID hashes the provider identity, route, canonical candidate payloads, and every
`FrozenActionBatch`.  A batch has the hard key `(operator,
parent_hypothesis_id)`.  It deterministically reveals unseen candidates up to
the action's slot limit and never mutates an internal cursor.  An action with no
exact batch match abstains as `frozen_action_unavailable`; it cannot fall back to
candidates generated for another operator or parent.  This prevents a typed
action label from becoming a post-hoc metadata preference.

The online merge evaluates only the newly emitted batch.  Repeated IDs must have
identical canonical payload, demo/query replay, verifier result, and rejection
status.  Any conflict quarantines every copy of that ID.  Quarantine propagates
to repair descendants so selected lineage remains parent-first and ID-closed.

## Metrics without leakage

The runtime report contains only oracle-free quantities: candidate-slot
utilization, eligible repair count per NCU, distinct selected query bundles, and
the complete action/state trajectory.  It never labels a query output correct.

`evaluate_online_report_with_oracle(report, task, pool_candidates=...)` is a
separate post-hoc step.  It first reconstructs the blind task, checks its content
hash against the frozen report, and verifies that every observed candidate is a
canonical member of the declared pool.  It then measures three distinct layers:

- full-pool raw and selectable oracle coverage;
- observed raw and selectable oracle coverage;
- final pass at k (normally k=2);
- exploration recall = observed selectable coverage / pool selectable coverage;
- selection utilization = pass / observed selectable coverage;
- total pool utilization = pass / pool selectable coverage;
- correct repair candidates and uniquely recovered tasks per NCU.

Omitting `pool_candidates` retains an explicitly labelled
`observed_only_fallback`; such a report has `strict_pool_metrics=false` and
cannot support a pool-utilization claim.  This separation makes it testable that
changing hidden test outputs cannot change controller states, actions, or
selections, while also preventing observed coverage from being mislabeled as
pool oracle coverage.

## Minimal API

```python
from afts_arc.hybrid import (
    BudgetVector,
    FrozenCandidatePoolProvider,
    OnlineControlConfig,
    OnlineFunctionalRouterSolver,
)

solver = OnlineFunctionalRouterSolver(
    providers=(frozen_dsl, frozen_ca, frozen_code, frozen_diffusion),
    config=OnlineControlConfig(
        budget_limit=BudgetVector(
            compute_units=64,
            controller_steps=16,
            provider_calls=8,
            repair_attempts=8,
            candidate_slots=48,
        ),
        provider_batch_size=4,
        max_selected_hypotheses=2,
    ),
)
report = solver.solve(blind_task)
```

The ordinary CLI now selects this controller by default.  Use
`--controller static` for the preserved v1 orchestration baseline.  CLI neural,
code, and DiffLogic sources remain explicit inference callbacks; no training is
started or resumed by this implementation.

## Evidence status and next experiment

Unit tests establish budget enforcement, typed cross-representation compilation,
repair-before-restart behavior, explicit STOP, deterministic content IDs,
strict action-batch access, pool/observed coverage separation, parent-conditioned
DSL resynthesis, strict-provider overrun rejection, and hidden-oracle invariance.  A synthetic
matched-budget fixture uses exactly 4 NCU and two candidate slots for both
policies: residual-first selects a successful color repair, while fixed run-all
spends the second action on a distractor and abstains.

`scripts/afts_arc_online_matched_budget.py` performs the next evidence step.  It
discovers typed DSL/CA actions without oracle access, freezes their action
mapping, replays residual-first, two fixed orders, round-robin, static routing,
deterministic random, and single-source baselines, and only then evaluates public
oracles.  The full frozen pool supplies the oracle-scheduler upper bound.  NCU is
strictly replay-matched; provider-native cost vectors are reported descriptively
and must not be conflated with matched FLOPs.
