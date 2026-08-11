# Cognitive Workspace + Object-Graph Rewrite v2 preregistration

## Single research question

Can a query-blind failure trace be stored in a typed shared workspace and
compiled into a bounded `fill_ast_hole` object-program composition that creates
new query-output candidates beyond the frozen one-stage object/program baseline?

This gate tests one representation bridge and one deterministic metacontrol
baseline.  It does not test learned routing, visual-posterior utility, ARC-AGI
leaderboard performance, or a biological brain model.

## Candidate and baseline

- **Candidate:** a replayable two-stage object-code program.  Stage 1 is an
  execution-valid near miss selected only from demonstration residuals.  Its
  node trace creates the typed hole `stages[1]`.  Stage 2 is synthesized on the
  intermediate demonstration grids and must close the hole exactly on every
  demonstration.
- **Frozen baseline:** all demo-exact one-stage candidates reached under the
  same first-stage grammar prefix, plus the complete Object–Program Workspace
  v1 candidate outputs.  Historical providers and results are unchanged.
- **Cold control:** the same number of second-stage program trials, ordered by a
  frozen content hash and not conditioned on the parent residual.
- **Residual controls:** clear or change the certificate type while keeping the
  option registry and native reservations fixed.

No query label participates in parent selection, residual compilation, program
enumeration, ranking, memory retrieval, or output-novelty calculation.

## Typed workspace state

The immutable state contains:

- an ordered goal stack;
- content-addressed hypotheses and failure certificates;
- registered capability options and a provider-native budget ledger;
- task-local episodic records;
- family-disjoint, demonstration-validated reusable schemas; and
- an append-only event trace sufficient for replay.

Memory scopes are enforced at retrieval time:

| Scope | Same task | New family confirmatory task |
|---|---:|---:|
| task-local demonstration/trace | yes | no |
| family-disjoint validated schema | yes | yes, only from another family |
| outcome-exposed diagnostic | diagnostic lane only | no |
| sealed query oracle | evaluator only | no |

The deterministic controller ranks only options matching the active
certificate and input representation.  It uses declared expected frontier
gain and native reservations; it does not learn from query outcomes.

## Search and cost contract

- Maximum retained parent near misses per task: **4**.
- First-stage grammar prefix: **2,000** program trials.
- Maximum second-stage trials per parent and arm: **512**.
- Maximum emitted demo-exact compositions per task: **32** before query-output
  deduplication.
- Candidate and cold arms reserve identical program trials, demonstration
  executions, and query executions.  Unused work is explicit padding.
- A parent must be execution-valid on every demonstration and query input.
- Identity is included as a baseline, not counted as a repair.
- A candidate is selectable only after exact demonstration replay, AST
  round-trip, content-address validation, and zero unfilled holes.

## Data lanes and oracle boundary

### Lane A — controlled semantic tests

Small generated examples isolate goal-stack behavior, memory scoping, typed
option applicability, budget refusal, certificate intervention, AST replay, and
the output-novelty guard.  They provide implementation evidence only.

### Lane B — outcome-exposed development opportunity screen

Use the frozen 50-family ARC-TGI development challenges.  Only challenge files
are read by the candidate builder.  This lane may guide bug fixes and failure
classification, but no score is a prospective generalization claim.

Pass condition: at least **5/50** tasks contain a demo-exact two-stage query
output not present in the frozen baseline.  If this fails, Lane C is not run and
the result is a representation null.

### Lane C — prospective reserve confirmation

The existing family-disjoint 18-family ARC-TGI reserve is materialized in two
phases.  Phase 1 writes only blind challenges and committed hashes of the sealed
oracle/witness.  Candidate outputs are frozen twice and must be byte-identical.

The oracle may be regenerated and opened only if at least **2/18** reserve tasks
have a demo-only novel-output opportunity.  Otherwise no reserve query score is
computed.

If opened, the minimum useful-recovery signal is **2/18** unique candidate
recoveries and at least one more recovery than the equal-cost cold arm.  This
small reserve is a gate, not a leaderboard estimate.

## Mechanism criteria

The mechanism claim requires all of the following:

1. certificate injection changes the applicable or selected option in the
   predicted direction;
2. removing the stage-1-to-stage-2 bridge selectively removes compositions but
   not one-stage candidates;
3. reducing an option budget makes it inapplicable without changing other
   providers' outputs;
4. every claimed transition has `novel_frontier_count > 0`; and
5. replayed state, programs, actions, costs, and candidate outputs are identical.

These tests may establish brain-inspired functional switching at a
computational level.  They cannot establish biological brain-region identity.

## Null and adverse outcomes

- **Representation null:** Lane B has fewer than 5/50 opportunities.
- **Prospective frontier null:** Lane C has fewer than 2/18 opportunities.
- **Selection null:** prospective frontier exists but typed and cold arms have
  equal recovery.
- **Mechanism null:** residual clear/change does not affect legal option choice.
- **Implementation failure:** replay, hashing, leakage, typing, reservation, or
  AST round-trip checks fail.  Such a run is invalid, not a negative algorithm
  result.
- **Adverse result:** accuracy rises only through extra native work or use of an
  outcome-exposed memory.  It must be reported as unfair/leaky, not progress.

## Claim ledger before execution

Supported by previous work: deterministic heterogeneous replay, provenance,
native-cost accounting, typed one-slot edits, and a controlled causal actuator.

Unknown: whether trace-conditioned program composition creates a natural output
frontier; whether episodic/schema memory improves selection; whether the
prospective reserve contains enough opportunities.

Prohibited before evidence: competitive solver, human-level planning,
biological brain mechanism, or learned dynamic switching claims.
