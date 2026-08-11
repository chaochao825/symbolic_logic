# Object–Program Workspace gate v1 — frozen 2026-08-11

## Accepted research direction and claim boundary

The long-term objective remains a brain-inspired functional-switching system for
ARC.  This gate tests only the missing causal middle of that claim:

```text
candidate execution failure
  -> typed object-correspondence certificate
  -> legal object_rematch action
  -> content-novel executable program frontier
  -> demonstration-exact candidate
  -> query recovery under a native-cost ledger
```

The active candidate is **Object–Program Workspace v1**.  Controller training,
GRU/MLP/diffusion routing, pixel voting, task-specific relations, canvas repair,
and biological claims are out of scope.  Existing providers, candidate IDs,
checkpoints, result files, and numerical behavior remain immutable; the new
provider is opt-in.

This gate was selected after the frozen 24-program visual transducer produced
zero novel candidates on 100 tasks.  That result was a representation-language
null: posterior interventions changed allocation, but the actuator could not
change any frontier.  The new gate therefore requires both an explicit
correspondence state and a positive novelty check before an action can be called
frontier-changing.

## Frozen v1 representation

For each input grid, the workspace retains several deterministic scene parses:

- modal/background hypotheses derived from visible inputs;
- 4- and 8-connected components;
- monochrome, foreground-connected, and color-group object groupings.

Each object slot records shape, size, topology, border contact, and its relations
to every other object.  A typed program has exactly four stages:

1. `parse_objects`;
2. `correspond_nary(selector)` over all object slots;
3. `recolor_component` or `erase_component` on complete selected objects;
4. same-canvas, conflict-free rendering.

The base prefix contains only coarse selectors (`all`, area extrema, and border
distance extrema).  The repair frontier contains relational selectors based on
shape/size/topology multiplicity and n-ary relation degree.  This split is fixed
by selector class and description length, never by task ID or query outcome.

An `object_rematch` child is legal only when it:

- preserves the parent parse, effect, canvas, and render semantics;
- changes exactly `ast.correspond.selector`;
- executes without partial-object writes;
- is exact on every demonstration before it is selectable; and
- has a program ID absent from the frozen base prefix.

Program novelty and candidate-frontier novelty are audited separately.  A new
program ID is only `program_frontier_changed`; the action may report
`frontier_changed` only when at least one demonstration-exact query output has a
content ID absent from the frozen base candidate pool.  This prevents a second
program that reproduces an existing grid from masquerading as a new candidate
region.

The executable trace records parse, relation-graph, correspondence, effect, and
render nodes plus the selected object IDs and complete write mask.

## Joint posterior bridge

Complete visual samples are never averaged into pixels.  Under a candidate
parse/effect, each whole-grid sample is either rejected as non-representable or
mapped to one complete object-assignment tuple.  Integer posterior mass is then
accumulated over assignment tuples:

```text
q(whole output grid)
  -> q(complete selected-object set | parse, effect)
  -> ranking over executable object_rematch programs
```

No independently estimated pixel or component marginal may synthesize a grid.
Every output is replayed from a serialized program.  The residual-only arm uses
the same typed frontier with all posterior masses cleared.  A shuffled arm uses
a deterministic cross-task permutation of frozen posterior views.  The cold arm
uses a deterministic hash order over the same content-novel complete grammar.

## Data lanes and leakage boundary

### Lane A — controlled semantic intervention

Deterministic fixtures inject a wrong correspondence selector while preserving a
valid parse and effect.  The gold child differs in exactly the typed selector
slot.  Whole-grid posterior samples and lesions are generated from declared
programs, not learned models.  This lane tests implementation controllability,
typing, causal intervention, and replay only; it is not evidence for visual
generalization.

### Lane B — outcome-exposed family-disjoint development

The already frozen 100-task ARC-TGI confirmation cohort and its frozen VARC
posterior may be scanned once with v1.  Because aggregate outcomes were exposed
by previous experiments, all results are development evidence.  Candidate
construction uses demonstrations, query inputs, and the already frozen posterior
only.  Query solutions are opened solely by a separate scorer after two
byte-identical candidate freezes.

No v1 operator, selector, ordering, budget, or threshold may change after Lane B
is scored.  A modification requires v2 and a new cohort.

### Lane C — future prospective confirmation

Lane C is not opened unless A passes and B shows enough demo-only opportunities.
It requires a new ARC-TGI/ARC-GEN family set disjoint from all prior authored and
generator families, frozen before query outcomes or new visual predictions are
inspected.

## Native-cost and action budget

Each repair arm may execute at most four content-novel full programs per task.
All arms reserve the same number of program trials, demonstration executions,
and query executions; missing trials replay the parent as padding and cannot
emit a candidate.  Shared certificate and posterior-to-assignment compilation
costs are reported separately as correspondence evaluations and are never hidden
inside a zero-cost residual token.

The base-prefix oracle, complete-grammar oracle, and typed/cold pass@2 are all
reported separately.  Full-grammar coverage cannot be presented as repair.

## Gates and outcome mapping

Lane A must satisfy all of the following:

- 100% JSON/content-ID/trace/topology/replay checks;
- all injected children change exactly the declared selector slot;
- `frontier_changed == (novel_frontier_count > 0)` for every arm/task;
- at least 80% typed recovery under the four-program budget;
- clearing or shuffling posterior mass changes the predicted proposal order on
  the fixtures designed to be posterior-disambiguated.

Lane B is a screening gate, not confirmation:

- at least five tasks must contain a demo-valid base near miss and a novel,
  demo-exact object-rematch child;
- all freeze, replay, leakage, and matched-reservation guards must pass;
- query recovery, visual-versus-lesion, and visual-versus-cold are reported but
  cannot support a general claim.

The unchanged prospective Lane C gate is:

- at least 20/100 tasks with a genuine novel typed frontier;
- at least 5/100 unique query recoveries over the frozen initial prefix;
- at least +3/100 recovery over equal-cost cold restart;
- clearing and shuffling posterior evidence reduce recovery by at least 3/100;
- zero demo-exactness, topology, replay, leakage, or cost violations.

Result classes are fixed before execution:

- **engineering/invalid**: schema, replay, trace, leakage, or cost guard fails;
- **representation null**: the complete grammar has too few demo-exact
  object-rematch children;
- **actuator null**: children exist but the typed action produces no novel
  frontier within budget;
- **sensor boundary**: typed repair works but posterior interventions do not
  improve proposal utility;
- **mechanism pass**: the prospective Lane C causal and recovery gates pass.

A null or adverse Lane B result ends v1.  It may motivate a separately frozen
representation family, but it does not authorize post-outcome selector tuning or
controller training.
