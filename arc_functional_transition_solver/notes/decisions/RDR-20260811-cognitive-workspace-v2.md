# RDR-20260811: revise the representation protocol around a cognitive workspace

## Status

Accepted by the user on 2026-08-11.  This decision supersedes only the proposed
next experiment after Object–Program Workspace v1.  It does not alter historical
algorithms, candidate freezes, checkpoints, result files, or claims.

## Decision

Keep the long-term goal of brain-inspired functional specialization and
switching, but test it through a smaller computational claim:

> A sparse, typed, content-addressed workspace can preserve goals, execution
> episodes, failure certificates, and reusable procedures; a budgeted
> metacontrol rule can use that state to choose a legal cross-representation
> option whose execution changes the candidate frontier.

The first open representation lane is an object-relation graph rewrite with a
typed second-stage AST hole.  Learned routing remains frozen.

## Why this revision is necessary

Object–Program Workspace v1 established a causal actuator on controlled tasks,
but produced zero new query-output frontiers on the 100-task natural lane.  Its
selector-only recolor/erase language could not express the dominant natural
failures.  Increasing router complexity cannot repair an empty legal frontier.

The thought experiment sharpens the architecture boundary.  If strong visual,
language, code, and symbolic capabilities are already callable, the missing
high-level function is not another perceptual provider.  It is the recurrent
organization of:

1. a limited shared task state and goal stack;
2. fast episodic storage of observations, executions, and failures;
3. slower admission of verified reusable schemas;
4. counterfactual rollout through typed capability options;
5. cost-aware gating, stopping, and credit assignment; and
6. verification that updates the shared state.

This is a functional analogy, not evidence for anatomical or biological
equivalence.

## Frozen boundaries

- Existing providers, routers, numerical behavior, checkpoints, and artifacts
  are immutable baselines.
- Query outputs are unavailable to candidate generation, memory retrieval,
  option ranking, and candidate freezing.
- Outcome-exposed records are diagnostic-only and cannot enter prospective
  memory retrieval.
- A program ID change is insufficient.  An action is frontier-changing only if
  it creates at least one new query-output content ID.
- Native provider costs remain a vector; no undeclared scalar conversion is
  allowed.
- No learned controller is trained until residual interventions predictably
  change legal actions and the prospective frontier gate passes.

## Alternatives rejected for this gate

- a larger GRU, MLP, bandit, or diffusion controller;
- unrestricted two-stage program Cartesian products;
- direct pixelwise posterior synthesis;
- cross-task retrieval from outcome-exposed ARC-TGI solutions or witnesses;
- hand-written rules for individual reserve families.

## Revisit condition

Revisit the controller freeze only after the preregistered v2 protocol records
both a non-empty prospective output frontier and a selective certificate/action
intervention under matched native cost.
