# RDR-20260812: test bounded multi-node counterfactual transitions

## Status

Accepted by the user on 2026-08-12. This opens one diagnostic successor to the
closed Stateful Object-Graph Rewrite v3 representation gate. It does not
modify v1-v3 algorithms, candidate freezes, scores, artifacts, or conclusions.

## Decision

Keep the v3 executable parent state and replace the single-node action with an
explicit, content-addressed transition over one to three declared AST regions:

```text
executable trace
-> persistent object/relation identities
-> node-level typed failure
-> bounded multi-node counterfactual
-> replay from the earliest changed node
-> exact full-replay verification
```

Evaluate three independently budgeted and preregistered search policies:

1. distance-tiered sampling, which does not use the diagnosed node to select
   programs;
2. typed semantic bundles selected by the v3 failure certificate; and
3. a two-phase residual beam that spends one quarter of its budget on
   single-node probes and extends the eight best demonstration residuals.

The controller, learned router, visual posterior, reserve oracle, and grammar
primitives remain frozen.

## Why this is the next bounded question

The valid v3 audit showed that a legal one-node rewrite improves 94/196 parents
but reaches no demonstration-exact program. Its compiler usually chooses a
different node from the best one, while even oracle node choice remains zero
exact. This separates a diagnosis defect from a joint-transition defect.

An exploratory multi-node run then failed on 47 tasks because the v3 executor
correctly rejected more than one declared program difference. That run is an
engineering failure and carries no scientific update. The v4 implementation
adds a separate multi-node API; the original single-node API remains strict.

## Frozen boundaries

- Regenerate exactly the same first 2,000 object/code trials, at most four
  parents, and the same certificates as the frozen v3 artifact.
- Reject the run if any regenerated certificate ID differs from v3.
- Use the same v3 scene grammar; no task-specific relation or new operator may
  be added after observing development outcomes.
- Each arm reserves 512 trials per parent and 2,048 trials per task. The frozen
  v3 cold restart is the single-arm cost control.
- A three-arm union consumes three times the single-arm transition reservation
  and cannot be compared with one cold arm as an efficiency result.
- Candidate construction may read demonstrations and query inputs, never query
  outputs. All three arms must be frozen before development solutions are read.
- A transition must declare every changed node in dependency order, cite a
  matching executable parent trace, and reuse only the prefix before the first
  changed node.
- Every emitted candidate must agree under suffix replay, fresh stateful replay,
  the historical scene executor, and serialized AST round-trip.
- Output content, not program identity, defines frontier novelty.

## Decision effect

A single arm must create novel query-output opportunities on at least 5/50
exposed development tasks before a fresh family-disjoint reserve may be
materialized. A union-only pass does not license the reserve because it uses
three transition reservations. If no arm passes, classify a bounded-transition
null and keep learned control frozen. If residual beam passes, residual
empty/shuffle/injection controls are required before any residual-causality
claim or router training.
