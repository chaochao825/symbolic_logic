# RDR-20260811: replace raster composition with stateful node rewrite

## Status

Accepted by the user on 2026-08-11. This decision opens one successor gate to
the closed Object-Graph Rewrite v2 representation null. It does not modify the
v2 protocol, algorithms, frozen candidates, scores, checkpoints, or claims.

## Decision

Replace the v2 bridge

```text
parent raster -> another complete program search
```

with an opt-in, content-addressed transition

```text
executable trace
-> persistent object/relation identities
-> node-level typed failure
-> one legal counterfactual AST rewrite
-> affected-subtree replay
-> exact full-replay verification
```

The active candidate is Stateful Object-Graph Rewrite v3. Learned routing,
cross-task outcome memory, unrestricted program products, and new perceptual
providers remain frozen.

## Evidence motivating the change

The valid v2 development run found one novel exact recovery in 50 tasks, below
its preregistered five-task opportunity gate. Forty-eight tasks had executable
parents but no exact raster-conditioned composition. The parent raster erased
which objects, relations, and AST nodes produced it, so the second program had
to rediscover the task from a lossy intermediate grid.

The v3 gate isolates that information-loss hypothesis. It does not claim that
the existing scene grammar is strong enough for competitive ARC performance.

## Frozen boundaries

- The exact v2 candidate freeze is the incumbent frontier and is read-only.
- Candidate construction may read demonstration inputs/outputs and query
  inputs, but never query outputs.
- One rewrite changes exactly one typed AST region: `parse`, `assignment`,
  `operate`, `canvas`, or `render`.
- A rewrite is legal only when it cites a parent trace and reuses all state
  strictly before the diagnosed node.
- Every candidate must agree across suffix replay, fresh stateful replay, the
  historical scene executor, and AST round-trip.
- Native trial reservations remain matched to an unconditioned cold restart.
- A new program ID is not frontier progress unless its query-output content ID
  is absent from the complete frozen v2 incumbent.
- No reserve scoring or controller training is authorized by implementation
  tests alone.

## Decision effect

If v3 reaches the preregistered development opportunity gate, materialize the
existing family-disjoint reserve under its two-phase oracle boundary. If it
does not, classify the result before explanation and keep the controller
frozen. A failure of replay, identity, leakage, or cost guards is invalid rather
than evidence against the representation hypothesis.
