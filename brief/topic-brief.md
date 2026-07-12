# Topic brief

## Question

Can logic-gate circuits serve as a useful, efficient component of symbolic AI?

## Bounded thesis

This project tests a deliberately narrow claim: a gate circuit is a plausible **learnable and compilable local predicate layer** for neuro-symbolic systems, especially for finite Boolean composition, local constraint checks, and repeated bit-level evaluation. The extension now includes a component-supervised pixel-to-predicate gridworld encoder, but the gate layer is not evaluated as a replacement for a general theorem prover, Datalog engine, planner, or perception encoder.

## Audience and deliverable

The deliverable is a reproducible Python experiment suite and a Chinese evidence report. It uses synthetic, fully enumerable Boolean predicates and layered reachability graphs so that both successes and failure modes can be checked exactly.

## Reference-document interpretation

The two supplied documents agree on the architecture `encoder -> named predicates -> LGN/Boolean DAG -> solver/planner/verifier`. They identify continuous relaxation, dense relation enumeration, hard constraint verification, grounding noise, training/hardening, and recursion as the central trade-offs. This project makes those claims falsifiable with local experiments.

## Scope limits

- Visual end-to-end evidence is limited to rendered gridworld pixels with component-level cell supervision; there is no claim for natural images or language grounding.
- No hardware PPA claim; CPU timing is a prototype measurement only.
- No claim that an arbitrary Boolean network is human-interpretable.
- No claim that a fixed-depth circuit replaces unbounded recursive search.
- The extension includes finite cyclic traversal and monotone planning/proof-frontier proxies to test state-transition loops, while keeping full solver and perception hardware evaluation out of scope.
