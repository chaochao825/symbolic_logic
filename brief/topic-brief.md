# Topic brief

## Question

Can logic-gate circuits serve as a useful, efficient component of symbolic AI?

## Bounded thesis

This project tests a deliberately narrow claim: a gate circuit is a plausible **learnable and compilable local predicate layer** for neuro-symbolic systems, especially for finite Boolean composition, local constraint checks, and repeated bit-level evaluation. It is not evaluated as a replacement for a general theorem prover, Datalog engine, planner, or perception encoder.

## Audience and deliverable

The deliverable is a reproducible Python experiment suite and a Chinese evidence report. It uses synthetic, fully enumerable Boolean predicates and layered reachability graphs so that both successes and failure modes can be checked exactly.

## Reference-document interpretation

The two supplied documents agree on the architecture `encoder -> named predicates -> LGN/Boolean DAG -> solver/planner/verifier`. They identify continuous relaxation, dense relation enumeration, hard constraint verification, grounding noise, training/hardening, and recursion as the central trade-offs. This project makes those claims falsifiable with local experiments.

## Scope limits

- No claim of end-to-end visual or language grounding.
- No hardware PPA claim; CPU timing is a prototype measurement only.
- No claim that an arbitrary Boolean network is human-interpretable.
- No claim that a fixed-depth circuit replaces unbounded recursive search.
