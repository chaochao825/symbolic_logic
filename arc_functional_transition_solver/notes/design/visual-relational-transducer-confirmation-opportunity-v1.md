# Visual relational-transducer confirmation opportunity — 2026-08-11

This pre-gold interpretation amendment addresses an absolute-gate ceiling that
was not explicit in the original confirmation decision table. It does not
change the 100-task cohort, providers, candidate grammar, allocation, cold
seed, native costs, novelty rule, scorer, or the requested threshold of five
unique recoveries.

The frozen NVARC anchor already solves 92/100 confirmation tasks at strict
top-10. The VARC/NVARC union can only leave the same or fewer failures. Define

```text
available opportunities = 100 - base union strict coverage.
```

Unique bridge recovery is upper-bounded by this quantity. Therefore:

- if available opportunities are fewer than five, the absolute `>=5/100` gate
  remains failed, but the random-cohort experiment is ineligible to diagnose a
  representation-language failure;
- if at least five opportunities exist and the static family recovers fewer
  than five, the original representation-failure diagnosis applies;
- static-family recovery must also be reported as a fraction of available
  opportunities;
- visual recovery must be reported as a fraction of static-family recoveries,
  because only tasks with a correct grammar candidate are controllable by the
  allocation;
- none of these normalized diagnostics replaces the absolute gate or licenses
  controller training.

An opportunity-limited result means that a strong base anchor made a generic
100-task cohort unsuitable for a five-recovery mechanism test. The next valid
instrument would be a separately frozen, family-disjoint set of at least 100
natural or injected anchor near-misses with known typed repair opportunities.
That set is a mechanism benchmark, not an end-to-end ARC generalization score.

This amendment is frozen while blind VARC inference is still running and before
the bridge candidate freeze or any access to confirmation solutions.
