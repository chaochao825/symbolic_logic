# Control-legend v0.1 development artifacts

This directory contains the outcome-exposed representation-reachability gate
for `afts-control-legend-dsl/v0.1`.

- `semantic_tests.xml`: receipt for the eight preregistered semantic fixtures;
- `candidate_freeze.json`: query-blind, replay-identical candidates for all 50
  frozen tasks, including native execution counts and v0.2 output novelty;
- `result.json`: separate post-freeze public-training query scoring;
- `*.stdout.log`: complete command outputs;
- `*.stderr.log`: retained empty error streams.

The bounded outcome is 2/50 standalone pass@2 and 2/50 unique pass@2 over the
frozen legacy plus relational-delta v0.2 system, giving a combined selectable
union of 4/50 on this exposed cohort.  Both recoveries are the two tasks used
to specify the family; no held-out generalization, typed repair, visual
proposal, controller, or causal switching claim is supported.
