# Visual relational-transducer confirmation decision — 2026-08-11

This decision table is frozen before the 100-task VARC candidate pool, bridge
candidates, and task-level confirmation outcomes are opened. It prevents a
failed gate from being reinterpreted as success by adding operators or changing
the allocation after scoring.

The confirmation endpoint is mechanism-track unique recovery beyond the raw
VARC/NVARC union. It is not an end-to-end pass@2 endpoint.

| Observed result | Diagnosis | Authorized next action |
|---|---|---|
| static-family unique recovery `< 5/100` | representation failure: the 24-program grammar does not contain enough useful confirmatory outputs | stop this family; do not tune the visual allocation or train a controller |
| static family `>= 5/100`, visual `< 5/100` | allocation/certificate failure: useful programs exist but the posterior score does not place budget on them | analyze posterior invariants and typed target localization; do not add grammar operators on the confirmation cohort |
| visual `>= 5/100` but visual `<=` cold | no matched-cost control advantage | reject the visual allocation claim; keep only static provider coverage if independently useful |
| posterior-cleared allocation set is unchanged | the visual posterior has no causal action effect under this score | reject posterior control regardless of recovery count |
| visual `>= 5/100`, visual `>` cold, and posterior clearing changes the set | bridge gate passes | open lesion, injected-certificate, and cost-intervention experiments before any learned router |

Any failure blocks XGBoost, bandit, GRU, diffusion-router, and brain-inspired
switching claims. A failure is split into implementation versus method causes
as follows:

- malformed grids, provider/task mismatch, unequal charged cost, replay
  mismatch, invalid content IDs, or candidates that are not demonstration
  exact are implementation/protocol failures and invalidate the run;
- a valid, replayable run with insufficient static-family recovery is a
  representation-language negative result;
- a valid run with static recovery but insufficient visual recovery is a
  certificate/allocation negative result;
- a valid run in which visual does not beat cold is a matched-cost mechanism
  negative result.

No task-specific relation, threshold, allocation fraction, candidate cap, or
seed may be changed after confirmation scoring. A later representation must be
motivated by aggregate failure certificates or a fresh generator-known
intervention cohort, not by writing rules for named confirmation tasks.
