# Visual relational-transducer allocation ablation — 2026-08-11

The frozen gate requires the visual certificate to have a causal effect on the
selected task set. Candidate freezing and scoring already compare the visual
allocation with a seeded equal-cost cold allocation, but the first
implementation did not serialize the direct certificate-clearing audit.

This pre-confirmation clarification adds a query-gold-free, content-addressed
audit without changing the representation family, visual ranking, selected
task count, cold seed, candidates, native costs, or any existing score.

The primary intervention clears only the visual-posterior preservation ratio.
It retains the pre-existing static pool descriptors in their frozen order:

1. total unique VARC plus NVARC candidate count, descending;
2. exact pool overlap, ascending;
3. task ID, ascending.

The selected task set must differ from the unablated visual allocation. This
tests whether the visual posterior itself, rather than only candidate-pool
size or task identity, affects the action frontier.

A secondary, stricter intervention clears every certificate feature and
selects by task ID only. It is reported for diagnosis but is not substituted
for the primary posterior intervention.

Both interventions operate only on the already frozen certificates and task
IDs. They do not read solutions or task outcomes. The audit must be frozen
before confirmatory scoring and records its source candidate-freeze ID and
this protocol's SHA-256.
