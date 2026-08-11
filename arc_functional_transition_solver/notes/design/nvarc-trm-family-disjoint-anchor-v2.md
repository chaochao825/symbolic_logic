# NVARC TRM family-disjoint anchor gate v2

## Supersession

This gate supersedes the accuracy comparison in
`nvarc-trm-reference-anchor-replication-v1.md`. The v1 software and compute
identity remain useful, but its proposed official-evaluation endpoint is not
clean held-out evidence: the released checkpoint's pretraining identifiers
overlap all 120 official evaluation tasks and all 50 old development tasks.

## Decision question

Can the frozen NVARC TRM checkpoint, with reference-scale test-time training,
produce a useful candidate anchor on post-checkpoint, generator-family-disjoint
ARC tasks without query-label-dependent candidate construction?

This is a candidate-generation gate. It does not train a router and it does not
test residual control.

## Frozen task and model contract

- cohort ID:
  `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`;
- primary endpoint: 100 confirmatory ARC-TGI ARC-Mini families, 113 queries;
- checkpoint: NVARC `step_220708`, SHA-256
  `dbf771737d9799b84faf009c24c5376831f095d2cfd66debbe9a505a445bfc31`;
- architecture: two L layers, H cycles 4, L cycles 4, halt maximum 10;
- test-time training: 2,000 epochs, global batch 128, EMA enabled;
- learning rate 1e-4 with 200 warmup steps;
- candidate endpoint: frozen top 10 per query;
- seed: zero as implemented upstream;
- upstream source edits: none.

The published example uses four GPUs. Server 210 currently has only GPU 0
available, so the run uses one process while preserving the global batch and
all numerical hyperparameters. It is reference-scale but not a bit-exact
four-device reproduction.

## Query-label boundary

The native compiler stores test labels because the upstream evaluator computes
loss and metrics. Candidate validity therefore requires all of the following:

1. executed-source hashes remain pinned;
2. the inner model's logit path consumes only inputs and puzzle identifiers;
3. evaluation mode forces ten inference steps rather than label-dependent
   halting;
4. evaluator labels affect metric comparison only, not candidate voting or
   ordering;
5. `submission.json` is content-addressed before independent solution scoring.

Any label-dependent update, halt, candidate filter, or rank invalidates the run.

## Task-level outcome boundary

The native submission is frozen before project-side solution scoring.  The
anchor gate then writes only aggregate rank-1/2/5/10 metrics plus a canonical
commitment to the complete task-level result.  It does not expose task IDs,
per-query hits, or solution hashes.  The committed full result may be opened
only after the visual bridge implementation and its confirmatory candidates
are frozen.  This prevents anchor qualification from becoming task-directed
bridge development while preserving a later exact consistency check.

## Decision rule

The run is operationally valid only if it completes all 6,288 planned training
steps, produces finite metrics and a complete top-10 submission, closes all
source/data/checkpoint hashes, and independently reconstructs the upstream
rank-1/2/5/10 metrics.

It qualifies as a **useful strong anchor on this cohort** only if strict top-10
task coverage is at least 10/100. This threshold is frozen before inspecting
the result. It is not derived from or compared numerically with NVARC's
contaminated official-task score.

If the run is valid but misses 10/100, record an adverse distribution-transfer
result and do not call it a strong anchor. The next permitted candidate anchor
is the already pinned query-blind VARC checkpoint under the same family split.
No router or repair experiment opens until one anchor passes its own frozen
coverage gate.
