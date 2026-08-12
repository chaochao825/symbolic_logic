# EXP-000 heterogeneous hypothesis population audit

This directory publishes the aggregate result of a post-hoc normalization
audit over two independently frozen providers on the exposed 100-task ARC-TGI
confirmatory cohort.  Complete task-level population and score artifacts remain
in the 210 content-addressed artifact store; `summary.json` commits to them.

## Result

| endpoint | strict tasks | queries |
|---|---:|---:|
| NVARC/TRM raw/top-10 | 92/100 | 105/113 |
| VARC raw | 89/100 | 101/113 |
| provider union | 95/100 | 108/113 |

The strict decomposition is:

- 87 tasks solved by both complete providers;
- 5 tasks solved only by NVARC/TRM;
- 2 tasks solved only by VARC;
- 1 two-query task solved only by composing one NVARC query answer with one
  VARC query answer;
- 5 tasks solved by neither population.

This corrects the earlier arithmetic inference that the 95-task union implied
6 recursive-exclusive and 3 visual-exclusive tasks.  The union value was
correct, but one task is a cross-provider query composition.  Consequently the
predeclared `visual-exclusive >= 3/100` threshold is not met on this exposed
cohort.  The outcome is a valid boundary result, not a provider-complement
confirmation.

## Replay and leakage boundary

- Candidate population construction reads no solutions and retains
  `query_gold_read=false` and `controller_training_started=false`.
- Provider-independent identities merge 5,183 memberships into 5,019 unique
  hypotheses: 437 recursive memberships, 4,746 visual memberships, and 164
  exact cross-provider overlaps.
- Original and replay provider freezes produce identical population, full
  result, and compact summary bytes.
- No model was trained or rerun for this audit.

## Scope

The result is post-hoc development evidence on synthetic ARC-TGI tasks.  It
supports continuing a bounded query-blind recruitment diagnostic, but it is not
an ARC-AGI score and cannot confirm the primary claim prospectively.
