# EXP-002 abstention-aware retrospective replication

This audit replays the frozen recursive NVARC/TRM and visual VARC providers on
the disjoint 50-task development split.  It extends the shared population only
enough to preserve an explicit provider abstention; all no-abstention v1
artifacts remain byte-identical.  The recruitment plan reads only the anchor
freeze and the manifest task list, never visual candidates or solutions.

## Population result

| endpoint | strict tasks | queries |
|---|---:|---:|
| recursive anchor | 45/50 | 55/60 |
| visual provider | 45/50 | 55/60 |
| population union | 47/50 | 57/60 |

Each provider contributes two strict-exclusive tasks and no task requires
cross-provider query composition.  The recursive provider abstains on one
task; that task remains in the denominator and is recorded as a miss rather
than silently removed.

## Recruitment result

| visual activation | strict coverage | marginal recovery | random upper median | observed VARC seconds |
|---:|---:|---:|---:|---:|
| 5/50 | 46/50 | 1/2 | 0 | 896/10,085 (8.9%) |
| 10/50 | 47/50 | 2/2 | 0 | 1,855/10,085 (18.4%) |
| 15/50 | 47/50 | 2/2 | 1 | 2,879/10,085 (28.5%) |
| 25/50 | 47/50 | 2/2 | 1 | 4,649/10,085 (46.1%) |

The frozen 30% gate passes.  One marginal task is the explicit anchor
abstention at priority rank 1.  The other is a genuine non-abstention
disagreement case at rank 9, with minimum top-output support 1/10 across its
two queries.  Therefore the recovery is not explained only by missing output.

## Interpretation boundary

This is a second positive retrospective split, not prospective confirmation:
there are only two marginal-value tasks and the historical provider outcomes
were already exposed.  It justifies a new blinded cohort with the unchanged
policy.  It does not justify training a router, claiming ARC-AGI performance,
or claiming a biological brain mechanism.

## Replay

Original and replay inputs produce byte-identical population, plan, full
results, and compact summaries.  `summary.json` and
`population_summary.json` omit task identities while committing to the full
external artifacts on server 210.
