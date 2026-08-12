# EXP-001 anchor-only functional recruitment diagnostic

This post-hoc development diagnostic freezes a task order from only NVARC/TRM
candidate disagreement, then asks how much independently frozen VARC complement
would have been recovered at fixed activation counts.  It does not train a
router and does not read visual candidates or solutions while constructing the
plan.

## Result

| visual activation | strict coverage | marginal recovery | random median | observed VARC seconds |
|---:|---:|---:|---:|---:|
| 10/100 | 94/100 | 2/3 | 0 | 1,991/19,503 (10.2%) |
| 20/100 | 94/100 | 2/3 | 1 | 3,725/19,503 (19.1%) |
| 30/100 | 94/100 | 2/3 | 1 | 5,663/19,503 (29.0%) |
| 50/100 | 94/100 | 2/3 | 2 | 9,333/19,503 (47.9%) |

The predeclared 30% development gate passes: recovery is 2, at least half of
the three marginal tasks, and strictly above the upper random median of 1.
Among 256 frozen random orders at 30%, 66 recovered at least two tasks.  The
secondary 10% prefix is more selective: only 7/256 random orders also recovered
two tasks.

Two recovered cases have anchor uncertainty ranks 3 and 4 and minimum query
top-support 1/10.  The unrecovered visual-exclusive task ranks 51 with minimum
top-support 9/10.  Thus disagreement detects two useful failures but cannot
detect a confident anchor error; expanding the same policy from 10% to 50%
adds cost without coverage.

## Interpretation boundary

This is the first positive signal for functional recruitment, but it rests on
only three marginal-value tasks in an outcome-exposed synthetic cohort.  It
supports freezing and testing the same policy on fresh tasks.  It does not yet
support learned routing, final pass@2 improvement, ARC-AGI competitiveness, or
a biological brain-mechanism claim.

## Replay

Original and replay upstream artifacts produce identical plan, full result,
and compact summary bytes.  `summary.json` commits to the external task-level
result without publishing task IDs or random-seed traces.
