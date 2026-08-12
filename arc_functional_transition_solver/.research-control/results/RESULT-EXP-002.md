# RESULT-EXP-002: Abstention-aware retrospective replication

- Status: completed
- Outcome: valid-positive retrospective evidence
- Date: 2026-08-12
- Claim: C-002 partially supported; prospective gate remains open
- External artifact root: `/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp002_v1`

## Integrity result

- The frozen development manifest contains 50 tasks.
- NVARC attempted 49 tasks and abstained on one; VARC attempted all 50.
- Population v2 preserves the union task universe and records the abstention as
  a provider miss.  It neither drops the task nor reads VARC to construct the
  recruitment plan.
- A/B population, plan, full score, and compact summaries are byte-identical.
- The no-abstention 100-task population still serializes as v1 and is
  byte-identical to `EXP-000` (SHA-256
  `6ebfd7c1eb4a4a56002990250e00342369d79bea16e081e976b58dc2d7ffe789`).
- New focused tests pass 10/10; the related provider regression set passes
  33/33; Ruff reports no violations.

## Population evidence

| endpoint | strict tasks | queries |
|---|---:|---:|
| recursive | 45/50 | 55/60 |
| visual | 45/50 | 55/60 |
| union | 47/50 | 57/60 |

There are 43 both-hit tasks, two recursive-exclusive tasks, two
visual-exclusive tasks, and three tasks solved by neither provider.  There is
no cross-provider composed strict task on this split.  Population v2 contains
3,300 unique raster hypotheses from 3,395 provider memberships, with 95 exact
cross-provider overlaps.

## Recruitment evidence

| activation | recovered marginal tasks | strict coverage | random upper median | observed VARC time |
|---:|---:|---:|---:|---:|
| 10% | 1/2 | 46/50 | 0 | 896/10,085 (8.88%) |
| 20% | 2/2 | 47/50 | 0 | 1,855/10,085 (18.39%) |
| 30% | 2/2 | 47/50 | 1 | 2,879/10,085 (28.55%) |
| 50% | 2/2 | 47/50 | 1 | 4,649/10,085 (46.10%) |

The unchanged 30% development gate passes.  The first target is the explicit
anchor abstention at priority rank 1.  The second is a non-abstention task at
rank 9 with minimum top-output support 1/10 across two queries.  Thus the
positive result is not wholly explained by trivial provider absence.

At 30%, the 256 deterministic random controls recover 0, 1, and 2 targets in
128, 99, and 29 orders respectively.  The fixed policy's recovery of two is
above the upper median but not above the random maximum; the sample contains
only two positive tasks.

## Artifact identities

- Population ID: `48c4e83b45fc958a2c9d436e47a7a60295bbc06a22ca352e3a3317b688c0a5c2`
- Population result ID: `4ec7819563f0705bf081d10687282888b2b889ef6a1d29b6e828db4566cd04ad`
- Plan ID: `323465f35a086a473701bb57036efa1a209f6e897fe0bcf7593b79230bb37c44`
- Recruitment result ID: `23761158c90b85479958de6a961141d760d7b36392584dd0976e002e04e2e1ca`
- Compact population summary ID: `954cb285d9be7dda80524c0a748b2cc989b8c7bd3be09d3a49bcbb12bbcf20e7`
- Compact recruitment summary ID: `3146e9c8d73b8f51e3e5e935755fcb5b6ed1d273f61f17c9fca2f82903a85095`

## Interpretation

The outcome strengthens the specific computational insight that cheap anchor
state can recruit a complementary expensive function.  It is still
retrospective, synthetic, and based on only two marginal tasks.  It does not
confirm final pass@2, equal-cost restart superiority, ARC-AGI generalization,
or a biological mechanism.

The abstention mismatch was an interface limitation, not an algorithmic
failure: the v1 shared population assumed every function emits on every task.
The v2 extension is the minimum honest representation of functional absence
and leaves all existing v1 numerical behavior intact.

## Decision

Retain `L-000` and keep controllers frozen.  Proceed only to a prospectively
sealed new cohort using the unchanged abstention-then-disagreement policy.
Require fresh added-provider strict complement before testing recruitment and
compare the fixed 30% policy with random and equal-native-cost controls.
