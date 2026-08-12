# RESULT-EXP-004: Prospective functional recruitment replication

- Status: completed
- Outcome: provider-null; recruitment descriptive-positive
- Date: 2026-08-13
- Claims: C-001 and C-002 remain partially supported, not confirmed
- External artifact root: `/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1`

## Integrity result

- The collision-free blind cohort contains 98 tasks, 113 queries, and 18
  explicitly sealed source families.
- Both NVARC boundary audits were byte-identical and reported no label,
  query-input, collision, or overflow violation.
- All 98 VARC tasks completed with exit code zero in three isolated single-GPU
  shards.  Their merged native receipt records 20,112 GPU seconds and no task
  failure.
- Recursive candidates, the 30-task policy prefix, 256 random controls, visual
  shard assignment, and visual runtime were frozen before query gold existed.
- Candidate, population, authorization, solution, and score A/B replays are
  byte-identical.
- The first closure attempt stopped before oracle authorization because the
  VARC environment lacked ARC-TGI's declared `shortuuid` dependency.  Commit
  `634dbab85086f94a34b4ae0b8d4b7111c8d71067` binds the non-semantic runtime
  correction to the cohort-generation environment.  No candidate, cost,
  policy, threshold, or score semantics changed.
- A self-referential first artifact manifest was preserved in experiment-local
  trash.  The corrected manifest excludes itself, verifies every listed file,
  and has SHA-256
  `2d2a7572b2839af2d1661a1065e127b410e435647b05b047bee5dc47866197cc`.

## Solver evidence

| endpoint | pass@1 | pass@2 | raw strict oracle |
|---|---:|---:|---:|
| recursive NVARC | 88/98 | 89/98 | 93/98 |
| visual VARC | 75/98 | 80/98 | 88/98 |
| heterogeneous union | — | — | 96/98 |

The union covers 109/113 queries.  At task level, 86 are solved by both
providers, seven only by recursive, two only by visual, and three by neither.
One additional two-query task is strict only after query-wise cross-provider
composition.  The population contains 6,514 unique raster hypotheses and 161
exact cross-provider overlaps.

VARC generated 57,630 raw samples, rejected four invalid grids, and retained
6,297 unique candidates.  The result therefore is not an empty-provider or
adapter-loss failure.

## Gate result

`G-002a` fails: the visual provider contributes 2/98 strict exclusive tasks,
below the preregistered minimum of three.  The valid outcome class is
**provider-null**.  Confirmatory `G-002b` is not applicable and `C-003` final
selection remains locked.

The family audit makes this stricter conclusion necessary.  All three
anchor-miss/union-hit tasks—the two visual-exclusive tasks and the composed
task—come from one source family.  Marginal family count is 1/18 and the
largest-family concentration is 3/3.  Treating the three replicas as three
independent successes would overstate generalization.

## Descriptive recruitment evidence

Because the population and policy were already frozen, recruitment was replayed
after the failed gate strictly as outcome-exposed diagnosis.

| selected tasks | recovered marginal tasks | visual GPU seconds | random median high | random maximum |
|---:|---:|---:|---:|---:|
| 10 | 3/3 | 1,910/20,112 | 0 | 2 |
| 20 | 3/3 | 3,676/20,112 | 0 | 3 |
| 30 | 3/3 | 6,076/20,112 | 1 | 3 |
| 49 | 3/3 | 10,004/20,112 | 1 | 3 |

The three marginal tasks have frozen priority ranks 6, 7, and 10.  At the
primary 30-task prefix, the policy recovers all three at 30.21% of full visual
cost.  Under the same 6,076-GPU-second budget, random orders select 26--33 tasks
and have median-high recovery one; the fixed policy is above the median but
ties the random maximum of three.

This replicates a cheap anchor-uncertainty recruitment signal, but only within
one complementary family.  It cannot rescue the failed provider gate and is
not evidence for broad dynamic switching.

## Interpretation and decision

The implementation succeeded; the main negative result is theoretical and
distributional.  NVARC and VARC are individually strong on this synthetic
ARC-TGI cohort, but their errors are too correlated for VARC to establish broad
functional specialization.  The prior retrospective policy effect was not
pure noise—the frozen ordering again concentrated all marginal value—but the
effective positive sample is one family.

Park this exact provider pair for confirmatory recruitment.  Keep the shared
population and cheap gating machinery, keep learned controllers frozen, and
require the next provider to add at least three exclusive sealed families.
Only after that gate passes should final pass@2 selection or a learned control
model be tested.  No ARC-AGI competitiveness or biological equivalence claim
is supported by this synthetic result.
