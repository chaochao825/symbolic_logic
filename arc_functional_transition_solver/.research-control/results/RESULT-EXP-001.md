# RESULT-EXP-001

- Experiment: EXP-001
- Date: 2026-08-12
- Protocol identity: `EXP-001-anchor-disagreement-v1`; SHA-256 `94187d7db2385f44b68220942c92da1b44a00558174b55807687cb66eb513fe2`
- Code identity: base `0014236d1ca2e99626fd20877e0ac7fcf75f892b`; core SHA-256 `8292df5d6748fe370fd2090f2cd13c33ecef8c0bbeabeaef16382efe98d269bd`; CLI SHA-256 `01177de657a159516ca2fdbb4fe59d395a99fda02e0934e3b52961a795c5806d`
- Data identity: exposed cohort `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`
- Configuration identity: integer lexicographic anchor disagreement; budgets 10/20/30/50; random seeds 0--255
- Evaluator identity: result ID `2840eef8b89128cb4ed681bd34532fde9405d17b654aba26df24993112fff0fb`
- Artifact identity/checksum: plan ID `359061aa07225af5932aac912d0666b4841b3741dd86e6856896ad936db7a45c`; summary ID `e6aa1b7038ead0d6c9639c161d184e893c3bee2561c573b4352eedd241773681`
- Evidence tier: post-hoc development diagnostic
- Independent unit: strict whole task
- Validity: valid
- Outcome class: valid-positive
- Protocol deviations: none

## Observations

- Full visual recruitment headroom over recursive anchor is 3 tasks: 92 to 95.
- The frozen policy recovers 2/3 targets in its first 10 tasks and no additional
  target through rank 50.
- At the primary 30-task prefix, coverage is 94/100; random recovery counts are
  0 for 76 seeds, 1 for 114, 2 for 57, and 3 for 9.  The policy recovery of 2
  exceeds the upper median 1.
- Actual selected VARC time at 30% is 5,663/19,503 GPU-seconds (29.04%).
- At 10%, random recovery counts are 0 for 181 seeds, 1 for 68, and 2 for 7;
  the policy's recovery is 2.
- Recovered targets have anchor ranks 3 and 4 and minimum top support 1/10.
  The missed target has rank 51 and minimum top support 9/10.
- A/B plan, result, and summary artifacts are byte-identical.

## Validity checks

The plan source hashes include only the anchor candidate freeze, anchor cost
receipt, and frozen protocol.  Its schema explicitly records that recruited
provider candidates and query gold were not read.  Population outcomes and the
VARC receipt are opened by a separate score command.  All 31 focused and
related regression tests pass, and static checks pass.

## Claim update

- `C-002`: partial development support.  Cheap anchor disagreement identifies
  two of three marginal-value tasks at 10%--30% activation and beats the frozen
  primary random-median control.
- `C-000`: still inconclusive.  The evidence is post-hoc, the positive count is
  two, no equal-cost restart candidate run exists, and no final pass@2 selector
  was tested.

## Gate recommendation

Close `G-001` as a valid-positive development result.  Freeze the same policy,
without new fields or weight changes, for a fresh family-disjoint cohort.  Do
not train a controller.  On the new cohort, evaluate complement first; only if
there are at least three marginal tasks should the 30% recruitment gate be
interpreted.

## Artifacts

- Repository: `results/functional_recruitment_exp001_20260812/summary.json`
- External A/B plan and full result:
  `/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp001_v1/`
- Checksums: `results/functional_recruitment_exp001_20260812/sha256.txt`
