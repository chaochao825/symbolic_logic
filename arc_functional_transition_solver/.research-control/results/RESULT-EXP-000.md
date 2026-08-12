# RESULT-EXP-000

- Experiment: EXP-000
- Date: 2026-08-12
- Protocol identity: `EXP-000-frozen-v1`; protocol SHA-256 `e3105a3141f05d67f8c953d40a81ed039c4f82fcf0d946e21aa396c444bf5c07`
- Code identity: base `0014236d1ca2e99626fd20877e0ac7fcf75f892b`; core SHA-256 `b3a15914654fd2aec07acb78f74c837c0d1b3ac2021996f16163fd95356ab214`; CLI SHA-256 `6330c9ba111e61f45f70cdd9470c981509b76789ff6ea70982ff22af601ab08d`
- Data identity: cohort `b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27`; solutions SHA-256 `0f1f6ee910bd222dd85950cc0d7305bd5439b3594b3a38330070669d780053b5`
- Configuration identity: `EXP-000-frozen-v1`, all raw NVARC top-10 and VARC posterior candidates
- Evaluator identity: result ID `118b1e153399383d187c45e033a5b9221588a14c2ead0075715d40947b30047b`
- Artifact identity/checksum: population ID `24dc8f299d5d4e263fba22ccd769150002f596d8f1689c3f6500f6efdcc170f4`; summary ID `8e638864e76ce8f27ace237ea8d0974a63305bc2510d3f775e8b9ad5b4897c54`
- Evidence tier: post-hoc audit
- Independent unit: strict whole ARC-TGI task; query counts secondary
- Validity: valid
- Outcome class: boundary
- Protocol deviations: none

## Observations

- Recursive provider: 92/100 strict tasks and 105/113 queries.
- Visual provider: 89/100 strict tasks and 101/113 queries.
- Provider union: 95/100 strict tasks and 108/113 queries.
- Strict exclusive coverage: recursive 5/100; visual 2/100.
- Cross-provider composed strict coverage: 1/100.  On that two-query task,
  query 0 is available only from recursive candidates and query 1 only from
  visual candidates.
- Population: 5,019 unique task/query hypotheses from 5,183 provider
  memberships, including 164 exact cross-provider overlaps.
- A and B replays are byte-identical for population, result, and summary.
- Targeted and related regression tests pass 27/27; the repository-wide suite
  exceeded the 180-second command cap without reporting a test failure.

## Validity checks

Both upstream freezes pass their canonical validators.  Task sets, query
counts, and query-input hashes agree.  The freeze command accepts no solution
path; scoring occurs in a separate command after the population artifact
exists.  Both provider and output replays are byte-identical.  The audit is
valid for evidence normalization, but outcome exposure prevents prospective
claim confirmation.

## Claim update

- `C-001`: partial support remains.  The two provider families are genuinely
  complementary, but the added provider contributes only 2 complete exclusive
  tasks, below the prospective threshold of 3.  One further union task requires
  query-level cross-provider composition.
- `C-000`: inconclusive.  No recruitment policy or final pass@2 selector was
  evaluated.

## Gate recommendation

Close `G-000` as a valid boundary audit.  Permit one CPU-only, explicitly
post-hoc recruitability diagnostic using only cheap recursive candidate-state
features to freeze the visual-provider priority order.  Do not train a router
and do not treat the exposed cohort as confirmation.  A new GPU run is allowed
only after that diagnostic demonstrates enough signal to justify a fresh
family-disjoint cohort.

## Artifacts

- Repository: `results/hypothesis_population_exp000_20260812/summary.json`
- External A/B population and full result:
  `/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp000_v1/`
- Source and upstream checksums:
  `results/hypothesis_population_exp000_20260812/sha256.txt`
