# Visual structure bridge v1 artifacts

This directory contains the immutable evidence for the 12-family ARC-GEN
confirmation run described in
`notes/results/visual-structure-bridge-20260809.md`.

## Result

- Visual raw oracle coverage: 8/12.
- Visual frequency pass@1/pass@2: 7/12 and 7/12.
- Frozen legacy portfolio raw/selectable coverage: 0/12 and 0/12.
- Hybrid structural pass@2: 7/12.
- Unique recovery/regression: 0/12 and 0/12.
- Decision: `null`; controller remains frozen.

This generated-family run supports complementary visual candidate generation,
but it does not support the current structural selector, typed repair, dynamic
switching, or a hidden ARC-AGI-2 accuracy claim.

## Evidence layout

- `cohort/`: content-addressed cohort manifest and compact blind/gold task
  package.
- `raw/`: compact query-blind VARC predictions, run receipts, and validation.
- `pre_gold/`: provider contract, first freeze, byte-identical replay, and the
  frozen structural candidates.
- `scores/`: frozen portfolio baseline, visual score, structural endpoint,
  post-outcome descriptive forensics, and gold-release receipts.
- `compatibility/`: byte-identical replay after changing `shape_delta` from a
  Python tuple to a JSON-native list. Metrics were not recomputed.
- `provenance/`: exact cohort, launch, validation, freeze, scoring, and analysis
  scripts plus their stdout receipts and pre-run source hashes.

`SHA256SUMS` hashes every published file except itself. The raw provider package
contains no gold path. `cohort/tasks.tar.gz` contains generated gold and is only
for endpoint replay after the pre-gold artifacts have been verified.

## Reproduction order

1. Verify `SHA256SUMS` and the cohort manifest.
2. Inspect `raw/validation.json`; it records `gold_accessed: false`, 12/12
   successful tasks, and zero sentinel mismatches.
3. Verify `pre_gold/freeze_receipt.json`, then compare the primary and replay
   frozen/candidate hashes.
4. Only then unpack `cohort/tasks.tar.gz` and replay the frozen baseline and
   score commands from `provenance/`.
5. Compare `scores/structure_score.json` and
   `scores/gold_release_receipt.json` by content ID and SHA-256.

The broader literature synthesis is in
`notes/literature/visual-posterior-to-typed-search-20260809.md`.
