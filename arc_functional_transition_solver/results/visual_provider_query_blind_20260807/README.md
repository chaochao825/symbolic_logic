# Query-blind visual-provider evidence package

This directory contains the immutable evidence for the 2026-08-07 visual
provider experiment.  See
`../../notes/results/visual-provider-query-blind-20260807.md` for the scientific
interpretation.

## Directory layout

- `aggregate/`: content-verified two-cohort aggregate.
- `v2_1/`: 20-task pilot cohort, provider contract, pre-gold and comparison
  freezes, raw validation, static baseline, object/code comparison, visual
  score, posterior-localization audit, and the one licensed pixel-consensus
  bridge.
- `v3_confirm/`: untouched 11-task confirmation with the same frozen provider
  semantics and fixed posterior-localization audit.
- `raw/`: compact transferred provider outputs and run receipts.  These retain
  the original task-level prediction file hashes.
- `provenance/`: experiment-specific scripts that generated the v3 pre-gold and
  comparison freeze receipts.
- `SHA256SUMS`: hashes every published artifact except `SHA256SUMS` itself.

## Invalidated runs

No score from v1 or strict v2 is included.  The v1 cohort overlapped a prior
development split and was invalidated before gold access.  Strict v2 rejected
the provider interface because raw VARC samples included non-ARC colors and one
empty grid.  Before gold access, v2.1 fixed a syntax-only rule that rejects an
invalid sample without cropping, mapping, coercing, or otherwise repairing it.
The same rule was reused unchanged in v3.

## Reproduction boundary

The external VARC checkpoint is not redistributed.  Its SHA-256, source commit,
executed-file hashes, runtime, configuration, and self-described training
metadata are recorded in each provider contract.  The released VARC loader was
run with an input-copy sentinel in every query-output field to prevent query
label access.  This is not an exact reproduction of the paper's released
evaluation script.
