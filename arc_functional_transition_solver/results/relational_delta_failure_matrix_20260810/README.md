# Relational-delta v0.2 terminal failure matrix artifacts

This first artifact is retained for audit history and is superseded by
`../relational_delta_failure_matrix_20260810_v2/`.  A formatting-only change to
the diagnostic script changed its source hash; no task row, category count, or
CSV byte changed.  The visual review sheets remain here and are still valid for
the identical task matrix.

This directory contains the read-only, outcome-exposed audit of the 48 tasks
without a demonstration-exact `afts-relational-delta-dsl/v0.2` program.

- `matrix.json`: content-addressed complete terminal-stage matrix;
- `matrix.csv`: flat per-task audit view;
- `parse_failure_sheet_*.png`: demonstration-only visual review sheets for the
  largest machine category;
- `run.stdout.log`: byte-identical copy of `matrix.json` emitted by the run;
- `run.stderr.log`: retained empty error stream.

The audit uses demonstration inputs and outputs and blind query inputs.  It
does not read query outputs or ARC-AGI-2 public-evaluation data.  Its categories
describe where tasks terminate in one frozen candidate language; they are not
ground-truth ARC semantic families.

The bounded conclusion is `18 parse_failure / 14 relation_missing / 12
canvas_incompatible / 3 legal_but_inexact / 1 ast_insufficient`.  Because the
largest bucket is heterogeneous, the next opt-in representation probe is
limited to the reusable border-legend-to-payload mapping subcluster.  This
does not authorize router training.
