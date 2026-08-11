# VARC ARC-TGI query-blind anchor v2

## Supersession and failure classification

This document supersedes only the task-path compatibility clause of
`varc-arc-tgi-query-blind-anchor-v1.md`. Development run v2 completed all 101
epochs and ten inference attempts for its first task, wrote predictions, and
then failed in the upstream diagnostic. VARC interprets every underscore as
the start of a geometric augmentation suffix, so an ARC-TGI identifier such
as `arc_tgi_...` was reduced to `arc`; the diagnostic then attempted to open
`arc.json`. This is an implementation failure after candidate generation, not
a negative model or bridge result. The partial prediction is ineligible.

## Content-preserving runtime aliases

Before any complete development outcome is inspected, v3 introduces a
deterministic path-only alias for each task (`t0000`, `t0001`, ...). The
runtime overlay:

- symlinks each query-blind evaluation JSON under its alias;
- symlinks every augmentation while preserving the complete transform/color
  suffix and lexical variant order;
- keeps the original task ID in status, canonical prediction filenames, and
  all project artifacts;
- records a content-addressed one-to-one alias manifest;
- leaves the checkpoint, model source, data bytes, seed, training order,
  101-epoch behavior, ten inference attempts, and numerical hyperparameters
  unchanged.

The first task must be replayed through the alias overlay and its canonical
prediction bytes must equal the v2 partial prediction before a full v3 run is
admissible. If they differ, the alias is not numerically neutral and the v3
cohort run must not start.

All query-label, candidate-freeze, development/confirmation, and no-router
boundaries from v1 remain in force.
