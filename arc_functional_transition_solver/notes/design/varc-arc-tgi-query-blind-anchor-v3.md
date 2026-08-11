# VARC ARC-TGI query-blind anchor v3

## Supersession

This document supersedes the runtime-alias proposal in
`varc-arc-tgi-query-blind-anchor-v2.md`. The one-task alias replay exited zero
but failed the frozen numerical-neutrality gate: its prediction JSON differed
from the v2 partial artifact and collapsed from nine unique grids to one. The
cause cannot be separated from GPU nondeterminism with one pair of runs, but
the alias changes model-visible task names and is therefore not admissible.

No complete development outcome was inspected before this correction. The v2
partial task remains ineligible.

## Diagnostic-only compatibility

The v3 overlay preserves every model-visible evaluation task name,
augmentation directory name, augmentation filename, file byte, and lexical
order. The sole added path is `data/evaluation/arc.json`, because upstream
reduces an ARC-TGI task name to `arc` only when its post-candidate diagnostic
runs.

Source order gives the causal boundary:

1. the train and evaluation loaders read only
   `data/eval_color_permute_ttt_9/<original-task-id>`;
2. logits, transform inversion, voting, and prediction serialization finish;
3. only then does `analyze_data` open `data/evaluation/arc.json` to print a
   sentinel identity diagnostic.

The diagnostic alias is therefore outside the candidate computation path. It
is changed serially before each task, archived instead of deleted, and the
launcher requires exactly one GPU to prevent alias races. Each alias record,
the exact-name overlay, runner, checkpoint, blind source, and executed model
source are content-addressed.

## Frozen v3 gate

The full development run may start only if tests prove that the overlay keeps
the original task and augmentation names and bytes, the launcher rejects more
than one GPU, and a task cannot overwrite an existing prediction or alias
record. Model, checkpoint, blind cohort, 101 executed epochs, seed, ten
attempts, timeout, and all numerical hyperparameters remain unchanged.

The run must produce one canonical prediction file and one zero-exit status
per each of the 50 development tasks. Predictions are frozen and replayed
before development solutions are opened. All no-router and unopened
confirmatory boundaries from v1 remain in force.
