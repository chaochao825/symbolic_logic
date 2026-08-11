# NVARC TRM family-disjoint anchor gate v3

## Post-run operational correction

This document supersedes only the completion-step clause in
`nvarc-trm-family-disjoint-anchor-v2.md`. It does not change the model,
dataset, seed, numerical hyperparameters, candidate ranking, accuracy metric,
or the frozen 10/100 qualification threshold.

The v2 value of 6,288 was copied from upstream `TrainState.total_steps`. Source
inspection after the run showed that this value is an estimate:

```text
int(epochs * total_groups * mean_puzzle_examples / global_batch_size)
```

It is not the number of batches emitted by `PuzzleDataset._iter_train`. The
iterator builds a 200,000-entry group order with NumPy Philox seed 1, samples
one augmented puzzle per group, packs examples into batches of 128, and drops
the final partial batch. An independent replay of those exact size decisions
emits 6,186 full batches, consumes all 200,000 group entries, and drops one
final batch of 31 examples. The completed upstream run reports step 6,186,
exits zero, evaluates all 108 batches, and writes its checkpoint and complete
top-10 submission at that step.

Because the upstream aggregate metric had already been printed when this
discrepancy was found, this is recorded as a post-run protocol correction, not
as a newly preregistered experiment. The correction is accepted only when a
content-addressed dataset-boundary audit independently reproduces all schedule
counts and proves that every training pair reconstructs to a demonstration
pair without reading test labels.

## Corrected completion contract

- metadata-estimated steps: 6,288;
- exact replayed full batches: 6,186;
- observed training steps: 6,186;
- group-order entries: 200,000;
- consumed group-order entries: 200,000;
- dropped final partial batch: 31 examples;
- evaluation batches: 108;
- tasks/queries/attempts: 100/113/10;
- process exit code: zero.

All other v2 boundaries and decision rules remain in force. A result passing
this gate is evidence only for a static candidate anchor on the synthetic,
generator-family-disjoint ARC-TGI cohort. It is not an ARC-AGI benchmark score
and does not establish residual control or functional switching.
