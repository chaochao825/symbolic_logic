# Cross-Version Dataset Overlap Audit

Audited on 2026-07-11 against the commits in `external/SOURCES.lock.md`.

## Finding

Task IDs and order-insensitive semantic fingerprints show substantial cross-version
reuse:

- ARC-AGI-2 public training and ARC-AGI-1 public evaluation share 376 task IDs;
- 375 of those are semantically identical after ignoring train/test pair order;
- ARC-AGI-2 public evaluation and ARC-AGI-1 public evaluation share 6 semantically
  identical tasks.

There is no same-version ID overlap between the Phase-0 training directories and their
respective evaluation directories. The Phase-0 D4 runs are therefore valid
training-only harness smoke. However, scoring ARC-AGI-2 training necessarily uses
answers for many tasks that ARC-AGI-1 labels evaluation.

## Policy consequence

Any model, rule, ranker, controller, threshold, prompt, or analysis developed using
ARC-AGI-2 public training must not describe ARC-AGI-1 public evaluation as held out.
Sealing is defined relative to the complete data lineage, not the directory name.

Before creating a split, the project must compute a semantic task fingerprint that:

1. preserves whether a pair belongs to `train` or `test`;
2. canonicalizes each grid exactly;
3. ignores the order of pairs within each partition;
4. records whether test outputs were included;
5. compares the proposed train/dev/holdout and every external dataset used by a model.

The primary C1 protocol therefore uses a split wholly inside ARC-AGI-2 public training
and makes no held-out ARC-AGI-1 claim.
