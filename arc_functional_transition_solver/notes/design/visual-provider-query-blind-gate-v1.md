# Query-blind static visual-provider gate v1

## Status: invalidated before scoring

The v1 cohort was invalidated during a stricter repository-wide exposure audit.
The automatic detector found task IDs in authored notes and summary artifacts,
but did not parse the semantic allowlists of the earlier M04a data-split
manifests.  Of the 20 selected IDs, 12 had been used by the earlier ARC-2 train
fold and four by its validation fold.  Four appeared only in source inventory.

All v1 processes were stopped before visual predictions were scored against
gold.  Partial provider and portfolio artifacts were moved to timestamped
experiment `trash/` directories and are infrastructure evidence only.  No v1
accuracy, coverage, or method conclusion is valid.  The corrected contract is
`visual-provider-query-blind-gate-v2.md`.

## Decision

The controller remains frozen.  The next deep-focus experiment tests whether an
independent visual test-time-training provider changes the candidate frontier on
ARC-AGI-2 development tasks.  This is a Solver-track experiment only; it cannot
support a residual-control or brain-like switching claim.

## Why this gate

The previous fixed 100-task audit placed almost all error in candidate-language
coverage, and object/code v0.3 obtained only 4 selectable tasks while spending
1,430,098 program trials.  Recent work points to visual priors, per-task
adaptation, recurrence, and perspective augmentation as complementary candidate
sources.  A static visual provider therefore has higher expected information
gain than another router experiment.

The released VARC code is not directly query-blind: its evaluation loader reads
`example["output"]`, and output dimensions affect resolution/translation
preprocessing.  This gate does not run labeled task JSON through that path.  It
replaces each test output with an exact copy of its test input, keeps gold files
outside the provider data root, freezes prediction files by content hash, and
only then scores them.

## Frozen pilot contract

- Provider: VARC-ViT-18M, upstream commit
  `bd478ecf362e6499a988b05f33223e5c5fc6a6be`.
- Offline checkpoint: `VisionARC/offline_train_ViT/checkpoint_best.pt`, expected
  SHA-256 `c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3`.
- Dataset: 20 ARC-AGI-2 public-training tasks selected without reading grid
  contents by ascending `sha256("visual-provider-v1-20260807" + NUL + task_id)`.
- Exclusions: every ARC-AGI-1 task ID and every task ID named by prior authored
  evidence or result artifacts.
- Provider-visible information: demonstration inputs/outputs and query inputs.
  Query outputs are input-derived sentinels, never gold labels.
- TTT configuration: released ARC-2 ViT settings, one TTT replica, seed 42,
  100 epochs, 10 inference perspectives, nine color permutations plus the
  released basic geometric views.  A one-task technical preflight may run first
  but may not change the 20-task cohort or hyperparameters.
- Candidate ranking: frequency across frozen perspectives; deterministic
  first-seen order breaks ties.  The selectable set is the top two grids for
  each query.
- Comparison pool: the frozen DSL/CA/scene baseline plus object/code v0.3 on the
  exact same task IDs.  No learned selector or controller is added.

## Endpoints and stop rules

Primary pilot endpoint: visual-provider unique selectable coverage.  At least
1/20 advances to a fixed 100-task provider gate; zero does not establish a true
zero rate, but stops immediate integration.  The existing full gate remains at
least 3 unique selectable tasks per 100 fixed tasks.

Secondary endpoints are raw visual coverage, standalone pass@1/pass@2,
baseline-plus-visual selectable union, invalid-grid rate, GPU-minutes per task,
and exact replay of one fixed seed.

The run is invalid if gold query output is present below the provider data root,
prediction artifacts are not frozen before scoring, source/checkpoint hashes do
not match, or the released model path cannot emit predictions without consulting
query labels.  Stop for engineering review above 40 GPU-minutes per task or on
repeated OOM/non-finite loss.

## Failure interpretation

- Query-blind execution cannot be made to run: implementation/protocol failure,
  not evidence against visual reasoning.
- Valid outputs but no unique coverage: candidate-distribution failure for this
  checkpoint/configuration; inspect distance and task-family overlap before any
  architectural conclusion.
- Raw hits lost outside top two: selection gap; only then is a verifier/ranker
  experiment justified.
- Unique selectable coverage without final portfolio gain: integration/ranking
  issue, not provider failure.

No result from this pilot authorizes GRU, MLP, bandit, or diffusion-router
training.
