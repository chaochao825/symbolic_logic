# NVARC TRM reference-scale anchor replication v1

## Decision question

Can the released NVARC Tiny Recursive Model component be reproduced on server
210 at its published model, checkpoint, data, test-time-training, and candidate
budget scale, producing a content-addressed heterogeneous-candidate anchor with
accuracy close to the authors' reported result?

This is a replication and portability gate, not a new-method comparison.  It
does not train a router and does not authorize visual-posterior repair.

## Frozen upstream identity

- NVARC repository:
  `https://github.com/ylamidon/kaggle-2025-arc-agi-2-nvarc.git`;
- NVARC commit:
  `846d0198efa752534594e321fc3289fc0a06c657`;
- TinyRecursiveModels submodule:
  `https://github.com/SamsungSAILMontreal/TinyRecursiveModels`;
- submodule commit:
  `e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9`;
- evaluation entry point: `TRM/eval-arc-k-10.py`;
- upstream source edits permitted in the scientific run: none.

Released assets:

- checkpoint dataset: `cpmpml/arc-prize-trm-031`, version 1, CC0,
  published 2025-11-09;
- checkpoint file: `step_220708`, declared size 2,159,719,349 bytes;
- evaluation dataset: `cpmpml/arc-prize-trm-evaluation-data`, version 1,
  CC0, published 2025-11-14, declared unpacked size 124,574,189 bytes.

Downloaded bytes must be hashed before execution.  Dataset and checkpoint
archives remain external to Git; only manifests, checksums, logs, metrics, and
candidate outputs enter this repository.

## Published reference

The NVARC repository reports the following command-level evaluation for
`step_220708`:

- TRM: 2 L layers, H cycles 4, L cycles 4, halt maximum 10;
- test-time training: 2,000 epochs, one evaluation after 2,000 epochs;
- global batch size: 128;
- learning-rate warmup: 200 steps;
- learning rate: 1e-4;
- EMA: enabled;
- four GPUs in the authors' portability run;
- reported `pass@1=0.0763889`, `pass@2=0.1013889`,
  `pass@100=0.1375`, and `pass@1000=0.1375`;
- reported Kaggle public score: 10.0%.

## Hardware and fairness boundary

Server 210 has four A800 80GB GPUs, but GPUs 1--3 are currently occupied by
other users.  The preregistered main run therefore uses only GPU 0 and preserves
the global batch, epochs, model, checkpoint, data, and candidate budget.  This
is a reference-scale single-device portability replication, not a bit-exact
four-GPU reproduction.  GPU sharing, model processes, wall time, peak memory,
and hardware identity must be logged.

If global batch 128 does not fit one A800, the reference run stops.  Reducing
batch size, epochs, checkpoint scale, augmentation count, or candidate count
would create a separate non-reference smoke result and cannot pass this gate.

## Split and leakage audit

Before scoring, record:

1. every puzzle identifier in the released train and test arrays;
2. duplicate IDs and content hashes within and across the two splits;
3. whether labels used during the 2,000-epoch adaptation are constructed only
   from observable demonstration transformations;
4. whether the released checkpoint's stated pretraining set includes the
   evaluation puzzle identities or query answers;
5. all overlap with the project's exposed 50-task cohort.

The upstream evaluator reads test labels to compute metrics after candidate
generation.  This access must remain inside the post-generation evaluator.
If query labels enter model updates or candidate ranking, the run is invalid.
If checkpoint training-data lineage cannot exclude public-evaluation exposure,
the result remains a replication anchor but is ineligible as clean held-out
evidence for our method.

## Execution phases

### A. Integrity preflight

- clone and detach both frozen commits;
- build a dedicated Python environment with a fully recorded package lock;
- download and hash the one checkpoint plus evaluation dataset;
- create the upstream-required symlinks without modifying source files;
- compile all Python entry points;
- load dataset metadata and checkpoint tensors on CPU;
- verify tensor keys, shapes, dtypes, finite values, and puzzle-embedding resize
  behavior.

### B. Bounded smoke

Run one deterministic, explicitly non-reference smoke that performs checkpoint
load, one train batch, and one evaluation batch on GPU 0.  It may only diagnose
environment or interface failures.  Its score cannot update the anchor claim.

### C. Reference-scale main run

Run the upstream command semantics with `CUDA_VISIBLE_DEVICES=0`, one process,
and unchanged global batch 128, 2,000 epochs, checkpoint, architecture, EMA,
and evaluator candidate budgets.  Capture stdout/stderr, environment, GPU
telemetry, wall time, peak memory, train-step count, inference-step count,
metrics, and the saved top-10 submission candidates.

Candidate construction must be replayed with the same seed.  Exact byte replay
is preferred; if CUDA kernels are nondeterministic, record per-task candidate
set overlap and metric deltas instead of claiming bit identity.

## Decision gate

The anchor passes only when all of the following hold:

1. upstream and asset identities close exactly;
2. no source patch or query-label training/ranking occurs;
3. the full reference-scale run completes with finite metrics;
4. observed pass@2 is at least 8% and within 2 percentage points of the
   published 10.1389%;
5. observed pass@1000 is within 2 percentage points of the published 13.75%;
6. the saved candidate artifact is content-addressed and reconstructs the
   reported metrics;
7. native costs and the single-device hardware deviation are fully disclosed.

`invalid` means a broken asset, source mismatch, label leakage into generation,
or incomplete cost/evidence identity.  `adverse` means a valid full run misses
the accuracy tolerance.  Environment/setup failures are implementation
failures until the frozen upstream semantics have been exercised.

Passing this gate freezes a strong candidate generator.  It does not support
our residual-control claim.  The next experiment must compare, at equal native
cost, an anchor cold restart against a visual-posterior-derived typed
object/AST proposal on a newly frozen task set.  Controller training remains
closed until that comparison yields at least 5/100 unique recoveries.
