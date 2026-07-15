# M04a global masked-grid candidate-source contract

## Registration status

This is the frozen pre-implementation contract for the first non-symbolic candidate
source. It was frozen on 2026-07-11 before M04a model code, training, checkpoint
selection, synthetic generation results, or a public-development M04a pool existed.
Its parent checkpoints are the verified M02c/M05f result and the limited next-source
diagnostic.

Independent final reviews report `PASS, P0=0, P1=0, P2=0` for each of: data
authority/leakage and episode construction; model/sampler/numerical closure; and
artifact lineage/cost/remote execution safety. Any material semantic change after
this line requires a new contract version and renewed reviews.

The intended versioned identities are:

```text
model:       afts-grid-cmlm/v0.1
stage:       M04a_global_masked_grid_generation
sampler:     afts-maskgit-cosine/v0.1
data folds:  afts-m04a-parent-folds/v0.1
episodes:    afts-grid-cmlm-episodes/v0.1
```

M04a is a task-conditional categorical masked generator. It is inspired by
[Mask-Predict](https://arxiv.org/abs/1904.09324),
[MaskGIT](https://arxiv.org/abs/2202.04200), and absorbing-mask discrete models such
as [D3PM](https://arxiv.org/abs/2107.03006) and
[MDLM](https://arxiv.org/abs/2406.07524). It is not a claim that a 9M-parameter
Grid-CMLM is equivalent to the 8B LLaDA system used by
[ARChitects](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/).
No author-published, ARC-ready checkpoint with an acceptable data boundary and full
sampler replay was identified. The primary model is therefore trained from scratch.

The first slice measures independent full-mask generation only. It does not consume
a candidate residual, re-mask a local region, revise committed cells, perform
per-task fine-tuning, use soft-logit recursion, or learn a functional controller.
[ReMDM](https://arxiv.org/abs/2503.00307), ARChitects-style global recursion, and
residual-directed local repair remain separately registered M11/E04 baselines.

## Scientific question and claim boundary

The narrow question is:

> Under a frozen checkpoint, blind shape source, 64-lane full-mask budget, and
> closed-world cost ledger, can a small categorical masked generator add any exact
> output that is absent from the frozen DSL v0.6 pool?

The generation population is every test pair of the literal fixed 20-task
ARC-AGI-2 public-training development-smoke manifest. That ordered task/pair list is
materialized and hashed from the blind-task manifest while the oracle loader is
unavailable; no oracle-derived solved/unsolved flag may select a task. In the frozen
blind DSL v0.6 pool, 12 of the 20 tasks have `candidate_count=0`; this defines a
predeclared blind subgroup, not pool eligibility. Post-freeze oracle evaluation
reports the full 20, the DSL-zero-candidate 12, and the DSL-nonempty eight. The key
marginal diagnostic is whether M04a contributes at least one exact output among the
12 after canonical DSL-union deduplication. A replayed zero is a valid negative
result and may not be tuned away.

This checkpoint cannot establish local-repair benefit, diffusion novelty,
heterogeneous-source complementarity, learned ranking, functional switching,
holdout transfer, or public-evaluation performance. E01b remains the later matched-
budget source-complementarity gate.

## Dataset authority and leakage folds

### Pinned parents

- ARC-AGI-1: commit
  `399030444e0ab0cc8b4e199870fb20b863846f34`; its 400 `data/training`
  tasks are the sole original-parent semantic authority for ReARC. ARC-AGI-1
  evaluation is not read. The unpinned `arc_original.zip` bundled by ReARC may not
  substitute for this source.
- ARC-AGI-2: commit
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`; only `data/training` may be read.
  `data/evaluation` is sealed.
- ReARC: commit
  `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce`; `re_arc.zip` is 50,913,541
  bytes with SHA-256
  `4a7c309499f450eb47c2117fa06f3c97fe741cf6e7912579c0b5b8cc759e9ac4`.
  It contains descendants of 400 ARC-AGI-1-training parent tasks.

The ReARC archive reader admits only members matching exactly
`re_arc/tasks/<8-lowercase-hex>.json`. The archive has 806 ZIP entries, but the
accepted set must contain exactly 400 unique task members whose IDs equal the
pinned ARC-AGI-1-training ID set; `__MACOSX/._*` resource-fork entries are ignored.
Every accepted member must contain exactly 1,000 examples and every example object
must have exactly the keys `input` and `output`, each a rectangular integer-color
grid. A missing,
duplicate, extra accepted task, schema mismatch, or color outside 0-9 stops the
build.

The source audit also records every member's maximum input/output dimensions. Model
v0.1 accepts only dimensions 1-30 and quarantines an entire semantic parent if any
descendant exceeds that bound; it never drops or resamples individual examples.
The exhaustive scan found one such parent, `e26a3af2`: 129 of its 1,000 examples
contain both an input and output of height 31-39 (258 grids total; maximum 39 by 30).
That parent is excluded from both ARC-AGI-2 and ReARC training sources.

The existing ARC-AGI-2 split remains authoritative: exact-semantic singleton tasks
are sorted by `afts-e01-public-train-split-v1`, reserving the first 150 for holdout,
the next 150 for validation, and 700 for development. The fixed smoke is the same
20-task `afts-e01-public-train-development-smoke-v1` subset of development.

Before orbit and dimension quarantine, the initial ARC-AGI-2 assignment is:

- initial ARC-AGI-2 train: the 680 development parents outside the fixed smoke;
- initial ARC-AGI-2 validation: the existing 150 validation parents;
- ARC-AGI-2 holdout: all 150 parents remain unread by training and checkpoint
  selection;
- ARC-AGI-2 smoke: all 20 parents and every augmentation or generated descendant
  are excluded from training and validation.

ReARC descendants inherit a parent fold before any sample is opened:

1. if a ReARC parent ID occurs in ARC-AGI-2 training, it inherits that task's
   existing holdout, validation, development, or fixed-smoke fold;
2. otherwise compute the order-insensitive full-task semantic fingerprint from
   `task_semantic_fingerprint(..., include_test_outputs=True)`; a semantic alias of
   an ARC-AGI-2 task inherits that task's fold and records both IDs;
3. fixed-smoke and holdout parents are excluded, not reassigned;
4. validation parents may contribute only validation descendants;
5. development parents outside the smoke may contribute training descendants;
6. a ReARC parent with neither an ID nor full-semantic ARC-AGI-2 match is assigned by

```text
uint32_be(SHA256(
  "m04a-rearc-parent-split/v0.1\0" + parent_task_id
)[0:4]) mod 100
```

   with buckets 0-79 train, 80-89 validation, and 90-99 internal diagnostic.

All 20 fixed-smoke IDs are explicitly deny-listed. Ten are also ARC-AGI-1-training
parents and therefore have ReARC generators; parent inheritance must exclude those
generators before descendant sampling. Task IDs are provenance fields only and are
never model tokens or features.

Before the orbit audits, parent-ID plus full-semantic-alias inheritance yields 280
ReARC training parents (274 inherited development plus six truly external
hash-train), 56 validation
parents, 52 excluded holdout parents, ten excluded smoke parents, and two external
internal-diagnostic parents. These counts are pre-registered closure expectations;
the orbit audits may only quarantine rows, never move them into a less protected
fold. The two internal-diagnostic parents never enter training, validation,
checkpoint/model/sampler selection, or hyperparameter decisions. They may be used
only after the contract and checkpoint are frozen in an explicitly post-hoc
diagnostic, or omitted. ReARC parent `40853293` is an exact semantic alias of
ARC-AGI-2 parent `070dd51e` with train-pair order exchanged and inherits its
development fold rather than being treated as external.

Before training, `afts-task-d4-color-orbit/v0.1` compares every initial train,
validation, holdout, and smoke parent. For each of the eight D4 transforms, it
visits the declared training pairs in file order followed by declared test pairs,
and visits each input then output in row-major order. Colors receive canonical
labels 0, 1, ... on first occurrence across the complete transformed task. The
canonical task serialization includes pair roles, grid shapes, and relabeled cells;
its orbit ID is the lexicographically smallest serialization across the eight
transforms, hashed by SHA-256. Pair order is intentionally not permuted, so this is
a D4/global-color orbit audit rather than a complete near-duplicate detector.

These orbit comparisons run in an oracle-aware, privileged split-audit process,
not in the model-training or checkpoint-selection process. The privileged process
may read protected input/output pairs solely to compile the split. It emits to the
training side only a sanitized manifest containing versioned algorithms, ordered
train and clean-validation parent allowlists, quarantine parent IDs, closure counts,
source commitments, and a commitment hash for the sealed privileged audit bundle.
The training and validation loaders cannot access protected ARC files, protected
ReARC members, protected orbit IDs, output hashes, or collision payloads. The full
collision rows and protected source hashes remain in the sealed sibling bundle and
are opened only for independent evidence verification after checkpoint freeze.

The privileged compiler runs locally or in a filesystem root isolated from the
remote training process. It publishes content-addressed sanitized shards containing
only the 662 usable ARC-AGI-2 training tasks, 145 clean-validation tasks, 262 usable
ReARC training members, and 51 clean-validation members. The remote training root
contains these shards plus their sanitized manifest, never the complete ARC-AGI-2
tree, complete ReARC archive, original-parent repository, sealed audit bundle,
fixed-smoke/holdout files, or any oracle sibling. Shards are read-only; the training
environment manifest enumerates every visible data file and SHA-256. The loader
rejects any unmanifested file, out-of-allowlist parent, or hash mismatch, and a
tamper test proves that fail-closed behavior before the preflight.

A stricter sample audit, `afts-sample-d4-color-orbit/v0.1`, operates on one complete
input/output example. It applies the same D4 transform to both grids, then assigns
canonical color labels on first occurrence while visiting the input followed by the
output in row-major order. Its serialization includes both shapes and both
relabeled grids; the orbit ID is the SHA-256 of the lexicographically smallest of
the eight serializations. The audit enumerates every original ARC-AGI-2 train/test
pair in the four initial folds and all 1,000 verified ReARC descendants for every
ReARC parent. It never samples a subset.

The protected sample universe is the union of original ARC-AGI-2 holdout/smoke
pairs and all ReARC descendants of inherited holdout/smoke parents. The initial
validation universe similarly unions original ARC-AGI-2 validation pairs and all
ReARC validation descendants. First, any semantic parent in the initial validation
universe with a sample or task-orbit collision against the protected universe is
quarantined from validation in both sources. Second, every training-source semantic
parent is compared conservatively against the complete *initial* validation
universe, including validation parents just quarantined, plus the protected
universe; any collision quarantines that parent from all ARC-AGI-2 and ReARC
training sources. A row is never reassigned to another usable fold.

The exhaustive pre-registration scan fixes the following replay expectations. Five
validation parents are quarantined:

```text
3af2c5a8 4c4377d9 67a3c6ac 74dd1130 c9e6f938
```

Seventeen development/training parents are quarantined:

```text
4258a5f9 44f52bb0 46442a0e 46f33fce 6150a2bd 62c24649
67e8384a 6d0aefbc 6fa7a44f 7fe24cdd 85c4e7cd 99b1bc43
ac0a08a4 b91ae062 cce03e0d ed36ccf7 f5b8619d
```

In addition, the dimension audit quarantines training parent `e26a3af2`, for 18
training quarantines total. Thus the final usable closure must be ARC-AGI-2 train
662 and validation 145, ReARC train 262 and validation 51, with the original 150
ARC-AGI-2 holdout and 20
smoke parents excluded and the ReARC 52 holdout, ten smoke, and two internal-
diagnostic parents unused. The sealed privileged audit bundle records both orbit
algorithm versions, every collision row and source side, initial and final parent
counts, all ARC parent file hashes, all 400 ReARC member hashes plus archive hash,
and final ordered parent IDs. The 400 ARC-AGI-1-training file hashes from the pinned
original-parent authority are included explicitly. The sanitized
`data_split_manifest.json` exposes only the allowlists, quarantines, counts, source
commitments, and sealed-bundle hash described above. If the deterministic audit
does not reproduce these IDs and counts exactly, training must stop and the contract
must be revised. No fold may be changed after model training begins.

### Episode stream

Every 16-microbatch accumulation window contains exactly eight ARC-AGI-2 episodes
and eight ReARC episodes in alternating slots. The deterministic training seed is
`20260711`.

For zero-based optimizer step `s=0,...,19999` and microbatch slot `j=0,...,15`,
define `g=16*s+j` and initialize a fresh Python CPU RNG from

```text
uint64_le(SHA256(
  "afts-grid-cmlm-episodes/v0.1\0" + decimal(20260711) +
  "\0" + decimal(g)
)[0:8])
```

The resulting integer initializes `random.Random(seed_u64)` from Python 3.10's
standard library. Even `j` selects ARC-AGI-2 and odd `j` selects ReARC. RNG calls
occur in this exact order: `randrange` into the ordered eligible-parent list; an
ARC-AGI-2 `randrange` into its ordered eligible-target list or ReARC
`sample(range(1000), 4)` whose first three returned indices are demonstrations and
fourth is target; `randrange(8)` for D4; `shuffle(list(range(10)))` for the mapping
from old color to indexed shuffled value; and `shuffle(list(range(D)))` for demo
order. The training-mask draws described below then continue from this same RNG.
No mutable global RNG or DataLoader worker state may affect an episode.

D4 indices 0-3 are identity and clockwise rotations by 90, 180, and 270 degrees.
Indices 4-7 first reflect left-to-right and then apply those four rotations. The
same indexed transform is applied to every grid in the episode. These conventions
also define the eight transform implementations used by both orbit audits, although
their lexicographic minima do not depend on enumeration order.

For an ARC-AGI-2 training parent, form an eligible target list from every original
test pair plus every original training pair for which at least one other training
pair remains. Choose uniformly from that nonempty list. If the target is an original
training pair, the demonstrations are the other original training pairs. If it is an
original test pair, demonstrations are all original training pairs. Outputs from
other test pairs never become demonstrations. No source slot is skipped or
resampled, so every accumulation window contains exactly 16 valid episodes.

For ReARC, select one parent uniformly from the training fold and four distinct
verified examples from its 1,000-example task file. Three are demonstrations and one
is the pseudo-query. The source stream, parent, example indices, target selection,
and all augmentations are generated from a counter-based seed derived from the
optimizer step and microbatch slot; training replay must not depend on worker count.

Each episode applies one uniform D4 transform, one uniform bijection of all ten ARC
colors, and one uniform demonstration permutation consistently to the complete
episode. It never crops, resizes, translates, pads inside a grid, or changes a target
shape. Descendants retain the parent ID and fold. The training stream is 50/50 by
construction rather than expectation.

## Grid representation and exact architecture

Every demonstration input/output and query input is encoded as its own grid. A
shared encoder processes the grid's `GRID_CLS` token and cells; all encoded grid
memories are concatenated. The target decoder receives a partially masked target
canvas during training and a fully masked proposed canvas during deployment. It
uses bidirectional self-attention plus cross-attention to the concatenated task
memory.

Canonical tokenization and ordering are:

| Sequence | Tokens | row/column | height/width | role | pair slot |
|---|---|---|---|---|---|
| demo input | `GRID_CLS`, then row-major color cells | CLS=`30/30`; cells literal | literal `H/W` | demo input | permuted demo index 0-9 |
| demo output | `GRID_CLS`, then row-major color cells | same | same | demo output | same as paired input |
| query input | `GRID_CLS`, then row-major color cells | same | same | query input | 15 |
| target decoder | exactly `H*W` row-major color-or-MASK cells; no CLS | literal | literal | target | 15 |
| PAD | `PAD` | `30/30` | `0/0` | enclosing sequence role | enclosing pair slot |

Demonstrations are permuted first and then assigned slots 0 through `D-1`; input
and output of one demonstration remain adjacent. Encoder memory concatenation is
permuted demo 0 input then output, demo 1 input then output, and so on, followed by
query input. For every training episode, those `2*D+1` grids form exactly one
encoder batch in that canonical order, padded to the longest encoder sequence in
that episode; there is no length bucketing and no separate encoder call per grid.
PAD keys are attention-masked and PAD query states are reset to zero after embedding
and every sublayer before being discarded. The unpadded outputs are concatenated
in canonical order to form the task memory. Training and validation use one episode
at a time, so task memories are never outer-batched or length-bucketed. Decoder PAD
positions obey the same rule and never enter target self-attention as keys,
cross-attention as queries, logits, or loss.

At deployment, a task/test pair with at least one accepted shape has its memory
encoded exactly once with dropout disabled under BF16 autocast, stripped of PAD,
and cached on the GPU as BF16; its key mask is cached as Boolean. A pair with no
accepted shape is not encoded. For an actual per-shape lane batch of size `B`, both
memory and its key mask are materialized as contiguous tensors by
`unsqueeze(0).expand(B, ...).contiguous()`. All proposed shapes, all lanes, and all
12 denoising steps reuse that cache and run only the decoder; the encoder is never
rerun per lane, proposed shape, or denoising step. The maximum-context preflight
includes the single 21-grid encoder batch induced by ten 30-by-30 demonstration
input/output pairs plus one 30-by-30 query input, followed by cached-memory decoder
inference.

The model has:

- `d_model=256`, 8 attention heads, FFN width 1,024;
- pre-layer normalization, GELU, dropout 0.1;
- three shared Transformer encoder layers;
- six bidirectional Transformer decoder layers;
- 13 token embeddings: colors 0-9, `MASK`, `PAD`, and `GRID_CLS`;
- row, column, grid-height, and grid-width embeddings of size 31 each;
- four role embeddings: demonstration input, demonstration output, query input,
  and target;
- 16 pair-slot embeddings; slots 0-9 cover the observed maximum of ten
  demonstrations and slot 15 is the query/target slot;
- one 256-to-10 output head; logits for `MASK`, `PAD`, or `GRID_CLS` are impossible.

The token, row, column, height, width, role, and pair-slot tables are shared by the
encoder and decoder; duplicating decoder embeddings would violate the parameter
closure. The three encoder layers have distinct parameters but the complete
three-layer encoder is reused for every grid. The six decoder layers likewise have
distinct parameters.

Each non-PAD token vector is the direct, unscaled sum of its seven embedding rows.
There is no embedding average, `sqrt(d_model)` scale, embedding layer norm, or
embedding dropout. The Transformer layers are separately constructed PyTorch
`TransformerEncoderLayer`/`TransformerDecoderLayer` equivalents with
`batch_first=True`, `norm_first=True`, bias-enabled attention and FFN projections,
LayerNorm epsilon `1e-5`, exact GELU (`approximate="none"`), attention dropout 0.1,
and the standard 0.1 residual/FFN dropout placements. Decoder self-attention is
bidirectional and has no causal mask.

Construction order is the seven embeddings in the order token, row, column,
height, width, role, pair slot; encoder layers 0-2; decoder layers 0-5; encoder and
decoder final LayerNorms; then output head. Layers are instantiated independently in
a `ModuleList`, not cloned from one initialized layer. After construction, reset all
parameters from a fresh CPU `torch.manual_seed(20260711)`: embeddings use
`Normal(0,0.02)`; MHA input projections and every linear weight use Xavier uniform;
all projection/linear biases are zero; LayerNorm weights are one and biases zero.
The model moves to CUDA only after this explicit reset.

Rows and columns 0-29 denote cells; index 30 is the `GRID_CLS` position. Height and
width indices 1-30 are literal dimensions and 0 is reserved. PAD positions neither
attend nor enter loss. A 30 by 30 grid is never truncated. A task exceeding ten
demonstrations is an explicit unsupported-input failure, not silently shortened.

The parameter closure is pre-registered:

| Component | Parameters |
|---|---:|
| one MHA | 263,168 |
| one FFN | 525,568 |
| one encoder layer | 789,760 |
| three encoder layers | 2,369,280 |
| one decoder layer | 1,053,440 |
| six decoder layers | 6,320,640 |
| Transformer core | 8,689,920 |
| all embeddings | 40,192 |
| encoder and decoder final layer norms | 1,024 |
| 256-to-10 output head | 2,570 |
| **total** | **8,733,706** |

Unit tests must instantiate exactly 8,733,706 trainable parameters. A smaller
`d_model=128`, four-decoder-layer fallback may be explored only under a new contract;
OOM may not trigger an unrecorded architecture change or grid truncation.

## Masked training objective

The true target shape is available for training episodes. Deployment shape is a
separate blind input described below.

For a target of `N` cells:

- if `N=1`, mask the only cell and consume no mask RNG draw;
- otherwise call the episode RNG's `random()` once and mask all `N` cells iff the
  result is less than 0.1;
- if that branch is not taken, set `p=rng.random()` on `[0,1)`, mask each row-major
  cell iff its next `rng.random()` is less than `p`, and deterministically resample
  until
  `1 <= masked_count < N`.

Every rejected non-all-mask attempt redraws both `p` and all `N` row-major uniforms
in that order; it does not reuse `p`. Thus Python 3.10 `random.Random`, including the
draw count, uniquely determines the corruption.

The decoder sees true values at unmasked target positions and `MASK` elsewhere.
The only objective is mean ten-class cross entropy over masked target cells. There
is no label smoothing, auxiliary shape loss, demo-output masking, self-conditioning,
EMA, oracle residual, or task-ID prediction.

Training is fixed to:

```text
optimizer             AdamW
learning rate         3e-4
betas                 (0.9, 0.95)
weight decay          0.1
warmup                2,000 optimizer steps
post-warmup schedule  cosine decay to zero
optimizer steps       20,000
microbatch            1 episode
gradient accumulation 16
effective batch       16 episodes
precision             BF16
gradient clip         global norm 1.0
EMA                    none
validation interval   2,000 steps
```

Model parameters, gradients, and AdamW state remain FP32. Forward passes use CUDA
BF16 autocast; masked logits are converted to FP32 before cross entropy. BF16 uses
no GradScaler. One AdamW parameter group contains every trainable parameter,
including biases and normalization parameters, at the stated weight decay. AdamW
additionally fixes `eps=1e-8`, `amsgrad=False`, `maximize=False`, `foreach=False`,
`fused=False`, `capturable=False`, and `differentiable=False`.

The remote launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the Python
process starts and therefore before any CUDA initialization; the exact value is
recorded in the environment manifest. Before model construction and training,
call `torch.manual_seed(20260711)` and `torch.cuda.manual_seed_all(20260711)`.
Enable deterministic algorithms, disable cuDNN benchmarking and TF32, and force
the mathematical SDPA backend by disabling flash and memory-efficient SDPA for
both training and inference. The 100-step preflight asserts the environment
variable before the first CUDA tensor is created and executes at least one
optimizer update under deterministic mode. Every checkpoint stores and restores
Python episode RNG metadata plus PyTorch CPU and every CUDA RNG state before a
resume; it may not reseed dropout at each step.

For optimizer update `t=1,...,20000`, set the learning rate before
`optimizer.step()` to

```text
3e-4 * t / 2000
    if 1 <= t <= 2000

3e-4 * 0.5 * (1 + cos(pi * (t - 2000) / 18000))
    if 2000 < t <= 20000.
```

Consequently update 1 uses `1.5e-7`, update 2000 uses `3e-4`, update 2001 begins
cosine decay, and update 20000 uses zero. For each of 16 microbatches, divide its
mean masked-cell loss by 16 and call `backward()`. Then directly clip the FP32
global gradient norm, assign this update's learning rate, call `optimizer.step()`,
and zero gradients. No unscale operation or separate scheduler state is used.
Validation checkpoints at 2,000-step multiples are evaluated after that update.

Loss is divided by 16 before each backward pass. Checkpoints are selected only by
the lowest task-grouped masked-cell CE on the fixed validation episode manifest;
ties choose the earlier step. Validation averages targets within a parent, then
parents, so tasks with more pairs do not receive more weight. Fixed-smoke outputs,
synthetic evaluation outputs, and downstream candidate coverage may not choose a
checkpoint, step count, learning rate, or sampler setting.

Every validation pass sets `model.eval()`, traverses the stored manifest in literal
row order, and evaluates exactly one episode per encoder/decoder call with no outer
batch or length bucketing. Masked logits are converted to FP32 and cross entropy is
computed and accumulated in FP32 before target-then-parent aggregation. Validation
does not consume or alter the training dropout RNG state.

`validation_episode_manifest.jsonl` is materialized and hashed before training. For
each clean ARC-AGI-2 validation parent, its ordered targets are original test pairs
in file order followed by original training pairs in file order for which another
training demonstration remains. Each target has four views with mask fractions
0.15, 0.35, 0.65, and 1.0 in that order. For each clean ReARC validation parent and
view, all 1,000 indices are ranked by SHA-256 of namespace, parent ID, view index,
and decimal example index; the first three are demonstrations and the fourth is the
target.

For every validation episode, a Python 3.10 `random.Random` seed is the first eight
little-endian bytes of SHA-256 over
`"afts-grid-cmlm-validation/v0.1\0"`, parent ID, target descriptor, and decimal view
index separated by NULs. Its only calls are `randrange(8)`, a ten-color `shuffle`,
and a demo-index `shuffle`, using the training conventions above. After
augmentation, rank all target cells by SHA-256 of the same episode identity plus
row and column; mask exactly `ceil(fraction*N)` lowest-ranked cells. This avoids a
zero-mask validation episode and introduces no Bernoulli or dropout RNG. The
manifest stores source descriptors, transformations, literal masked coordinates,
complete episode hashes, and its own SHA-256, not merely seeds. Validation averages
views within target, targets within semantic parent across both sources, then
semantic parents.

The maximum primary training campaign is 20,000 optimizer updates on one RTX 4090
within 24 GPU-hours. GPU-hours are the sum of all wall intervals during which the
campaign holds the GPU lock, from preflight lock handshake through final checkpoint
selection, including the preflight, fresh reconstruction, primary training, all 10
validation passes, checkpoint writes/loads, and any resumed lock-held intervals.
CPU-only privileged split compilation and later candidate-pool inference are outside
this cap and receive separate ledgers. A projection or actual cumulative time above
24 hours yields `BUDGET_EXCEEDED` and an incomplete result; no partial-training
checkpoint may be selected or used for M04a generation.

The preflight is exactly 100 synthetic optimizer updates, each with 16
microbatch-one forward/backward passes, followed by one complete 12-step,
eight-lane inference pass. It reads zero dataset files, uses a versioned synthetic
constant-grid fixture, and cannot alter hyperparameters. It exercises the worst
declared context: ten 30-by-30 demo inputs and outputs plus one 30-by-30 query input,
a 30-by-30 target, math SDPA, and the cached-memory inference path. The outer wall
timer for every synthetic update starts before reconstructing all 16 typed episodes
and production mask-audit rows; each slot then uses the same production target
masking, CPU tokenization, tensor construction, host-to-device transfer, masked CE,
backward, clipping, and AdamW update functions as primary training. No pre-tokenized
training batch is reused across measured updates or microbatches. Its
optimizer/checkpoint state is discarded; the primary run reconstructs the model and
optimizer from the frozen seeds and begins at update 1. OOM or a projected budget
overrun stops the run and triggers a new contract; it does not authorize a silent
fallback. The projection includes the measured launcher-handshake interval, the
handshake-to-preflight gap, all measured preflight/campaign-overhead probes, and a
fixed five-minute conservative margin for final report construction, pure evidence
replay, and handoff to primary training. The actual handshake-to-final-selection
lock endpoint remains authoritative.

A terminal `BUDGET_EXCEEDED` diagnostic publishes a separate closed-world
`preflight-failure/` bundle with the exact 100-row ledger, training and inference
summaries, complete overhead report, budget projection, launch/runtime/lock parents,
and diagnostic-checkpoint bytes/commitment. It never publishes `preflight-gate/`,
remains ineligible for training evidence, marks the diagnostic checkpoint
unselectable, forbids a same-run retry, and returns a nonzero process status. Earlier
OOM/nonfinite/incomplete-projection failures use the strict partial-evidence branch.
PASS and full-budget-failure reports both record `training_checkpoint_writes=0` and
`diagnostic_checkpoint_writes=1`; a partial failure records the diagnostic count as
zero unless that complete write-and-verify boundary was reached.

The checkpoint probe's `write_wall_ns` ends after serialization, payload `fsync`,
content hashing, identity checks, and durable publication of the hidden staging
entry. The cost probe returns ownership of that verified but uncommitted staging
file to exact-preflight. Verified loading, fresh reconstruction, selection, immutable
snapshot verification, both inner and outer RNG restoration/cleanup, budget
projection, full-budget-failure or PASS report sealing, `PreflightFailure` or
`PreflightResult` construction and validation, and the final held-lock check all
occur while only that hidden staging name exists. Exact-preflight then publishes
the content-addressed name as its last mutation with a no-replace rename. That
constant-time namespace
operation is covered by the fixed five-minute post-preflight margin rather than
silently mixed into the measured checkpoint-write interval. The intermediate
scratch name alone makes no crash-durability claim; the atomically published PASS
or failure evidence bundle containing the verified checkpoint bytes is the durable
authority. Windows execution is a CPU semantic fixture only; production crash and
directory-handle guarantees are Linux-only.

Any ordinary projection, outer-report validation, final lock-check, or no-replace
rename failure before that commit is converted to the strict
`PROJECTION_INCOMPLETE` partial branch with `diagnostic_checkpoint_writes=0`; the
hidden staging file is not a committed diagnostic checkpoint. On every failed exact
preflight, lock release is attempted before failure publication. The failure manifest
states this attempt explicitly and treats process exit as the final descriptor-release
backstop rather than claiming that a fallible close definitely succeeded.

`afts-training-cost-ledger/v0.1` writes one row per preflight update, primary update,
validation episode, checkpoint operation, and resume segment. Rows carry phase,
step/episode/checkpoint identity; ARC-AGI-2 and ReARC episode counts; encoder and
decoder forward calls; backward calls; optimizer updates; masked-cell predictions;
CUDA-event nanoseconds; CPU and wall nanoseconds; checkpoint I/O bytes/nanoseconds;
and CUDA peak allocated/reserved bytes. The primary-run closure is exactly 20,000
optimizer updates, 320,000 microbatches, 160,000 ARC-AGI-2 episodes, 160,000 ReARC
episodes, 320,000 encoder forwards, 320,000 decoder forwards, and 320,000 backward
calls. It additionally closes 10 ordered validation passes, their manifest-derived
episode/encoder/decoder counts, and 10 checkpoint writes. Masked-cell counts are
data-dependent but must equal the sum of per-episode mask manifests.

Every measured GPU phase calls `torch.cuda.synchronize()`, resets peak memory
statistics, records CUDA start/end events, synchronizes the end event, and records a
nonoverlapping host `perf_counter_ns` interval. The run summary reports preflight,
training, validation, checkpoint, and resume/setup costs separately and jointly;
overlapping sub-intervals are never added twice. It binds the runtime source
snapshot, remote launcher SHA-256, frozen config, environment, sanitized-shard,
schema, checkpoint, and resume manifests. Exact closure and the 24-hour cap are
machine-checked before the selected checkpoint manifest can be published.
The campaign coordinator additionally checks cumulative lock-held wall time at every
optimizer-update, validation-episode, checkpoint-publication, and final-selection
boundary, and checks it once more immediately before selected-checkpoint publication.
Exactly 24 GPU-hours is allowed; 24 GPU-hours plus one nanosecond is terminal, and no
partial checkpoint is selectable.
The first lock interval starts at the full handshake acquisition timestamp, and
the selected-checkpoint manifest records the same final-selection completion
timestamp as the last lock endpoint. M04a v0.5 evidence permits zero resume
segments; resumes remain fail-closed until each segment has its own committed lock
handshake chain.

## Blind shape boundary

M04a does not predict output dimensions in v0.1. It consumes only distinct,
content-addressed per-test shapes replayed from the existing
`afts-output-shape/v0.1` M03a proposals in the blind DSL v0.6 pool. Proposal order is
preserved, duplicate shapes retain the first proposal identity, and at most the
first four shapes are accepted. The cap and every truncation are recorded without
oracle access.

No query output, query-output shape, output hash, or oracle shape may enter the
primary pool. A pair with no blind M03a shape receives `NO_SHAPE_PROPOSAL` and zero
lanes. On the current 12 zero-candidate tasks, nine have an identity-shape M03a
proposal and three have none; this is a pre-generation support count, not output
coverage. Oracle-shape generation, if ever run, is a distinct diagnostic bundle and
cannot rescue the primary result.

## Frozen 12-step, 64-lane sampler

For a test pair with `K` accepted shapes, allocate exactly 64 lanes:

| K | Lanes per ordered shape |
|---:|---|
| 1 | 64 |
| 2 | 32, 32 |
| 3 | 22, 21, 21 |
| 4 | 16, 16, 16, 16 |

For ordered shape `i`, `global_lane` is its cumulative allocation offset plus
`local_lane`. Within each shape, local lanes are evaluated in consecutive batches
`[0:8], [8:16], ...`; the final batch has its literal remaining size and is not
padded, and batches never mix shapes. Local lane 0 for every shape is greedy with
ascending color as the logit-tie break. All other lanes sample at temperature 1.0.
Every lane records the following derived CPU seed:

```text
uint64_le(SHA256(
  "afts-maskgit-cosine/v0.1\0" +
  blind_task_id + "\0" + decimal(test_index) + "\0" +
  shape_proposal_id + "\0" + decimal(local_lane)
)[0:8])
```

Inference starts with all `N` target cells masked. For steps `k=1,...,12`, predict
only cells masked at the start of the step, fill them, and retain as MASK the
lowest-confidence

```text
m_k = 0                                      if k = 12
      (N + 1) // 2                           if k = 8
      ceil(N * cos(pi * k / 24))             otherwise
```

The non-special cases use Python 3.10 binary64 `math.pi`, `math.cos`, multiplication,
and `math.ceil` in that order. The exact k=8 branch avoids the binary64
`cos(pi/3)=0.5000000000000001` boundary; in particular N=2 gives one retained mask,
not two. Unit tests enumerate all `N=1,...,900`, require monotonicity, and pin the
complete schedule-table SHA-256.

among those just predicted. Committed cells never change. Confidence is the sampled
color probability; confidence ties are resolved in row-major order. Every lane owns
a dedicated `torch.Generator(device="cpu")` initialized by
`manual_seed(seed_u64)`. Greedy lanes record the seed but consume zero random draws.
At each step, masked coordinates are visited in
row-major order. Their logits are cast to CPU float64 and converted with float64
`log_softmax`/`softmax`. A greedy lane chooses the smallest color with maximum
probability. A sampled lane draws exactly one float64 uniform per masked coordinate
with `torch.rand((), dtype=torch.float64, generator=lane_generator).item()` and
applies inverse CDF, choosing the smallest color whose cumulative probability is
strictly greater than the draw, with color 9 as the rounding fallback. This is one
row-major scalar draw sequence, not `multinomial` or a batched categorical call.
Tokens are then transferred back to the GPU. The final step commits every cell.

Any non-finite input logit makes the entire pool run incomplete with
`NONFINITE_LOGITS`; it is not converted into a normal invalid candidate. Each cell
stores the probability and log probability from the step at which it is finally
committed. `mean_log_probability` is the arithmetic mean of those final natural log
probabilities and `minimum_probability` is their minimum probability; provisional
draws that are re-masked do not enter either summary but remain covered by the trace
hash.

Inference has a maximum per-shape microbatch of eight lanes under the batching rule
above. Each pair with a shape therefore records 64 raw candidates and
`64 * 12 = 768` sample-equivalent decoder-forward calls, including duplicates.
Actual decoder-batch forward calls are `12 * sum_i ceil(shape_lanes_i / 8)`: 96 for
K=1, 2, or 4 and 108 for K=3. There is additionally exactly one encoder-batch
forward call when `K>0` and none when `K=0`. All three measures are reported and
never conflated.
No confidence threshold, DSL agreement, verifier result, candidate novelty, or
oracle match may stop a lane early. Independent lanes are full-mask cold starts, not
local repairs. Test-time D4/color ensembles, linear Mask-Predict, MDLM, ReMDM,
ARChitects recursion, per-task fine-tuning, and committed-token revision are outside
the primary sampler.

## Candidate and cost ledger

The immutable inference schemas are:

```text
lane row       afts-m04a-lane/v0.1
lane trace     afts-maskgit-trace/v0.1
encoder call   afts-encoder-forward-ledger/v0.1
decoder call   afts-decoder-forward-ledger/v0.1
pair cost      afts-m04a-pair-cost/v0.1
pool setup     afts-m04a-pool-setup-cost/v0.1
```

Every lane records at least:

```text
blind_task_id / test_index
shape_proposal_id / proposed_height / proposed_width
checkpoint_sha256 / model_semantics_version / sampler_semantics_version
global_lane / local_lane / greedy / seed_u64 / temperature
denoising_steps / mask_count_trace / trace_sha256
output_grid / output_key / duplicate_class
mean_log_probability / minimum_probability
format_status / failure_code
sample_equivalent_forward_calls / batch_forward_call_ids / masked_token_predictions
cpu_sampling_ns / lane_wall_time_ns
```

Each lane has one trace object containing ordered steps. A step stores its index,
ordered masked linear cell indices at entry, a SHA-256 over the little-endian raw
float64 logits in masked-cell then color-0-to-9 order, and one prediction record for
every entry cell. A prediction stores linear index, row/column, chosen color,
`float.hex()` probability and log probability, sampled `float.hex()` uniform or
`null` for greedy, and whether the provisional value is re-masked. The step also
stores ordered re-masked indices and the committed/masked state after the step. The
trace terminates with the final row-major grid and output key, so every provisional
draw, greedy decision, confidence tie, re-mask, and final commit is covered.

`trace_sha256` hashes the UTF-8 encoding of that trace object serialized with Python
3.10 `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)`. Timing and machine metadata are excluded from the semantic trace.
Every trace and ledger row carries its schema identity; JSONL row order is the
canonical order already declared for candidates, and the artifact manifest hashes
the exact bytes. A verifier reconstructs mask states, output grids, RNG draw counts,
summary probabilities, and all closure totals from traces rather than trusting lane
summaries.

A separate `encoder_forward_ledger.jsonl` records the one encoder call for each
task/test pair with `K>0`, with the ordered grid descriptors, padded batch shape,
unpadded memory length, cache dtype, GPU forward nanoseconds, wall nanoseconds, and
CUDA peak allocated/reserved bytes. A `K=0` pair has no encoder-ledger row and its
per-pair summary records `encoder_batch_calls=0` plus `NO_SHAPE_PROPOSAL`. A
`batch_forward_ledger.jsonl` records every actual
decoder call with test, shape, step, ordered global lanes, batch size, cached-memory
descriptor, GPU forward nanoseconds, wall nanoseconds, and CUDA peak
allocated/reserved bytes. Encoder and decoder GPU times are summed once from their
respective ledgers and reported separately and jointly; neither is copied into
every lane as if independently incurred.

For every encoder or decoder call, timing synchronizes CUDA before measurement,
calls `torch.cuda.reset_peak_memory_stats()`, brackets kernels with CUDA events,
synchronizes the end event, and records the enclosing nonoverlapping host
`perf_counter_ns` interval. `pair_cost_ledger.jsonl` measures one endpoint interval
from the first task-input transfer through the completed pair artifact serialization
and separately records H2D, D2H, encoder GPU, decoder GPU, CPU sampling, hashing,
and serialization time plus pair peak memory. Model construction, checkpoint load,
and one-time warmup occur once in `pool_setup_cost.json`; they are not amortized or
copied into pair/lane rows. Pool finalization is another nonoverlapping run-summary
interval. Per-lane wall spans may overlap because lanes share decoder batches; they
are diagnostics only and may never be summed as endpoint cost. Total pool wall time
closes from setup plus sequential pair intervals plus finalization, while GPU kernel
time closes independently from actual encoder/decoder ledgers.

Required per-pair closure includes:

```text
raw_lanes = sum(shape_lane_allocations) = 64 when K > 0 else 0
sample_equivalent_forward_calls = raw_lanes * 12
encoder_batch_calls = 1 when K > 0 else 0
decoder_batch_calls = 12 * sum_i ceil(shape_lane_allocations_i / 8)
total_actual_batch_calls = encoder_batch_calls + decoder_batch_calls
mask_trace begins at N and ends at 0
mask counts are monotonically non-increasing
masked_token_predictions_per_lane = sum(mask_trace[0:12])
format_valid + format_invalid = raw_lanes
unique_outputs + duplicate_outputs = format_valid
candidate rows = raw_lanes
```

Format validity requires the declared rectangular shape and integer colors 0-9.
Candidate and trace rows serialize in `(test_index, shape_order, local_lane)` order.
All valid lane rows enter the immutable candidate store; duplicate rows retain one
canonical output class rather than disappearing. DSL-output deduplication is
computed only after both pools are frozen. It may define marginal coverage but may
not prune, stop, or reorder M04a generation.

## Training, pool, and evaluation artifacts

The closed-world evidence lineage is:

```text
source/data lock -> sealed privileged split audit
  -> sanitized shards + split/episode manifests
reviewed runtime source/test snapshot + remote launcher + frozen schema/config
  -> environment + preflight + training cost ledger
  -> validation metrics + all checkpoint/resume manifests
  -> selected frozen checkpoint
  -> blind tasks + blind M03a shape sidecar
  -> M04a generation ledger and candidate pool
  -> post-freeze oracle evaluation and DSL-union analysis
```

The training bundle must contain source commitments and the sanitized split
manifest, ordered usable fold IDs, quarantine parent IDs, a fixed validation episode
manifest, model config, parameter count, environment manifest, seed policy,
optimizer/schedule state, per-checkpoint validation metrics, selected-checkpoint
rule, every training-cost row and summary, weight SHA-256, and measured time/memory.
It also contains the externally committed `launch_plan.json`, the full inherited
lock-handshake artifact, the complete campaign-overhead cost report, and the
retained nonselectable diagnostic preflight checkpoint. It carries the canonical
`python-runtime-lock.json` plus its byte SHA-256 and semantic runtime-lock ID through
the environment, training, pool, and evaluation lineages. The diagnostic checkpoint
may never appear in the 10 selectable checkpoint identities.
`training_artifact_manifest.json` content-addresses the complete reviewed runtime
source and test snapshot, remote launcher, every schema/config file, sanitized-shard
manifest, environment/preflight artifacts, training/validation ledgers, every
checkpoint/resume manifest, and the selected-checkpoint manifest. It must not
contain protected collision payloads or protected output-derived hashes. Checkpoint
loading uses `torch.load(..., weights_only=True)`.

The pool bundle must bind that exact training artifact manifest and selected
checkpoint artifact manifest, blind-task
artifact manifest, blind shape sidecar, source snapshot, environment, sampler
config, every raw lane, candidate rows, summary, and cost funnel. The evaluator may
read query outputs only after pool publication. It reports source-only coverage,
DSL-only coverage, canonical union coverage, M04a marginal exact outputs, coverage
by shape support, raw/valid/unique/duplicate counts, and cost.

Full pool replay means exact inference from the frozen checkpoint and blind parents
on the recorded hardware/software stack, not retraining the checkpoint on every
verification. Training reproducibility is separately supported by deterministic
data/config manifests, a short exact replay fixture, and an independently reviewed
rerun protocol. Any nondeterministic kernel must be declared; inference enables
deterministic algorithms, disables TF32 and flash/memory-efficient SDPA, and uses
the mathematical SDPA backend plus CPU sampling.

## Proposed execution target

The pre-registration target is one GPU on SSH profile `237`:

- two NVIDIA GeForce RTX 4090 GPUs are installed, each reporting 24,564 MiB;
- the read-only preflight found both GPUs idle;
- driver `580.82.07`;
- existing `mixbit` environment: Python 3.10.20, PyTorch `2.10.0+cu128`, CUDA
  runtime 12.8, cuDNN 91,002;
- the code imports only Python standard-library modules and PyTorch.

Availability is not frozen by this observation. Before execution, the exact conda
explicit list, Python/PyTorch/CUDA/cuDNN versions, host/GPU IDs, free disk, and
`nvidia-smi` state enter an environment artifact. Before any GPU query, the launcher
validates a canonical Python runtime lock against the exact CPython executable,
all installed distribution metadata, the recursively reached critical dependency
closure, every critical RECORD-owned byte, and Torch source/version provenance.
Preflight repeats the live check before importing Torch, checks the imported Torch
immediately afterward, and revalidates the bound roots before publishing its gate.
The exact remote identities are:

```text
REMOTE_PROJECT_ROOT=/srv/afts/arc_functional_transition_solver
RUN_ID=m04a-grid-cmlm-v0.1-seed20260711-r1
RUN_ROOT=/srv/afts/arc_functional_transition_solver/runs/m04a-grid-cmlm-v0.1-seed20260711-r1
```

`RUN_ROOT` is exclusive-create and an existing path is a hard failure, never an
overwrite. The reviewed source snapshot, tests, launcher, frozen contract/config,
sanitized data, and their expected hashes are named in `launch_plan.json`. Before
any remote mutation, the pinned-host-key `ssh-dev` wrapper produces a `PrintOnly`
dry-run artifact containing the exact commands, arguments, resolved roots, and
hashes; independent remote-safety review must approve it. `labctl` OpenSSH aliases
are currently unavailable on the Windows host, so this wrapper is the validated
fallback.

The externally committed plan is stored at exactly
`REMOTE_PROJECT_ROOT/launch_plan.json`, outside the visible input root. The flat
`REMOTE_PROJECT_ROOT/input` directory is a closed world containing exactly the 17
plan-bound input artifacts plus `conda-explicit.txt`; placing another plan copy or
any other file there is a hard failure. One of those 17 artifacts is
`python-runtime-lock.json`; its external SHA-256 and semantic ID are independently
reparsed by each launcher/preflight/evidence consumer rather than trusted as a
shared mutable object.

The existing `mixbit` environment is read-only: no `pip`, `conda install/update`,
or environment-file edit is allowed. The isolated neural launcher retains Python
`-I -B -S`, inserts only the recorded `mixbit` site-packages and reviewed project
source paths, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python/CUDA starts.
The environment manifest records the launcher/source/test hashes, import paths,
complete visible-file inventory, conda explicit list, exact runtime versions, and
the Python runtime-lock SHA-256/semantic ID.

Immediately before launch, a GPU is eligible only if `nvidia-smi` reports no compute
process, zero percent compute utilization, and at most 256 MiB used memory. If no
GPU qualifies, the attempt records `NO_IDLE_GPU` and stops; it never kills another
process, relaxes the thresholds, or oversubscribes a device. The lock file is
`REMOTE_PROJECT_ROOT/locks/gpu-<GPU_UUID>.lock`. A detached lock-holder acquires
`flock -n`, repeats the GPU eligibility check, and writes a PID/GPU-UUID/run-ID
handshake before it execs the training process in the same PID. The canonical
handshake additionally binds the prelaunch attempt nonce, boot ID, process start
ticks, exact lock device/inode/path, inherited descriptor, launcher SHA, launch-plan
SHA, and monotonic endpoints. The verifier never acquires a lock itself: Linux
`/proc/self/fdinfo/<fd>` must show that the specific inherited open-file description
carries the exclusive flock. Training immediately duplicates that descriptor into
a private non-inheritable lifecycle guard and retains it through final checkpoint
selection. A missing/invalid/stale handshake, wrong descriptor, or second-check
failure starts no training.

The run also fails closed unless available bytes on the `RUN_ROOT` filesystem are
at least `max(30 GiB, projected_artifact_bytes + 20 GiB)` and sufficient inodes are
available. The dry run records the projection and the launch rechecks it. Structured
`train-events.jsonl` logs every 50 optimizer updates and every validation,
checkpoint, resume, and error event. Captured stdout and stderr each rotate at
64 MiB with at most two backups. Monitoring uses only the bounded summary command
or `tail` of at most 200 lines; no unbounded log dump is allowed.

No remote file is permanently deleted. Any replacement or cleanup first verifies
that resolved source and destination remain under `REMOTE_PROJECT_ROOT`, then moves
the target into
`REMOTE_PROJECT_ROOT/trash/<UTC timestamp>-m04a/<original relative path>`.
Run-root relative structure is preserved, trash is never published, and `rm`,
`unlink`, or recursive permanent cleanup is forbidden without explicit user
approval. A real launch requires the reviewed dry run, idle-GPU and disk rechecks,
flock handshake, bounded logging, and remote review/safety gate above.

## Evidence gate

Before any M04a claim upgrade:

1. the contract receives independent data-leakage and model/sampler reviews with no
   unresolved P0, P1, or P2 findings;
2. ReARC and ARC dataset identities, fold inheritance, fixed-smoke/holdout exclusion,
   and all descendant provenance replay exactly;
3. model parameter closure is exactly 8,733,706 and maximum-size forward/backward,
   padding, masking, sampling, seed, schedule, lane-allocation, and tamper tests pass;
4. training uses only the frozen stream and selects its checkpoint only by the
   frozen validation metric within the 24-GPU-hour cap;
5. a generator-known post-freeze suite reports semantic/output coverage, D4/color
   counterfactuals, identity/nonidentity shapes, raw diversity, and exact replay;
   no positive coverage threshold is required for infrastructure validity;
6. the public M04a pool is frozen before its oracle sidecar is opened, and all 64
   lanes or explicit no-shape failures close against the cost ledger;
7. source-only, DSL-only, union, and marginal coverage are reported even when M04a
   adds zero correct outputs;
8. independent code, remote-safety, and formal-evidence audits report no unresolved
   P0, P1, or P2 findings.

Only after this gate may the checkpoint enter an E01b matched-source comparison or
serve as the starting point for M11 local repair.
