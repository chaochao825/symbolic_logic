# USRL CVPR 2026 paper-spec reproduction: bounded results

Date: 2026-07-26

Hardware: three NVIDIA A800 80GB GPUs for parallel pilots; one A800 for probes

Seed: 20260726

## Outcome

The paper-level architecture and tensor contracts were reconstructed, the
implementation optimizes successfully on a fixed synthetic task, and strict and
paper-transductive ARC protocols are content-addressed. The bounded ARC pilots
did **not** reproduce the paper's 47.2% pass@2: every 3,000-update condition
obtained 0 exact test grids. These pilots emit one deterministic candidate, so
they are pass@1 diagnostics and must not be compared numerically to pass@2 as if
the metrics were identical.

The negative result is informative. The best shallow model learned background,
boundaries, and some foreground colours, while the deeper under-trained models
fell into a background fixed point. A cosine-mask training ablation lowered
training CE but worsened test-time rule execution when evaluated from all MASK.
Comparison with the official DRM code shows why: the ablation lacks DRM's
multi-timestep remask-and-reinject inference sampler. It is therefore labeled a
training-corruption ablation, not a DRM reproduction.

## Fixed-budget results

All rows use a 30x30 canvas, width 96, one block in UM and one in SM, batch 8,
3,000 optimizer updates, task-balanced sampling, random D4, non-background colour
permutation, and shared input/output translation. `1x1` and `4x3` denote
inner/outer recurrence. Accuracy columns use the adaptively selected step.

| Protocol / condition | Recurrence | Params | Exact | Valid cells | Non-black colours | Black / colour 0 | EOS | Shape | Mean residual | <=5 residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strict, direct CE | 1x1 | 177,121 | 0/87 | 77.84% | 44.50% | 96.69% | 88.50% | 70.11% | 30.67 | 16.09% |
| strict, CE + 0.5 contrastive | 1x1 | 177,121 | 0/87 | 74.66% | 39.43% | 93.86% | 88.87% | 65.52% | 35.06 | 18.39% |
| transductive, direct CE | 1x1 | 177,121 | 0/419 | **79.32%** | **56.30%** | 98.29% | 86.56% | **65.63%** | **53.58** | **2.86%** |
| transductive, cosine-mask training only | 1x1 | 177,265 | 0/419 | 68.75% | 31.59% | 98.59% | 84.46% | 56.32% | 80.99 | 1.67% |
| transductive, direct CE | 4x3 | 177,121 | 0/419 | 50.21% | 0.00% | 99.66% | 24.12% | 30.07% | 129.00 | 1.43% |
| transductive, cosine-mask training only | 4x3 | 177,265 | 0/419 | 55.52% | 0.002% | 99.55% | 81.88% | 60.38% | 115.27 | 1.43% |

The strict and transductive validation sets are different, so their row-to-row
delta is descriptive, not a controlled causal estimate of evaluation-demo
training. The controlled comparisons are within each protocol and recurrence.
Here, `black / colour 0` and `non-black colours` are token-level operational
classes, not a task-specific modal-background inference.

## What the ablations establish

### 1. The model is trainable, but the ARC campaign is under-scaled

On eight fixed synthetic recolouring episodes, a 95,553-parameter smoke model
improved from 4.065 to 0.348 loss, 4.69% to 98.96% valid-cell accuracy, and 0% to
75% exact grids in 100 updates. This rules out a completely broken tensor or
gradient path. It does not establish compositional ARC generalization.

The ARC pilot has only 177K parameters, 2.49% of the 7.10M paper-spec model, and
3,000 optimizer updates. Under the epoch semantics used by the closely related
official TRM training code, the paper's disclosed 100,000 epochs, 960 base tasks,
mean examples per task, and batch 768 imply on the order of 400K optimizer
updates. The pilot is therefore below 1% of that update scale and omits the
reported 876K augmented tasks, ConceptARC exposure, EMA, and distributed global
batch.

### 2. Overall cell accuracy hid black-token collapse

The deep cosine-mask run initially looked better than its depth-matched direct
run by 5.30 percentage points of valid-cell accuracy. The class breakdown shows
that both deep models predict essentially no non-black colours correctly. The
apparent gain comes from EOS/shape recovery while black-token accuracy stays near 100%. Exact match
correctly assigns both runs zero.

This is why future neural candidates must report non-black, black, EOS,
shape, residual distribution, and exact match—not a single cell average.

### 3. Partial-target corruption creates a train/test shortcut without a sampler

The shallow cosine-mask run ended at training CE 0.361 versus 0.770 for direct
CE, yet its test valid-cell accuracy was 10.58 points lower and foreground was
24.71 points lower. At training time, an average random fraction of the target
is visible in the draft; evaluation begins from all MASK. A complete DRM system
bridges this gap through 16 scheduled prediction/remasking steps and recurrent
state reinjection. The current ablation does not, so low CE reflects an easier
conditional completion objective rather than robust from-noise solution.

### 4. The paper contrastive term is not helpful at this budget

On the controlled strict split, weight 0.5 reduced valid-cell accuracy by 3.17
points, non-black accuracy by 5.07 points, and shape by 4.60 points. The paper's own
large-scale ablation reports a positive effect, so the present result should be
read as a budget interaction: representation clustering consumes useful capacity
before the small model has learned grid transformation.

### 5. Representation-consistency halting is not calibrated to correctness

Mean halt depth ranged from 1.86 to 2.00, while exact accuracy stayed zero. On
the best transductive shallow run, adaptive selection slightly increased cell
accuracy over the fixed final step but did not solve a grid. Similarity crossing
is therefore not a correctness certificate. It should be calibrated against
demo-exact reconstruction, fixed-point distance, and residual reduction, with a
fixed-depth baseline on identical trajectories.

## Paper-shape audit and cost

The independent public-spec implementation has 7,103,505 trainable parameters
and matches the supplement's three-demo tensor shapes exactly. On an A800,
warmed batch-one inference measured 0.0631 s, 0.0988 s, and 0.1728 s for one,
two, and four reasoning steps. An affine projection gives approximately 0.599 s
for 16 steps. This demonstrates that inference is feasible; it does not estimate
the multi-day training campaign reliably.

## Can the reported result be reached?

There is currently no evidence that this independent implementation can reach
47.2%, and also no evidence that the central UM/SM idea is incapable of it. The
experiment is underdetermined for two separate reasons:

1. the public paper omits the exact gate, block, episode, stochastic-halting,
   optimization schedule, EMA decay, state-gradient, and pass@2 construction;
2. the bounded campaign is far below the disclosed model, data, update, and
   hardware scale.

The correct conclusion is **architecture reconstructed; optimization path
validated; paper score not reproduced under bounded compute**.

## Recommended gated continuation

1. Run the pinned official TRM or URM code and dataset pipeline on the same
   content-addressed protocol. Do not scale USRL until the environment reproduces
   a published, code-available anchor within a declared tolerance.
2. Port the official DRM 16-timestep remask/reinjection sampler as a distinct
   provider. Require a foreground and exact-match gain over direct CE before
   retaining it.
3. Add EMA, LR warm-up/cosine decay, ConceptARC, deduplicated 1000x augmentation,
   and scaling checkpoints at 10K/30K/100K updates. Stop early if exact remains
   zero or foreground regresses.
4. At matched parameter and update budgets, ablate shared recurrence, separate
   UM/SM, the assumed rule gate, URM short convolution/truncated backpropagation,
   and fixed-point halting.
5. Produce two content-addressed neural trajectories, verify them on all demos,
   and pass their residuals into the repository's typed DSL/CA/local-repair
   controller. Report candidate oracle coverage and final pass@2 separately.

Raw artifacts are indexed in
`results/usrl_cvpr2026_reproduction/README.md`; protocol and reconstruction
assumptions are in `notes/design/usrl-paper-spec-reproduction.md`.
