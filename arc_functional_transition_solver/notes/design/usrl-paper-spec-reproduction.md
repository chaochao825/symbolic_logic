# USRL paper-spec reproduction contract

Date: 2026-07-26

## Scope and claim boundary

This directory contains an independent, paper-spec implementation of
*Human-like Abstract Visual Reasoning via Understanding and Solving Reasoning
Loop* (CVPR 2026). No official implementation or checkpoint was linked from the
CVF paper page or found in the authors' public material during this audit.
Consequently, this is a transparent reconstruction of the public specification,
not a bit-exact reproduction and not evidence for the reported 47.2% ARC-AGI-1
pass@2 result.

The implementation is intentionally kept separate from the production hybrid
solver. It can establish tensor contracts, optimization viability, protocol
differences, and compute requirements before any expensive campaign is approved.

## Public specification implemented directly

| Item | Paper value | Implementation |
|---|---:|---|
| hidden width | 368 | `USRLConfig.paper().hidden_size` |
| attention heads | 8 | independent UM and SM recurrence stacks |
| UM / SM blocks | 2 / 2 | two blocks in each module |
| query tokens | 4 | zero-initialized learned UM queries |
| inner / outer recurrence | 4 / 3 | shared-weight recurrent blocks |
| maximum reasoning steps | 16 | iterative UM-gate-SM-draft loop |
| grid canvas | 30 x 30 | 900 tokens |
| three-demo UM input | `(3, 904, 368)` | 900 pair tokens plus four queries |
| raw rule representation | `(3, 4, 368)` | cached demo rules |
| gated rule representation | `(1, 16, 368)` | three demos plus answer rule |
| SM sequence | `(1, 916, 368)` | 900 problem/draft plus 16 rules |
| loss | CE + 0.5 contrastive | supervised contrastive temperature 0.07 |
| reported model size | approximately 7M | 7,103,505 trainable parameters |
| adaptive halt | answer consistency exceeds intra-example consistency | evaluated after re-encoding the newly generated draft |

The 1.48% parameter difference from the rounded 7M claim is within rounding
range, but it does not prove that the unpublished block and gate definitions
match the authors' code.

## Explicit reconstruction assumptions

The following details are not fixed sufficiently by the paper or supplement and
are isolated in code so that later author clarification or an official release
can replace them:

1. **Transformer block.** The recurrence uses a TRM-style pre-norm block with
   RMSNorm, RoPE attention, and SwiGLU. The paper does not specify all of these.
2. **Rule gate.** A learned scalar gate conditions every demonstration rule
   token on the current answer-rule summary and SM state, then concatenates the
   current answer rule. The paper gives the resulting tensor shape but no gate
   equation.
3. **Episode construction.** Training demonstrations are converted to supervised
   samples by leave-one-demonstration-out. The held-out pair is never present in
   its own UM context.
4. **Tokenization and shape.** PAD=0, EOS=1, ARC colours=2..11 follows the public
   TRM convention. Output shape is decoded from the PAD/EOS frontier.
5. **State and gradients.** Draft tokens and recurrent state are detached between
   outer reasoning steps. The paper does not state these boundaries.
6. **Stochastic depth.** A uniformly sampled supervised reasoning depth selects
   each sample's deep-supervision loss. This approximates, but cannot identify,
   the paper's unspecified stochastic-halting and dynamic-replacement procedure.
7. **Pass@2.** The paper does not explain how two candidates are generated or
   selected. Current pilots report deterministic fixed-depth and adaptive
   pass@1-style exact match; they must not be relabeled pass@2.

## Two non-interchangeable data protocols

`strict` sorts the 400 ARC-AGI-1 training tasks, fits on 320 tasks, and evaluates
on the test pairs of the remaining 80 tasks. It never performs gradient updates
on demonstrations belonging to validation tasks.

`paper-transductive` follows the paper's disclosure: demonstrations from all 400
training and all 400 evaluation tasks are used for gradient updates, while only
evaluation test outputs are withheld. This is a legitimate transductive task
adaptation protocol, but it is not the same claim as generalization to wholly
unseen tasks. Results from the two protocols are always labeled separately.

The content-addressed protocol audit records 1,037 fit episodes / 87 validation
episodes for `strict`, and 2,665 fit episodes / 419 validation episodes for
`paper-transductive`.

## Bounded pilot versus paper campaign

The pilot keeps the 30x30 token geometry but uses width 96, one UM block, one SM
block, two reasoning steps, batch 8, and 3,000 optimizer updates. Its purpose is
to reject broken data flow and compare mechanisms cheaply. It is roughly four
orders of magnitude below the paper's disclosed augmented training exposure and
does not include ConceptARC or the reported 876,000-task augmentation corpus.

The full paper-shape model has 7,103,505 parameters. On one NVIDIA A800 80GB,
warmed batch-one inference measured 0.0631 s at one reasoning step, 0.0988 s at
two, and 0.1728 s at four; a simple affine fit projects about 0.599 s at 16
steps. This is an inference planning probe, not a training-throughput benchmark.

## Reproduction commands

From `arc_functional_transition_solver/`:

```bash
python3 scripts/afts_arc_usrl_reproduce.py paper-audit \
  --output results/usrl_cvpr2026_reproduction/paper_architecture_audit.json

python3 scripts/afts_arc_usrl_reproduce.py protocol-audit \
  --arc-root /path/to/ARC-AGI-1/data \
  --output results/usrl_cvpr2026_reproduction/protocol_audit.json

python3 scripts/afts_arc_usrl_reproduce.py overfit-smoke \
  --device cuda --steps 100 \
  --output results/usrl_cvpr2026_reproduction/synthetic_overfit_smoke.json

python3 scripts/afts_arc_usrl_reproduce.py arc-pilot \
  --arc-root /path/to/ARC-AGI-1/data \
  --protocol strict --device cuda --steps 3000 --batch-size 8 \
  --contrastive-weight 0 --curriculum none \
  --inner-loops 1 --outer-loops 1 \
  --output results/usrl_cvpr2026_reproduction/final_strict_shallow_3000.json
```

Every generated JSON includes a claim boundary, environment receipt, seed,
configuration, sampler, augmentation description, optimization trace, and final
metrics.
