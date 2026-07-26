# USRL and recent recursive ARC reasoning

Literature audit date: 2026-07-26. Only papers and official repositories are
used for method and score claims. Reported numbers below are authors' results,
not results reproduced by this repository.

## Primary target

**USRL (CVPR 2026).** Chen et al. split reasoning into an Understanding Module
that re-encodes demonstration and answer pairs, and a Solving Module that updates
a draft under gated rule representations. A representation-consistency test
halts the loop. The paper reports 47.2% ARC-AGI-1 pass@2 with about 7M parameters
under fixed inference and 47.0% with adaptive inference. Its disclosed training
pool includes the demonstrations of all 400 ARC-AGI-1 evaluation tasks, so the
score is transductive at task level. The CVF page provides paper and supplement,
but no official code or checkpoint was linked during this audit.

- Paper: <https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Human-like_Abstract_Visual_Reasoning_via_Understanding_and_Solving_Reasoning_Loop_CVPR_2026_paper.html>
- Supplement: <https://openaccess.thecvf.com/content/CVPR2026/supplemental/Chen_Human-like_Abstract_Visual_CVPR_2026_supplemental.pdf>

## Closest reproducible anchors

| Work | Main idea relevant here | Authors' ARC result | Public implementation |
|---|---|---|---|
| TRM (2025) | two-layer shared network; latent recursion and deep supervision | about 45% ARC-AGI-1 and 8% ARC-AGI-2 with 7M parameters | <https://github.com/SamsungSAILMontreal/TinyRecursiveModels> |
| URM (2025) | recurrence itself is the key bias; short convolution and truncated backpropagation | 53.8% pass@1 ARC-AGI-1; 16.0% pass@1 ARC-AGI-2 | <https://github.com/zitian-gao/URM> |
| DRM (2026) | cosine discrete corruption plus **multi-step** recursive denoising aligns train and test trajectories | paper reports gains over TRM; its ARC-Easy ablation reaches 50.5 at four denoising recursions | <https://github.com/wwwwwwwwz/DenoisingRecursionModels> |
| Mamba-2 hybrid (2026) | replace recursive Transformer operators with parameter-matched attention/SSM hybrid | 45.88% versus 43.88% pass@2; larger gain at pass@100 | paper only found: <https://arxiv.org/abs/2602.12078> |
| CompressARC (2025) | per-puzzle inference-time training and MDL, with no pretraining | 20% ARC-AGI-1 evaluation with 76K parameters | <https://github.com/iliao2345/CompressARC> |

Primary papers:

- TRM: <https://arxiv.org/abs/2510.04871>
- URM: <https://arxiv.org/abs/2512.14693>
- DRM: <https://arxiv.org/abs/2604.18839>
- CompressARC: <https://arxiv.org/abs/2512.06104>

## Newer control and stability evidence

**Fixed-Point Reasoners** diagnose signal propagation in deeply looped models
and combine pre-norm, residual scaling, and convergence-based adaptive halting.
That is a stronger candidate for USRL stopping than a raw cross-example cosine
threshold because the stopping signal is tied to state convergence. Paper:
<https://arxiv.org/abs/2606.18206>.

**Mechanistic analysis of hierarchical reasoners** reports multiple fixed
points and abrupt grokking across reasoning steps. This cautions against assuming
that monotonic representation consistency means correctness, and motivates
multiple candidate trajectories rather than a single greedy halt. Paper:
<https://arxiv.org/abs/2601.10679>.

**ARC Prize 2025 Technical Report** identifies the refinement loop—program-space,
application-space, or weight-space iterative improvement—as the year's central
pattern. It also warns that knowledge coverage and contamination can dominate
apparently strong generalization. Paper: <https://arxiv.org/abs/2601.10904>.

## Implications for this repository

1. **Reproduce a code-available anchor first.** Exact runs of TRM, URM, and DRM
   on the same content-addressed ARC protocol are more informative than scaling
   an under-specified USRL reconstruction immediately.
2. **Keep UM as an ablation, not a premise.** Compare a shared recurrence, USRL's
   separate UM/SM, and URM's short-convolution operator at matched parameters and
   optimizer updates. This tests whether explicit rule re-encoding adds value.
3. **Use trajectory-aligned corruption.** The bounded pilot already shows that a
   one-step diffusion-style objective can hurt when inference begins from all
   MASK. DRM's multi-recursion curriculum should be used only with sufficient
   recurrent depth and warm-up states.
4. **Replace heuristic halting with calibrated control.** Track residual change,
   fixed-point distance, answer/demo consistency, entropy, and verifier evidence.
   Train a budget-aware stop/branch/repair policy and evaluate it against fixed
   depth on identical candidate trajectories.
5. **Generate pass@2 honestly.** Use two replayable stochastic trajectories or
   operator-diverse candidates, then let demo-exact verification and MDL rank
   them. A deterministic single output is pass@1, regardless of the paper label.
6. **Join neural recursion to typed repair.** The existing content-addressed
   blackboard can compile neural residuals into DSL, CA, or local repair actions.
   This turns USRL's implicit internal loop into the repository's stated
   cross-representation, budgeted reasoning loop.
