# ARC visual learning and frontier control (2026-08-07)

## Research decision

The strongest current opportunity is not another learned router.  It is a
two-stage contribution:

1. add a genuinely independent visual candidate distribution under a
   query-blind, content-frozen protocol; then
2. compile stable structure in that distribution into typed symbolic actions
   that demonstrably enter a new candidate region.

This preserves the original motivation—specialized functional systems that can
exchange information and switch under failure—while moving the claim from a
brain metaphor to an intervention-testable computational mechanism.  The Solver
track and Mechanism track remain separate.  A better visual provider can improve
the former without supporting the latter.

## Accepted internal evidence

- The frozen 100-task audit put almost all error in candidate coverage, not
  final selection or controller choice.
- Object/code v0.3 adds exactly 3/100 unique selectable tasks, but the combined
  selectable union is only 12/100 and costs 1,430,098 program trials.
- All exact object/code v0.3 programs use the crop lane.  Copy, count, arrange,
  compose, and natural typed repair add no exact recovery.
- Residual deletion and shuffling previously left action traces unchanged.
  Consequently the controller remains frozen.

These facts rule out “train a stronger router over the same providers” as the
next experiment.  They do not imply that residuals are intrinsically
uninformative: the present failures are mostly outside the candidate language,
and the present compiler cannot map them to actions that reach a useful region.

## What the recent literature actually changes

| Line of work | Evidence and limitation | Implication here |
|---|---|---|
| [VARC / *ARC Is a Vision Problem*](https://arxiv.org/abs/2511.14761), [code](https://github.com/lillian039/VARC) | An 18M ViT with two-dimensional canvases, synthetic pretraining, perspective augmentation, and per-task TTT reports strong ARC-AGI-1 results and a much smaller ARC-AGI-2 result.  The released evaluation loader reads query outputs during preprocessing, so its public script is not directly query-blind. | Visual TTT is a plausible independent candidate source, but published predictions cannot be inserted into this system without a label-blind protocol repair and fresh scoring. |
| [LoopViT](https://arxiv.org/abs/2602.02156), [code](https://github.com/WenjieShu/LoopViT) | Weight-tied recurrent visual blocks and entropy-based early exit report 65.8% on ARC-AGI-1 for the 18M model.  This is adaptive compute *inside* one provider, not evidence of switching among heterogeneous representations. | Recurrent depth and calibrated stopping are attractive provider internals.  They do not rescue the current residual controller claim. |
| [Human-like Understanding and Solving Reasoning Loop](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Human-like_Abstract_Visual_Reasoning_via_Understanding_and_Solving_Reasoning_Loop_CVPR_2026_paper.html) | A recurrent understanding/solving loop strengthens a single neural representation. | Its loop can be nested inside a provider; it is not a substitute for content-addressed cross-representation actions or matched-cost repair tests. |
| [Think Visually, Reason Textually](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Think_Visually_Reason_Textually_Vision-Language_Synergy_in_Abstract_Reasoning_CVPR_2026_paper.html) | The paper explicitly separates global visual abstraction/verification from symbolic textual formulation/execution and reports that naive visual rendering can hurt. | This supports modality-aligned specialists and typed bridges rather than forcing every task through one representation. |
| [L-VARC](https://arxiv.org/abs/2606.12847), [code](https://github.com/GZHU-DVL/L-VARC) | Language descriptions are privileged training-time supervision and are removed at inference.  The reported ARC-AGI-2 increment is modest. | Semantic supervision may improve a visual prior, but it is a provider-training project, not an immediate controller intervention. |
| [Multi-Perspective Transformers](https://arxiv.org/abs/2605.01154) | The headline 21.7% ARC-AGI-2 evaluation number is the pretrained-only row.  In the paper's table, TTT, product-of-experts, and their combination score zero on evaluation, and the authors attribute this to overfitting. | TTT is not uniformly beneficial.  Every provider must be tested on a frozen, unseen cohort; training fit is not a proxy for candidate coverage. |
| [CompressARC](https://arxiv.org/abs/2512.06104), [code](https://github.com/iliao2345/CompressARC) | Per-task compression without pretraining reports nontrivial ARC-AGI-1 and lower ARC-AGI-2 performance at substantial per-task compute.  Its released preprocessing is query-blind on code inspection. | Compression is a credible fallback candidate source if the visual checkpoint cannot pass the protocol gate, but running both opportunistically would weaken the current single-gate decision. |
| [Tiny Recursive Models](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) and [test-time adaptation of recursive models](https://arxiv.org/abs/2511.02886) | Small recurrent networks show that depth, synthetic tasks, and per-task adaptation can matter more than raw parameter count. | Recurrence should be evaluated as adaptive provider compute and calibrated against cost, not advertised as brain-region switching. |
| [Modality-Driven Search](https://arxiv.org/abs/2606.31543) | Independent image, text, and code channels plus a holistic judge report high semi-private accuracy at high per-task cost.  The paper also reports that prescriptive iterative refinement can reduce diversity. | Preserve independent candidate channels.  Failure feedback should constrain a typed frontier, not homogenize all generators through one natural-language conversation. |
| [ARCANA](https://arxiv.org/abs/2607.09059) | The July preprint is conceptually close to the original proposal: scene graphs, DSL proposals, execution, reflective feedback, shared blackboard, and a learned meta-controller.  No public implementation or replay artifacts are linked, and the evaluation description is insufficient for this project's evidence standard. | A generic “reflective multi-agent neuro-symbolic ARC system” is no longer a defensible novelty claim.  Novelty must come from legal frontier-changing actions, content addressing, and causal matched-cost evidence. |
| [Tycho](https://arxiv.org/abs/2607.28287) on ARC-AGI-3 | The newest interactive work separates model construction, repair, use, and bypass under action cost.  It reports that more accurate repaired world models do not automatically produce the best next action. | This is strong conceptual support for *active abstraction*: the controller should value an action by expected downstream decision improvement, not residual reduction alone.  ARC-AGI-3 results are not direct ARC-AGI-2 evidence. |
| [Rethinking Visual Intelligence](https://arxiv.org/abs/2510.24448) | Video-diffusion pretraining supplies spatiotemporal priors and reports better data efficiency than a language comparison across ARC, ConceptARC, visual games, routes, and cellular automata. | A future masked/video-diffusion provider is plausible, especially for local dynamics, but it must first show independent query-blind coverage.  The paper does not establish a cross-provider repair policy. |
| [ARC-TGI](https://arxiv.org/abs/2603.05099) and [ARC-GEN](https://github.com/google/ARC-GEN) | Programmatic task families expose latent rules, controlled variation, and in ARC-TGI partially evaluated code/reasoning templates.  They also make contamination and family overlap explicit. | These generators are better suited than public-task labels for constructing typed failure injections and testing whether a certificate identifies the known corrupted representation. |
| [CogARC](https://arxiv.org/abs/2602.22408) | Human edit trajectories include direct solutions, extended exploration, and partial restarts; incorrect final outputs can converge even when trajectories differ. | This offers a defensible human-comparison target: action-sequence and restart/repair signatures.  It warns that agreement or residual reduction can converge on a shared wrong attractor and should not be treated as correctness evidence. |

The official [ARC Prize 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
similarly emphasizes test-time adaptation, refinement, synthetic data, and
ensembles.  The common denominator is a stronger task-conditioned candidate
distribution.  It is not evidence that a learned inter-provider router helps
when the correct output is absent from every provider.

Two newer open implementations raise the practical Solver-track bar.  The
[ARCgentica repository](https://github.com/symbolica-ai/arcgentica) reports
85.28% on the ARC-AGI-2 public evaluation set at $6.94/task using up to ten
sub-agents that generate and execute Python programs; it publishes logs, but
this is a self-reported public-set result rather than a private competition
score.  [Multi-LLM AB-MCTS](https://github.com/SakanaAI/ab-mcts-arc2) similarly
treats LLM calls as adaptive tree-search branches.  Both primarily expand and
select code hypotheses.  They are strong evidence that an ARC solver needs a
much broader candidate distribution, but they do not make a typed residual
bridge redundant: their agent/search traces do not by themselves establish
which failure certificate caused a legal frontier change or whether that change
beat an equal-cost restart.

The [ARC Prize 2026 rules](https://arcprize.org/competitions/2026) require
open-source, reproducible submissions and run the Kaggle evaluation without
internet access.  The current experiment is compatible with that offline
direction, but it is a 31-task development study on ARC-AGI-2 training tasks,
not a competition submission or leaderboard-comparable result.

## Evidence update from the query-blind visual gate

The static VARC provider passed a pilot and an untouched confirmation cohort.
Across 31 disjoint exposure-audited ARC-AGI-2 training tasks, it adds 13 unique
raw and 8 unique selectable solutions over the frozen portfolio plus object/code
provider.  The 95% Wilson interval for selectable coverage is 13.7%-43.2%.
This is the first direct evidence in this project that a visual provider enters
a useful complementary candidate region.

The posterior contains more information than its top-2 grids.  Thirteen query
answers occur somewhere in the raw pool, but only eight are rank 1; the five
minority answers have ranks 3, 5, 6, 14, and 16.  This matches the selection
problem emphasized by modality-driven search while preserving a crucial
difference: this project requires a replayable certificate and legal action,
not an unrestricted holistic judge.

Posterior disagreement also localizes errors: over 27 incorrect same-shape
queries, median pixel-error AUROC is 0.942 and the fixed top-10% mask obtains
5.87x recall lift over random.  Yet per-pixel modal composition creates zero
unique recovery.  The resulting insight is structural: visual uncertainty is a
useful *search-location* signal, but pixel marginals destroy correlations among
objects, transformations, and AST decisions.  The next bridge should use the
mask to constrain object/program synthesis, not directly choose replacement
colors.

## New synthesis: posterior-to-certificate bridges

The visual and symbolic approaches need not meet only at final grids.  A visual
provider emits a task-conditioned sample distribution.  Before reading query
gold, that distribution exposes stable structural statistics:

- modal output canvas and uncertainty over height/width;
- D4/color-equivariant consensus across perspectives;
- stable foreground masks, object count, topology, and relative placement;
- spatially localized disagreement between visual candidates and a symbolic
  execution; and
- entropy or cluster mass that can decide whether more provider compute is
  worthwhile.

The proposed bridge converts only stable, typed statistics into failure
certificates such as `canvas_mismatch`, `mask_mismatch`, `object_count_mismatch`,
or `equivariance_break`.  A certificate may trigger `canvas_reinfer`,
`grid_remask`, `object_rematch`, or `fill_ast_hole` only when the action is legal
for its source and target representations.  An action counts as
frontier-changing only if it produces at least one new content-addressed program
or grid after parent-pool subtraction.

This is more specific than a generic ensemble and more testable than a learned
blackboard controller:

```text
visual samples
  -> invariant posterior summary
  -> typed failure certificate
  -> legal bridge action
  -> novel content-addressed frontier
  -> demonstration-exact verification
```

The key hypothesis is not that neural uncertainty predicts task difficulty.  It
is that a query-blind posterior statistic can identify a representation change
whose matched-cost frontier contains useful candidates more often than a cold
restart.

Generator-known tasks make this bridge testable without learning from benchmark
answers.  Corrupt one latent factor at a time (canvas, object match, mask,
transform parameter, or AST node), execute the corrupted hypothesis, and ask
whether the posterior summary recovers the known failure type and selects the
corresponding legal action.  Natural ARC near misses remain a separate external
validity set; they must not be used to tune the certificate taxonomy.

## Experiment ladder

1. **Static visual-provider gate.**  On a frozen 20-task ARC-AGI-2 cohort that
   excludes ARC-AGI-1 and all previously named tasks, measure independent and
   unique raw/selectable coverage.  Freeze predictions before gold scoring.
2. **Near-miss audit.**  For every miss, record output-shape agreement, closest
   same-shape Hamming distance, candidate diversity, and oracle rank.  This
   distinguishes absent representations from repairable local errors.
3. **Bridge-only gate.**  On generator-known and natural near misses, compile
   visual summaries into typed actions.  Keep action selection deterministic;
   do not train a router.
4. **Matched-cost repair test.**  Require at least 5/100 unique recoveries and a
   gain over equal native-cost cold restart.  Require `novel_frontier_count > 0`
   for every claimed frontier change.
5. **Mechanism interventions.**  Only after step 4, test residual injection,
   bridge lesion, module lesion, provider-cost intervention, and insertion of a
   new provider with little or no retraining.
6. **Controller learning.**  Only if residual perturbations already cause the
   predicted deterministic action changes and yield at least +3/100 should
   XGBoost, a bandit, GRU, or diffusion controller be compared in that order.

## Falsification and claim boundary

- A valid visual run with zero unique coverage rejects this checkpoint/config as
  a useful provider on the pilot; it does not reject visual reasoning in
  general.
- Visual unique coverage without bridge gains supports the Solver track but not
  functional switching.
- Bridge repair gains below equal-cost restart reject the proposed bridge action
  semantics even if residual classification accuracy is high.
- Selective module and bridge lesions, predictable residual interventions, and
  cost adaptation are required for a “brain-inspired functional specialization
  and switching” claim.  No biological mechanism claim is licensed.

The near-term paper claim should therefore remain:

> On a replayable, content-addressed heterogeneous candidate DAG, execution and
> posterior failures are compiled into typed, budgeted cross-representation
> actions, and those actions are tested for causal frontier change and repair
> efficiency against matched-cost restart.
